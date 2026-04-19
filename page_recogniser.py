"""
Page Recogniser
----------------
Takes a full page image containing multiple lines of handwriting and
transcribes it to text.

Pipeline:
    1. Preprocess  — grayscale, denoise, binarize
    2. Line segmentation  — horizontal projection profile to find text rows
    3. Word segmentation  — vertical projection within each line to find words
    4. Word recognition   — existing CRNN model on each word crop
    5. Reassembly         — words → lines → full text

Usage:
    python page_recogniser.py --image path/to/page.jpg
    python page_recogniser.py --image path/to/page.jpg --show
    python page_recogniser.py --image path/to/page.jpg --beam --beam-width 10
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from dataset import preprocess_image
from model import CRNN, _greedy_decode
from evaluate import beam_decode


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

def preprocess_page(img: np.ndarray) -> np.ndarray:
    """
    Convert a page photo to a clean binary image ready for segmentation.
    Handles phone photos with uneven lighting, shadows, and grey backgrounds.
    """
    # Convert colour photo to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Upscale small images — segmentation works poorly below ~1200px height
    h, w = gray.shape
    if h < 1200:
        scale = 1200 / h
        gray = cv2.resize(gray, (int(w * scale), 1200), interpolation=cv2.INTER_CUBIC)
    h, w = gray.shape
    # Cap width to avoid very wide images that slow everything down
    if w > 3000:
        scale = 3000 / w
        gray = cv2.resize(gray, (3000, int(h * scale)), interpolation=cv2.INTER_AREA)

    # Smooth out noise before thresholding
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold: the block size scales with image height so it works on
    # both small crops and large full-page photos
    h, w = gray.shape
    block = max(51, (h // 15) | 1)   # must be odd — the | 1 ensures that
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block, C=20
    )

    # Adaptive threshold can produce either black-on-white or white-on-black —
    # flip if needed so text is always dark on a white background
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)

    # Morphological opening removes small noise dots (paper grain, scanner specks)
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)

    return binary


# ---------------------------------------------------------------------------
# 2. Line Segmentation
# ---------------------------------------------------------------------------

def segment_lines(binary: np.ndarray, min_line_height: int = 10,
                  padding: int = 6) -> list:
    """
    Use horizontal projection profile to find text lines.
    Returns list of (y1, y2) row spans for each line.
    """
    # Invert so ink pixels = 1, background = 0
    inverted = cv2.bitwise_not(binary) // 255

    # Sum ink pixels across each row — rows with text have high sums
    row_sums = inverted.sum(axis=1)

    # Smooth the profile to merge ascenders/descenders back into their line
    # Kernel size scales with image height for consistent behaviour
    smooth_size = max(5, binary.shape[0] // 80)
    kernel = np.ones(smooth_size) / smooth_size
    row_sums_smooth = np.convolve(row_sums.astype(float), kernel, mode='same')

    # Any row above 4% of the peak count is considered a text row
    threshold = max(1, row_sums_smooth.max() * 0.04)
    in_line = row_sums_smooth > threshold

    lines = []
    in_region = False
    start = 0

    for i, val in enumerate(in_line):
        if val and not in_region:
            in_region = True
            start = i
        elif not val and in_region:
            in_region = False
            y1 = max(0, start - padding)
            y2 = min(binary.shape[0], i + padding)
            if (y2 - y1) >= min_line_height:
                lines.append((y1, y2))

    # Handle line that goes to end of image
    if in_region:
        y1 = max(0, start - padding)
        y2 = binary.shape[0]
        if (y2 - y1) >= min_line_height:
            lines.append((y1, y2))

    return lines


# ---------------------------------------------------------------------------
# 3. Word Segmentation
# ---------------------------------------------------------------------------

def segment_words(line_img: np.ndarray, min_word_width: int = 10,
                  padding: int = 4) -> list:
    """
    Use morphological dilation + vertical projection to find words in a line.
    Dilation merges spaced letters within a word into solid blobs first,
    then projection finds the gaps between words.
    Returns list of (x1, x2) column spans for each word.
    """
    inverted = cv2.bitwise_not(line_img)

    # Horizontally dilate to bridge gaps between letters within the same word.
    # Without this, natural letter spacing causes each letter to look like its own "word".
    # Kernel width scales with line height — taller lines have wider letter gaps.
    line_h = line_img.shape[0]
    dilation_w = max(6, line_h // 5)
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilation_w, 1)
    )
    dilated = cv2.dilate(inverted, kernel_dilate, iterations=1)

    # Column projection on the dilated image — now gaps only appear between words, not letters
    col_sums = (dilated // 255).sum(axis=0).astype(float)

    threshold = max(1, col_sums.max() * 0.05)
    in_word = col_sums > threshold

    words = []
    in_region = False
    start = 0

    for i, val in enumerate(in_word):
        if val and not in_region:
            in_region = True
            start = i
        elif not val and in_region:
            in_region = False
            x1 = max(0, start - padding)
            x2 = min(line_img.shape[1], i + padding)
            if (x2 - x1) >= min_word_width:
                words.append((x1, x2))

    if in_region:
        x1 = max(0, start - padding)
        x2 = line_img.shape[1]
        if (x2 - x1) >= min_word_width:
            words.append((x1, x2))

    return words


# ---------------------------------------------------------------------------
# 4. Word Recognition
# ---------------------------------------------------------------------------

def recognise_word(word_img: np.ndarray, model, device,
                   use_beam: bool, beam_width: int) -> str:
    """Run the CRNN model on a single word crop."""
    # word_img is already binary (white bg, dark text)
    arr = preprocess_image(word_img)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(tensor)   # (T, 1, C)

    lp = log_probs[:, 0, :]

    if use_beam:
        return beam_decode(lp, beam_width=beam_width)
    else:
        return _greedy_decode(lp.argmax(dim=1).cpu().tolist())


# ---------------------------------------------------------------------------
# 5. Full Page Recognition
# ---------------------------------------------------------------------------

def recognise_page(img: np.ndarray, model, device,
                   use_beam: bool = True, beam_width: int = 10,
                   verbose: bool = False, save_crops: bool = False) -> str:
    """
    Full pipeline: page image → transcribed text.

    Returns a string with newlines separating text lines.
    """
    # Step 1: Preprocess
    binary = preprocess_page(img)

    # Step 2: Find lines
    lines = segment_lines(binary)
    if verbose:
        print(f"  Found {len(lines)} line(s)")

    if not lines:
        return ""

    result_lines = []

    for line_idx, (y1, y2) in enumerate(lines):
        line_img = binary[y1:y2, :]

        # Step 3: Find words in this line
        words = segment_words(line_img)
        if verbose:
            print(f"  Line {line_idx + 1}: {len(words)} word(s)")

        if not words:
            continue

        line_words = []
        for word_idx, (x1, x2) in enumerate(words):
            word_crop = line_img[:, x1:x2]

            # Skip crops that are too small to be a real word
            if word_crop.shape[1] < 15 or word_crop.shape[0] < 10:
                continue

            # Optionally save crops for debugging
            if save_crops:
                Path("debug_crops").mkdir(exist_ok=True)
                cv2.imwrite(f"debug_crops/line{line_idx}_word{word_idx}.png", word_crop)

            # Step 4: Recognise word
            pred = recognise_word(word_crop, model, device, use_beam, beam_width)
            if verbose:
                print(f"    Word {word_idx}: '{pred}'")
            # Filter out results that are purely punctuation/noise
            clean = pred.strip().strip('.,!?;:\'"()[]#@-_')
            if clean:
                line_words.append(pred.strip())

        # Only keep lines with at least one real word (length > 1)
        real_words = [w for w in line_words if len(w.strip('.,!?;:\'"()[]#@-_')) > 1]
        if real_words:
            result_lines.append(" ".join(line_words))

    # Step 5: Reassemble
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def visualise_segmentation(img: np.ndarray, output_path: str = "segmentation_debug.png"):
    """
    Save a debug image showing detected lines and words with bounding boxes.
    """
    binary = preprocess_page(img)
    lines = segment_lines(binary)

    # Convert binary to colour for drawing
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    colours = [(255, 0, 0), (0, 200, 0), (0, 0, 255),
               (200, 200, 0), (0, 200, 200), (200, 0, 200)]

    for line_idx, (y1, y2) in enumerate(lines):
        colour = colours[line_idx % len(colours)]
        cv2.rectangle(vis, (0, y1), (vis.shape[1], y2), colour, 2)

        line_img = binary[y1:y2, :]
        words = segment_words(line_img)

        for x1, x2 in words:
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 1)

    cv2.imwrite(output_path, vis)
    print(f"Segmentation debug saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> CRNN:
    model = CRNN().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model: {checkpoint_path}  (epoch {ckpt['epoch']})")
    return model


def main():
    parser = argparse.ArgumentParser(description="Recognise handwriting on a full page")
    parser.add_argument("--image",      required=True,  help="Path to page image")
    parser.add_argument("--checkpoint", default=str(config.CKPT_DIR / "best.pt"))
    parser.add_argument("--beam",       action="store_true", default=True)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--show",        action="store_true",
                        help="Show segmentation debug image")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save individual word crops to debug_crops/")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read image: {args.image}")
        return

    if args.show:
        visualise_segmentation(img)

    print("\nRecognising page...\n")
    text = recognise_page(img, model, device,
                          use_beam=args.beam,
                          beam_width=args.beam_width,
                          verbose=args.verbose,
                          save_crops=args.save_crops)

    print("=" * 50)
    print(text)
    print("=" * 50)


if __name__ == "__main__":
    main()
