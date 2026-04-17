"""
Synthetic Handwriting Data Generator
--------------------------------------
Generates word images that look like handwriting using system fonts,
augmented to simulate real handwriting variation.

Output structure mirrors the IAM dataset so the existing dataset.py
loader can use it directly alongside real IAM data.

Usage:
    python generate_synthetic.py --count 50000 --output data/synthetic
    python generate_synthetic.py --count 10000 --output data/synthetic --preview
"""

import argparse
import csv
import os
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import config

# ---------------------------------------------------------------------------
# Fonts — handwriting/script fonts available on Windows
# ---------------------------------------------------------------------------

HANDWRITING_FONTS = [
    r"C:\Windows\Fonts\Inkfree.ttf",
    r"C:\Windows\Fonts\LHANDW.TTF",
    r"C:\Windows\Fonts\BRADHITC.TTF",
    r"C:\Windows\Fonts\comic.ttf",
    r"C:\Windows\Fonts\comicbd.ttf",
    r"C:\Windows\Fonts\comici.ttf",
    r"C:\Windows\Fonts\FREESCPT.TTF",
    r"C:\Windows\Fonts\FRSCRIPT.TTF",
    r"C:\Windows\Fonts\SCRIPTBL.TTF",
    r"C:\Windows\Fonts\KUNSTLER.TTF",
    r"C:\Windows\Fonts\BRUSHSCI.TTF",
    r"C:\Windows\Fonts\PALSCRI.TTF",
    r"C:\Windows\Fonts\MATURASC.TTF",
]

# Keep only fonts that actually exist on this machine
HANDWRITING_FONTS = [f for f in HANDWRITING_FONTS if Path(f).exists()]


# ---------------------------------------------------------------------------
# Word list — load from IAM words.txt or use a built-in fallback
# ---------------------------------------------------------------------------

def load_word_list(max_words: int = 50000) -> list:
    """Load words from IAM words.txt labels (ground truth transcriptions)."""
    words = set()

    # Primary: pull unique words from the IAM ground truth
    words_txt = config.WORDS_TXT
    if words_txt.exists():
        with open(words_txt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9 or parts[1] == "err":
                    continue
                word = parts[8]
                # Only keep words made of characters in our alphabet
                if word and all(c in config.CHAR2IDX for c in word):
                    words.add(word)

    # Supplement with the downloaded words.txt in Downloads if available
    extra = Path(r"C:\Users\raiya\Downloads\words.txt")
    if extra.exists():
        with open(extra, encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = line.strip()
                if word and all(c in config.CHAR2IDX for c in word) and 1 <= len(word) <= 15:
                    words.add(word)

    words = list(words)
    random.shuffle(words)
    print(f"[Wordlist] {len(words):,} unique words available")
    return words[:max_words]


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def make_word_image(
    word: str,
    font_path: str,
    font_size: int,
    target_h: int = config.IMG_HEIGHT,
    target_w: int = config.IMG_WIDTH,
) -> np.ndarray:
    """
    Render a word in the given font and apply handwriting-like augmentations.
    Returns a grayscale numpy array of shape (target_h, target_w) uint8.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Measure text size
    dummy = Image.new("L", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), word, font=font)
    text_w = bbox[2] - bbox[0] + 10
    text_h = bbox[3] - bbox[1] + 10

    # Render on white canvas
    canvas = Image.new("L", (text_w, text_h), color=255)
    draw   = ImageDraw.Draw(canvas)
    draw.text((5, 5), word, font=font, fill=0)

    # --- Augmentations ---

    # 1. Random slight rotation
    angle = random.uniform(-4, 4)
    canvas = canvas.rotate(angle, expand=True, fillcolor=255)

    img = np.array(canvas, dtype=np.uint8)

    # 2. Random brightness/contrast jitter
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.randint(-20, 20)    # brightness
    img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 3. Gaussian noise (paper texture)
    noise = np.random.normal(0, random.uniform(2, 8), img.shape)
    img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Occasional slight blur (simulate out-of-focus scan)
    if random.random() < 0.3:
        sigma = random.uniform(0.3, 0.8)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=sigma)

    # 5. Random horizontal stretch (simulate writing speed variation)
    if random.random() < 0.4:
        scale = random.uniform(0.85, 1.15)
        new_w = max(1, int(img.shape[1] * scale))
        img = cv2.resize(img, (new_w, img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # --- Resize to model input dimensions (same as preprocess_image) ---
    h, w = img.shape
    scale = target_h / h
    new_w = max(1, int(w * scale))
    img   = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    # Pad or crop to target_w
    if new_w < target_w:
        img = np.pad(img, ((0, 0), (0, target_w - new_w)),
                     mode="constant", constant_values=255)
    else:
        img = img[:, :target_w]

    return img


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(count: int, output_dir: Path, preview: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    if not HANDWRITING_FONTS:
        print("ERROR: No handwriting fonts found. Check font paths.")
        return

    print(f"[Generator] Using {len(HANDWRITING_FONTS)} fonts:")
    for f in HANDWRITING_FONTS:
        print(f"  {Path(f).name}")

    words = load_word_list(max_words=count * 2)
    if len(words) < count:
        # Repeat words if we don't have enough unique ones
        words = (words * ((count // len(words)) + 2))[:count]
    else:
        words = words[:count]

    labels = []   # (filename, word) pairs for the CSV

    print(f"\n[Generator] Generating {count:,} images into {output_dir} ...")

    for i, word in enumerate(words):
        font_path = random.choice(HANDWRITING_FONTS)
        font_size = random.randint(28, 48)   # vary font size for diversity

        try:
            img = make_word_image(word, font_path, font_size)
        except Exception as e:
            print(f"  Skipping '{word}': {e}")
            continue

        filename = f"syn_{i:07d}.png"
        cv2.imwrite(str(img_dir / filename), img)
        labels.append((filename, word))

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,}/{count:,} generated")

        if preview and i < 10:
            cv2.imshow(f"Preview: {word}", img)
            cv2.waitKey(500)

    if preview:
        cv2.destroyAllWindows()

    # Save labels CSV
    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "word"])
        writer.writerows(labels)

    print(f"\n[Generator] Done! {len(labels):,} images saved.")
    print(f"  Images : {img_dir}")
    print(f"  Labels : {csv_path}")
    return len(labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic handwriting data")
    parser.add_argument("--count",   type=int,  default=50000,
                        help="Number of images to generate (default: 50000)")
    parser.add_argument("--output",  type=str,  default="data/synthetic",
                        help="Output directory (default: data/synthetic)")
    parser.add_argument("--preview", action="store_true",
                        help="Show first 10 generated images")
    args = parser.parse_args()

    generate(
        count=args.count,
        output_dir=Path(args.output),
        preview=args.preview,
    )
