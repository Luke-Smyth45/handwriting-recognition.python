"""
Single-image inference script.

Usage
-----
    python inference.py --image path/to/word.png
    python inference.py --image path/to/word.png --beam --beam-width 10
    python inference.py --image path/to/word.png --checkpoint checkpoints/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from dataset import preprocess_image
from model import CRNN, _greedy_decode
from evaluate import beam_decode


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def predict(
    image_path: str | Path,
    model: CRNN,
    device: torch.device,
    use_beam: bool = False,
    beam_width: int = config.BEAM_WIDTH,
) -> str:
    """
    Load a word image, run the model, return decoded string.

    Args:
        image_path : path to a grayscale (or colour) word image
        model      : loaded CRNN
        device     : torch device
        use_beam   : use beam search (True) or greedy (False)
        beam_width : beam width when use_beam=True
    Returns:
        Predicted transcription string.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    arr = preprocess_image(img)                         # (1, H, W)  float32
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 1, H, W)

    model.eval()
    with torch.no_grad():
        log_probs = model(tensor)   # (T, 1, C)

    lp = log_probs[:, 0, :]        # (T, C)

    if use_beam:
        return beam_decode(lp, beam_width=beam_width)
    else:
        return _greedy_decode(lp.argmax(dim=1).cpu().tolist())


# ---------------------------------------------------------------------------
# Visualise prediction
# ---------------------------------------------------------------------------

def visualise(image_path: str | Path, prediction: str):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(8, 2))
    plt.imshow(img, cmap="gray")
    plt.title(f'Prediction: "{prediction}"', fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict handwriting from a word image")
    parser.add_argument("--image",      required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default=str(config.CKPT_DIR / "best.pt"),
                        help="Model checkpoint path")
    parser.add_argument("--beam",       action="store_true",
                        help="Use beam search decoding")
    parser.add_argument("--beam-width", type=int, default=config.BEAM_WIDTH)
    parser.add_argument("--show",       action="store_true",
                        help="Display image with prediction (requires display)")
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load model
    model = CRNN().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Predict
    result = predict(args.image, model, device,
                     use_beam=args.beam, beam_width=args.beam_width)

    mode = f"beam(w={args.beam_width})" if args.beam else "greedy"
    print(f'[{mode}]  "{result}"')

    if args.show:
        visualise(args.image, result)


if __name__ == "__main__":
    main()
