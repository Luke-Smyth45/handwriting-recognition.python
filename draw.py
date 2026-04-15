"""
Interactive drawing GUI for handwriting recognition.

Draw a word with your mouse, then click Predict to transcribe it.

Usage
-----
    python draw.py
    python draw.py --checkpoint checkpoints/best.pt
    python draw.py --beam --beam-width 10
"""

import argparse
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import config
from dataset import preprocess_image
from model import CRNN, _greedy_decode
from evaluate import beam_decode


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_from_pil(pil_img: Image.Image, model, device, use_beam, beam_width) -> str:
    """Convert a PIL image to grayscale, preprocess, and run the model."""
    gray = np.array(pil_img.convert("L"))

    # Invert if background is dark (canvas draws white-on-black by default here)
    if gray.mean() < 128:
        gray = 255 - gray

    arr = preprocess_image(gray)                              # (1, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)   # (1, 1, H, W)

    model.eval()
    with torch.no_grad():
        log_probs = model(tensor)   # (T, 1, C)

    lp = log_probs[:, 0, :]        # (T, C)

    if use_beam:
        return beam_decode(lp, beam_width=beam_width)
    else:
        return _greedy_decode(lp.argmax(dim=1).cpu().tolist())


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class DrawApp:
    CANVAS_W = 640
    CANVAS_H = 160
    PEN_WIDTH = 12   # thick pen to mimic real handwriting stroke width

    def __init__(self, root: tk.Tk, model, device, use_beam: bool, beam_width: int):
        self.root       = root
        self.model      = model
        self.device     = device
        self.use_beam   = use_beam
        self.beam_width = beam_width

        root.title("Handwriting Recognition — Draw a word")
        root.resizable(False, False)

        # PIL image that mirrors the canvas (used for inference)
        self._pil_img  = Image.new("RGB", (self.CANVAS_W, self.CANVAS_H), "white")
        self._pil_draw = ImageDraw.Draw(self._pil_img)

        self._build_ui()
        self._last_xy = None

    # ------------------------------------------------------------------
    def _build_ui(self):
        # Instructions
        tk.Label(
            self.root,
            text="Draw a single handwritten word below, then click  Predict",
            font=("Helvetica", 12),
            pady=6,
        ).pack()

        # Canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.CANVAS_W,
            height=self.CANVAS_H,
            bg="white",
            cursor="pencil",
            relief=tk.SUNKEN,
            bd=2,
        )
        self.canvas.pack(padx=12, pady=(0, 8))

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",        self._on_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_release)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=(0, 8))

        tk.Button(
            btn_frame, text="Predict", width=12, bg="#4a9eff", fg="white",
            font=("Helvetica", 11, "bold"), command=self._predict,
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            btn_frame, text="Clear", width=12,
            font=("Helvetica", 11), command=self._clear,
        ).pack(side=tk.LEFT, padx=8)

        # Result label
        self.result_var = tk.StringVar(value='Click "Predict" after drawing')
        tk.Label(
            self.root,
            textvariable=self.result_var,
            font=("Helvetica", 16, "bold"),
            fg="#222",
            pady=8,
        ).pack()

    # ------------------------------------------------------------------
    # Drawing callbacks
    # ------------------------------------------------------------------

    def _on_press(self, event):
        self._last_xy = (event.x, event.y)

    def _on_drag(self, event):
        if self._last_xy is None:
            return
        x0, y0 = self._last_xy
        x1, y1 = event.x, event.y
        r = self.PEN_WIDTH // 2

        # Draw on tkinter canvas
        self.canvas.create_line(
            x0, y0, x1, y1,
            fill="black", width=self.PEN_WIDTH,
            capstyle=tk.ROUND, smooth=True,
        )

        # Mirror onto PIL image
        self._pil_draw.line([x0, y0, x1, y1], fill="black", width=self.PEN_WIDTH)

        self._last_xy = (x1, y1)

    def _on_release(self, event):
        self._last_xy = None

    # ------------------------------------------------------------------
    # Predict / Clear
    # ------------------------------------------------------------------

    def _predict(self):
        self.result_var.set("Running model…")
        self.root.update_idletasks()

        prediction = predict_from_pil(
            self._pil_img, self.model, self.device,
            self.use_beam, self.beam_width,
        )

        mode = f"beam(w={self.beam_width})" if self.use_beam else "greedy"
        self.result_var.set(f'[{mode}]  "{prediction}"')

    def _clear(self):
        self.canvas.delete("all")
        self._pil_img  = Image.new("RGB", (self.CANVAS_W, self.CANVAS_H), "white")
        self._pil_draw = ImageDraw.Draw(self._pil_img)
        self.result_var.set('Click "Predict" after drawing')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Draw-and-predict handwriting GUI")
    parser.add_argument("--checkpoint", default=str(config.CKPT_DIR / "best.pt"))
    parser.add_argument("--beam",       action="store_true")
    parser.add_argument("--beam-width", type=int, default=config.BEAM_WIDTH)
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = CRNN().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt['epoch']})")

    root = tk.Tk()
    DrawApp(root, model, device, use_beam=args.beam, beam_width=args.beam_width)
    root.mainloop()


if __name__ == "__main__":
    main()
