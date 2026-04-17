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
from tkinter import filedialog
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import config
from dataset import preprocess_image
from model import CRNN, _greedy_decode
from evaluate import beam_decode
from page_recogniser import recognise_page, load_model, visualise_segmentation


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def simulate_scanned(gray: np.ndarray) -> np.ndarray:
    """
    Make mouse-drawn strokes look more like scanned handwriting.
    Keep it minimal — over-processing hurts more than it helps.
    """
    # Thicken strokes to match pen-on-paper thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray = cv2.dilate(gray, kernel, iterations=1)

    # Very slight blur to soften the perfectly sharp digital edges
    gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.5)

    return gray


def clean_real_image(gray: np.ndarray) -> np.ndarray:
    """
    Clean up a real photo of handwriting so it looks like a clean scan.
    Handles uneven lighting, shadows, and colour variation.
    """
    # Adaptive thresholding handles uneven lighting/shadows from phone photos
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=15
    )
    # Mild dilation to strengthen thin strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    gray = cv2.dilate(gray, kernel, iterations=1)
    return gray


def predict_from_array(gray: np.ndarray, model, device, use_beam, beam_width,
                       is_real_image: bool = False) -> str:
    """Run model on a grayscale numpy array."""
    # Invert if background is dark
    if gray.mean() < 128:
        gray = 255 - gray

    if is_real_image:
        gray = clean_real_image(gray)
    else:
        gray = simulate_scanned(gray)

    # Save debug image so we can see exactly what the model receives
    debug_preprocessed = preprocess_image(gray)  # (1, H, W) in [-1, 1]
    debug_vis = ((debug_preprocessed[0] + 1) * 127.5).astype(np.uint8)
    cv2.imwrite("debug_model_input.png", debug_vis)

    arr = debug_preprocessed
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)   # (1, 1, H, W)

    model.eval()
    with torch.no_grad():
        log_probs = model(tensor)   # (T, 1, C)

    lp = log_probs[:, 0, :]        # (T, C)

    if use_beam:
        return beam_decode(lp, beam_width=beam_width)
    else:
        return _greedy_decode(lp.argmax(dim=1).cpu().tolist())


def predict_from_pil(pil_img: Image.Image, model, device, use_beam, beam_width,
                     is_real_image: bool = False) -> str:
    """Convert a PIL image to grayscale and run the model."""
    gray = np.array(pil_img.convert("L"))
    return predict_from_array(gray, model, device, use_beam, beam_width, is_real_image)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class DrawApp:
    CANVAS_W = 640
    CANVAS_H = 160
    PEN_WIDTH = 18   # thicker pen to better match training image stroke widths

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
        self._loaded_real_image = False
        self._real_img_array = None   # stores original cv2 array for loaded images

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
            btn_frame, text="Load Word", width=12, bg="#6c757d", fg="white",
            font=("Helvetica", 11), command=self._load_image,
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            btn_frame, text="Load Page", width=12, bg="#28a745", fg="white",
            font=("Helvetica", 11, "bold"), command=self._load_page,
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

        if self._loaded_real_image and self._real_img_array is not None:
            # Use original full-resolution array — skip canvas resize
            prediction = predict_from_array(
                self._real_img_array, self.model, self.device,
                self.use_beam, self.beam_width, is_real_image=True,
            )
        else:
            prediction = predict_from_pil(
                self._pil_img, self.model, self.device,
                self.use_beam, self.beam_width, is_real_image=False,
            )

        mode = f"beam(w={self.beam_width})" if self.use_beam else "greedy"
        self.result_var.set(f'[{mode}]  "{prediction}"')

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select a handwritten word image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return

        # Load image and display it on the canvas
        img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            self.result_var.set("Could not load image.")
            return

        # Convert to PIL and resize to fit canvas
        pil_loaded = Image.fromarray(img_cv).convert("RGB")
        pil_loaded = pil_loaded.resize((self.CANVAS_W, self.CANVAS_H), Image.LANCZOS)

        # Update the internal PIL image used for inference
        self._pil_img  = pil_loaded
        self._pil_draw = ImageDraw.Draw(self._pil_img)

        # Show it on the tkinter canvas
        from PIL import ImageTk
        self._tk_img = ImageTk.PhotoImage(pil_loaded)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        self._loaded_real_image = True
        self._real_img_array = img_cv   # keep original resolution for inference
        self.result_var.set('Image loaded — click "Predict"')

    def _load_page(self):
        path = filedialog.askopenfilename(
            title="Select a full page handwriting image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return

        img_cv = cv2.imread(path)
        if img_cv is None:
            self.result_var.set("Could not load image.")
            return

        # Show the page image on canvas (scaled to fit)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_page = Image.fromarray(img_rgb)
        pil_page_resized = pil_page.resize((self.CANVAS_W, self.CANVAS_H), Image.LANCZOS)

        from PIL import ImageTk
        self._tk_img = ImageTk.PhotoImage(pil_page_resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        self.result_var.set("Recognising page... please wait")
        self.root.update_idletasks()

        # Run full page recognition
        text = recognise_page(
            img_cv, self.model, self.device,
            use_beam=self.use_beam,
            beam_width=self.beam_width,
            verbose=False,
        )

        if not text.strip():
            self.result_var.set("No text detected. Try a clearer image.")
            return

        # Show result in a popup window
        self._show_page_result(text)
        self.result_var.set("Page recognised — see result window")

    def _show_page_result(self, text: str):
        """Show the full page transcription in a scrollable popup window."""
        win = tk.Toplevel(self.root)
        win.title("Page Transcription")
        win.geometry("600x400")
        win.resizable(True, True)

        tk.Label(win, text="Transcribed Text:", font=("Helvetica", 12, "bold"),
                 pady=8).pack()

        frame = tk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_box = tk.Text(frame, font=("Courier", 12), wrap=tk.WORD,
                           yscrollcommand=scrollbar.set)
        text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_box.yview)

        text_box.insert(tk.END, text)
        text_box.config(state=tk.DISABLED)

        tk.Button(win, text="Copy to Clipboard", font=("Helvetica", 11),
                  command=lambda: [win.clipboard_clear(),
                                   win.clipboard_append(text)]).pack(pady=(0, 8))

    def _clear(self):
        self.canvas.delete("all")
        self._pil_img  = Image.new("RGB", (self.CANVAS_W, self.CANVAS_H), "white")
        self._pil_draw = ImageDraw.Draw(self._pil_img)
        self._loaded_real_image = False
        self._real_img_array = None
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
    model  = load_model(args.checkpoint, device)

    root = tk.Tk()
    DrawApp(root, model, device, use_beam=args.beam, beam_width=args.beam_width)
    root.mainloop()


if __name__ == "__main__":
    main()
