# Handwriting Recognition AI

A deep learning system for recognising handwritten text — both individual words and full pages of handwriting.
Built with PyTorch using a CRNN (Convolutional Recurrent Neural Network) trained on the IAM Handwriting Database.

> **Branch `Rayans-Model`** — Python/PyTorch implementation by Rayan Rakib  
> Best model: **7.51% CER / 22.3% WER** on the IAM test set

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Background & Design Decisions](#background--design-decisions)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Training History](#training-history)
6. [Results](#results)
7. [Setup & Installation](#setup--installation)
8. [How to Run](#how-to-run)
9. [File Structure](#file-structure)
10. [Technical Details](#technical-details)
11. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Project Overview

This project takes a photograph or scan of handwritten text and transcribes it to a string.
It handles both individual word crops and full pages with multiple lines.

**Full-page recognition pipeline:**
1. **Preprocess** — grayscale, denoise (Gaussian blur), adaptive binarization that handles grey/textured paper from phone photos
2. **Line segmentation** — horizontal projection profile finds text rows
3. **Word segmentation** — horizontal morphological dilation + vertical projection finds individual words within each line
4. **Word recognition** — CRNN model run on each word crop
5. **Reassembly** — words → lines → full text string

---

## Background & Design Decisions

### The IAM Handwriting Database
The IAM Handwriting Database is a standard benchmark dataset for handwriting recognition research, originally collected at the University of Bern, Switzerland. It contains handwritten English text from 657 different writers, with images scanned at 300 DPI on white paper. The word-level subset (used here) contains approximately 115,000 labelled word images across all splits. It is the most widely used dataset for this task and allows direct comparison of results against published academic work.

### Why CRNN?
CRNN (Convolutional Recurrent Neural Network) was chosen because handwriting recognition is fundamentally a sequence problem — a word image is a sequence of visual strokes that maps to a sequence of characters. A pure CNN would treat the image as a fixed spatial pattern and cannot naturally produce a variable-length output sequence. A pure RNN has no efficient way to extract spatial features from an image. CRNN combines both:
- The **CNN** extracts visual features from the image (stroke shapes, curves, pen direction)
- The **RNN (BiLSTM)** reads those features as a left-to-right sequence, capturing how characters flow and connect

This architecture was first proposed by Shi et al. (2015) in "An End-to-End Trainable Neural Network for Image-based Sequence Recognition" and remains the standard approach for word-level handwriting recognition.

### Why CTC Loss?
CTC (Connectionist Temporal Classification) is used instead of standard cross-entropy because handwriting does not come with character-level alignment. When training, we only know what the word says ("hello") — we do not know which pixel columns correspond to which letters. Cross-entropy would require this alignment. CTC solves this by summing over all possible ways the output sequence could be aligned to the label, so the model learns to recognise characters without needing per-character position labels. It also naturally handles characters of different widths (e.g. 'i' is much narrower than 'w') and spaces between characters.

### Why Bidirectional LSTM?
A standard (unidirectional) LSTM reads the sequence left to right, so when predicting a character it only knows what came before it. A **bidirectional** LSTM runs two passes — one left-to-right and one right-to-left — and combines both. This is important in handwriting because the shape of a letter is often ambiguous without context from the letters around it. For example, a poorly written 'a' might look like a 'u' in isolation, but the surrounding letters make the correct reading clear.

### Why OneCycleLR?
OneCycleLR is a learning rate schedule that starts low, ramps up to a peak over the first ~10% of training, then gradually decreases to near zero using a cosine curve. Compared to a fixed learning rate, this consistently reaches lower loss values faster because the warm-up phase stabilises early training and the annealing phase allows fine-grained convergence at the end.

---

## Architecture

### CRNN (Convolutional Recurrent Neural Network)

```
Input: (B, 1, 32, 128)  — grayscale word image, height=32, width=128

CNN Backbone
  Block 1: Conv(1→64)   × 2  + MaxPool(2,2)   → (B,  64, 16, W/2)
  Block 2: Conv(64→128) × 2  + MaxPool(2,2)   → (B, 128,  8, W/4)
  Block 3: Conv(128→256)× 3  + MaxPool(2,1)   → (B, 256,  4, W/4)
  Block 4: Conv(256→512)× 3  + MaxPool(4,1)   → (B, 512,  1, W/4)
                                               ↑ height fully collapsed to 1

Map-to-Sequence: squeeze + permute → (T, B, 512)   T = W/4 = 32 time steps

BiLSTM Stack
  2 stacked bidirectional LSTM layers
  Hidden units: 256 per direction (512 combined)
  Dropout: 0.3 between layers

Linear Head: 512 → 96 classes (95 printable ASCII + CTC blank)

Output: (T, B, 96)  — log-softmax scores, fed to CTC loss
```

**Parameter count:** ~7.2M trainable parameters

**Conv-BN-ReLU blocks** with Kaiming weight init for CNN and Xavier for the linear head.

**CTC Decoding strategies:**
- **Greedy** — argmax per time step, collapse repeated tokens, remove blanks. Fast.
- **Beam search** — prefix beam search, configurable width (default 5–10). More accurate on ambiguous strokes. Pure Python, no external language model.

---

## Dataset

### IAM Handwriting Database (Local, Kaggle) — Runs 1 & 2
- ~38,000 word-level images from scanned handwriting by 657 different writers
- Split by **writer ID** (writer-independent): no writer appears in more than one split
  - Train: ~29,800 samples (~80% of writers)
  - Val: ~3,700 samples (~10% of writers)
  - Test: ~7,500 samples (~10% of writers)
- Approximate because the 80/10/10 ratio applies to the number of writers, and writers vary in how many samples they contributed
- Source: `nibinv23/iam-handwriting-word-database` on Kaggle

### HuggingFace IAM Dataset (Runs 3 & 4)
- `priyank-m/IAM_words_text_recognition` — 115,318 total word samples
- Comes with pre-defined splits (60/20/20):
  - Train: **69,190 samples**
  - Val: **23,064 samples**
  - Test: **23,064 samples**
- Larger and cleaner than the Kaggle version
- Loaded via `dataset_hf.py`, auto-cached by HuggingFace

### Synthetic Data (Run 4)
- 50,000 word images generated from 13 Windows handwriting fonts:
  `Inkfree`, `Comic Sans MS`, `Bradley Hand ITC`, `Segoe Print`, `Segoe Script`,
  `Lucida Handwriting`, `Mistral`, `Brush Script MT`, `Kristen ITC`, `MV Boli`,
  `Freestyle Script`, `French Script MT`, `Gigi`
- Generated with random backgrounds, stroke widths, and augmentations
- Script: `generate_synthetic.py` — outputs to `data/synthetic/`
- Combined with HF IAM via `dataset_combined.py` (ConcatDataset)

### Vocabulary
95 printable ASCII characters + CTC blank token (index 0):
```
 !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ
abcdefghijklmnopqrstuvwxyz
```
Total: **96 classes**

---

## Training History

All runs used: **Adam optimizer, OneCycleLR scheduler, AMP (fp16), gradient clipping (max norm 5.0), weight decay 1e-4**

Hardware: **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM), CUDA 11.8**

---

### Run 1 — Baseline
- **Dataset:** Local IAM Kaggle (~38k samples)
- **Epochs:** 50, Batch size: 64, LR: 3e-4
- **Augmentation:** rotation ±5°, scale ±15%, horizontal shear ±0.15, brightness ±40, Gaussian noise σ=8
- **Result:** CER ~16.6% — model learned the task but struggled heavily with real handwriting due to domain gap

---

### Run 2 — Stronger Augmentation
- **Dataset:** Local IAM Kaggle (~38k samples)
- **Epochs:** 50
- **Changes from Run 1:**
  - Added **elastic distortion** (α=12, σ=4) applied 50% of the time
  - Added **random erosion/dilation** (30% probability) — simulates pen width variation
  - Increased `RNN_DROPOUT` from 0.1 → 0.3
- **Result:** CER ~12% — less overfitting, meaningfully better on unseen writers

---

### Run 3 — Larger Dataset (HuggingFace)
- **Dataset:** HuggingFace `priyank-m/IAM_words_text_recognition` (~69k samples)
- **Epochs:** 50
- All augmentations from Run 2 carried over
- **Result:** CER ~9% — more diverse training data made a significant difference

---

### Run 4 — Overnight Training (Best Model)
- **Training data:** HuggingFace IAM train (69,190) + Synthetic (50,000) = **119,190 total training samples**
- **Validation data:** HuggingFace IAM val (23,064 samples, real handwriting only — no synthetic)
- **Test data:** HuggingFace IAM test (23,064 samples, real handwriting only — no synthetic)
- **Epochs:** 50
- All augmentations active
- Training time: ~5 hours overnight on RTX 3050 Laptop
- The best checkpoint was selected based on lowest **validation CER** across all 50 epochs; saved automatically whenever val CER improved
- **Result:** Test CER **7.51%**, Test WER **22.3%** — evaluated on the 23,064 sample test set
- Checkpoint backups: `best_checkpoint_rayans_model.pt`, `best_checkpoint_v2_page_recogniser.pt`

---

## Results

| Run | Train Samples | Test Samples | Augmentation | CER (test) | WER (test) |
|-----|--------------|-------------|-------------|-----------|-----------|
| 1 | ~29,800 (Kaggle IAM) | ~7,500 | Basic | ~16.6% | ~45% |
| 2 | ~29,800 (Kaggle IAM) | ~7,500 | + Elastic + Erosion | ~12% | ~35% |
| 3 | 69,190 (HF IAM) | 23,064 | Full | ~9% | ~28% |
| **4** | **119,190 (HF IAM + Synthetic)** | **23,064** | **Full** | **7.51%** | **22.3%** |

### What CER and WER mean

**CER (Character Error Rate)** measures what percentage of individual characters were wrong. It is calculated as the edit distance (number of insertions, deletions, and substitutions needed to turn the prediction into the correct answer) divided by the length of the correct answer. A CER of 7.51% means that on average, about 1 in 13 characters is incorrect. For context, commercial OCR systems on clean printed text achieve below 1%, while state-of-the-art handwriting recognition on IAM achieves around 3–5%. A CER of 7.51% is a reasonable result for a from-scratch implementation trained on limited hardware.

**WER (Word Error Rate)** applies the same edit-distance logic but at the word level — a word counts as wrong if even a single character in it is incorrect. WER is always higher than CER. A WER of 22.3% means roughly 1 in 4 words contains at least one mistake. This is why full-sentence recognition still has noticeable errors even when individual characters are mostly correct.

**Sample predictions from the best model (Run 4):**
```
GT  : 'the'       PRED: 'the'       ✓
GT  : 'writing'   PRED: 'writng'    ✗ (one char dropped)
GT  : 'quickly'   PRED: 'quickly'   ✓
GT  : 'beautiful' PRED: 'beautifl'  ✗ (minor)
```

### Context Against Published Results

| System | CER on IAM |
|--------|-----------|
| This project (Run 4) | 7.51% |
| Typical CRNN baseline (published) | 8–12% |
| State-of-the-art (2023, Transformer-based) | ~3–4% |
| Commercial OCR on printed text | <1% |

A CER of 7.51% is competitive with published CRNN baselines on IAM, and was achieved on a consumer laptop GPU (RTX 3050, 4GB VRAM) in approximately 5 hours of training.

**Full page test** — "How are you guys? Would you like chocolates? If so why or why not?"
```
Predicted: "to are you quys / Would you like hocolates' / IF so why or ehiy no"
```
The model correctly identifies the number of lines and the number of words per line, but makes character substitution errors. This is primarily caused by the domain gap — the model was trained on clean 300 DPI scans, but the test input was a phone photo with different lighting, perspective, and background texture.

---

## Setup & Installation

### Requirements
- **Python 3.12** (PyTorch has no wheels for Python 3.13+)
- NVIDIA GPU with CUDA recommended (CPU works but is very slow for training)
- Windows/Linux/macOS

### 1. Clone the repo and switch to this branch
```bash
git clone https://github.com/Luke-Smyth45/Handwriting-Recognition-AI.git
cd Handwriting-Recognition-AI
git checkout Rayans-Model
```

### 2. Install PyTorch with CUDA (do this FIRST before requirements.txt)
```bash
# CUDA 11.8 (tested — RTX 3050 Laptop)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (for newer GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset

**Option A — HuggingFace (recommended, 69k samples, auto-downloads):**
```bash
python -c "from datasets import load_dataset; ds = load_dataset('priyank-m/IAM_words_text_recognition'); print('Done')"
```

**Option B — Local IAM from Kaggle (~38k samples):**
1. Download from: `https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database`
2. Extract so that this path exists:
   ```
   data/raw/archive/iam_words/words.txt
   data/raw/archive/iam_words/words/a01/...
   ```

**Option C — Generate synthetic data (Windows only, requires handwriting fonts):**
```bash
python generate_synthetic.py
# Outputs 50,000 images to data/synthetic/
```

### 5. Place the trained checkpoint
Copy the provided checkpoint file to:
```
checkpoints/best.pt
```

---

## How to Run

### Interactive Drawing GUI

```bash
# Default (greedy decoding)
python draw.py

# With beam search (more accurate, slightly slower)
python draw.py --beam --beam-width 10
```

The GUI has four buttons:
| Button | What it does |
|--------|-------------|
| **Predict** | Runs the model on whatever is drawn/loaded on the canvas |
| **Load Word** | Opens a file picker to load a single word image (jpg/png) |
| **Load Page** | Opens a file picker to load a full page image; runs the complete page pipeline and shows a scrollable result popup with a Copy button |
| **Clear** | Resets the canvas |

**Tips for best results with drawn input:**
- Draw slowly and clearly with consistent stroke thickness
- Fill the canvas height — small strokes are harder to recognise
- The model was trained on scanned handwriting, not mouse strokes, so results vary

### Full Page Recognition (CLI)

```bash
# Basic
python page_recogniser.py --image path/to/page.jpg

# With beam search
python page_recogniser.py --image path/to/page.jpg --beam --beam-width 10

# Save a debug image showing line and word bounding boxes
python page_recogniser.py --image path/to/page.jpg --show

# Save individual word crops to debug_crops/ for inspection
python page_recogniser.py --image path/to/page.jpg --save-crops

# Verbose output (line/word counts + per-word predictions)
python page_recogniser.py --image path/to/page.jpg --verbose
```

### Training

```bash
# Train on local IAM dataset
python train.py

# Train on HuggingFace IAM dataset (recommended)
python train.py --hf

# Train on combined HF + synthetic dataset (Run 4 config — best results)
python train.py --combined

# Resume from a checkpoint
python train.py --resume checkpoints/best.pt
```

### Evaluation

```bash
# Evaluate on test set with greedy decoding
python evaluate.py --checkpoint checkpoints/best.pt

# Evaluate with beam search
python evaluate.py --checkpoint checkpoints/best.pt --beam --beam-width 10

# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/best.pt --split val
```

### Single Image Inference

```bash
python inference.py --image path/to/word.png
python inference.py --image path/to/word.png --beam --beam-width 10
```

---

## File Structure

```
AI Project/
├── config.py              # All hyperparameters, paths, vocabulary (edit this first)
├── model.py               # CRNN architecture (CNNBackbone + BiLSTM + CTC head)
├── dataset.py             # IAM local dataset + WordAugmenter (elastic distortion etc.)
├── dataset_hf.py          # HuggingFace IAM dataset loader
├── dataset_combined.py    # ConcatDataset: HF IAM + synthetic
├── generate_synthetic.py  # Synthetic word image generator (Windows fonts)
├── train.py               # Training loop (AMP, OneCycleLR, TensorBoard, checkpointing)
├── evaluate.py            # CER/WER metrics + beam search decoder + test-set eval script
├── page_recogniser.py     # Full page recognition pipeline (preprocess→lines→words→text)
├── draw.py                # Tkinter drawing/loading GUI
├── inference.py           # Standalone single-image inference
├── requirements.txt       # Python dependencies
├── checkpoints/           # Saved model weights — NOT committed (too large for git)
│   └── best.pt            # ← place your checkpoint here
└── data/                  # Dataset files — NOT committed (too large for git)
    ├── raw/archive/iam_words/
    │   ├── words.txt
    │   └── words/
    └── synthetic/
```

---

## Technical Details

### Image Preprocessing
- **Fixed size:** 32px height, up to 128px width (aspect ratio preserved → pad with white to 128px)
- **Normalisation:** `(pixel / 255 - 0.5) / 0.5` → range [-1, 1]
- **Real photo preprocessing** (phone photos): adaptive Gaussian threshold (blockSize=31, C=15) + mild dilation to strengthen thin strokes

### Data Augmentation (training only, CPU-side via OpenCV)

| Augmentation | Parameters | Notes |
|---|---|---|
| Random rotation | ±5° | Simulates tilted writing |
| Random width scale | 0.85×–1.15× | Handles wide/narrow letter spacing |
| Horizontal shear | ±0.15 | Simulates italic/slanted handwriting |
| **Elastic distortion** | α=12, σ=4 (50% prob) | Most effective — random pixel-level warping |
| Brightness jitter | ±40 pixel value | Handles varying ink density |
| Gaussian noise | σ=8 | Paper grain / scanner noise |
| Random erosion/dilation | 2×2 kernel (30% prob) | Simulates pen width variation |

### Page Segmentation Algorithm

**Line detection (horizontal projection):**
1. Invert binary image so text pixels = 1
2. Sum pixels per row → horizontal projection profile
3. Smooth with moving average (kernel size scales with image height: `max(5, height // 80)`)
4. Threshold at 4% of peak → text rows vs. blank rows
5. Extract contiguous text regions with 6px padding

**Word detection within each line (morphological approach):**
1. Horizontally dilate with a wide kernel (`width = line_height // 5`) to merge spaced letters within a word into solid blobs
2. Column projection on dilated image to find word boundaries
3. Apply 4px padding to each word bounding box

> **Why dilate first?** Without it, natural letter spacing causes individual letters to be
> detected as separate "words". Dilation merges them before projection, so gaps between words
> are detected rather than gaps between letters.

### Page Preprocessing for Phone Photos
- Resize to minimum 1200px height, cap at 3000px width
- Gaussian blur 5×5 for denoising
- Adaptive threshold: `blockSize = max(51, (height // 15) | 1)`, C=20 — blockSize scales with image so it works on both small and large photos
- Morphological opening (3×3 ellipse kernel) removes paper texture noise
- Auto-invert if text came out white on black

### Hyperparameters (Run 4 — best model)

```python
IMG_HEIGHT   = 32      # fixed image height
IMG_WIDTH    = 128     # fixed image width (padded)
RNN_HIDDEN   = 256     # BiLSTM hidden units per direction (512 total)
RNN_LAYERS   = 2       # stacked BiLSTM layers
RNN_DROPOUT  = 0.3     # dropout between LSTM layers
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 3e-4    # OneCycleLR peak learning rate
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 5.0     # gradient clipping max norm
BEAM_WIDTH   = 5       # default beam search width
USE_AMP      = True    # fp16 mixed precision
```

---

## Known Limitations & Future Work

### Domain Gap (Main Issue)
The model was trained on **clean scanned IAM images**. When given a **phone photo** of
handwriting, performance drops noticeably due to differences in lighting, background texture,
perspective, and ink colour. The `clean_real_image()` function in `draw.py` partially bridges
this with adaptive thresholding, but fully fixing it requires:
- Fine-tuning on phone photos of real handwriting
- Adding augmentations that simulate phone photo degradation (motion blur, perspective warp, non-uniform lighting)

### Accuracy Improvements
- **Language model decoding:** beam search currently has no language model. Integrating a
  character n-gram LM via `pyctcdecode` would significantly lower WER by preferring real words
- **Wider input:** `IMG_WIDTH=128` crops long words. Increasing to 256 would preserve more of
  each word at the cost of memory
- **More synthetic data:** 50k was helpful; scaling to 500k+ with more font diversity would
  likely push CER below 5%
- **Transformer encoder:** replacing BiLSTM with a Transformer (e.g. ViT-based) would give
  global attention across the full word sequence

### Segmentation Improvements
- **Slanted lines:** horizontal projection fails on pages where lines aren't axis-aligned
- **Touching characters:** adaptive dilation kernel width could be tuned per-image
- **Punctuation:** short word crops (apostrophes, single letters) are currently filtered out
  as noise — a smarter filter would keep them
