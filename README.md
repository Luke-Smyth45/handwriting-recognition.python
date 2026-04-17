# Handwriting Recognition — CRNN on IAM

A deep learning pipeline that reads images of handwritten words and transcribes them to text.
Built with PyTorch using a CRNN (Convolutional Recurrent Neural Network) trained on the
[IAM Handwriting Word Database](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database).

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Dataset](#dataset)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Inference](#inference)
8. [Interactive Drawing GUI](#interactive-drawing-gui)
9. [Configuration Reference](#configuration-reference)
10. [Results](#results)
11. [What Can Be Improved](#what-can-be-improved)

---

## How It Works

### Architecture

The model is a **CRNN** — a CNN feature extractor feeding into a recurrent sequence model,
trained end-to-end with CTC loss.

```
Input image (1, 32, 128)
        │
        ▼
┌───────────────┐
│  CNN Backbone │  4 blocks of Conv-BN-ReLU + MaxPool
│               │  Collapses height to 1, preserves width
│  Output:      │  (512, 1, 32)
└───────┬───────┘
        │  reshape to sequence
        ▼
┌───────────────┐
│  BiLSTM x2   │  2 stacked bidirectional LSTM layers
│  hidden=256   │  captures left and right context
│  Output:      │  (32, 512)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Linear Head  │  projects to vocabulary size (96 classes)
└───────┬───────┘
        │
        ▼
  CTC Decode  →  "hello"
```

**CNN Backbone** — Four convolutional blocks progressively reduce the 32px-tall input
image down to a single-row feature map of width W/4. This turns the 2D image into a
1D sequence of 32 column vectors, each summarising a vertical slice of the word.

**BiLSTM** — Two stacked bidirectional LSTM layers read the column sequence left-to-right
and right-to-left simultaneously, giving each time step full context of the whole word.

**CTC Loss** — Connectionist Temporal Classification allows the model to be trained without
knowing which output character aligns to which pixel. It marginalises over all valid
alignments, making it ideal for handwriting where character widths vary.

### Decoding

Two decoding strategies are available after training:

- **Greedy** — picks the highest-probability character at each time step, then collapses
  repeated tokens and removes blanks. Fast but suboptimal.
- **Beam Search** — maintains the top-N candidate sequences at each step and picks the
  best final sequence. More accurate, especially for ambiguous strokes.

### Data Pipeline

Images are loaded as grayscale, resized to a fixed height of 32px (width scaled
proportionally then padded/cropped to 128px), and normalised to `[-1, 1]`.

During training, light augmentations are applied:
- Random rotation ±3°
- Random width scaling ±10%
- Random brightness jitter ±30
- Gaussian noise (σ=5)

Splits are generated deterministically from the data by grouping samples by **writer ID**,
ensuring no writer appears in more than one split (writer-independent evaluation):
- **Train** — 29,851 samples (~80% of writers)
- **Val** — 888 samples (~10% of writers)
- **Test** — 7,566 samples (~10% of writers)

---

## Project Structure

```
handwriting-recognition/
├── config.py          # All hyperparameters, paths, and vocabulary
├── model.py           # CRNN architecture (CNN + BiLSTM + CTC head)
├── dataset.py         # IAM dataset loader, augmentations, split generation
├── train.py           # Training loop with AMP, checkpointing, TensorBoard
├── evaluate.py        # CER/WER metrics, beam search decoder, test-set eval
├── inference.py       # Single-image prediction from file
├── draw.py            # Interactive GUI — draw a word and predict it
├── requirements.txt   # Python dependencies
└── data/
    └── raw/
        └── archive/
            └── iam_words/
                ├── words.txt       # Ground-truth labels
                └── words/          # Word image tree (a01/a01-000u/...)
```

Directories created automatically at runtime:
```
checkpoints/    # Saved model weights (.pt files)
runs/           # TensorBoard logs
```

---

## Setup

**Requirements:** Python 3.10+, an NVIDIA GPU (recommended)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA support

The default `pip install torch` installs the CPU-only build. For GPU support:

```bash
# CUDA 12.1 (works with driver versions reporting CUDA 12.x or 13.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Dataset

Download the **IAM Handwriting Word Database** from Kaggle:
[nibinv23/iam-handwriting-word-database](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)

Extract `archive.zip` into `data/raw/`:
```bash
# Windows PowerShell
Expand-Archive -Path archive.zip -DestinationPath data\raw\

# Linux / macOS
unzip archive.zip -d data/raw/
```

The expected layout after extraction:
```
data/raw/archive/iam_words/
├── words.txt
└── words/
    ├── a01/
    ├── a02/
    └── ...
```

No further setup is needed — train/val/test splits are generated automatically.

---

## Training

```bash
python train.py
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 50 | Number of training epochs |
| `--batch-size N` | 64 | Batch size (reduce if GPU runs out of memory) |
| `--resume PATH` | None | Resume training from a saved checkpoint |

**Examples:**
```bash
python train.py --epochs 100 --batch-size 32
python train.py --resume checkpoints/epoch_050.pt
```

**What gets saved:**
- `checkpoints/best.pt` — saved whenever validation CER improves
- `checkpoints/epoch_NNN.pt` — saved every 5 epochs

**Monitor training in a separate terminal:**
```bash
tensorboard --logdir runs
```
Then open `http://localhost:6006`. Logged metrics: `train/loss`, `val/loss`, `val/CER`, `train/lr`.

**Training features:**
- Automatic Mixed Precision (fp16) for ~2x speed on CUDA
- OneCycleLR learning rate schedule
- Gradient clipping (max norm 5.0)
- AdamW optimiser with weight decay

---

## Evaluation

Run the full test set (7,566 samples) and print CER, WER, and sample predictions:

```bash
python evaluate.py
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | `checkpoints/best.pt` | Checkpoint to load |
| `--split` | `test` | Dataset split: `train`, `val`, or `test` |
| `--beam` | off | Use beam search instead of greedy decoding |
| `--beam-width N` | 5 | Number of beams for beam search |
| `--batch-size N` | 64 | Batch size |

**Examples:**
```bash
python evaluate.py --beam --beam-width 10
python evaluate.py --split val --checkpoint checkpoints/epoch_030.pt
```

**Output:**
```
[greedy]  CER=0.1640  WER=0.3200  (7,566 samples)

Sample predictions:
  GT  : 'hello'
  PRED: 'hello'

  GT  : 'world'
  PRED: 'worId'
```

**Metrics:**
- **CER** (Character Error Rate) — edit distance between predicted and ground-truth string,
  divided by ground-truth length. 0.16 = ~16% of characters are wrong.
- **WER** (Word Error Rate) — same but at the word level. Always higher than CER.

---

## Inference

Predict the text in a single word image:

```bash
python inference.py --image path/to/word.png
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | `checkpoints/best.pt` | Checkpoint to load |
| `--beam` | off | Use beam search |
| `--beam-width N` | 5 | Beam width |
| `--show` | off | Display image + prediction in a matplotlib window |

**Examples:**
```bash
python inference.py --image word.png --beam --beam-width 10
python inference.py --image word.png --show
```

**Output:**
```
[greedy]  "recognition"
```

---

## Interactive Drawing GUI

Draw a word with your mouse and have the model transcribe it in real time:

```bash
python draw.py
```

A window appears with a white 640×160 canvas. Draw a single word, then click **Predict**.
Click **Clear** to reset and try again.

```
┌──────────────────────────────────────────┐
│  Draw a single handwritten word below    │
├──────────────────────────────────────────┤
│                                          │
│   [white canvas — draw here]             │
│                                          │
├──────────────────────────────────────────┤
│   [ Predict ]   [ Clear ]                │
│                                          │
│   [greedy]  "hello"                      │
└──────────────────────────────────────────┘
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | `checkpoints/best.pt` | Checkpoint to load |
| `--beam` | off | Use beam search |
| `--beam-width N` | 5 | Beam width |

Note: the model was trained on scanned handwriting samples, so mouse-drawn input will look
different to the training data. Results are best when drawing slowly and clearly.

---

## Configuration Reference

All settings live in `config.py`. Key values:

| Setting | Default | Description |
|---------|---------|-------------|
| `DEVICE` | `"cuda"` | `"cpu"` if no GPU available |
| `IMG_HEIGHT` | `32` | Fixed image height (px) |
| `IMG_WIDTH` | `128` | Fixed image width after padding (px) |
| `EPOCHS` | `50` | Training epochs |
| `BATCH_SIZE` | `64` | Training batch size |
| `LR` | `3e-4` | Peak learning rate |
| `RNN_HIDDEN` | `256` | BiLSTM hidden units per direction |
| `RNN_LAYERS` | `2` | Number of stacked BiLSTM layers |
| `RNN_DROPOUT` | `0.1` | Dropout between LSTM layers |
| `BEAM_WIDTH` | `5` | Default beam search width |
| `USE_AMP` | `True` | Mixed precision (fp16) — CUDA only |
| `GRAD_CLIP` | `5.0` | Gradient clipping max norm |

---

## Results

Trained for 50 epochs on an NVIDIA RTX 3070 Ti (~32s/epoch):

| Metric | Value |
|--------|-------|
| Best val CER | ~0.164 |
| Training time | ~27 min total |

Overfitting was observed from around epoch 35 — training loss continued to fall while
validation CER plateaued. The best checkpoint is saved automatically.

---

## What Can Be Improved

### Accuracy

**Increase regularisation**
The model overfits after ~35 epochs. Raising `RNN_DROPOUT` from `0.1` to `0.3` in
`config.py` and adding dropout after CNN blocks would reduce this.

**Stronger augmentations**
The current augmentations are minimal. Adding elastic distortion, random perspective
warps, and random erosion/dilation would better simulate real handwriting variation.
The `albumentations` library (already in `requirements.txt`) supports all of these.

**Larger input width**
`IMG_WIDTH = 128` crops longer words. Increasing to `256` or `512` would preserve more
information at the cost of more memory and slower training.

**Attention mechanism**
Replacing the BiLSTM with a Transformer encoder would give the model global attention
over the full sequence rather than only local context, typically improving accuracy on
longer words.

**Language model decoding**
The beam search has no language model — it scores sequences purely on acoustic probability.
Integrating a character-level n-gram LM via `pyctcdecode` or `ctcdecode` would significantly
reduce WER by favouring real words over nonsense sequences.

### Data

**Pre-training on synthetic data**
IAM contains ~38k word samples, which is small by modern standards. Generating millions of
synthetic handwriting images using fonts and augmentations (e.g. with the `TextRecognitionDataGenerator`
library) and pre-training on those before fine-tuning on IAM is the single highest-leverage
improvement available.

**Additional real datasets**
Other handwriting datasets that can supplement IAM: RIMES (French), CVL, ICDAR competitions.

### Training

**Early stopping**
Currently training runs for a fixed number of epochs. Adding early stopping (halt when
val CER hasn't improved for N epochs) would prevent wasted compute and save the best model
more reliably.

**Learning rate tuning**
The OneCycleLR scheduler works well but the peak LR (`3e-4`) and warmup fraction (`10%`)
were not tuned. A learning rate finder pass before training would identify the optimal value.

### Deployment

**ONNX export**
The trained model can be exported to ONNX for deployment outside of Python/PyTorch:
```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

**Batch inference API**
`inference.py` processes one image at a time. Wrapping it in a FastAPI server would allow
batch requests and integration into other applications.

**Line-level recognition**
The current model operates on pre-segmented word images. Adding a text detection stage
(e.g. CRAFT or DBNet) would allow recognising full lines or paragraphs from a photograph.
