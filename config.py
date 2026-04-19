"""
Central configuration for the IAM handwriting recognition pipeline.
All paths, hyperparameters, and vocabulary live here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.resolve()   # root folder of the project
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw" / "archive" / "iam_words"
WORDS_IMG  = RAW_DIR  / "words"         # word image tree: words/a01/a01-000u/...
WORDS_TXT  = RAW_DIR  / "words.txt"     # ground-truth labels file
SPLITS_DIR = None                        # not provided; splits are auto-generated
CKPT_DIR   = BASE_DIR / "checkpoints"   # saved model .pt files go here
LOG_DIR    = BASE_DIR / "runs"          # TensorBoard training logs

# Create these folders automatically if they don't exist
for _d in (DATA_DIR, CKPT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Vocabulary  (95 printable ASCII chars + blank token for CTC)
# ---------------------------------------------------------------------------
# Every character the model can recognise — anything outside this is filtered out
ALPHABET   = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BLANK_IDX  = 0                           # CTC blank must be index 0
CHAR2IDX   = {c: i + 1 for i, c in enumerate(ALPHABET)}   # char → integer (1-indexed)
IDX2CHAR   = {i + 1: c for i, c in enumerate(ALPHABET)}   # integer → char
IDX2CHAR[BLANK_IDX] = "<B>"
NUM_CLASSES = len(ALPHABET) + 1          # 95 chars + 1 blank = 96 total output classes

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
IMG_HEIGHT = 32    # every image is resized to exactly this height before entering the model
IMG_WIDTH  = 128   # width is padded or cropped to this after proportional resize

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
CNN_CHANNELS    = [1, 64, 128, 256, 256, 512, 512]   # channel sizes through each CNN block
RNN_HIDDEN      = 256    # hidden units per direction in the BiLSTM (512 total combined)
RNN_LAYERS      = 2      # number of stacked BiLSTM layers
RNN_DROPOUT     = 0.3    # dropout between LSTM layers to prevent overfitting

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE      = 64     # images per training step
NUM_WORKERS     = 4      # parallel CPU workers for loading data
EPOCHS          = 50     # how many full passes through the training data
LR              = 3e-4   # peak learning rate used by OneCycleLR
LR_STEP         = 10     # (unused with OneCycleLR, kept for reference)
LR_GAMMA        = 0.5
WEIGHT_DECAY    = 1e-4   # L2 penalty on weights — discourages memorising training data
GRAD_CLIP       = 5.0    # caps gradient magnitude to prevent unstable training spikes
VAL_INTERVAL    = 1      # run validation every N epochs
SAVE_INTERVAL   = 5      # save a periodic checkpoint every N epochs

# ---------------------------------------------------------------------------
# Inference / decoding
# ---------------------------------------------------------------------------
BEAM_WIDTH      = 5      # how many candidate sequences beam search keeps at each time step

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
DEVICE          = "cuda"  # uses GPU if available; train.py falls back to "cpu" automatically
PIN_MEMORY      = True    # keeps data tensors in pinned memory for faster GPU transfer
USE_AMP         = True    # fp16 mixed precision — halves VRAM usage, speeds up training

# ---------------------------------------------------------------------------
# Kaggle dataset slug  (adjust if the exact dataset differs on your account)
# ---------------------------------------------------------------------------
KAGGLE_DATASET  = "nibinv23/iam-handwriting-word-database"
