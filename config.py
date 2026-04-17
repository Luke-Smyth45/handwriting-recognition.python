"""
Central configuration for the IAM handwriting recognition pipeline.
All paths, hyperparameters, and vocabulary live here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw" / "archive" / "iam_words"
WORDS_IMG  = RAW_DIR  / "words"         # word image tree: words/a01/a01-000u/...
WORDS_TXT  = RAW_DIR  / "words.txt"     # ground-truth file
SPLITS_DIR = None                        # not provided; splits are auto-generated
CKPT_DIR   = BASE_DIR / "checkpoints"
LOG_DIR    = BASE_DIR / "runs"          # TensorBoard logs

for _d in (DATA_DIR, CKPT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Vocabulary  (95 printable ASCII chars + blank token for CTC)
# ---------------------------------------------------------------------------
ALPHABET   = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BLANK_IDX  = 0                           # CTC blank is index 0
CHAR2IDX   = {c: i + 1 for i, c in enumerate(ALPHABET)}   # 1-indexed
IDX2CHAR   = {i + 1: c for i, c in enumerate(ALPHABET)}
IDX2CHAR[BLANK_IDX] = "<B>"
NUM_CLASSES = len(ALPHABET) + 1          # +1 for CTC blank

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
IMG_HEIGHT = 32    # fixed height; width is kept proportional then padded
IMG_WIDTH  = 128   # maximum width after resize + pad

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
CNN_CHANNELS    = [1, 64, 128, 256, 256, 512, 512]   # in/out channels per block
RNN_HIDDEN      = 256    # BiLSTM hidden units (each direction)
RNN_LAYERS      = 2      # stacked BiLSTM layers
RNN_DROPOUT     = 0.3

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE      = 64
NUM_WORKERS     = 4
EPOCHS          = 50
LR              = 3e-4
LR_STEP         = 10     # StepLR decay every N epochs
LR_GAMMA        = 0.5
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 5.0    # gradient clipping max norm
VAL_INTERVAL    = 1      # run validation every N epochs
SAVE_INTERVAL   = 5      # save checkpoint every N epochs

# ---------------------------------------------------------------------------
# Inference / decoding
# ---------------------------------------------------------------------------
BEAM_WIDTH      = 5      # beam search width (1 = greedy)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
DEVICE          = "cuda"  # override to "cpu" if needed
PIN_MEMORY      = True
USE_AMP         = True    # Automatic Mixed Precision (fp16) for speed on CUDA

# ---------------------------------------------------------------------------
# Kaggle dataset slug  (adjust if the exact dataset differs on your account)
# ---------------------------------------------------------------------------
KAGGLE_DATASET  = "nibinv23/iam-handwriting-word-database"
