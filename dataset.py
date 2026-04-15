"""
IAM Handwriting Word Dataset
-----------------------------
Parses words.txt, loads word images, applies augmentations,
and returns (image_tensor, label_indices, label_length) triples
suitable for CTC training.
"""

import os
import re
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import config


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_words_txt(words_txt: Path) -> List[dict]:
    """
    Parse the IAM words.txt ground-truth file.
    Each valid line produces a dict:
        { 'id': str, 'ok': bool, 'path': Path, 'label': str }
    Lines starting with '#' and lines with segmentation result 'err' are skipped.
    """
    records = []
    with open(words_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            word_id, ok_flag = parts[0], parts[1]
            transcription = parts[8]

            # Skip poorly segmented samples
            if ok_flag == "err":
                continue

            # Build image path: a01-000u-00-00  -> words/a01/a01-000u/a01-000u-00-00.png
            segs = word_id.split("-")
            img_path = (
                config.WORDS_IMG
                / segs[0]
                / f"{segs[0]}-{segs[1]}"
                / f"{word_id}.png"
            )

            # Filter characters not in our alphabet
            label = "".join(c for c in transcription if c in config.CHAR2IDX)
            if not label:
                continue

            records.append({
                "id":    word_id,
                "ok":    ok_flag == "ok",
                "path":  img_path,
                "label": label,
            })
    return records


def make_splits(
    records: List[dict],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Deterministically split records into train/val/test by writer ID so that
    no writer appears in more than one split (writer-independent evaluation).
    Returns {'train': [...], 'val': [...], 'test': [...]}.
    """
    # Group by writer (first segment of word id, e.g. "a01")
    from collections import defaultdict
    writers: dict = defaultdict(list)
    for r in records:
        writer = r["id"].split("-")[0]
        writers[writer].append(r)

    writer_ids = sorted(writers.keys())
    rng = random.Random(seed)
    rng.shuffle(writer_ids)

    n = len(writer_ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_writers = set(writer_ids[:n_train])
    val_writers   = set(writer_ids[n_train:n_train + n_val])

    splits: dict = {"train": [], "val": [], "test": []}
    for r in records:
        writer = r["id"].split("-")[0]
        if writer in train_writers:
            splits["train"].append(r)
        elif writer in val_writers:
            splits["val"].append(r)
        else:
            splits["test"].append(r)

    return splits


# ---------------------------------------------------------------------------
# Augmentations  (training only, CPU-side via OpenCV/numpy for portability)
# ---------------------------------------------------------------------------

class WordAugmenter:
    """Light augmentation pipeline for handwriting word images (grayscale)."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = self._random_rotate(img)
        img = self._random_scale(img)
        img = self._random_brightness(img)
        img = self._gaussian_noise(img)
        return img

    @staticmethod
    def _random_rotate(img, max_deg=3.0):
        h, w = img.shape
        angle = np.random.uniform(-max_deg, max_deg)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _random_scale(img, lo=0.9, hi=1.1):
        scale = np.random.uniform(lo, hi)
        h, w = img.shape
        new_w = max(1, int(w * scale))
        img = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
        return img

    @staticmethod
    def _random_brightness(img, delta=30):
        img = img.astype(np.int16)
        img += np.random.randint(-delta, delta)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _gaussian_noise(img, sigma=5):
        noise = np.random.randn(*img.shape) * sigma
        img = img.astype(np.float32) + noise
        return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Core preprocessing: resize to fixed height, pad width, normalise
# ---------------------------------------------------------------------------

def preprocess_image(
    img: np.ndarray,
    target_h: int = config.IMG_HEIGHT,
    target_w: int = config.IMG_WIDTH,
) -> np.ndarray:
    """
    Resize to target_h (keep aspect ratio), pad/crop width to target_w.
    Returns float32 array in [0, 1] with shape (1, H, W).
    """
    h, w = img.shape
    scale = target_h / h
    new_w = max(1, int(w * scale))
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    # Pad or crop width
    if new_w < target_w:
        pad = target_w - new_w
        img = np.pad(img, ((0, 0), (0, pad)), mode="constant", constant_values=255)
    else:
        img = img[:, :target_w]

    img = img.astype(np.float32) / 255.0          # [0, 1]
    img = (img - 0.5) / 0.5                        # [-1, 1]  (normalise)
    return img[np.newaxis, :, :]                   # (1, H, W)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IAMWordDataset(Dataset):
    """
    PyTorch Dataset for the IAM word-level handwriting recognition task.

    Args:
        split:    'train', 'val', or 'test'
        augment:  apply training augmentations
    """

    def __init__(self, split: str = "train", augment: bool = False):
        assert split in ("train", "val", "test"), \
            "split must be one of ('train', 'val', 'test')"
        self.augment = augment
        self.aug_fn  = WordAugmenter() if augment else None

        # Load all records from words.txt, keep only those whose image exists
        all_records = [r for r in parse_words_txt(config.WORDS_TXT) if r["path"].exists()]

        # Auto-generate writer-independent splits
        splits = make_splits(all_records)
        self.records = splits[split]

        if len(self.records) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}'. "
                "Check that data/raw/archive/iam_words/ has the correct structure."
            )

        print(f"[Dataset] {split:5s}  {len(self.records):>6,} samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.records[idx]

        # Load grayscale
        img = cv2.imread(str(rec["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: white image
            img = np.full((config.IMG_HEIGHT, config.IMG_WIDTH), 255, dtype=np.uint8)

        if self.aug_fn is not None:
            img = self.aug_fn(img)

        img_tensor = torch.from_numpy(preprocess_image(img))  # (1, H, W)

        # Encode label
        label_enc = torch.tensor(
            [config.CHAR2IDX[c] for c in rec["label"] if c in config.CHAR2IDX],
            dtype=torch.long,
        )

        return img_tensor, label_enc, len(label_enc)


# ---------------------------------------------------------------------------
# Collate  (variable-length labels -> padded batch)
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        images      (B, 1, H, W)
        targets     (sum_of_label_lengths,)   — flat, as CTC expects
        target_lens (B,)
    """
    images, labels, lengths = zip(*batch)
    images      = torch.stack(images, 0)
    targets     = torch.cat(labels, 0)
    target_lens = torch.tensor(lengths, dtype=torch.long)
    return images, targets, target_lens


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def get_loaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = config.PIN_MEMORY,
):
    train_ds = IAMWordDataset("train", augment=True)
    val_ds   = IAMWordDataset("val",   augment=False)
    test_ds  = IAMWordDataset("test",  augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader
