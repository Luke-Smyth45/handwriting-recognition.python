"""
Combined Dataset Loader
------------------------
Merges the HuggingFace IAM dataset with synthetic generated data
for training. Val and test sets use real IAM data only.

Usage:
    from dataset_combined import get_loaders_combined
    train_loader, val_loader, test_loader = get_loaders_combined()
"""

import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import config
from dataset import WordAugmenter, preprocess_image, collate_fn
from dataset_hf import IAMWordDatasetHF


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Loads pre-generated synthetic word images from data/synthetic/.
    Expects: data/synthetic/labels.csv and data/synthetic/images/*.png
    """

    def __init__(self, synthetic_dir: str = "data/synthetic", augment: bool = True):
        self.img_dir = Path(synthetic_dir) / "images"
        csv_path     = Path(synthetic_dir) / "labels.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Synthetic labels not found at {csv_path}. "
                "Run generate_synthetic.py first."
            )

        self.records = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row["word"]
                # Filter chars outside alphabet
                label = "".join(c for c in word if c in config.CHAR2IDX)
                if label:
                    self.records.append((row["filename"], label))

        self.augment = augment
        self.aug_fn  = WordAugmenter() if augment else None
        print(f"[Synthetic] {len(self.records):>6,} samples loaded from {synthetic_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        filename, label = self.records[idx]

        img = cv2.imread(str(self.img_dir / filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((config.IMG_HEIGHT, config.IMG_WIDTH), 255, dtype=np.uint8)

        # Synthetic images are already preprocessed to model dims — just normalise
        if self.aug_fn is not None:
            img = self.aug_fn(img)

        img_tensor = torch.from_numpy(preprocess_image(img))

        label_enc = torch.tensor(
            [config.CHAR2IDX[c] for c in label if c in config.CHAR2IDX],
            dtype=torch.long,
        )

        return img_tensor, label_enc, len(label_enc)


# ---------------------------------------------------------------------------
# Combined loaders
# ---------------------------------------------------------------------------

def get_loaders_combined(
    batch_size: int = config.BATCH_SIZE,
    pin_memory: bool = config.PIN_MEMORY,
    synthetic_dir: str = "data/synthetic",
):
    """
    Train  = HuggingFace IAM train + synthetic data (combined)
    Val    = HuggingFace IAM val   (real data only)
    Test   = HuggingFace IAM test  (real data only)
    """
    # Real IAM data
    iam_train = IAMWordDatasetHF("train", augment=True)
    val_ds    = IAMWordDatasetHF("val",   augment=False)
    test_ds   = IAMWordDatasetHF("test",  augment=False)

    # Synthetic data
    syn_train = SyntheticDataset(synthetic_dir=synthetic_dir, augment=True)

    # Combine for training
    train_ds = ConcatDataset([iam_train, syn_train])
    print(f"[Combined] Total train samples: {len(train_ds):,} "
          f"(IAM: {len(iam_train):,} + Synthetic: {len(syn_train):,})")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader
