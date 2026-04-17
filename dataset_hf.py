"""
HuggingFace IAM Dataset Loader
--------------------------------
Downloads and loads the full IAM word dataset (~115k samples) from:
  https://huggingface.co/datasets/priyank-m/IAM_words_text_recognition

Images are stored as bytes in parquet files — this loader decodes them
on the fly and applies the same preprocessing pipeline as dataset.py.

Usage:
    from dataset_hf import get_loaders_hf
    train_loader, val_loader, test_loader = get_loaders_hf()
"""

import io
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

import config
from dataset import WordAugmenter, preprocess_image, collate_fn

HF_DATASET_ID = "priyank-m/IAM_words_text_recognition"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IAMWordDatasetHF(Dataset):
    """
    PyTorch Dataset backed by the HuggingFace IAM words dataset.
    Downloads automatically on first use and caches locally.
    """

    def __init__(self, split: str = "train", augment: bool = False):
        assert split in ("train", "val", "test"), \
            "split must be one of ('train', 'val', 'test')"

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required. "
                "Install it with: pip install datasets"
            )

        print(f"[HF Dataset] Loading '{split}' split from {HF_DATASET_ID} ...")
        hf_split = split  # dataset uses 'train', 'val', 'test' directly
        ds = load_dataset(HF_DATASET_ID, split=hf_split)

        # Filter out samples whose label contains characters outside our alphabet
        def has_valid_label(example):
            label = example.get("label", "") or example.get("text", "") or example.get("transcription", "")
            label = "".join(c for c in label if c in config.CHAR2IDX)
            return len(label) > 0

        # Find the label column name
        sample = ds[0]
        self.label_col = None
        for col in ("label", "text", "transcription", "word"):
            if col in sample:
                self.label_col = col
                break
        if self.label_col is None:
            raise RuntimeError(
                f"Could not find label column in dataset. "
                f"Available columns: {list(sample.keys())}"
            )

        # Find the image column name
        self.image_col = None
        for col in ("image", "img", "image_path"):
            if col in sample:
                self.image_col = col
                break
        if self.image_col is None:
            raise RuntimeError(
                f"Could not find image column in dataset. "
                f"Available columns: {list(sample.keys())}"
            )

        ds = ds.filter(has_valid_label)
        self.ds      = ds
        self.augment = augment
        self.aug_fn  = WordAugmenter() if augment else None

        print(f"[HF Dataset] {split:5s}  {len(self.ds):>6,} samples  "
              f"(image_col='{self.image_col}', label_col='{self.label_col}')")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.ds[idx]

        # Decode image — HuggingFace returns PIL Images for image columns
        pil_img = row[self.image_col]
        if hasattr(pil_img, "convert"):
            # PIL Image
            pil_img = pil_img.convert("L")  # grayscale
            img = np.array(pil_img, dtype=np.uint8)
        elif isinstance(pil_img, bytes):
            # Raw bytes
            arr = np.frombuffer(pil_img, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.full((config.IMG_HEIGHT, config.IMG_WIDTH), 255, dtype=np.uint8)

        if img is None:
            img = np.full((config.IMG_HEIGHT, config.IMG_WIDTH), 255, dtype=np.uint8)

        if self.aug_fn is not None:
            img = self.aug_fn(img)

        img_tensor = torch.from_numpy(preprocess_image(img))  # (1, H, W)

        # Encode label
        label_str = row[self.label_col] or ""
        label_enc = torch.tensor(
            [config.CHAR2IDX[c] for c in label_str if c in config.CHAR2IDX],
            dtype=torch.long,
        )

        return img_tensor, label_enc, len(label_enc)


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def get_loaders_hf(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,   # HF datasets work better with 0 workers on Windows
    pin_memory: bool = config.PIN_MEMORY,
):
    train_ds = IAMWordDatasetHF("train", augment=True)
    val_ds   = IAMWordDatasetHF("val",   augment=False)
    test_ds  = IAMWordDatasetHF("test",  augment=False)

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
