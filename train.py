"""
Training script for the IAM CRNN handwriting recogniser.

Features
--------
- CUDA + Automatic Mixed Precision (AMP/fp16) for fast GPU training
- CTC loss
- OneCycleLR scheduler
- Gradient clipping
- TensorBoard logging  (loss, CER, LR)
- Checkpoint save/resume
- Greedy CER tracked on validation set each epoch
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_loaders
from dataset_hf import get_loaders_hf
from dataset_combined import get_loaders_combined
from model import CRNN, _greedy_decode
from evaluate import compute_cer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path):
    torch.save(state, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scaler):
    ckpt = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"Resumed from checkpoint: {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt.get("best_cer", 1.0)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, scheduler, device, writer, epoch
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True)
    for step, (images, targets, target_lens) in enumerate(pbar):
        # Move data to GPU
        images      = images.to(device, non_blocking=True)
        targets     = targets.to(device, non_blocking=True)
        target_lens = target_lens.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass in fp16 (AMP) for speed and memory savings
        with autocast(enabled=config.USE_AMP):
            log_probs = model(images)                  # (T, B, C)
            T, B, _   = log_probs.shape
            # CTC needs to know the sequence length for each sample (all equal here)
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lens, target_lens)

        # Backward pass — scaler handles fp16 gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Clip gradients to prevent large update spikes
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        # OneCycleLR steps every batch (not every epoch)
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Log per-step metrics to TensorBoard
        global_step = epoch * len(loader) + step
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    avg_loss = total_loss / len(loader)
    writer.add_scalar("train/loss_epoch", avg_loss, epoch)
    return avg_loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    all_preds, all_gts = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False, dynamic_ncols=True)
    for images, targets, target_lens in pbar:
        images      = images.to(device, non_blocking=True)
        targets     = targets.to(device, non_blocking=True)
        target_lens = target_lens.to(device, non_blocking=True)

        with autocast(enabled=config.USE_AMP):
            log_probs  = model(images)
            T, B, _    = log_probs.shape
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)
            loss       = criterion(log_probs, targets, input_lens, target_lens)

        total_loss += loss.item()

        # Greedy decode each sample in the batch to measure CER
        preds_idx = log_probs.argmax(dim=2).permute(1, 0)  # (B, T)
        offset = 0
        for i in range(B):
            pred_str = _greedy_decode(preds_idx[i].cpu().tolist())
            tgt_len  = target_lens[i].item()
            # Targets are packed flat — use offset to slice each sample's label
            gt_str   = "".join(
                config.IDX2CHAR.get(targets[offset + j].item(), "")
                for j in range(tgt_len)
            )
            all_preds.append(pred_str)
            all_gts.append(gt_str)
            offset += tgt_len

    avg_loss = total_loss / len(loader)
    cer = compute_cer(all_preds, all_gts)

    writer.add_scalar("val/loss",  avg_loss, epoch)
    writer.add_scalar("val/CER",   cer,      epoch)

    return avg_loss, cer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train IAM CRNN")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--hf", action="store_true",
                        help="Use full HuggingFace IAM dataset (~115k samples)")
    parser.add_argument("--combined", action="store_true",
                        help="Use HuggingFace IAM + synthetic data combined")
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    if args.combined:
        print("Using HuggingFace IAM + synthetic data combined")
        train_loader, val_loader, _ = get_loaders_combined(
            batch_size=args.batch_size,
            pin_memory=(device.type == "cuda"),
        )
    elif args.hf:
        print("Using HuggingFace IAM dataset (~115k samples)")
        train_loader, val_loader, _ = get_loaders_hf(
            batch_size=args.batch_size,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader, val_loader, _ = get_loaders(
            batch_size=args.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )

    # Model
    model = CRNN(num_classes=config.NUM_CLASSES).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # CTC loss — handles variable-length labels without needing alignment
    # zero_infinity=True silences rare NaN losses from degenerate short sequences
    criterion = nn.CTCLoss(blank=config.BLANK_IDX, reduction="mean", zero_infinity=True)

    # AdamW adds weight decay directly to parameters (not through gradients)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    # OneCycleLR: warms up LR for 10% of training then cosine-anneals to near zero
    # This gives faster convergence than a fixed LR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LR,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,       # 10% warmup
        anneal_strategy="cos",
    )

    # GradScaler manages the fp16 loss scaling to prevent underflow
    scaler = GradScaler(enabled=config.USE_AMP)

    # TensorBoard writer — run `tensorboard --logdir runs` to view live training curves
    writer = SummaryWriter(log_dir=str(config.LOG_DIR))

    # Resume
    start_epoch = 1
    best_cer    = 1.0
    if args.resume:
        start_epoch, best_cer = load_checkpoint(
            Path(args.resume), model, optimizer, scaler
        )
        start_epoch += 1

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, scheduler, device, writer, epoch
        )

        val_loss, val_cer = validate(
            model, val_loader, criterion, device, writer, epoch
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_CER={val_cer:.4f}  "
            f"time={elapsed:.1f}s"
        )

        # Save best
        if val_cer < best_cer:
            best_cer = val_cer
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "best_cer": best_cer},
                config.CKPT_DIR / "best.pt",
            )
            print(f"  -> New best CER: {best_cer:.4f}  (saved best.pt)")

        # Periodic save
        if epoch % config.SAVE_INTERVAL == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "best_cer": best_cer},
                config.CKPT_DIR / f"epoch_{epoch:03d}.pt",
            )

    writer.close()
    print(f"\nTraining complete. Best val CER: {best_cer:.4f}")
    print(f"Best checkpoint: {config.CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
