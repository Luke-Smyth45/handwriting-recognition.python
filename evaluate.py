"""
Evaluation utilities and full test-set evaluation script.

Metrics
-------
CER  — Character Error Rate  (edit distance / ref length)
WER  — Word Error Rate       (edit distance on word tokens)

Decoding
--------
Greedy  : argmax per time-step, collapse repeats, remove blanks.
Beam    : simple prefix-beam-search (pure Python, no language model).
          For production, swap in a KenLM-backed beam decoder via
          the `ctcdecode` or `pyctcdecode` library.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import editdistance
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

import config
from dataset import get_loaders
from model import CRNN, _greedy_decode


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def cer_single(pred: str, gt: str) -> Tuple[int, int]:
    """Return (edit_distance, ref_length) for one sample."""
    return editdistance.eval(pred, gt), max(len(gt), 1)


def wer_single(pred: str, gt: str) -> Tuple[int, int]:
    p_words = pred.split()
    g_words = gt.split()
    return editdistance.eval(p_words, g_words), max(len(g_words), 1)


def compute_cer(preds: List[str], gts: List[str]) -> float:
    total_dist, total_len = 0, 0
    for p, g in zip(preds, gts):
        d, l = cer_single(p, g)
        total_dist += d
        total_len  += l
    return total_dist / total_len if total_len > 0 else 0.0


def compute_wer(preds: List[str], gts: List[str]) -> float:
    total_dist, total_len = 0, 0
    for p, g in zip(preds, gts):
        d, l = wer_single(p, g)
        total_dist += d
        total_len  += l
    return total_dist / total_len if total_len > 0 else 0.0


# ---------------------------------------------------------------------------
# Beam search decoder  (prefix beam search, no LM)
# ---------------------------------------------------------------------------

def beam_decode(log_probs: torch.Tensor, beam_width: int = config.BEAM_WIDTH) -> str:
    """
    Pure-Python prefix beam search on a single sample.

    Args:
        log_probs : (T, num_classes)  log-softmax scores for one sample
        beam_width: number of beams to keep
    Returns:
        Decoded string.
    """
    T, C = log_probs.shape
    probs = log_probs.exp().cpu().numpy()

    # beam: dict  prefix -> (prob_blank, prob_non_blank)
    NEG_INF = float("-inf")
    beams = {(): (1.0, 0.0)}  # empty prefix: (p_blank=1, p_nb=0)

    for t in range(T):
        p_t = probs[t]   # (C,)
        new_beams = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = p_b + p_nb

            # Extend with blank
            new_p_b = p_total * p_t[config.BLANK_IDX]
            _merge(new_beams, prefix, new_p_b, 0.0)

            # Extend with each non-blank character
            for c in range(1, C):
                p_c = p_t[c]
                if len(prefix) > 0 and prefix[-1] == c:
                    # Same char: only blank path can extend without doubling
                    new_p_nb = p_b * p_c
                else:
                    new_p_nb = p_total * p_c
                _merge(new_beams, prefix + (c,), 0.0, new_p_nb)

        # Prune to top beam_width beams
        beams = dict(
            sorted(new_beams.items(),
                   key=lambda x: x[1][0] + x[1][1],
                   reverse=True)[:beam_width]
        )

    # Best beam
    best_prefix = max(beams, key=lambda p: beams[p][0] + beams[p][1])
    return "".join(config.IDX2CHAR.get(i, "") for i in best_prefix)


def _merge(beams: dict, prefix: tuple, p_b: float, p_nb: float):
    if prefix in beams:
        old_b, old_nb = beams[prefix]
        beams[prefix] = (old_b + p_b, old_nb + p_nb)
    else:
        beams[prefix] = (p_b, p_nb)


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: CRNN,
    loader,
    device: torch.device,
    use_beam: bool = False,
    beam_width: int = config.BEAM_WIDTH,
    verbose: bool = True,
) -> dict:
    model.eval()
    all_preds, all_gts = [], []

    pbar = tqdm(loader, desc="Evaluating", leave=True, dynamic_ncols=True)
    for images, targets, target_lens in pbar:
        images      = images.to(device, non_blocking=True)
        targets     = targets.to(device, non_blocking=True)
        target_lens = target_lens.to(device, non_blocking=True)

        with autocast(enabled=config.USE_AMP):
            log_probs = model(images)   # (T, B, C)

        T, B, C = log_probs.shape

        offset = 0
        for i in range(B):
            lp_i = log_probs[:, i, :]   # (T, C)

            if use_beam:
                pred_str = beam_decode(lp_i, beam_width=beam_width)
            else:
                pred_str = _greedy_decode(lp_i.argmax(dim=1).cpu().tolist())

            tgt_len = target_lens[i].item()
            gt_str  = "".join(
                config.IDX2CHAR.get(targets[offset + j].item(), "")
                for j in range(tgt_len)
            )
            all_preds.append(pred_str)
            all_gts.append(gt_str)
            offset += tgt_len

    cer = compute_cer(all_preds, all_gts)
    wer = compute_wer(all_preds, all_gts)

    results = {"CER": cer, "WER": wer, "n_samples": len(all_preds)}

    if verbose:
        mode = f"beam(w={beam_width})" if use_beam else "greedy"
        print(f"\n[{mode}]  CER={cer:.4f}  WER={wer:.4f}  "
              f"({results['n_samples']:,} samples)")
        # Show a few examples
        print("\nSample predictions:")
        for i in range(min(10, len(all_preds))):
            print(f"  GT  : {all_gts[i]!r}")
            print(f"  PRED: {all_preds[i]!r}")
            print()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate IAM CRNN on test set")
    parser.add_argument("--checkpoint", type=str,
                        default=str(config.CKPT_DIR / "best.pt"),
                        help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--beam", action="store_true",
                        help="Use beam search instead of greedy decoding")
    parser.add_argument("--beam-width", type=int, default=config.BEAM_WIDTH)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = CRNN().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt['epoch']})")

    # Data
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}

    evaluate(
        model,
        loader_map[args.split],
        device,
        use_beam=args.beam,
        beam_width=args.beam_width,
    )


if __name__ == "__main__":
    main()
