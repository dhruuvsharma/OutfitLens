"""Evaluation metrics for single-label specialist training: top-1/5 accuracy, per-item recall, confusion matrix."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def topk_accuracy(
    y_true: np.ndarray,
    logits: np.ndarray,
    k: int = 1,
) -> float:
    """Compute top-k accuracy: fraction of samples where true label is in top-k predictions.

    Args:
        y_true: integer ground-truth labels, shape (N,).
        logits: raw model logits or scores, shape (N, C).
        k: number of top predictions to consider.
    """
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]  # (N, k)
    correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
    return float(correct.mean())


def top1_accuracy(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Return top-1 accuracy over all samples."""
    return topk_accuracy(y_true, logits, k=1)


def top5_accuracy(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Return top-5 accuracy over all samples (capped at num_classes)."""
    k = min(5, logits.shape[1])
    return topk_accuracy(y_true, logits, k=k)


# ---------------------------------------------------------------------------
# Per-item recall
# ---------------------------------------------------------------------------

def per_item_recall_at_k(
    y_true: np.ndarray,
    logits: np.ndarray,
    item_names: List[str],
    k: int = 5,
) -> Dict[str, float]:
    """Compute per-item recall@k: did the correct item appear in the top-k predictions?

    Returns a dict mapping item_name → recall@k value in [0, 1].
    """
    k = min(k, logits.shape[1])
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]  # (N, k)
    recall: Dict[str, float] = {}
    for cls_idx, name in enumerate(item_names):
        mask = y_true == cls_idx
        if mask.sum() == 0:
            recall[name] = 0.0
            continue
        hit = np.any(top_k_preds[mask] == cls_idx, axis=1)
        recall[name] = float(hit.mean())
    return recall


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, logits: np.ndarray, num_classes: int) -> np.ndarray:
    """Return a (C, C) confusion matrix where entry [i, j] = samples with true=i predicted=j."""
    preds = np.argmax(logits, axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, preds):
        cm[int(t), int(p)] += 1
    return cm


def save_confusion_matrix_csv(path: Path, cm: np.ndarray, item_names: List[str]) -> None:
    """Write a confusion matrix to CSV with row/column headers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + item_names)
        for i, row in enumerate(cm):
            writer.writerow([item_names[i]] + [str(v) for v in row])


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def save_epoch_csv(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    top1: float,
    top5: float,
) -> None:
    """Append one epoch row (loss + top-1/5 accuracy) to the training log CSV."""
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "top1_acc", "top5_acc"])
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{top1:.4f}", f"{top5:.4f}"])


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_and_log(
    y_true: np.ndarray,
    logits: np.ndarray,
    item_names: List[str],
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> Tuple[float, float]:
    """Compute top-1/5 accuracy, log to CSV, and return (top1, top5)."""
    t1 = top1_accuracy(y_true, logits)
    t5 = top5_accuracy(y_true, logits)
    save_epoch_csv(log_path, epoch, train_loss, val_loss, t1, t5)
    return t1, t5
