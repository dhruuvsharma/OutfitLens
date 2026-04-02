"""Evaluation metrics: per-item precision/recall/F1, mAP, and confusion matrix export."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-item metrics
# ---------------------------------------------------------------------------

def per_item_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Compute per-class precision, recall, and F1 from probability predictions.

    Args:
        y_true: ground-truth multi-hot array, shape (N, C), values in {0, 1}.
        y_pred: predicted probabilities, shape (N, C), values in [0, 1].
        threshold: decision threshold for converting probabilities to binary predictions.

    Returns:
        dict with keys 'precision', 'recall', 'f1', each an array of shape (C,).
    """
    preds = (y_pred >= threshold).astype(np.float32)

    tp = (preds * y_true).sum(axis=0)
    fp = (preds * (1.0 - y_true)).sum(axis=0)
    fn = ((1.0 - preds) * y_true).sum(axis=0)

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    return {"precision": precision, "recall": recall, "f1": f1}


def mean_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Return the macro-averaged F1 across all classes."""
    return float(per_item_metrics(y_true, y_pred, threshold)["f1"].mean())


# ---------------------------------------------------------------------------
# Mean Average Precision
# ---------------------------------------------------------------------------

def average_precision(y_true_col: np.ndarray, y_score_col: np.ndarray) -> float:
    """Compute average precision (AP) for a single class using the step-function AUC."""
    sorted_idx = np.argsort(y_score_col)[::-1]
    y_true_sorted = y_true_col[sorted_idx]

    tp_cum = np.cumsum(y_true_sorted)
    total_pos = y_true_col.sum()
    if total_pos == 0:
        return 0.0

    n = len(y_true_sorted)
    denom = np.arange(1, n + 1, dtype=np.float32)
    precision_at_k = tp_cum / denom
    recall_at_k = tp_cum / total_pos

    # Δrecall between adjacent thresholds
    delta_recall = np.diff(recall_at_k, prepend=0.0)
    return float((precision_at_k * delta_recall).sum())


def mean_average_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged AP (mAP) across all classes."""
    num_classes = y_true.shape[1]
    aps = [average_precision(y_true[:, c], y_pred[:, c]) for c in range(num_classes)]
    return float(np.mean(aps))


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def binary_confusion_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Return a (C, 2, 2) array of per-class binary confusion matrices [[TN,FP],[FN,TP]]."""
    preds = (y_pred >= threshold).astype(np.int32)
    n_classes = y_true.shape[1]
    cms = np.zeros((n_classes, 2, 2), dtype=np.int64)
    for c in range(n_classes):
        for true_val in (0, 1):
            for pred_val in (0, 1):
                mask = (y_true[:, c] == true_val) & (preds[:, c] == pred_val)
                cms[c, true_val, pred_val] = mask.sum()
    return cms


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def save_metrics_csv(
    path: Path,
    item_names: List[str],
    metrics_dict: Dict[str, np.ndarray],
    epoch: int,
    loss: float,
) -> None:
    """Append a per-item metrics row (precision, recall, F1) to a CSV log file."""
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["epoch", "loss"] + [
                f"{name}_{metric}"
                for name in item_names
                for metric in ("precision", "recall", "f1")
            ] + ["mean_f1"]
            writer.writerow(header)
        row: List = [epoch, f"{loss:.6f}"]
        for idx in range(len(item_names)):
            row += [
                f"{metrics_dict['precision'][idx]:.4f}",
                f"{metrics_dict['recall'][idx]:.4f}",
                f"{metrics_dict['f1'][idx]:.4f}",
            ]
        row.append(f"{metrics_dict['f1'].mean():.4f}")
        writer.writerow(row)


def compute_and_log(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    item_names: List[str],
    log_path: Path,
    epoch: int,
    loss: float,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Compute per-item metrics + mAP, log to CSV, and return results."""
    metrics = per_item_metrics(y_true, y_pred)
    mAP = mean_average_precision(y_true, y_pred)
    metrics["mAP"] = np.array([mAP])
    save_metrics_csv(log_path, item_names, metrics, epoch, loss)
    return metrics, mAP
