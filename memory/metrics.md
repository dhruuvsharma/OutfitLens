---
name: metrics
description: Per-item precision/recall/F1, mAP, confusion matrix, and CSV logging for training evaluation
type: project
---

# Memory: metrics.py

## Purpose
Provides all evaluation metrics needed to assess model quality on the multi-label item recognition task. Metrics are computed from raw numpy arrays (no sklearn dependency in the hot path) to avoid import-time overhead. Functions cover: per-class precision/recall/F1, macro-averaged F1, per-class average precision and mAP, per-class binary confusion matrices, and a CSV append helper used by the training loop to build `logs/train_log.csv`.

## Location
`training/metrics.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| per_item_metrics | fn | Returns precision/recall/F1 arrays of shape (C,) |
| mean_f1 | fn | Macro-averaged F1 scalar |
| average_precision | fn | AP for a single class |
| mean_average_precision | fn | mAP across all classes |
| binary_confusion_per_class | fn | (C, 2, 2) confusion matrices |
| save_metrics_csv | fn | Appends metrics row to a CSV log file |
| compute_and_log | fn | Convenience wrapper: compute + log + return |

## Inputs & Outputs
- **Inputs:** `y_true` and `y_pred` numpy arrays of shape `(N, C)`; item name list; log file path
- **Outputs:** dict of metric arrays; scalar mAP; CSV row appended to log file

## Dependencies
- Internal: none
- External: `numpy`, `csv`, `pathlib`

## Config Keys Used
None directly (threshold can be passed as argument; default 0.5).

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
| 2026-04-05 | v2 rewrite — single-label metrics: top-1/top-5 accuracy, per-item recall@5, confusion matrix CSV; removed multi-label mAP/F1 functions |
