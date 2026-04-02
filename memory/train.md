---
name: train
description: Training loop with BCE/focal loss, Adam optimiser, per-epoch validation, and checkpoint saving
type: project
---

# Memory: train.py

## Purpose
Implements the complete training pipeline. `run_training(config)` is the single entry point called by `scripts/run_training.py`. It builds DataLoaders from the synthetic dataset, instantiates the ResNet-34 backbone and classifier head, selects the loss function from config, and runs the train/val loop for the configured number of epochs. The best checkpoint (by mean validation F1) is saved to `logs/best_model.pt`. Per-epoch metrics are appended to `logs/train_log.csv`.

## Location
`training/train.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| run_training | fn | Main entry point — executes full training from config dict |
| _train_epoch | fn | One forward-backward pass over the training DataLoader |
| _val_epoch | fn | One evaluation pass; returns loss + stacked y_true/y_pred |
| _save_checkpoint | fn | Saves backbone+head state dicts to best_model.pt |
| _ensure_splits | fn | Creates train/val split files if missing |

## Inputs & Outputs
- **Inputs:** config dict (from yaml); synthetic images + labels.json on disk
- **Outputs:** `logs/best_model.pt` (best checkpoint); `logs/train_log.csv` (per-epoch metrics)

## Dependencies
- Internal: `data_pipeline/augmentation.py`, `data_pipeline/dataset.py`, `models/backbone.py`, `models/classifier_head.py`, `training/losses.py`, `training/metrics.py`
- External: `torch`, `numpy`, `tqdm`

## Config Keys Used
- `synthetic_dir`, `checkpoint_dir`, `val_split`, `image_size`, `batch_size`
- `embedding_dim`, `learning_rate`, `epochs`, `focal_loss`

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
