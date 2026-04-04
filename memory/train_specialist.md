---
name: train_specialist
description: Trains one specialist model for one clothing category; populates per-category embedding DB post-training
type: project
---

# Memory: train_specialist.py

## Purpose
Implements the specialist training pipeline for a single clothing category. `train_specialist(config, category)` runs cross-entropy classification training on synthetic composite images, then after training it embeds all real item renders from `data/raw/<category>/` and populates the reference embedding DB. Saves checkpoint to `logs/<category>/best_model.pt`, DB to `logs/<category>/embedding_db.npz`, and metrics to `logs/<category>/train_log.csv`.

## Location
`training/train_specialist.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| train_specialist | fn | Main entry point — full train + DB population for one category |
| _train_epoch | fn | One forward-backward pass over the training DataLoader |
| _val_epoch | fn | Validation pass; returns loss + predictions |
| _populate_db | fn | Embeds raw renders and saves reference EmbeddingDB |
| _save_checkpoint | fn | Saves backbone state dict + item_names mapping |

## Inputs & Outputs
- **Inputs:** config dict; category name string; synthetic images + labels.json in `data/synthetic/<category>/`; raw renders in `data/raw/<category>/`
- **Outputs:** `logs/<category>/best_model.pt`; `logs/<category>/embedding_db.npz`; `logs/<category>/train_log.csv`

## Dependencies
- Internal: `data_pipeline/augmentation.py`, `data_pipeline/dataset.py`, `models/backbone.py`, `training/losses.py`, `training/metrics.py`, `models/embedding_db.py`
- External: `torch`, `numpy`, `tqdm`, `json`

## Config Keys Used
- `synthetic_dir`, `logs_dir`, `raw_data_dir`, `val_split`
- `image_size`, `batch_size`, `embedding_dim`, `learning_rate`, `epochs`
- `focal_loss`, `single_angle`

## Change Log
| Date | Change |
|------|--------|
| 2026-04-05 | Initial creation — v2 specialist training |
