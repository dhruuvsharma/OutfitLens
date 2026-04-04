---
name: dataset
description: PyTorch Dataset and split-builder for synthetic composite outfit images
type: project
---

# Memory: dataset.py

## Purpose
Provides the `OutfitDataset` PyTorch Dataset class that reads composite images and their multi-hot labels from disk. It consumes a split JSON file (train.json or val.json listing filenames) and the master `labels.json` produced by the compositor. Also includes `build_splits()` which creates the train/val split files from the master labels in the configured ratio. The dataset is the bridge between the data pipeline and the training loop.

## Location
`data_pipeline/dataset.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| OutfitDataset | class | PyTorch Dataset returning (image_tensor, multi_hot_label) |
| OutfitDataset.class_names | property | Returns ordered list of item names |
| build_splits | fn | Creates train.json / val.json split files |

## Inputs & Outputs
- **Inputs:** split JSON file path, master labels.json path, images directory, optional transform callable
- **Outputs:** `(torch.Tensor [3,H,W], torch.Tensor [num_classes])` tuples per sample; split JSON files on disk

## Dependencies
- Internal: none (uses `data_pipeline/augmentation.py` transforms passed in by the caller)
- External: `torch`, `Pillow`, `json`, `pathlib`

## Config Keys Used
- `synthetic_dir` — root containing images/ and labels.json
- `val_split` — fraction of samples held out for validation

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
| 2026-04-05 | v2 rewrite — single-label (item_index int) instead of multi-hot; `SpecialistDataset` class reads per-category labels.json list format; `build_splits` updated for list-of-dicts structure |
