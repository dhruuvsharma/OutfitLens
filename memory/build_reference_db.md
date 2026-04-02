---
name: build_reference_db
description: CLI script that embeds all raw asset renders and saves the reference EmbeddingDB
type: project
---

# Memory: build_reference_db.py

## Purpose
Reads all raw clothing item renders from the configured assets directory, embeds each item using the trained ResNet-34 backbone, and saves the resulting reference EmbeddingDB as a `.npz` file. For 4-angle mode, embeddings from all available angle renders are averaged (then re-normalised) into a single item vector. For 1-angle mode, only the front-view embedding is used. This script must be re-run whenever new items are added — no retraining is needed.

## Location
`scripts/build_reference_db.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| parse_args | fn | Parses --config, --checkpoint, --output CLI args |
| build_db | fn | Main logic: loads backbone, embeds items, saves DB |
| main | fn | Entry point |

## Inputs & Outputs
- **Inputs:** config yaml path; checkpoint .pt path; asset image files in `assets_dir`
- **Outputs:** `logs/reference_db.npz` (or custom --output path)

## Dependencies
- Internal: `data_pipeline/augmentation.py`, `models/backbone.py`, `models/embedding_db.py`
- External: `torch`, `numpy`, `Pillow`, `pyyaml`, `tqdm`

## Config Keys Used
- `assets_dir` — source of raw renders
- `single_angle` — controls whether to average angle embeddings
- `image_size`, `embedding_dim`, `checkpoint_dir`

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
