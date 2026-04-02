---
name: predict
description: CLI script for single-image outfit item prediction using trained backbone + reference DB
type: project
---

# Memory: predict.py

## Purpose
User-facing CLI script for running inference on a single outfit image. Loads the config, backbone checkpoint, and reference embedding DB, then runs `OutfitRecognizer.predict_path()`. Prints detected items with confidence scores sorted descending. Optionally writes results to a JSON file. Provides overrides for checkpoint/DB path and threshold via CLI flags.

## Location
`scripts/predict.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| parse_args | fn | Parses --image, --config, --checkpoint, --db, --threshold, --output-json |
| main | fn | Loads model, runs inference, prints + optionally writes results |

## Inputs & Outputs
- **Inputs:** outfit image path; config yaml; checkpoint .pt; reference_db.npz
- **Outputs:** stdout table of {item: score}; optional JSON file

## Dependencies
- Internal: `inference/recognizer.py`
- External: `pyyaml`, `json`, `pathlib`

## Config Keys Used
- `checkpoint_dir` — default location for checkpoint and DB
- `confidence_threshold` — can be overridden via --threshold
- `image_size`, `embedding_dim` — forwarded to recognizer

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
