---
name: run_training
description: CLI entry point that orchestrates compositor + training loop from a config yaml
type: project
---

# Memory: run_training.py

## Purpose
Top-level CLI script that ties together the data generation and training steps. Accepts a config yaml path and an optional `--skip-composite` flag. Without the flag it first calls `generate_dataset()` to produce synthetic composites, then calls `run_training()`. With the flag it skips compositor if `labels.json` already exists, allowing re-runs after the dataset is built.

## Location
`scripts/run_training.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| parse_args | fn | Parses --config and --skip-composite CLI args |
| main | fn | Orchestrates compositor → training |

## Inputs & Outputs
- **Inputs:** config yaml; raw asset images (must already be in assets_dir)
- **Outputs:** delegates to compositor (images + labels.json) and train.py (checkpoint + logs)

## Dependencies
- Internal: `data_pipeline/compositor.py`, `training/train.py`
- External: `pyyaml`

## Config Keys Used
All keys delegated to compositor and train; this script reads `synthetic_dir` to check for labels.json.

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
