---
name: train_all
description: CLI script that loops over all category folders and runs compositor + training for each
type: project
---

# Memory: train_all.py

## Purpose
Top-level orchestration script for the v2 specialist architecture. Discovers all category subdirectories in `data/raw/`, optionally filters to a single category via `--category`, then for each category runs: (1) `generate_category_dataset()` to produce synthetic composites, and (2) `train_specialist()` to train and populate the embedding DB. Replaces v1 `scripts/run_training.py`.

## Location
`scripts/train_all.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| parse_args | fn | Parses --config, --categories-config, --category, --skip-composite |
| main | fn | Discovers categories, loops compositor + training |

## Inputs & Outputs
- **Inputs:** base_config.yaml path; categories.yaml path; raw renders in `data/raw/<category>/`
- **Outputs:** delegates to compositor (images + labels.json per category) and train_specialist (checkpoints + DBs per category)

## Dependencies
- Internal: `data_pipeline/compositor.py`, `training/train_specialist.py`
- External: `pyyaml`

## Config Keys Used
- All keys delegated to compositor and train_specialist; reads `raw_data_dir` to discover categories

## Change Log
| Date | Change |
|------|--------|
| 2026-04-05 | Initial creation — v2 multi-category training orchestration |
