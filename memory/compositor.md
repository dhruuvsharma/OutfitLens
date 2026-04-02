---
name: compositor
description: Synthetic outfit compositor that layers random items onto backgrounds and writes labels.json
type: project
---

# Memory: compositor.py

## Purpose
Reads raw asset renders from the configured asset directory and generates synthetic composite outfit images for training. Each composite layers 2–6 randomly chosen items onto a plain-colour or noise background at randomised scales and positions, simulating natural occlusion via z-ordering (later items drawn on top). Writes JPEG images to `data/synthetic/images/` and a `labels.json` mapping each filename to the list of items present (plus a multi-hot vector indexed by the full item list).

## Location
`data_pipeline/compositor.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| generate_dataset | fn | Main entry point — reads config dict, generates N composites |
| _load_assets | fn | Dispatcher: routes to single- or multi-angle loader |
| _load_single_angle | fn | Loads one front-view image per item |
| _load_multi_angle | fn | Loads 4-angle renders; uses front view as pixel representative |
| _make_background | fn | Creates random plain-colour or noise background canvas |
| _compose | fn | Layers a list of item images onto a background with random scale/position |

## Inputs & Outputs
- **Inputs:** config dict with keys `assets_dir`, `synthetic_dir`, `image_size`, `composite_count`, `single_angle`
- **Outputs:** JPEG images in `<synthetic_dir>/images/`; `<synthetic_dir>/labels.json`

## Dependencies
- Internal: none
- External: `numpy`, `Pillow`, `json`, `pathlib`, `random`

## Config Keys Used
- `assets_dir` — source folder for raw renders
- `synthetic_dir` — destination for generated images and labels.json
- `image_size` — canvas side length in pixels
- `composite_count` — number of composites to generate
- `single_angle` — if true, load only front-view images

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
