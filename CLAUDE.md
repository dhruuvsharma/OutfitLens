# CLAUDE.md — Instructions for Claude Code

## Before making any change
1. Read `memory/_index.md` to find the relevant file(s)
2. Read the memory doc for each file you plan to touch
3. Update the Change Log in the memory doc BEFORE editing the .py file

## Code principles
- Keep library modules decoupled — files in `models/` and `data_pipeline/` should import from at most 2 other project files. Orchestration files (`training/train.py`, `inference/recognizer.py`, `scripts/`) are exempted because they are intentional integration points.
- All config is read from yaml — no hardcoded paths or hyperparameters in .py files
- Every function must have a one-line docstring
- No file should exceed 300 lines — split if needed

## Adding a new asset item
- Drop renders into the correct `data/raw/` subfolder (`assets_4angle/` or `assets_1angle/`)
- Re-run `scripts/build_reference_db.py` — no retraining needed

## Adding a new Python file
- Create the .py file
- Immediately create its memory doc at `memory/<filename>.md`
- Add it to `memory/_index.md`

## Architecture rules
- `data_pipeline/` never imports from `training/` or `inference/`
- `training/` never imports from `inference/`
- `inference/` never imports from `training/`
- `models/` imports nothing from this project
- `scripts/` may import from any layer

## Config discipline
- Always load config via `yaml.safe_load(open(args.config))`
- Never hardcode paths, sizes, or hyperparameters inside `.py` files
- Both `configs/config_4angle.yaml` and `configs/config_1angle.yaml` must always contain every key used by any module

## Testing a change
1. Run compositor with a small composite_count (e.g. 20) to verify image generation
2. Run `build_reference_db.py` to verify embedding pipeline
3. Run a 2-epoch training pass to verify the training loop
4. Run `predict.py` on a synthetic image to verify end-to-end inference
