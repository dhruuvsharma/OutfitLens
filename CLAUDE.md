# CLAUDE.md — Instructions for Claude Code

## Before making any change
1. Read `memory/_index.md` to find the relevant file(s)
2. Read the memory doc for each file you plan to touch
3. Update the Change Log in the memory doc BEFORE editing the .py file

## Code principles
- Keep library modules decoupled — files in `models/` and `data_pipeline/` should import from at most 2 other project files. Orchestration files (`training/train_specialist.py`, `inference/recognizer.py`, `scripts/`) are exempted because they are intentional integration points.
- All config is read from yaml — no hardcoded paths or hyperparameters in .py files
- Every function must have a one-line docstring
- No file should exceed 300 lines — split if needed
- Type hints on all function signatures
- Use `pathlib.Path` everywhere, not `os.path`

## Adding a new clothing category
1. Create `data/raw/<new_category>/` and drop renders in
2. Add entry to `configs/categories.yaml`
3. Run `python scripts/train_all.py --category <new_category>`
4. No other code changes needed

## Adding a new item to an existing category
1. Drop renders into `data/raw/<category>/`
2. Run `python scripts/build_reference_db.py --category <category>`
3. No retraining needed — the DB just gets the new embedding added

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
- `configs/base_config.yaml` must always contain every key used by any module
- Per-category overrides go in `configs/categories.yaml`

## Changing angle mode
Set `single_angle: true` in `configs/base_config.yaml` and retrain.

## Testing a change
1. Run compositor with a small composite_count (e.g. 20) to verify image generation
2. Run `build_reference_db.py` to verify embedding pipeline
3. Run a 2-epoch training pass (`python scripts/train_all.py --category hats`) to verify training loop
4. Run `predict.py` on a synthetic image to verify end-to-end inference
