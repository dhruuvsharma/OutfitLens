# SUPERSEDED by scripts/train_all.py (v2 specialist architecture).
# This v1 file is kept for reference only — imports will fail against the v2 API.
"""CLI entry point: generate synthetic data and run the full training pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.compositor import generate_dataset
from training.train import run_training


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training runner."""
    p = argparse.ArgumentParser(description="OutfitLens training runner")
    p.add_argument("--config", required=True, help="Path to config yaml (e.g. configs/config_4angle.yaml)")
    p.add_argument(
        "--skip-composite",
        action="store_true",
        help="Skip dataset generation if synthetic images already exist",
    )
    return p.parse_args()


def main() -> None:
    """Load config, optionally generate dataset, then run training."""
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    labels_file = Path(config["synthetic_dir"]) / "labels.json"

    if args.skip_composite and labels_file.exists():
        print("Skipping compositor — labels.json already exists.")
    else:
        print("=== Step 1/2: Generating synthetic composite dataset ===")
        generate_dataset(config)

    print("=== Step 2/2: Training ===")
    run_training(config)


if __name__ == "__main__":
    main()
