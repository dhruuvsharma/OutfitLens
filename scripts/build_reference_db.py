"""Build the reference embedding DB for one or all categories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_specialist import _populate_db
import torch


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the reference DB builder."""
    p = argparse.ArgumentParser(description="OutfitLens — build reference embedding DB")
    p.add_argument("--config", default="configs/base_config.yaml", help="Path to base_config.yaml")
    p.add_argument("--category", default=None, help="Category to build DB for (omit for --all)")
    p.add_argument("--all", action="store_true", help="Build DB for all categories with a trained checkpoint")
    return p.parse_args()


def build_category_db(config: dict, category: str) -> None:
    """Build and save the reference DB for one category from its trained checkpoint."""
    logs_dir = Path(config["logs_dir"])
    raw_dir = Path(config["raw_data_dir"]) / category
    checkpoint_path = logs_dir / category / "best_model.pt"
    db_path = logs_dir / category / "embedding_db.npz"

    if not raw_dir.is_dir():
        print(f"ERROR: raw directory not found: {raw_dir}")
        return
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _populate_db(config, category, raw_dir, checkpoint_path, db_path, device)


def main() -> None:
    """Entry point for build_reference_db script."""
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.all:
        raw_data_dir = Path(config["raw_data_dir"])
        categories = [d.name for d in sorted(raw_data_dir.iterdir()) if d.is_dir()]
        if not categories:
            print(f"No category folders found in {raw_data_dir}")
            sys.exit(1)
        for cat in categories:
            build_category_db(config, cat)
    elif args.category:
        build_category_db(config, args.category)
    else:
        print("ERROR: specify --category <name> or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
