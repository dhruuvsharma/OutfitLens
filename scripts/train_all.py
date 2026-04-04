"""Orchestrate compositor + specialist training for all (or one) clothing categories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.compositor import generate_category_dataset
from training.train_specialist import train_specialist


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the multi-category training script."""
    p = argparse.ArgumentParser(description="OutfitLens — train all specialist models")
    p.add_argument("--config", default="configs/base_config.yaml", help="Path to base_config.yaml")
    p.add_argument(
        "--categories-config",
        default="configs/categories.yaml",
        help="Path to categories.yaml",
    )
    p.add_argument(
        "--category",
        default=None,
        help="Train only this category (omit to train all discovered categories)",
    )
    p.add_argument(
        "--skip-composite",
        action="store_true",
        help="Skip compositor if labels.json already exists for a category",
    )
    return p.parse_args()


def _discover_categories(raw_data_dir: Path) -> list:
    """Return sorted list of category folder names found in raw_data_dir."""
    return sorted(d.name for d in raw_data_dir.iterdir() if d.is_dir())


def main() -> None:
    """Load configs, discover categories, run compositor + training for each."""
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.categories_config) as f:
        cats_cfg = yaml.safe_load(f)

    # Merge per-category overrides into config copies
    override_map = {c["name"]: c for c in cats_cfg.get("categories", [])}

    raw_data_dir = Path(config["raw_data_dir"])
    if args.category:
        categories = [args.category]
    else:
        categories = _discover_categories(raw_data_dir)

    if not categories:
        print(f"No category folders found in {raw_data_dir}")
        sys.exit(1)

    print(f"Categories to train: {categories}")

    for category in categories:
        cat_dir = raw_data_dir / category
        if not cat_dir.is_dir():
            print(f"WARNING: {cat_dir} does not exist — skipping {category}")
            continue

        # Build effective config for this category (merge overrides)
        effective_config = dict(config)
        if category in override_map:
            for k, v in override_map[category].items():
                if k != "name":
                    effective_config[k] = v

        # Step 1: Compositor
        labels_file = Path(effective_config["synthetic_dir"]) / category / "labels.json"
        if args.skip_composite and labels_file.exists():
            print(f"\n[{category}] Skipping compositor — labels.json exists")
        else:
            print(f"\n[{category}] === Step 1/2: Generating synthetic dataset ===")
            generate_category_dataset(effective_config, category)

        # Step 2: Training
        print(f"\n[{category}] === Step 2/2: Training specialist ===")
        train_specialist(effective_config, category)

    print("\nAll categories complete.")


if __name__ == "__main__":
    main()
