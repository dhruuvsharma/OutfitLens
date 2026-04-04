"""CLI entry point: predict clothing items in an outfit image using all specialist models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.aggregator import Aggregator
from inference.recognizer import load_recognizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the prediction script."""
    p = argparse.ArgumentParser(description="OutfitLens — predict items in an outfit image")
    p.add_argument("--image", required=True, help="Path to the outfit image")
    p.add_argument("--config", default="configs/base_config.yaml", help="Path to base_config.yaml")
    p.add_argument(
        "--categories-config",
        default="configs/categories.yaml",
        help="Path to categories.yaml",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override confidence threshold from config",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Write structured results to this JSON file",
    )
    return p.parse_args()


def main() -> None:
    """Load all specialists, run inference, aggregate, and print per-category results."""
    args = parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)
    with open(args.categories_config) as f:
        cats_cfg = yaml.safe_load(f)

    categories_list = cats_cfg.get("categories", [])

    if args.threshold is not None:
        base_config["confidence_threshold"] = args.threshold

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    # Load recognizer (all specialists)
    recognizer = load_recognizer(base_config, categories_list)

    if not recognizer.specialists:
        print("ERROR: No specialists loaded. Run scripts/train_all.py first.")
        sys.exit(1)

    # Build aggregator with per-category overrides
    category_overrides = {
        c["name"]: c
        for c in categories_list
        if any(k not in ("name",) for k in c)
    }
    aggregator = Aggregator(
        confidence_threshold=base_config.get("confidence_threshold", 0.70),
        top_n_results=base_config.get("top_n_results", 5),
        category_overrides=category_overrides,
    )

    # Run inference
    raw_results = recognizer.recognize(image_path)
    final_output = aggregator.aggregate(raw_results)

    # Print results
    print(f"\nOutfit analysis: {args.image}")
    print("=" * 50)
    for category, items in final_output.items():
        print(f"\n{category.upper()}:")
        if items:
            for entry in items:
                print(f"  {entry['item']:<30s}  {entry['confidence']:.4f}")
        else:
            print("  (no items above threshold)")
    print()

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(final_output, f, indent=2)
        print(f"Results written to: {args.output_json}")


if __name__ == "__main__":
    main()
