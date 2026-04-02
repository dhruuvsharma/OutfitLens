"""CLI inference script: identify clothing items in an outfit image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.recognizer import load_recognizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the prediction script."""
    p = argparse.ArgumentParser(description="OutfitLens — predict items in an outfit image")
    p.add_argument("--image", required=True, help="Path to the outfit image")
    p.add_argument("--config", required=True, help="Path to config yaml")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint .pt (default: <checkpoint_dir>/best_model.pt)",
    )
    p.add_argument(
        "--db",
        default=None,
        help="Path to reference DB .npz (default: <checkpoint_dir>/reference_db.npz)",
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
        help="If set, write prediction dict to this JSON file",
    )
    return p.parse_args()


def main() -> None:
    """Load model + DB, run inference, and print results."""
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.threshold is not None:
        config["confidence_threshold"] = args.threshold

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_dir / "best_model.pt"
    db_path = Path(args.db) if args.db else checkpoint_dir / "reference_db.npz"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not db_path.exists():
        print(f"ERROR: Reference DB not found: {db_path}")
        sys.exit(1)

    recognizer = load_recognizer(config, checkpoint_path, db_path)
    results = recognizer.predict_path(Path(args.image))

    # Sort by confidence descending
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"\nDetected items in: {args.image}")
    print(f"Threshold: {config['confidence_threshold']}")
    print("-" * 40)
    if sorted_results:
        for item_name, score in sorted_results:
            print(f"  {item_name:<30s}  {score:.4f}")
    else:
        print("  No items detected above threshold.")
    print()

    if args.output_json:
        output = {name: float(score) for name, score in sorted_results}
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to: {args.output_json}")


if __name__ == "__main__":
    main()
