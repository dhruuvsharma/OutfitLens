"""Build the reference embedding database from raw asset renders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.augmentation import ValAugmentation
from models.backbone import ResNet34Backbone
from models.embedding_db import EmbeddingDB
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the reference DB builder."""
    p = argparse.ArgumentParser(description="Build OutfitLens reference embedding database")
    p.add_argument("--config", required=True, help="Path to config yaml")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt checkpoint")
    p.add_argument("--output", default=None, help="Output .npz path (default: logs/reference_db.npz)")
    return p.parse_args()


def build_db(config: dict, checkpoint_path: Path, output_path: Path) -> None:
    """Embed all assets and save the reference database to output_path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets_dir = Path(config["assets_dir"])
    single_angle: bool = config.get("single_angle", False)
    image_size: int = config["image_size"]
    embedding_dim: int = config["embedding_dim"]

    # Load backbone
    backbone = ResNet34Backbone(embedding_dim=embedding_dim).to(device).eval()
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    print(f"Loaded backbone from {checkpoint_path}")

    transform = ValAugmentation(image_size=image_size)
    angle_suffixes = ("_front", "_back", "_left", "_right")

    # Discover items
    item_paths: dict[str, list[Path]] = {}
    for img_path in sorted(assets_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = img_path.stem
        if single_angle:
            item_name = stem.replace("_front", "")
            item_paths.setdefault(item_name, []).append(img_path)
        else:
            matched = next((s for s in angle_suffixes if stem.endswith(s)), None)
            if matched:
                item_name = stem[: -len(matched)]
                item_paths.setdefault(item_name, []).append(img_path)

    if not item_paths:
        raise RuntimeError(f"No asset images found in {assets_dir}")

    db = EmbeddingDB()
    with torch.no_grad():
        for item_name, paths in tqdm(item_paths.items(), desc="Embedding items"):
            angle_embeds = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                emb = backbone(tensor).squeeze(0).cpu().numpy()
                angle_embeds.append(emb)

            if single_angle or len(angle_embeds) == 1:
                final_emb = angle_embeds[0]
            else:
                # Average angle embeddings, then re-normalise
                import numpy as np
                avg = sum(angle_embeds) / len(angle_embeds)
                norm = float((avg ** 2).sum() ** 0.5)
                final_emb = avg / max(norm, 1e-8)

            db.add(item_name, final_emb)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(output_path)
    print(f"Reference DB saved: {output_path}  ({len(db)} items)")


def main() -> None:
    """Entry point for build_reference_db script."""
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else Path(config["checkpoint_dir"]) / "reference_db.npz"
    build_db(config, checkpoint_path, output_path)


if __name__ == "__main__":
    main()
