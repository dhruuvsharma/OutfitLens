"""Outfit recognizer: loads all specialist models and returns per-category ranked results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

from data_pipeline.augmentation import ValAugmentation
from models.specialist import Specialist


class OutfitRecognizer:
    """Runs all specialist models on an outfit image and returns per-category ranked candidates.

    Each specialist is independent — they do not share computation or backbone weights.
    Results are returned as raw cosine-similarity scores before any threshold filtering;
    use Aggregator to apply threshold and top-N.
    """

    def __init__(
        self,
        specialists: Dict[str, Specialist],
        image_size: int,
        top_k: int = 10,
    ) -> None:
        """Attach specialist dict, build val transform, set query top-k."""
        self.specialists = specialists
        self.transform = ValAugmentation(image_size=image_size)
        self.top_k = top_k

    def recognize(self, image_path: Path) -> Dict[str, List[Tuple[str, float]]]:
        """Run all specialists on one image; return {category: [(item_name, score), ...]}."""
        img = Image.open(Path(image_path)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)  # (1, 3, H, W)

        results: Dict[str, List[Tuple[str, float]]] = {}
        for category, specialist in self.specialists.items():
            results[category] = specialist.query(tensor, top_k=self.top_k)
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_recognizer(
    base_config: Dict,
    categories_config: List[Dict],
    logs_dir: Path | None = None,
) -> OutfitRecognizer:
    """Build OutfitRecognizer from base config and categories list.

    For each category entry, loads checkpoint from ``logs/<category>/best_model.pt``
    and DB from ``logs/<category>/embedding_db.npz``.
    """
    logs_dir = Path(logs_dir) if logs_dir else Path(base_config["logs_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size: int = base_config["image_size"]
    top_n: int = base_config.get("top_n_results", 5)

    specialists: Dict[str, Specialist] = {}
    for cat_entry in categories_config:
        category: str = cat_entry["name"]
        checkpoint_path = logs_dir / category / "best_model.pt"
        db_path = logs_dir / category / "embedding_db.npz"

        if not checkpoint_path.exists():
            print(f"WARNING: checkpoint not found for '{category}' at {checkpoint_path} — skipping")
            continue
        if not db_path.exists():
            print(f"WARNING: embedding DB not found for '{category}' at {db_path} — skipping")
            continue

        specialists[category] = Specialist.from_checkpoint(
            config=base_config,
            checkpoint_path=checkpoint_path,
            db_path=db_path,
            device=device,
        )
        print(f"Loaded specialist: {category}  ({len(specialists[category].db)} items in DB)")

    return OutfitRecognizer(specialists=specialists, image_size=image_size, top_k=top_n * 2)
