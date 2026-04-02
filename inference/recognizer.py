"""Outfit recognizer: global + regional embedding inference against a reference database."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from data_pipeline.augmentation import ValAugmentation
from models.backbone import ResNet34Backbone
from models.embedding_db import EmbeddingDB


class OutfitRecognizer:
    """Identifies clothing items in an outfit image using embedding similarity.

    Two inference paths are combined:
    1. **Global path** — the full image is embedded and queried against the DB.
    2. **Regional path** — the image is divided into a 3×3 grid of overlapping crops;
       each crop is embedded and queried independently.

    Results from both paths are merged by taking the highest confidence score
    per item, then filtered by the configured threshold.
    """

    GRID_SIZE: int = 3  # 3×3 sliding window for regional inference

    def __init__(
        self,
        config: Dict,
        backbone: ResNet34Backbone,
        db: EmbeddingDB,
        device: torch.device | None = None,
    ) -> None:
        """Attach a loaded backbone and embedding DB; build the val transform."""
        self.config = config
        self.backbone = backbone
        self.db = db
        self.device = device or torch.device("cpu")
        self.backbone.to(self.device).eval()
        self.transform = ValAugmentation(image_size=config["image_size"])
        self.threshold: float = config.get("confidence_threshold", 0.75)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image) -> Dict[str, float]:
        """Return {item_name: confidence} for all items above the confidence threshold."""
        global_hits = self._global_inference(image)
        regional_hits = self._regional_inference(image)
        merged = _merge_hits([global_hits, regional_hits])
        return {name: score for name, score in merged.items() if score >= self.threshold}

    def predict_path(self, image_path: Path) -> Dict[str, float]:
        """Load an image from disk and run predict()."""
        img = Image.open(Path(image_path)).convert("RGB")
        return self.predict(img)

    # ------------------------------------------------------------------
    # Inference paths
    # ------------------------------------------------------------------

    def _global_inference(self, image: Image.Image) -> Dict[str, float]:
        """Embed the full image and return cosine-similarity scores for all DB items."""
        vec = self._embed(image)
        hits = self.db.query(vec, top_k=len(self.db))
        return {name: score for name, score in hits}

    def _regional_inference(self, image: Image.Image) -> Dict[str, float]:
        """Embed each cell of a 3×3 grid and merge per-item max scores."""
        crops = _grid_crops(image, self.GRID_SIZE)
        all_hits: List[Dict[str, float]] = []
        for crop in crops:
            vec = self._embed(crop)
            hits = self.db.query(vec, top_k=len(self.db))
            all_hits.append({name: score for name, score in hits})
        return _merge_hits(all_hits)

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed(self, image: Image.Image) -> np.ndarray:
        """Preprocess a PIL image and return a 1-D numpy embedding vector."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.backbone(tensor)  # L2-normalised
        return emb.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_recognizer(config: Dict, checkpoint_path: Path, db_path: Path) -> OutfitRecognizer:
    """Build a ready-to-use OutfitRecognizer from a config dict and file paths."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = ResNet34Backbone(embedding_dim=config["embedding_dim"])
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])

    db = EmbeddingDB()
    db.load(Path(db_path))

    return OutfitRecognizer(config=config, backbone=backbone, db=db, device=device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_crops(image: Image.Image, grid: int) -> List[Image.Image]:
    """Divide an image into grid×grid non-overlapping crops and return them."""
    w, h = image.size
    cell_w = w // grid
    cell_h = h // grid
    crops: List[Image.Image] = []
    for row in range(grid):
        for col in range(grid):
            left = col * cell_w
            top = row * cell_h
            right = left + cell_w
            bottom = top + cell_h
            crops.append(image.crop((left, top, right, bottom)))
    return crops


def _merge_hits(hit_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Merge multiple hit dicts by keeping the highest confidence score per item."""
    merged: Dict[str, float] = {}
    for hits in hit_dicts:
        for name, score in hits.items():
            if name not in merged or score > merged[name]:
                merged[name] = score
    return merged
