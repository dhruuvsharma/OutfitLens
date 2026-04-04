"""Specialist model: wraps one ResNet-34 backbone + one per-category EmbeddingDB."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from models.backbone import ResNet34Backbone
from models.embedding_db import EmbeddingDB


class Specialist:
    """Combines a backbone and embedding DB for a single clothing category.

    Stateless after loading — no mutable state is modified during inference.
    Each specialist has its own independent backbone weights; nothing is shared
    across categories.
    """

    def __init__(
        self,
        backbone: ResNet34Backbone,
        db: EmbeddingDB,
        device: torch.device,
    ) -> None:
        """Attach backbone and DB; move backbone to device and set eval mode."""
        self.backbone = backbone.to(device).eval()
        self.db = db
        self.device = device

    def query(self, image_tensor: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """Embed image_tensor and return top-k [(item_name, cosine_score), ...] sorted descending."""
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            embedding = self.backbone(image_tensor)  # L2-normalised
        vec: np.ndarray = embedding.squeeze(0).cpu().numpy()
        return self.db.query(vec, top_k=top_k)

    @classmethod
    def from_checkpoint(
        cls,
        config: Dict,
        checkpoint_path: Path,
        db_path: Path,
        device: torch.device | None = None,
    ) -> "Specialist":
        """Load a Specialist from a checkpoint .pt and an embedding DB .npz file."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(checkpoint_path)
        db_path = Path(db_path)

        backbone = ResNet34Backbone(embedding_dim=config["embedding_dim"])
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        backbone.load_state_dict(ckpt["backbone_state"])

        db = EmbeddingDB()
        db.load(db_path)

        return cls(backbone=backbone, db=db, device=device)
