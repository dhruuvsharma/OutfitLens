"""PyTorch Dataset for single-label specialist training: reads per-category labels.json."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class SpecialistDataset(Dataset):
    """Loads composite images and their single-label item index for one clothing category.

    The labels.json produced by the compositor is a list of dicts:
    ``[{"image": filename, "item_name": "Cowboy_Hat", "category": "hats"}, ...]``

    Each sample returns ``(image_tensor, item_index)`` where item_index is the
    integer position of item_name in the sorted list of unique item names.
    """

    def __init__(
        self,
        samples: List[Dict],
        images_dir: Path,
        item_names: List[str],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Build dataset from a pre-filtered sample list and the category item name index."""
        self.images_dir = Path(images_dir)
        self.item_names = item_names
        self.transform = transform
        self._name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(item_names)}
        self.samples = samples

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return (image_tensor, item_index) for the given index."""
        entry = self.samples[idx]
        img_path = self.images_dir / entry["image"]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            import numpy as np
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        item_idx = self._name_to_idx[entry["item_name"]]
        return tensor, item_idx

    @property
    def num_classes(self) -> int:
        """Return the number of unique item classes for this category."""
        return len(self.item_names)

    @property
    def class_names(self) -> List[str]:
        """Return the ordered list of item names (index → name mapping)."""
        return list(self.item_names)


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def build_splits(
    labels_file: Path, val_split: float
) -> Tuple[List[Dict], List[Dict]]:
    """Split samples from labels.json into train and val lists; return (train, val)."""
    labels_file = Path(labels_file)
    with open(labels_file) as f:
        all_samples: List[Dict] = json.load(f)

    shuffled = list(all_samples)
    random.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_split))
    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]

    print(f"Splits: {len(train_samples)} train / {len(val_samples)} val")
    return train_samples, val_samples


def load_item_names(labels_file: Path) -> List[str]:
    """Return sorted unique item names from a labels.json file."""
    labels_file = Path(labels_file)
    with open(labels_file) as f:
        samples: List[Dict] = json.load(f)
    return sorted({s["item_name"] for s in samples})
