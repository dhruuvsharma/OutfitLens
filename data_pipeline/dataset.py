"""PyTorch Dataset for synthetic outfit images: reads labels.json and returns tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class OutfitDataset(Dataset):
    """Loads composite outfit images and their multi-hot labels from a split JSON file.

    The split file (train.json / val.json) is a list of filename strings that index
    into the master labels.json produced by the compositor.  For each sample the
    dataset returns ``(image_tensor, multi_hot_label_tensor)``.
    """

    def __init__(
        self,
        split_file: Path,
        labels_file: Path,
        images_dir: Path,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Load split list, master labels, and item name index from disk."""
        split_file = Path(split_file)
        labels_file = Path(labels_file)
        self.images_dir = Path(images_dir)
        self.transform = transform

        with open(labels_file) as f:
            master = json.load(f)
        self.item_names: List[str] = master["item_names"]
        self.num_classes: int = len(self.item_names)
        all_samples: Dict[str, Dict] = master["samples"]

        with open(split_file) as f:
            split_filenames: List[str] = json.load(f)

        self.samples: List[Tuple[str, List[int]]] = [
            (fname, all_samples[fname]["multi_hot"])
            for fname in split_filenames
            if fname in all_samples
        ]

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (image_tensor, multi_hot_label_tensor) for the given index."""
        filename, multi_hot = self.samples[idx]
        img_path = self.images_dir / filename
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = torch.from_numpy(
                _pil_to_numpy(img)
            ).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(multi_hot, dtype=torch.float32)
        return tensor, label

    @property
    def class_names(self) -> List[str]:
        """Return the ordered list of item names corresponding to label columns."""
        return list(self.item_names)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_numpy(img: Image.Image):  # type: ignore[return]
    """Convert a PIL image to a numpy uint8 array without importing numpy at module level."""
    import numpy as np  # deferred to avoid hard dep at import time
    return np.array(img)


def build_splits(labels_file: Path, synthetic_dir: Path, val_split: float) -> None:
    """Create train.json and val.json split files from the master labels.json."""
    import random

    labels_file = Path(labels_file)
    splits_dir = synthetic_dir.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_file) as f:
        master = json.load(f)
    filenames = list(master["samples"].keys())
    random.shuffle(filenames)

    n_val = max(1, int(len(filenames) * val_split))
    val_files = filenames[:n_val]
    train_files = filenames[n_val:]

    with open(splits_dir / "train.json", "w") as f:
        json.dump(train_files, f, indent=2)
    with open(splits_dir / "val.json", "w") as f:
        json.dump(val_files, f, indent=2)

    print(f"Splits: {len(train_files)} train / {len(val_files)} val")
