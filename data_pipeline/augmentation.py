"""Image augmentation transforms for training: flip, colour jitter, scale, occlusion, normalise."""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance


# ImageNet channel statistics used as normalisation constants (no pretrained dependency).
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class TrainAugmentation:
    """Stochastic augmentation pipeline applied to PIL images during training.

    Stages (applied in order):
    1. Random horizontal flip (p=0.5)
    2. Hue/saturation jitter ±20 %
    3. Random scale (0.8–1.2×) with centre-crop back to target size
    4. Random occlusion patch (black box, 10–20 % of image area)
    5. ToTensor + normalise with ImageNet mean/std
    """

    def __init__(self, image_size: int = 224) -> None:
        """Initialise augmentation pipeline for the given square image size."""
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply the full augmentation pipeline and return a normalised tensor."""
        img = _random_flip(img)
        img = _colour_jitter(img, hue_range=0.10, sat_range=0.20)
        img = _random_scale_crop(img, self.image_size, scale_range=(0.8, 1.2))
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = TF.to_tensor(img)  # [0, 1] float32
        tensor = _random_occlusion(tensor, area_fraction_range=(0.10, 0.20))
        tensor = TF.normalize(tensor, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))
        return tensor


class ValAugmentation:
    """Deterministic preprocessing pipeline for validation/inference images."""

    def __init__(self, image_size: int = 224) -> None:
        """Initialise the validation pipeline for the given square image size."""
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Resize, convert to tensor, and normalise — no stochastic transforms."""
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))
        return tensor


# ---------------------------------------------------------------------------
# Individual transform helpers
# ---------------------------------------------------------------------------

def _random_flip(img: Image.Image) -> Image.Image:
    """Randomly flip the image horizontally with probability 0.5."""
    if random.random() < 0.5:
        return TF.hflip(img)
    return img


def _colour_jitter(img: Image.Image, hue_range: float, sat_range: float) -> Image.Image:
    """Apply random hue rotation and saturation scaling within the given ranges."""
    # Saturation jitter: factor drawn from [1 - sat_range, 1 + sat_range]
    sat_factor = 1.0 + random.uniform(-sat_range, sat_range)
    img = ImageEnhance.Color(img).enhance(max(0.0, sat_factor))

    # Hue jitter via HSV shift on the raw pixel values (±hue_range in fraction of 360°)
    hue_shift = random.uniform(-hue_range, hue_range)
    img = TF.adjust_hue(img, hue_shift)
    return img


def _random_scale_crop(img: Image.Image, target_size: int, scale_range: Tuple[float, float]) -> Image.Image:
    """Scale the image by a random factor then centre-crop to target_size × target_size."""
    scale = random.uniform(*scale_range)
    w, h = img.size
    new_w = max(target_size, int(w * scale))
    new_h = max(target_size, int(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Centre crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    return img.crop((left, top, left + target_size, top + target_size))


def _random_occlusion(tensor: torch.Tensor, area_fraction_range: Tuple[float, float]) -> torch.Tensor:
    """Zero-out a random rectangular patch covering area_fraction of the image."""
    _, h, w = tensor.shape
    area = h * w
    frac = random.uniform(*area_fraction_range)
    patch_area = int(area * frac)

    # Aspect ratio of the patch drawn uniformly from [0.5, 2.0]
    aspect = random.uniform(0.5, 2.0)
    patch_h = max(1, int((patch_area / aspect) ** 0.5))
    patch_w = max(1, int(patch_area / patch_h))
    patch_h = min(patch_h, h)
    patch_w = min(patch_w, w)

    top = random.randint(0, h - patch_h)
    left = random.randint(0, w - patch_w)
    tensor = tensor.clone()
    tensor[:, top : top + patch_h, left : left + patch_w] = 0.0
    return tensor
