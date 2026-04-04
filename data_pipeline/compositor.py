"""Specialist compositor: generates per-category synthetic images with cross-category distractors."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_category_dataset(config: Dict, category: str) -> None:
    """Generate N synthetic composites for one category with other-category distractors."""
    raw_data_dir = Path(config["raw_data_dir"])
    category_dir = raw_data_dir / category
    synthetic_dir = Path(config["synthetic_dir"]) / category
    image_size: int = config["image_size"]
    n_samples: int = config["composite_count"]
    distractor_count: int = config.get("distractor_count", 2)
    single_angle: bool = config.get("single_angle", False)

    images_dir = synthetic_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    target_items = _load_category_items(category_dir, single_angle)
    if not target_items:
        raise RuntimeError(f"No valid asset images found in {category_dir}")

    distractor_pool = _load_distractor_pool(raw_data_dir, category, single_angle)

    labels: List[Dict] = []
    for idx in range(n_samples):
        target_name, target_img = random.choice(target_items)

        n_dist = min(distractor_count, len(distractor_pool))
        distractors = [img for _, img in random.sample(distractor_pool, n_dist)] if n_dist > 0 else []

        composite = _compose(target_img, distractors, image_size)
        filename = f"composite_{idx:05d}.jpg"
        composite.save(str(images_dir / filename), quality=90)

        labels.append({"image": filename, "item_name": target_name, "category": category})

    labels_path = synthetic_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    item_names = sorted({e["item_name"] for e in labels})
    print(f"[{category}] Generated {n_samples} composites → {synthetic_dir}")
    print(f"[{category}] {len(item_names)} unique items, {len(distractor_pool)} distractor renders available")


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------

def _load_category_items(
    category_dir: Path, single_angle: bool
) -> List[Tuple[str, Image.Image]]:
    """Return [(item_name, representative_PIL_image)] for all items in a category folder."""
    if single_angle:
        return _load_single_angle(category_dir)
    return _load_multi_angle(category_dir)


def _load_distractor_pool(
    raw_data_dir: Path, exclude_category: str, single_angle: bool
) -> List[Tuple[str, Image.Image]]:
    """Return flat list of (item_name, image) from all categories except exclude_category."""
    pool: List[Tuple[str, Image.Image]] = []
    for cat_dir in sorted(raw_data_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == exclude_category:
            continue
        pool.extend(_load_category_items(cat_dir, single_angle))
    return pool


def _load_single_angle(folder: Path) -> List[Tuple[str, Image.Image]]:
    """Load one front-view (or any) image per item from a folder."""
    item_map: Dict[str, Image.Image] = {}
    for img_path in sorted(folder.glob("*_front.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        item_name = img_path.stem.replace("_front", "")
        item_map[item_name] = Image.open(img_path).convert("RGBA")
    if not item_map:
        for img_path in sorted(folder.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            item_name = img_path.stem
            if item_name not in item_map:
                item_map[item_name] = Image.open(img_path).convert("RGBA")
    return [(name, img) for name, img in sorted(item_map.items())]


def _load_multi_angle(folder: Path) -> List[Tuple[str, Image.Image]]:
    """Load multi-angle renders; use the front view as the pixel representative."""
    angle_suffixes = ("_front", "_back", "_left", "_right")
    items_found: Dict[str, Image.Image] = {}

    for img_path in sorted(folder.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = img_path.stem
        matched = next((s for s in angle_suffixes if stem.endswith(s)), None)
        if matched is None:
            continue
        item_name = stem[: -len(matched)]
        # Keep the first (alphabetically = typically front) render as representative
        if item_name not in items_found:
            items_found[item_name] = Image.open(img_path).convert("RGBA")

    return [(name, img) for name, img in sorted(items_found.items())]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def _make_background(size: int) -> Image.Image:
    """Create a plain or mild noise RGB background canvas."""
    if random.random() < 0.5:
        r, g, b = random.randint(180, 255), random.randint(180, 255), random.randint(180, 255)
        bg = Image.new("RGB", (size, size), (r, g, b))
    else:
        arr = np.random.randint(180, 256, (size, size, 3), dtype=np.uint8)
        noise = np.random.randint(-15, 16, (size, size, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(arr, mode="RGB")
    return bg


def _paste_item(canvas: Image.Image, item_img: Image.Image, size: int) -> None:
    """Paste one item onto canvas at a random scale and position (in-place)."""
    scale = random.uniform(0.25, 0.65)
    item_w = max(10, int(size * scale))
    item_h = max(10, int(item_img.height * item_w / max(item_img.width, 1)))
    item_resized = item_img.resize((item_w, item_h), Image.LANCZOS)

    max_x = max(0, size - item_w)
    max_y = max(0, size - item_h)
    x = random.randint(-item_w // 4, max_x + item_w // 4)
    y = random.randint(-item_h // 4, max_y + item_h // 4)
    x = max(-item_w + 1, min(x, size - 1))
    y = max(-item_h + 1, min(y, size - 1))

    if item_resized.mode == "RGBA":
        paste_x, paste_y = max(0, x), max(0, y)
        crop_x, crop_y = max(0, -x), max(0, -y)
        crop_w = min(item_w - crop_x, size - paste_x)
        crop_h = min(item_h - crop_y, size - paste_y)
        if crop_w <= 0 or crop_h <= 0:
            return
        region = item_resized.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        canvas.paste(region.convert("RGB"), (paste_x, paste_y), region.split()[3])
    else:
        canvas.paste(item_resized.convert("RGB"), (x, y))


def _compose(target_img: Image.Image, distractors: List[Image.Image], size: int) -> Image.Image:
    """Layer distractors then target item onto a background canvas."""
    canvas = _make_background(size)
    # Distractors go behind the target item
    for dist_img in distractors:
        _paste_item(canvas, dist_img, size)
    # Target item always drawn on top (positive signal)
    _paste_item(canvas, target_img, size)
    return canvas
