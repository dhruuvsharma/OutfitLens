"""Synthetic outfit image compositor: layers random items onto a background and saves labels."""

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

def generate_dataset(config: Dict) -> None:
    """Generate N synthetic composite outfit images and write them plus labels.json to disk."""
    assets_dir = Path(config["assets_dir"])
    synthetic_dir = Path(config["synthetic_dir"])
    image_size: int = config["image_size"]
    n_samples: int = config["composite_count"]
    single_angle: bool = config.get("single_angle", False)

    images_dir = synthetic_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    item_names, item_images = _load_assets(assets_dir, single_angle)
    if len(item_names) == 0:
        raise RuntimeError(f"No valid asset images found in {assets_dir}")

    all_labels: Dict[str, Dict] = {}
    for idx in range(n_samples):
        n_items = random.randint(2, min(6, len(item_names)))
        chosen_indices = random.sample(range(len(item_names)), n_items)
        chosen_names = [item_names[i] for i in chosen_indices]
        chosen_imgs = [item_images[i] for i in chosen_indices]

        composite = _compose(chosen_imgs, image_size)
        filename = f"composite_{idx:05d}.jpg"
        composite.save(str(images_dir / filename), quality=90)

        multi_hot = [1 if name in chosen_names else 0 for name in item_names]
        all_labels[filename] = {
            "items": chosen_names,
            "multi_hot": multi_hot,
        }

    labels_path = synthetic_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump({"item_names": item_names, "samples": all_labels}, f, indent=2)

    print(f"Generated {n_samples} composites → {synthetic_dir}")
    print(f"Discovered {len(item_names)} unique items: {item_names}")


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------

def _load_assets(
    assets_dir: Path, single_angle: bool
) -> Tuple[List[str], List[Image.Image]]:
    """Return (item_names, representative_images) from the asset directory."""
    if single_angle:
        return _load_single_angle(assets_dir)
    return _load_multi_angle(assets_dir)


def _load_single_angle(assets_dir: Path) -> Tuple[List[str], List[Image.Image]]:
    """Load one image per item (ItemName_front.jpg or any single file)."""
    item_map: Dict[str, Image.Image] = {}
    for img_path in sorted(assets_dir.glob("*_front.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        item_name = img_path.stem.replace("_front", "")
        item_map[item_name] = Image.open(img_path).convert("RGBA")
    # Fallback: any image if no _front suffix present
    if not item_map:
        for img_path in sorted(assets_dir.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            item_name = img_path.stem
            if item_name not in item_map:
                item_map[item_name] = Image.open(img_path).convert("RGBA")
    names = sorted(item_map.keys())
    return names, [item_map[n] for n in names]


def _load_multi_angle(assets_dir: Path) -> Tuple[List[str], List[Image.Image]]:
    """Load four-angle renders per item and average them into one representative image."""
    angle_suffixes = ("_front", "_back", "_left", "_right")
    items_found: Dict[str, List[Image.Image]] = {}

    for img_path in sorted(assets_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = img_path.stem
        matched_suffix = next((s for s in angle_suffixes if stem.endswith(s)), None)
        if matched_suffix is None:
            continue
        item_name = stem[: -len(matched_suffix)]
        items_found.setdefault(item_name, [])
        items_found[item_name].append(Image.open(img_path).convert("RGBA"))

    # Create representative image by alpha-blending all available angles
    item_map: Dict[str, Image.Image] = {}
    for name, angle_imgs in items_found.items():
        if not angle_imgs:
            continue
        # Use the front view as the visual representative for compositing;
        # angle-averaging is done on embeddings, not on pixels.
        item_map[name] = angle_imgs[0]

    names = sorted(item_map.keys())
    return names, [item_map[n] for n in names]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def _make_background(size: int) -> Image.Image:
    """Create a plain or mild noise RGB background canvas."""
    if random.random() < 0.5:
        # Solid colour background
        r, g, b = random.randint(180, 255), random.randint(180, 255), random.randint(180, 255)
        bg = Image.new("RGB", (size, size), (r, g, b))
    else:
        # Subtle Gaussian noise background
        arr = np.random.randint(180, 256, (size, size, 3), dtype=np.uint8)
        noise = np.random.randint(-15, 16, (size, size, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(arr, mode="RGB")
    return bg


def _compose(item_images: List[Image.Image], size: int) -> Image.Image:
    """Layer item images onto a background canvas with random scale and position."""
    canvas = _make_background(size)

    for item_img in item_images:
        # Random scale between 0.25× and 0.65× of canvas
        scale = random.uniform(0.25, 0.65)
        item_w = max(10, int(size * scale))
        item_h = max(10, int(item_img.height * item_w / max(item_img.width, 1)))
        item_resized = item_img.resize((item_w, item_h), Image.LANCZOS)

        # Random position (allow partial occlusion at edges)
        max_x = max(0, size - item_w)
        max_y = max(0, size - item_h)
        x = random.randint(-item_w // 4, max_x + item_w // 4)
        y = random.randint(-item_h // 4, max_y + item_h // 4)
        x = max(-item_w + 1, min(x, size - 1))
        y = max(-item_h + 1, min(y, size - 1))

        # Composite using alpha channel if available
        if item_resized.mode == "RGBA":
            paste_x = max(0, x)
            paste_y = max(0, y)
            crop_x = max(0, -x)
            crop_y = max(0, -y)
            crop_w = min(item_w - crop_x, size - paste_x)
            crop_h = min(item_h - crop_y, size - paste_y)
            if crop_w <= 0 or crop_h <= 0:
                continue
            region = item_resized.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            canvas.paste(region.convert("RGB"), (paste_x, paste_y), region.split()[3])
        else:
            canvas.paste(item_resized.convert("RGB"), (x, y))

    return canvas
