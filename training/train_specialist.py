"""Specialist training loop: cross-entropy classification + reference DB population for one category."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline.augmentation import TrainAugmentation, ValAugmentation
from data_pipeline.dataset import SpecialistDataset, build_splits, load_item_names
from models.backbone import ResNet34Backbone
from models.embedding_db import EmbeddingDB
from training.losses import build_loss
from training.metrics import compute_and_log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_specialist(config: Dict, category: str) -> None:
    """Train one specialist model for one category, then populate its embedding DB."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training specialist: {category}  |  device: {device}")
    print(f"{'='*60}")

    synthetic_dir = Path(config["synthetic_dir"]) / category
    logs_dir = Path(config["logs_dir"]) / category
    raw_dir = Path(config["raw_data_dir"]) / category
    logs_dir.mkdir(parents=True, exist_ok=True)

    labels_file = synthetic_dir / "labels.json"
    item_names = load_item_names(labels_file)
    num_classes = len(item_names)
    print(f"[{category}] {num_classes} unique items")

    # Build train/val split
    train_samples, val_samples = build_splits(labels_file, config["val_split"])
    images_dir = synthetic_dir / "images"
    image_size: int = config["image_size"]
    batch_size: int = config["batch_size"]

    train_ds = SpecialistDataset(train_samples, images_dir, item_names, TrainAugmentation(image_size))
    val_ds = SpecialistDataset(val_samples, images_dir, item_names, ValAugmentation(image_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    backbone = ResNet34Backbone(embedding_dim=config["embedding_dim"]).to(device)
    head = nn.Linear(config["embedding_dim"], num_classes).to(device)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)

    criterion = build_loss(config)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()),
        lr=config["learning_rate"],
    )

    log_path = logs_dir / "train_log.csv"
    best_top1 = -1.0
    epochs: int = config["epochs"]

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(backbone, head, train_loader, criterion, optimizer, device)
        val_loss, y_true, logits = _val_epoch(backbone, head, val_loader, criterion, device)

        top1, top5 = compute_and_log(y_true, logits, item_names, log_path, epoch, train_loss, val_loss)
        print(
            f"[{category}] Epoch {epoch:03d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"top1={top1:.4f}  top5={top5:.4f}"
        )

        if top1 > best_top1:
            best_top1 = top1
            _save_checkpoint(backbone, item_names, epoch, best_top1, logs_dir)

    print(f"[{category}] Training complete. Best val top-1: {best_top1:.4f}")

    # Populate reference embedding DB from raw renders
    best_ckpt = logs_dir / "best_model.pt"
    db_path = logs_dir / "embedding_db.npz"
    _populate_db(config, category, raw_dir, best_ckpt, db_path, device)


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def _train_epoch(
    backbone: ResNet34Backbone,
    head: nn.Linear,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean batch loss."""
    backbone.train()
    head.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        features = backbone.forward_features(images)
        logits = head(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def _val_epoch(
    backbone: ResNet34Backbone,
    head: nn.Linear,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run validation; return (mean_loss, y_true int array, logits float array)."""
    backbone.eval()
    head.eval()
    total_loss = 0.0
    all_true: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            features = backbone.forward_features(images)
            logits = head(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_logits.append(logits.cpu().numpy())
            all_true.append(labels.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    return total_loss / max(len(loader), 1), y_true, logits_arr


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    backbone: ResNet34Backbone,
    item_names: List[str],
    epoch: int,
    best_top1: float,
    logs_dir: Path,
) -> None:
    """Save backbone state dict and item name mapping as the best checkpoint."""
    ckpt = {
        "epoch": epoch,
        "best_top1": best_top1,
        "backbone_state": backbone.state_dict(),
        "item_names": item_names,
    }
    torch.save(ckpt, logs_dir / "best_model.pt")
    print(f"  -> Saved best checkpoint (epoch {epoch}, top-1 {best_top1:.4f})")


# ---------------------------------------------------------------------------
# Reference DB population
# ---------------------------------------------------------------------------

def _populate_db(
    config: Dict,
    category: str,
    raw_dir: Path,
    checkpoint_path: Path,
    db_path: Path,
    device: torch.device,
) -> None:
    """Embed all raw renders for the category and save the reference EmbeddingDB."""
    from PIL import Image
    from data_pipeline.augmentation import ValAugmentation

    print(f"[{category}] Populating reference DB from {raw_dir} ...")

    backbone = ResNet34Backbone(embedding_dim=config["embedding_dim"]).to(device).eval()
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])

    transform = ValAugmentation(image_size=config["image_size"])
    angle_suffixes = ("_front", "_back", "_left", "_right")
    single_angle: bool = config.get("single_angle", False)

    item_paths: Dict[str, List[Path]] = {}
    for img_path in sorted(raw_dir.glob("*.*")):
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

    db = EmbeddingDB()
    with torch.no_grad():
        for item_name, paths in tqdm(item_paths.items(), desc=f"embed {category}"):
            angle_embeds = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                emb = backbone(tensor).squeeze(0).cpu().numpy()
                angle_embeds.append(emb)

            if single_angle or len(angle_embeds) == 1:
                final_emb = angle_embeds[0]
            else:
                avg = np.mean(np.stack(angle_embeds, axis=0), axis=0)
                norm = float(np.linalg.norm(avg))
                final_emb = avg / max(norm, 1e-8)

            db.add(item_name, final_emb)

    db.save(db_path)
    print(f"[{category}] Reference DB saved: {db_path}  ({len(db)} items)")
