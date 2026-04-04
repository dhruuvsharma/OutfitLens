# SUPERSEDED by training/train_specialist.py (v2 specialist architecture).
# This v1 file is kept for reference only — imports will fail against the v2 API.
"""Training loop: binary cross-entropy multi-label training with periodic validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline.augmentation import TrainAugmentation, ValAugmentation
from data_pipeline.dataset import OutfitDataset, build_splits
from models.backbone import ResNet34Backbone
from models.classifier_head import ClassifierHead
from training.losses import build_loss
from training.metrics import compute_and_log, mean_f1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_training(config: Dict) -> None:
    """Execute the full training pipeline from a config dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    synthetic_dir = Path(config["synthetic_dir"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    labels_file = synthetic_dir / "labels.json"
    _ensure_splits(labels_file, synthetic_dir, config["val_split"])

    splits_dir = synthetic_dir.parent / "splits"
    image_size: int = config["image_size"]
    batch_size: int = config["batch_size"]

    train_ds = OutfitDataset(
        split_file=splits_dir / "train.json",
        labels_file=labels_file,
        images_dir=synthetic_dir / "images",
        transform=TrainAugmentation(image_size),
    )
    val_ds = OutfitDataset(
        split_file=splits_dir / "val.json",
        labels_file=labels_file,
        images_dir=synthetic_dir / "images",
        transform=ValAugmentation(image_size),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = train_ds.num_classes
    item_names = train_ds.class_names
    print(f"Classes: {num_classes}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    backbone = ResNet34Backbone(embedding_dim=config["embedding_dim"]).to(device)
    head = ClassifierHead(embedding_dim=config["embedding_dim"], num_classes=num_classes).to(device)

    criterion = build_loss(config)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()),
        lr=config["learning_rate"],
    )

    log_path = checkpoint_dir / "train_log.csv"
    best_f1 = -1.0
    epochs: int = config["epochs"]

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(backbone, head, train_loader, criterion, optimizer, device)
        val_loss, y_true, y_pred = _val_epoch(backbone, head, val_loader, criterion, device)

        metrics, mAP = compute_and_log(y_true, y_pred, item_names, log_path, epoch, val_loss)
        mf1 = float(metrics["f1"].mean())

        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mean_F1={mf1:.4f}  mAP={mAP:.4f}"
        )

        if mf1 > best_f1:
            best_f1 = mf1
            _save_checkpoint(backbone, head, optimizer, epoch, best_f1, checkpoint_dir)

    print(f"Training complete. Best val mean-F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def _train_epoch(
    backbone: ResNet34Backbone,
    head: ClassifierHead,
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
    head: ClassifierHead,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run one validation epoch; return (mean_loss, y_true, y_pred_proba)."""
    backbone.eval()
    head.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            features = backbone.forward_features(images)
            logits = head(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_pred.append(probs)
            all_true.append(labels.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return total_loss / max(len(loader), 1), y_true, y_pred


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    backbone: ResNet34Backbone,
    head: ClassifierHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_f1: float,
    checkpoint_dir: Path,
) -> None:
    """Save backbone + head state dicts as the best model checkpoint."""
    ckpt = {
        "epoch": epoch,
        "best_f1": best_f1,
        "backbone_state": backbone.state_dict(),
        "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(ckpt, checkpoint_dir / "best_model.pt")
    print(f"  ✓ Saved best checkpoint (epoch {epoch}, mean-F1 {best_f1:.4f})")


def _ensure_splits(labels_file: Path, synthetic_dir: Path, val_split: float) -> None:
    """Create train/val splits if they do not exist yet."""
    splits_dir = synthetic_dir.parent / "splits"
    if not (splits_dir / "train.json").exists():
        build_splits(labels_file, synthetic_dir, val_split)
