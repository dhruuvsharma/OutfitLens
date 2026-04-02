"""Loss functions: binary cross-entropy and focal loss for multi-label item prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Standard binary cross-entropy loss with optional per-class positive weighting."""

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        """Initialise BCE loss; pos_weight balances positive vs negative examples per class."""
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean BCE loss over the batch from raw logits and float targets in {0,1}."""
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="mean"
        )


class FocalLoss(nn.Module):
    """Focal loss for multi-label classification, down-weighting easy negatives.

    Focal loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    where p_t is the model's estimated probability for the true class.
    γ > 0 reduces the relative loss for well-classified examples, focusing
    training on hard, misclassified ones.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        """Initialise focal loss with modulating factor gamma and balance factor alpha."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean focal loss over the batch from raw logits and float targets."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def build_loss(config: dict, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Construct and return the appropriate loss module from config flags."""
    if config.get("focal_loss", False):
        return FocalLoss(gamma=2.0, alpha=0.25)
    return BCELoss(pos_weight=pos_weight)
