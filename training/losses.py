"""Loss functions: cross-entropy and focal loss for single-label specialist classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss for single-label classification."""

    def __init__(self) -> None:
        """Initialise cross-entropy loss."""
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean cross-entropy loss from raw logits and integer class targets."""
        return F.cross_entropy(logits, targets)


class FocalLossCE(nn.Module):
    """Focal loss variant of cross-entropy for single-label classification.

    Focal loss: FL(p_t) = -(1 - p_t)^γ * log(p_t)
    Reduces the contribution of easy examples, focusing training on hard ones.
    Useful when most synthetic images are correctly classified quickly.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        """Initialise focal loss with modulating exponent gamma."""
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean focal loss from raw logits and integer class targets."""
        log_probs = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_probs, targets, reduction="none")
        probs = torch.exp(log_probs)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * ce).mean()


def build_loss(config: dict) -> nn.Module:
    """Construct and return the appropriate loss module from config flags."""
    if config.get("focal_loss", False):
        return FocalLossCE(gamma=2.0)
    return CrossEntropyLoss()
