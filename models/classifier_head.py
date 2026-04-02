"""Supervised classification head that sits on top of the ResNet-34 backbone."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Linear classification head for multi-label outfit item prediction.

    Accepts the 512-d embedding produced by ResNet34Backbone.forward_features()
    and projects it to a logit vector of length num_classes.  Binary
    cross-entropy loss is applied per class (items co-exist; no softmax).

    The head is intentionally kept separate from the backbone so that the
    backbone weights can be shared between the embedding path and the
    supervised path without entanglement.
    """

    def __init__(self, embedding_dim: int = 512, num_classes: int = 1) -> None:
        """Initialise the head with the given input and output dimensions."""
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(embedding_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project backbone features to per-class logits (no activation applied)."""
        return self.fc(features)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities in [0, 1] for each class."""
        return torch.sigmoid(self.forward(features))
