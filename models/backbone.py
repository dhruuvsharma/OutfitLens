"""ResNet-34 CNN backbone built from scratch, producing 512-d L2-normalized embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BasicBlock(nn.Module):
    """Standard residual block with two 3×3 convolutions and a skip connection."""

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialise block, creating a downsample shortcut when dimensions change."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample: nn.Sequential | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block: F(x) + shortcut(x)."""
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
    """Build a sequence of residual blocks for one ResNet stage."""
    layers = [BasicBlock(in_channels, out_channels, stride)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class ResNet34Backbone(nn.Module):
    """ResNet-34 backbone producing a 512-d L2-normalised embedding vector.

    Architecture follows the original He et al. (2016) ResNet-34 design:
    [3,4,6,3] blocks across four stages with channel widths [64,128,256,512].
    The final global average pool collapses spatial dimensions to a single vector,
    which is then L2-normalised for cosine-similarity queries.
    """

    STAGE_CHANNELS: Tuple[int, ...] = (64, 128, 256, 512)
    STAGE_BLOCKS: Tuple[int, ...] = (3, 4, 6, 3)

    def __init__(self, embedding_dim: int = 512) -> None:
        """Initialise ResNet-34 with the given embedding dimensionality."""
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = _make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = _make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = _make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = _make_layer(256, 512, blocks=3, stride=2)

        # Projection head (identity if embedding_dim == 512)
        self.proj: nn.Linear | nn.Identity
        if embedding_dim != 512:
            self.proj = nn.Linear(512, embedding_dim)
        else:
            self.proj = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-initialise conv layers; constant-init BN layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone and return the raw 512-d pooled vector before normalisation."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embedding; use forward_features for raw vector."""
        features = self.forward_features(x)
        return F.normalize(features, p=2, dim=1)
