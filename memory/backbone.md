---
name: backbone
description: ResNet-34 CNN backbone built from scratch in PyTorch, producing L2-normalized 512-d embeddings
type: project
---

# Memory: backbone.py

## Purpose
Implements a ResNet-34 convolutional neural network built entirely from scratch using PyTorch primitives (no torchvision pretrained weights). The backbone serves as the shared feature extractor for both the embedding-based retrieval path and the supervised classification head. It outputs a 512-dimensional L2-normalized embedding suitable for cosine-similarity queries against the reference embedding database.

## Location
`models/backbone.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| BasicBlock | class | Standard residual block with two 3×3 convs and optional downsample shortcut |
| _make_layer | fn | Factory that builds a sequential stage of N BasicBlock residuals |
| ResNet34Backbone | class | Full ResNet-34 backbone; forward() returns L2-normalized embedding |
| ResNet34Backbone.forward_features | fn | Returns raw pooled vector before L2 normalization |
| ResNet34Backbone.forward | fn | Returns L2-normalized 512-d embedding |

## Inputs & Outputs
- **Inputs:** RGB image tensor of shape `(B, 3, H, W)` — expected H=W=224
- **Outputs:** L2-normalized embedding tensor `(B, embedding_dim)` from `forward()`; raw vector from `forward_features()`

## Dependencies
- Internal: none
- External: `torch`, `torch.nn`, `torch.nn.functional`

## Config Keys Used
- `embedding_dim` (default 512) — controls optional linear projection after global avg pool

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
