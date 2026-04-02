---
name: augmentation
description: Stochastic training augmentation and deterministic val preprocessing — PIL → normalised tensor
type: project
---

# Memory: augmentation.py

## Purpose
Implements all image transforms needed for training and validation. `TrainAugmentation` applies a stochastic pipeline (flip, colour jitter, random scale+crop, random occlusion patch) and normalises to ImageNet mean/std. `ValAugmentation` is the deterministic counterpart used during validation and inference. All transforms are applied directly to PIL images and return PyTorch float32 tensors. The ImageNet statistics are hard-coded constants — they do not imply use of pretrained weights.

## Location
`data_pipeline/augmentation.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| TrainAugmentation | class | Full stochastic pipeline for training images |
| ValAugmentation | class | Deterministic resize+normalise for val/inference |
| _random_flip | fn | Horizontal flip with p=0.5 |
| _colour_jitter | fn | Random hue shift and saturation scale |
| _random_scale_crop | fn | Random scale then centre-crop to target_size |
| _random_occlusion | fn | Zero-out a random rectangular patch on the tensor |

## Inputs & Outputs
- **Inputs:** PIL `Image` object (RGB)
- **Outputs:** normalised `torch.Tensor` of shape `(3, image_size, image_size)`, dtype float32

## Dependencies
- Internal: none
- External: `torch`, `torchvision` (transforms.functional only — no pretrained weights), `Pillow`

## Config Keys Used
- `image_size` — controls output tensor spatial dimensions

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
