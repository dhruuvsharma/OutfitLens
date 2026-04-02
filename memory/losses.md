---
name: losses
description: BCE and focal loss modules for multi-label clothing item classification
type: project
---

# Memory: losses.py

## Purpose
Provides two loss functions for training the multi-label outfit recogniser. `BCELoss` wraps PyTorch's `binary_cross_entropy_with_logits` for the standard case. `FocalLoss` applies the focal modulation factor `(1 - p_t)^γ` to down-weight easy negatives and focus training on hard examples — useful when item classes are heavily imbalanced. `build_loss()` is a factory that selects between them based on the `focal_loss` config flag.

## Location
`training/losses.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| BCELoss | class | Standard BCE with optional pos_weight |
| FocalLoss | class | Focal loss with configurable gamma and alpha |
| build_loss | fn | Factory: returns BCELoss or FocalLoss from config |

## Inputs & Outputs
- **Inputs:** logit tensor `(B, num_classes)`, target tensor `(B, num_classes)` in {0.0, 1.0}
- **Outputs:** scalar loss tensor

## Dependencies
- Internal: none
- External: `torch`, `torch.nn.functional`

## Config Keys Used
- `focal_loss` — bool; if true, use FocalLoss instead of BCELoss

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
