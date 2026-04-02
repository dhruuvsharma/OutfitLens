---
name: classifier_head
description: Linear multi-label classifier head that projects backbone embeddings to per-item logits
type: project
---

# Memory: classifier_head.py

## Purpose
Provides the supervised classification head that sits on top of the ResNet-34 backbone during training. It accepts the raw (un-normalised) 512-d pooled features from `ResNet34Backbone.forward_features()` and produces per-class logits. Binary cross-entropy (not softmax) is used because multiple items can be present simultaneously. The head is decoupled from the backbone so that both the embedding path and the classification path can share backbone weights cleanly.

## Location
`models/classifier_head.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| ClassifierHead | class | Linear projection from embedding_dim → num_classes |
| ClassifierHead.forward | fn | Returns raw logits (no sigmoid) |
| ClassifierHead.predict_proba | fn | Returns sigmoid probabilities |

## Inputs & Outputs
- **Inputs:** feature tensor `(B, embedding_dim)` from `ResNet34Backbone.forward_features()`
- **Outputs:** logit tensor `(B, num_classes)` — apply BCEWithLogitsLoss during training

## Dependencies
- Internal: none (used alongside `models/backbone.py` but does not import it)
- External: `torch`, `torch.nn`

## Config Keys Used
- `embedding_dim` — input feature size
- num_classes is derived at runtime from the dataset label count

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
