---
name: specialist
description: Specialist model wrapping one ResNet-34 backbone + one per-category EmbeddingDB for single-category inference
type: project
---

# Memory: specialist.py

## Purpose
Wraps one `ResNet34Backbone` and one `EmbeddingDB` into a single stateless inference unit for a specific clothing category. At inference, an image tensor is embedded by the backbone and queried against the category's embedding DB to produce a ranked top-k list of candidate items. Each specialist is completely independent; they do not share backbone weights.

## Location
`models/specialist.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| Specialist | class | Backbone + DB wrapper for one clothing category |
| Specialist.query | fn | Embed image tensor, query DB, return [(item_name, score), ...] |
| Specialist.from_checkpoint | classmethod | Factory: load backbone checkpoint + DB file → ready Specialist |

## Inputs & Outputs
- **Inputs:** image tensor `(1, 3, H, W)` to `query()`; checkpoint `.pt` path and DB `.npz` path to `from_checkpoint()`
- **Outputs:** `list[(item_name: str, score: float)]` sorted descending by cosine similarity

## Dependencies
- Internal: `models/backbone.py`, `models/embedding_db.py`
- External: `torch`, `numpy`

## Config Keys Used
- `embedding_dim` — backbone config
- `image_size` — expected input size (validated but not enforced here)

## Change Log
| Date | Change |
|------|--------|
| 2026-04-05 | Initial creation — v2 specialist architecture |
