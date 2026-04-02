---
name: embedding_db
description: Pure-numpy embedding database mapping item names to 512-d vectors with cosine top-k search
type: project
---

# Memory: embedding_db.py

## Purpose
Provides a lightweight in-memory key-value store that maps clothing item names to their 512-dimensional L2-normalised embedding vectors. The database is persisted as a compressed NumPy `.npz` archive. Cosine-similarity search is implemented as a plain matrix-vector dot product (no external vector DB library). Used by both the reference DB builder script and the inference recognizer.

## Location
`models/embedding_db.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| EmbeddingDB | class | Main database class |
| EmbeddingDB.add | fn | Add/replace a single item embedding |
| EmbeddingDB.build_from_dict | fn | Populate DB from a dict of embeddings |
| EmbeddingDB.save | fn | Persist to .npz file |
| EmbeddingDB.load | fn | Load from .npz file |
| EmbeddingDB.query | fn | Top-k cosine similarity search |
| EmbeddingDB.get_vector | fn | Retrieve stored vector by name |
| _ensure_1d_float32 | fn | Internal helper — flatten and cast array |

## Inputs & Outputs
- **Inputs:** numpy float32 vectors (1-D, length = embedding_dim); item name strings; file paths for save/load
- **Outputs:** `.npz` file on disk; list of `(item_name, score)` tuples from query

## Dependencies
- Internal: none
- External: `numpy`

## Config Keys Used
None directly — embedding_dim is implicit in the vectors added by callers.

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
