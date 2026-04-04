---
name: recognizer
description: Inference engine combining global + 3×3 regional embedding paths against the reference DB
type: project
---

# Memory: recognizer.py

## Purpose
Provides the `OutfitRecognizer` class that identifies which clothing items appear in an outfit image at inference time. It runs two complementary paths: a global embed of the full image, and a 3×3 regional grid where each cell is embedded independently. Results are merged by max-score per item and filtered by the confidence threshold from config. Also provides `load_recognizer()` as a factory that assembles the backbone + DB from file paths.

## Location
`inference/recognizer.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| OutfitRecognizer | class | Main inference class |
| OutfitRecognizer.predict | fn | Full predict pipeline → {item_name: score} above threshold |
| OutfitRecognizer.predict_path | fn | Load image from disk then predict |
| OutfitRecognizer._global_inference | fn | Full-image embedding query |
| OutfitRecognizer._regional_inference | fn | 3×3 crop grid query with merge |
| OutfitRecognizer._embed | fn | PIL image → L2-normalised numpy vector |
| load_recognizer | fn | Factory: builds recognizer from config + file paths |
| _grid_crops | fn | Divides image into grid×grid crop list |
| _merge_hits | fn | Max-score merge of multiple hit dicts |

## Inputs & Outputs
- **Inputs:** PIL Image (or path); config dict; loaded ResNet34Backbone; EmbeddingDB
- **Outputs:** `{item_name: confidence_score}` dict filtered to scores ≥ threshold

## Dependencies
- Internal: `data_pipeline/augmentation.py`, `models/backbone.py`, `models/embedding_db.py`
- External: `torch`, `numpy`, `Pillow`

## Config Keys Used
- `image_size` — for the val transform
- `confidence_threshold` — minimum score to include item in output
- `embedding_dim` — backbone config

## Change Log
| Date | Change |
|------|--------|
| 2026-04-03 | Initial creation |
| 2026-04-05 | v2 rewrite — specialist-per-category architecture; loads all Specialist models from categories.yaml; `recognize(image_path)` runs all specialists and returns per-category ranked lists; removed global+regional grid inference paths |
