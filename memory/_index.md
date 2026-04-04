# OutfitLens Memory Index

| File | Memory Doc | Purpose |
|------|-----------|---------|
| models/backbone.py | memory/backbone.md | ResNet-34 CNN backbone built from scratch — L2-normalised 512-d embeddings |
| models/embedding_db.py | memory/embedding_db.md | Pure-numpy DB: item_name → vector, cosine top-k query, npz save/load |
| models/specialist.py | memory/specialist.md | Wraps one backbone + one DB = one specialist model for a single category |
| data_pipeline/augmentation.py | memory/augmentation.md | Stochastic train augmentation + deterministic val preprocessing |
| data_pipeline/compositor.py | memory/compositor.md | Per-category synthetic compositor with cross-category distractors; single-label labels.json |
| data_pipeline/dataset.py | memory/dataset.md | SpecialistDataset: single-label (item_index) PyTorch Dataset for one category |
| training/losses.py | memory/losses.md | CrossEntropyLoss and FocalLossCE modules; build_loss() factory (single-label) |
| training/metrics.py | memory/metrics.md | Top-1/5 accuracy, per-item recall@5, confusion matrix, CSV logging |
| training/train_specialist.py | memory/train_specialist.md | Trains one specialist + populates reference embedding DB for one category |
| inference/aggregator.py | memory/aggregator.md | Applies confidence threshold + top-N per category to raw recognizer output |
| inference/recognizer.py | memory/recognizer.md | Loads all specialist models; recognize() returns per-category ranked lists |
| scripts/build_reference_db.py | memory/build_reference_db.md | CLI: --category or --all; embeds raw renders; saves embedding_db.npz |
| scripts/train_all.py | memory/train_all.md | CLI: loops over categories, runs compositor + train_specialist for each |
| scripts/predict.py | memory/predict.md | CLI: predict --image outfit.jpg; prints per-category ranked results |

## v1 Artifacts (superseded, kept for reference)
| File | Notes |
|------|-------|
| models/classifier_head.py | v1 multi-label head — superseded by specialist.py |
| training/train.py | v1 multi-label training loop — superseded by train_specialist.py |
| scripts/run_training.py | v1 single-config orchestration — superseded by train_all.py |
| configs/config_4angle.yaml | v1 config — superseded by base_config.yaml + categories.yaml |
| configs/config_1angle.yaml | v1 config — superseded by base_config.yaml + categories.yaml |
