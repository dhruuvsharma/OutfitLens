# OutfitLens Memory Index

| File | Memory Doc | Purpose |
|------|-----------|---------|
| models/backbone.py | memory/backbone.md | ResNet-34 CNN backbone built from scratch — L2-normalised 512-d embeddings |
| models/embedding_db.py | memory/embedding_db.md | Pure-numpy DB: item_name → vector, cosine top-k query, npz save/load |
| models/classifier_head.py | memory/classifier_head.md | Linear multi-label head (embedding_dim → num_classes logits) |
| data_pipeline/compositor.py | memory/compositor.md | Generates N synthetic composite outfit images with multi-hot labels |
| data_pipeline/augmentation.py | memory/augmentation.md | Stochastic train augmentation + deterministic val preprocessing |
| data_pipeline/dataset.py | memory/dataset.md | PyTorch Dataset for synthetic images; also creates train/val splits |
| training/losses.py | memory/losses.md | BCELoss and FocalLoss modules; build_loss() factory |
| training/metrics.py | memory/metrics.md | Per-item precision/recall/F1, mAP, confusion matrix, CSV logging |
| training/train.py | memory/train.md | Full training loop with validation, checkpoint saving, metric logging |
| inference/recognizer.py | memory/recognizer.md | Global + 3×3 regional inference paths merged by max-score per item |
| scripts/build_reference_db.py | memory/build_reference_db.md | Embeds all asset renders; averages multi-angle vectors; saves DB |
| scripts/run_training.py | memory/run_training.md | Orchestrates compositor → training from a config yaml |
| scripts/predict.py | memory/predict.md | CLI: predict items in a single outfit image; print + optional JSON output |
