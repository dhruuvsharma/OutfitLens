# OutfitLens

OutfitLens is a fully offline machine learning system that identifies individual clothing and accessory items present in a combined outfit image. Given a render of a complete NPC outfit, the system outputs which individual items are present along with a confidence score for each. All training and inference run locally — no external APIs or pretrained model weights are required.

The system is built on a ResNet-34 backbone trained from scratch using PyTorch. During inference, a query image is embedded into a 512-dimensional vector space and matched against a reference database of known item embeddings using cosine similarity. A complementary supervised classifier head is also trained and can be used alongside the embedding path. Confidence scores are produced via two complementary inference paths (global full-image embedding and a 3×3 regional crop grid) whose results are merged by max-score.

---

## Folder Structure

```
outfitlens/
├── CLAUDE.md                        ← instructions for Claude Code
├── README.md
├── requirements.txt
├── configs/
│   ├── config_4angle.yaml           ← primary: 4-angle renders per item
│   └── config_1angle.yaml           ← secondary: front-only renders
├── data/
│   ├── raw/
│   │   ├── assets_4angle/           ← drop 4-angle renders here
│   │   └── assets_1angle/           ← drop 1-angle renders here
│   ├── synthetic/
│   │   ├── images/                  ← generated composite images
│   │   └── labels.json              ← generated labels
│   └── splits/
│       ├── train.json
│       └── val.json
├── models/
│   ├── backbone.py                  ← ResNet-34 from scratch
│   ├── embedding_db.py              ← numpy cosine DB
│   └── classifier_head.py          ← linear multi-label head
├── data_pipeline/
│   ├── compositor.py                ← synthetic outfit generator
│   ├── augmentation.py              ← image transforms
│   └── dataset.py                  ← PyTorch Dataset + split builder
├── training/
│   ├── train.py                     ← training loop
│   ├── losses.py                    ← BCE + focal loss
│   └── metrics.py                  ← F1, mAP, confusion matrix
├── inference/
│   └── recognizer.py               ← global + regional inference
├── scripts/
│   ├── build_reference_db.py        ← embed assets → save DB
│   ├── run_training.py              ← generate data + train
│   └── predict.py                  ← CLI prediction
├── memory/                          ← Claude Code memory docs
└── logs/                            ← checkpoints + training logs
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended. All packages are CPU-compatible; a CUDA-capable GPU is used automatically if available.

---

## Step-by-step Usage

### 1. Drop renders into `data/raw/`

**4-angle mode** — place files named `ItemName_front.jpg`, `ItemName_back.jpg`, `ItemName_left.jpg`, `ItemName_right.jpg` into `data/raw/assets_4angle/`.

**1-angle mode** — place files named `ItemName_front.jpg` (or any single file per item) into `data/raw/assets_1angle/`.

### 2. Generate synthetic training data

```bash
python scripts/run_training.py --config configs/config_4angle.yaml
```

This runs the compositor first (generating 5 000 composite images by default) then launches the training loop. To skip regeneration on subsequent runs:

```bash
python scripts/run_training.py --config configs/config_4angle.yaml --skip-composite
```

You can also run the compositor separately:

```python
import yaml
from data_pipeline.compositor import generate_dataset

with open("configs/config_4angle.yaml") as f:
    config = yaml.safe_load(f)
generate_dataset(config)
```

### 3. Build the reference embedding database

Run this after training completes (or after adding new items — no retraining needed):

```bash
python scripts/build_reference_db.py \
    --config configs/config_4angle.yaml \
    --checkpoint logs/best_model.pt
```

Output: `logs/reference_db.npz`

### 4. Predict items in an outfit image

```bash
python scripts/predict.py \
    --image path/to/outfit.jpg \
    --config configs/config_4angle.yaml
```

Optional flags:
- `--threshold 0.8` — override the confidence threshold
- `--output-json results.json` — write predictions to a JSON file
- `--checkpoint path/to/model.pt` — use a specific checkpoint
- `--db path/to/db.npz` — use a specific reference DB

---

## Config Reference

| Key | Default | Description |
|-----|---------|-------------|
| `single_angle` | `false` | Use front-only renders (`true`) or 4-angle renders (`false`) |
| `image_size` | `224` | Input image size (square, pixels) |
| `embedding_dim` | `512` | Embedding vector dimensionality |
| `batch_size` | `32` | Training batch size |
| `learning_rate` | `0.0003` | Adam learning rate |
| `epochs` | `50` | Number of training epochs |
| `val_split` | `0.15` | Fraction of synthetic data held out for validation |
| `composite_count` | `5000` | Number of synthetic composite images to generate |
| `confidence_threshold` | `0.75` | Minimum cosine similarity to report an item as detected |
| `focal_loss` | `false` | Use focal loss instead of standard BCE |
| `assets_dir` | `data/raw/assets_4angle` | Source folder for raw renders |
| `synthetic_dir` | `data/synthetic` | Destination for generated images and labels |
| `checkpoint_dir` | `logs` | Where to save checkpoints and training logs |

---

## 4-Angle vs 1-Angle Mode

Both modes share the same codebase. The only difference is the `single_angle` config flag:

- **4-angle (`single_angle: false`)** — `build_reference_db.py` embeds all four renders of each item and averages the resulting vectors into a single item signature. This produces a more view-invariant embedding. Use `configs/config_4angle.yaml`.

- **1-angle (`single_angle: true`)** — only the front-view render is embedded. Faster to set up; useful when only front renders are available. Use `configs/config_1angle.yaml`.

The compositor always uses the front-view render as the pixel-level representative for compositing regardless of mode. Angle-averaging is an embedding-space operation, not a pixel-space one.

---

## Adding New Items (No Retraining Required)

1. Drop the new item renders into the correct `data/raw/` subfolder.
2. Re-run `build_reference_db.py` to update the reference database.
3. The recognizer will immediately pick up the new item on the next `predict.py` call.

Retraining is only needed if the model's general feature quality needs to improve (e.g. after adding many new visual categories).
