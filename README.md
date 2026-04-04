# OutfitLens

OutfitLens is a fully offline machine learning system that identifies individual clothing and accessory items present in a combined outfit image. Given a render of a complete NPC outfit, the system outputs which individual items are present along with a confidence score for each. All training and inference run locally — no external APIs or pretrained model weights are required.

---

## Why Specialist-Per-Category?

A single multi-label model must learn to distinguish hats from shirts from shoes simultaneously. Specialists divide this problem: each model only ever sees items from one category, so it learns finer-grained intra-category features without cross-category noise. Adding a new category never requires retraining existing specialists.

---

## Folder Structure

```
outfitlens/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/
│   ├── base_config.yaml         ← shared hyperparameters
│   └── categories.yaml          ← active categories + per-category overrides
├── data/
│   └── raw/
│       ├── hats/                ← all hat renders go here
│       ├── shirts/
│       ├── pants/
│       ├── shoes/
│       ├── jackets/
│       └── accessories/
├── models/
│   ├── backbone.py              ← ResNet-34 from scratch (512-d embeddings)
│   ├── embedding_db.py          ← per-category numpy cosine DB
│   └── specialist.py            ← backbone + DB = one specialist model
├── data_pipeline/
│   ├── compositor.py            ← per-category synthetic compositor with distractors
│   ├── augmentation.py          ← image transforms
│   └── dataset.py               ← SpecialistDataset (single-label)
├── training/
│   ├── train_specialist.py      ← trains one specialist + populates embedding DB
│   ├── losses.py                ← cross-entropy + focal loss (single-label)
│   └── metrics.py               ← top-1/5 accuracy, per-item recall, confusion matrix
├── inference/
│   ├── recognizer.py            ← loads all specialists, runs recognize()
│   └── aggregator.py            ← confidence threshold + top-N per category
├── scripts/
│   ├── build_reference_db.py    ← embed raw renders → save DB (--category or --all)
│   ├── train_all.py             ← orchestrate compositor + training for all categories
│   └── predict.py               ← CLI: python predict.py --image outfit.jpg
├── memory/                      ← Claude Code memory docs
└── logs/
    ├── hats/                    ← best_model.pt, embedding_db.npz, train_log.csv
    ├── shirts/
    └── ...
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended. A CUDA GPU is used automatically if available.

---

## Step-by-step Usage

### 1. Organise renders into category folders

Place item renders into `data/raw/<category>/`. File naming convention:

- **4-angle mode** (default): `ItemName_front.jpg`, `ItemName_back.jpg`, `ItemName_left.jpg`, `ItemName_right.jpg`
- **1-angle mode**: `ItemName_front.jpg` or any single file per item

Example:
```
data/raw/hats/Cowboy_Hat_front.jpg
data/raw/hats/Cowboy_Hat_back.jpg
data/raw/shirts/Plaid_Shirt_front.jpg
```

### 2. Register categories

Edit `configs/categories.yaml` to list your categories (or just ensure the folders exist — `train_all.py` auto-discovers them):

```yaml
categories:
  - name: hats
  - name: shirts
  - name: accessories
    top_n_results: 3   # optional per-category override
```

### 3. Train all specialists

```bash
python scripts/train_all.py
```

This runs for each category:
1. **Compositor** — generates 3 000 synthetic composite images (target item + cross-category distractors)
2. **Training** — trains a ResNet-34 specialist with cross-entropy loss
3. **DB population** — embeds all raw renders and saves `logs/<category>/embedding_db.npz`

To train a single category only:
```bash
python scripts/train_all.py --category hats
```

Skip regenerating composites if already built:
```bash
python scripts/train_all.py --skip-composite
```

### 4. Predict items in an outfit image

```bash
python scripts/predict.py --image path/to/outfit.jpg
```

Example output:
```
Outfit analysis: outfit.jpg
==================================================

HATS:
  Cowboy_Hat                      0.9412

SHIRTS:
  Plaid_Shirt                     0.9104

PANTS:
  Cargo_Pants                     0.9631
```

Optional flags:
- `--threshold 0.8` — override confidence threshold
- `--output-json results.json` — write structured JSON output

---

## Adding a New Category

1. Create `data/raw/<new_category>/` and add renders
2. Add entry to `configs/categories.yaml`
3. Run `python scripts/train_all.py --category <new_category>`
4. No changes to existing specialists

## Adding a New Item to an Existing Category

1. Drop renders into `data/raw/<category>/`
2. Run `python scripts/build_reference_db.py --category <category>`
3. No retraining needed — the DB is updated immediately

---

## Config Reference

### `configs/base_config.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `single_angle` | `false` | `true` = front-only; `false` = average all angle embeddings |
| `image_size` | `224` | Input image size in pixels (square) |
| `embedding_dim` | `512` | Embedding vector dimensionality |
| `batch_size` | `32` | Training batch size |
| `learning_rate` | `0.0003` | Adam learning rate |
| `epochs` | `60` | Training epochs per specialist |
| `val_split` | `0.15` | Validation fraction |
| `composite_count` | `3000` | Synthetic images generated per category |
| `distractor_count` | `2` | Other-category items added per composite |
| `confidence_threshold` | `0.70` | Minimum cosine score to include in output |
| `top_n_results` | `5` | Max results returned per category |
| `focal_loss` | `false` | Use focal cross-entropy instead of standard CE |
| `raw_data_dir` | `data/raw` | Root folder containing category subfolders |
| `synthetic_dir` | `data/synthetic` | Root for generated composites (one subfolder per category) |
| `logs_dir` | `logs` | Root for checkpoints and logs (one subfolder per category) |

### `configs/categories.yaml`

Per-category overrides can set: `top_n_results`, `confidence_threshold`, `epochs`, `batch_size`.

---

## 4-Angle vs 1-Angle Mode

Set `single_angle` in `configs/base_config.yaml`:

- **`false` (default)** — all available angle renders are embedded and averaged into one item signature. More view-invariant.
- **`true`** — only the front-view render is used. Faster to set up when only one render per item is available.

The compositor always uses the first available render as the pixel representative. Angle-averaging is an embedding-space operation only.
