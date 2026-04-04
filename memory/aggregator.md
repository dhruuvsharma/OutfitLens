---
name: aggregator
description: Post-processes raw per-category specialist results — applies confidence threshold and returns top-N
type: project
---

# Memory: aggregator.py

## Purpose
Takes the raw per-category ranked output from `OutfitRecognizer.recognize()` and applies a configurable confidence threshold and top-N limit per category. Produces the final structured output dict with `{"item": name, "confidence": score}` entries. Per-category overrides (e.g. accessories returning fewer results) are applied from `categories.yaml`.

## Location
`inference/aggregator.py`

## Key Classes / Functions
| Name | Type | Description |
|------|------|-------------|
| Aggregator | class | Stateless filter/formatter for recognizer output |
| Aggregator.aggregate | fn | Apply threshold + top-N, return structured output dict |

## Inputs & Outputs
- **Inputs:** `{category: [(item_name, score), ...]}` raw recognizer output; category override dicts
- **Outputs:** `{category: [{"item": name, "confidence": score}, ...]}` — only items above threshold, capped at top_n

## Dependencies
- Internal: none
- External: none (pure Python)

## Config Keys Used
- `confidence_threshold` — default minimum score
- `top_n_results` — default max items per category (per-category override in categories.yaml)

## Change Log
| Date | Change |
|------|--------|
| 2026-04-05 | Initial creation — v2 aggregation layer |
