"""Aggregator: applies confidence threshold and top-N per category to raw recognizer output."""

from __future__ import annotations

from typing import Dict, List


class Aggregator:
    """Filters and formats raw per-category recognizer output into the final structured result.

    Applies a configurable confidence threshold and top-N cap per category.
    Per-category overrides (e.g. accessories returning fewer results) are applied
    when provided via category_overrides.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        top_n_results: int = 5,
        category_overrides: Dict[str, Dict] | None = None,
    ) -> None:
        """Initialise aggregator with default threshold, top-N, and optional per-category overrides."""
        self.confidence_threshold = confidence_threshold
        self.top_n_results = top_n_results
        self.category_overrides: Dict[str, Dict] = category_overrides or {}

    def aggregate(
        self, raw_results: Dict[str, List[tuple]]
    ) -> Dict[str, List[Dict]]:
        """Apply threshold + top-N to each category; return structured output.

        Args:
            raw_results: ``{category: [(item_name, score), ...]}`` from OutfitRecognizer.

        Returns:
            ``{category: [{"item": name, "confidence": score}, ...]}``
            Only items meeting the threshold are included; list capped at top_n per category.
        """
        output: Dict[str, List[Dict]] = {}
        for category, hits in raw_results.items():
            overrides = self.category_overrides.get(category, {})
            threshold = overrides.get("confidence_threshold", self.confidence_threshold)
            top_n = overrides.get("top_n_results", self.top_n_results)

            filtered = [
                {"item": name, "confidence": round(float(score), 4)}
                for name, score in hits
                if score >= threshold
            ][:top_n]
            output[category] = filtered
        return output
