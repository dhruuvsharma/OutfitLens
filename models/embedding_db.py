"""Embedding database: stores item_name → 512-d vector, supports cosine top-k queries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class EmbeddingDB:
    """In-memory key-value store mapping item names to L2-normalised embedding vectors.

    Vectors are stored in a matrix for vectorised cosine-similarity search.
    The database is saved/loaded as a compressed NumPy `.npz` archive — no external
    vector DB library is required.
    """

    def __init__(self) -> None:
        """Initialise an empty embedding database."""
        self._names: List[str] = []
        self._matrix: np.ndarray | None = None  # shape (N, D)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def add(self, item_name: str, vector: np.ndarray) -> None:
        """Add or replace a single item embedding (vector must be 1-D)."""
        vector = _ensure_1d_float32(vector)
        if item_name in self._names:
            idx = self._names.index(item_name)
            self._matrix[idx] = vector  # type: ignore[index]
        else:
            self._names.append(item_name)
            if self._matrix is None:
                self._matrix = vector[np.newaxis, :]
            else:
                self._matrix = np.vstack([self._matrix, vector[np.newaxis, :]])

    def build_from_dict(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Populate the database from a {item_name: vector} mapping, replacing any existing data."""
        self._names = []
        self._matrix = None
        for name, vec in embeddings.items():
            self.add(name, vec)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the database to a compressed .npz file at the given path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            names=np.array(self._names, dtype=object),
            matrix=self._matrix if self._matrix is not None else np.empty((0,)),
        )

    def load(self, path: Path) -> None:
        """Load the database from a .npz file, replacing any existing data."""
        path = Path(path)
        data = np.load(str(path), allow_pickle=True)
        self._names = list(data["names"])
        matrix = data["matrix"]
        self._matrix = matrix if matrix.ndim == 2 else None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-k items most similar to vector as [(item_name, cosine_score), ...]."""
        if self._matrix is None or len(self._names) == 0:
            return []
        vector = _ensure_1d_float32(vector)
        # Both stored vectors and the query are expected to be L2-normalised,
        # so dot product == cosine similarity.
        scores: np.ndarray = self._matrix @ vector  # (N,)
        top_k = min(top_k, len(self._names))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [(self._names[i], float(scores[i])) for i in indices]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of items stored in the database."""
        return len(self._names)

    @property
    def item_names(self) -> List[str]:
        """Return a list of all stored item names."""
        return list(self._names)

    def get_vector(self, item_name: str) -> np.ndarray | None:
        """Return the stored embedding for item_name, or None if not found."""
        if item_name not in self._names:
            return None
        idx = self._names.index(item_name)
        return self._matrix[idx].copy()  # type: ignore[index]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ensure_1d_float32(v: np.ndarray) -> np.ndarray:
    """Flatten and cast a numpy array to float32, raising if it is empty."""
    v = np.asarray(v, dtype=np.float32).ravel()
    if v.size == 0:
        raise ValueError("Embedding vector must not be empty.")
    return v
