"""
Lightweight embedding hooks for spectrum similarity.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np
from matchms import Spectrum

from yogimass.similarity.metrics import spec2vec_vectorize


def _token_index(token: str, dimension: int) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % max(1, dimension)


def _dense_embedding(vector: Mapping[str, float], dimension: int) -> np.ndarray:
    dense = np.zeros(dimension, dtype=float)
    for token, weight in vector.items():
        dense[_token_index(token, dimension)] += float(weight)
    return dense


def build_embeddings(model_name: str, spectra: Iterable[Spectrum], *, dimension: int = 64) -> np.ndarray:
    """
    Build simple hashed embeddings for a collection of spectra.
    Placeholder for future model-backed embeddings.
    """
    rows = []
    for spectrum in spectra:
        spec2vec = spec2vec_vectorize(spectrum)
        rows.append(_dense_embedding(spec2vec, dimension))
    if not rows:
        return np.zeros((0, dimension), dtype=float)
    return np.vstack(rows)


def embedding_vectorizer(
    spectrum: Spectrum,
    *,
    model_name: str = "spec2vec-lite",
    dimension: int = 64,
) -> dict[str, float]:
    """
    Vectorize a spectrum into a sparse embedding token map.
    Uses hashed buckets for stability without external models.
    """
    dense = _dense_embedding(spec2vec_vectorize(spectrum), dimension)
    return {f"{model_name}:{idx}": value for idx, value in enumerate(dense) if value != 0.0}


__all__ = ["build_embeddings", "embedding_vectorizer"]
