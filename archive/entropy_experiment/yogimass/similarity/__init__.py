"""Similarity metrics and processing utilities for Yogimass core."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_base_dir = Path(__file__).resolve().parent
_repo_root = _base_dir.parent.parent
_extra_path = (
    _repo_root / "splinters" / "similarity-search" / "yogimass" / "similarity"
)
if _extra_path.is_dir():
    extra_str = str(_extra_path)
    if extra_str not in __path__:
        __path__.append(extra_str)

from yogimass.similarity.metrics import (
    cosine_from_vectors,
    cosine_similarity,
    modified_cosine_similarity,
    spec2vec_vectorize,
)
from yogimass.similarity.processing import SpectrumProcessor

__all__ = [
    "SpectrumProcessor",
    "cosine_from_vectors",
    "cosine_similarity",
    "modified_cosine_similarity",
    "spec2vec_vectorize",
]
