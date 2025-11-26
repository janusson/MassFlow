"""
Similarity and spectral library utilities for Yogimass.
"""

from yogimass.similarity.batch import batch_process_folder
from yogimass.similarity.library import LibraryEntry, LocalSpectralLibrary, SearchHit
from yogimass.similarity.metrics import (
    cosine_from_vectors,
    cosine_similarity,
    modified_cosine_similarity,
    spec2vec_vectorize,
)
from yogimass.similarity.processing import SpectrumProcessor
from yogimass.similarity.search import LibrarySearcher, SearchResult

__all__ = [
    "batch_process_folder",
    "cosine_from_vectors",
    "cosine_similarity",
    "modified_cosine_similarity",
    "spec2vec_vectorize",
    "SpectrumProcessor",
    "LibraryEntry",
    "LocalSpectralLibrary",
    "SearchHit",
    "LibrarySearcher",
    "SearchResult",
]
