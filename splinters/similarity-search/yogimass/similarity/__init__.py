"""
Similarity and spectral library utilities for Yogimass.
"""

from yogimass.similarity.batch import batch_process_folder
from yogimass.similarity.backends import create_index_backend
from yogimass.similarity.embeddings import build_embeddings, embedding_vectorizer
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
    "build_embeddings",
    "embedding_vectorizer",
    "SpectrumProcessor",
    "LibraryEntry",
    "LocalSpectralLibrary",
    "SearchHit",
    "LibrarySearcher",
    "SearchResult",
    "create_index_backend",
]
