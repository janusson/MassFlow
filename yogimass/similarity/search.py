"""
Search utilities for finding nearest spectra in a local library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from matchms import Spectrum

from yogimass.similarity.metrics import spec2vec_vectorize
from yogimass.similarity.library import LocalSpectralLibrary, SearchHit
from yogimass.similarity.processing import SpectrumProcessor
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """
    Similarity search result with score and library metadata.
    """

    identifier: str
    score: float
    metadata: dict[str, Any]
    precursor_mz: float | None


class LibrarySearcher:
    """
    Thin wrapper to process query spectra and return nearest library matches.
    """

    def __init__(
        self,
        library: LocalSpectralLibrary,
        *,
        processor: SpectrumProcessor | None = None,
        vectorizer: Callable[[Spectrum], dict[str, float]] = spec2vec_vectorize,
        backend: str = "naive",
    ):
        self.library = library
        self.processor = processor or SpectrumProcessor()
        self.vectorizer = vectorizer
        self.backend_name = backend
        self._index = self._build_index()

    def search_spectrum(
        self,
        spectrum: Spectrum,
        *,
        top_n: int = 5,
        min_score: float = 0.0,
        reference_mz: Sequence[float] | None = None,
    ) -> list[SearchResult]:
        """
        Process a query spectrum and return nearest matches from the library.
        """
        processed = self.processor.process(spectrum, reference_mz=reference_mz)
        query_vector = self.vectorizer(processed)
        hits = self._index.query(query_vector, top_n=top_n, min_score=min_score) if self._index else []
        return [self._to_result(hit) for hit in hits]

    def _to_result(self, hit: SearchHit) -> SearchResult:
        return SearchResult(
            identifier=hit.entry.identifier,
            score=hit.score,
            metadata=hit.entry.metadata,
            precursor_mz=hit.entry.precursor_mz,
        )

    def _build_index(self):
        from yogimass.similarity.backends import create_index_backend

        return create_index_backend(self.backend_name, entries=self.library.iter_entries())
