"""
Search backends for spectrum libraries.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol

from yogimass.similarity.library import LibraryEntry, SearchHit
from yogimass.similarity.metrics import cosine_from_vectors
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


class SpectrumIndexBackend(Protocol):
    """Interface for spectrum search backends."""

    name: str

    def add(self, entry: LibraryEntry) -> None: ...

    def build(self) -> None: ...

    def query(
        self, vector: Mapping[str, float], *, top_n: int, min_score: float
    ) -> list[SearchHit]: ...


class NaiveSpectrumIndex:
    """In-memory cosine search over stored vectors."""

    name = "naive"

    def __init__(self):
        self._entries: list[LibraryEntry] = []

    def add(self, entry: LibraryEntry) -> None:
        self._entries.append(entry)

    def build(self) -> None:  # pragma: no cover - nothing to build
        return None

    def query(
        self, vector: Mapping[str, float], *, top_n: int, min_score: float
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for entry in self._entries:
            score = cosine_from_vectors(entry.vector, vector)
            if score >= min_score:
                hits.append(SearchHit(entry=entry, score=score))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_n]


class AnnoySpectrumIndex:
    """Approximate nearest neighbor search powered by ``annoy``."""

    name = "annoy"

    def __init__(self, num_trees: int = 10):
        try:
            from annoy import AnnoyIndex  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Annoy backend requires `annoy`; install with `pip install yogimass[annoy]`."
            ) from exc
        self._annoy_cls = AnnoyIndex
        self._num_trees = num_trees
        self._entries: list[LibraryEntry] = []
        self._vocab: dict[str, int] = {}
        self._index = None

    def add(self, entry: LibraryEntry) -> None:
        self._entries.append(entry)
        for token in entry.vector:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)

    def build(self) -> None:
        dimension = max(1, len(self._vocab))
        self._index = self._annoy_cls(dimension, "angular")
        for idx, entry in enumerate(self._entries):
            dense = self._to_dense(entry.vector, dimension)
            self._index.add_item(idx, dense)
        if self._entries:
            self._index.build(self._num_trees)

    def query(
        self, vector: Mapping[str, float], *, top_n: int, min_score: float
    ) -> list[SearchHit]:
        if not self._entries:
            return []
        if self._index is None:
            self.build()
        assert self._index is not None  # for type checkers
        dimension = max(1, len(self._vocab))
        query_dense = self._to_dense(vector, dimension)
        candidate_ids = self._index.get_nns_by_vector(
            query_dense,
            n=min(top_n, len(self._entries)),
            include_distances=False,
        )
        hits: list[SearchHit] = []
        for idx in candidate_ids:
            entry = self._entries[idx]
            score = cosine_from_vectors(entry.vector, vector)
            if score >= min_score:
                hits.append(SearchHit(entry=entry, score=score))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_n]

    def _to_dense(self, vector: Mapping[str, float], dimension: int) -> list[float]:
        dense = [0.0] * dimension
        for token, weight in vector.items():
            pos = self._vocab.get(token)
            if pos is None:
                continue
            dense[pos] = float(weight)
        return dense


def create_index_backend(
    name: str, *, entries: Iterable[LibraryEntry]
) -> SpectrumIndexBackend:
    """
    Construct and populate a search backend.
    """
    normalized = name.lower()
    if normalized == "naive":
        backend: SpectrumIndexBackend = NaiveSpectrumIndex()
    elif normalized == "annoy":
        backend = AnnoySpectrumIndex()
    elif normalized == "faiss":
        logger.warning("FAISS backend not yet implemented; using naive search.")
        backend = NaiveSpectrumIndex()
    else:
        raise ValueError(f"Unsupported search backend: {name}")
    for entry in entries:
        backend.add(entry)
    backend.build()
    return backend


__all__ = [
    "AnnoySpectrumIndex",
    "NaiveSpectrumIndex",
    "SpectrumIndexBackend",
    "create_index_backend",
]
