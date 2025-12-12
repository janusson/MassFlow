"""
Lightweight local spectral library storage for similarity search.

Responsibilities:
- Persist spectra and precomputed vectors (JSON or SQLite).
- Add/update entries, iterate/search via cosine, and expose simple data classes.

Depends on:
- ``matchms`` for ``Spectrum`` objects.
- ``yogimass.similarity.metrics`` for vectorization and cosine calculations.
- Logging utilities for basic warnings.

Avoids:
- CLI/config parsing or workflow orchestration.
- Reporting/curation/network logic (those consume this layer).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping
from uuid import uuid4

from matchms import Spectrum

from yogimass.similarity.metrics import cosine_from_vectors, spec2vec_vectorize
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "1.0"


@dataclass
class LibraryEntry:
    """
    Stored representation of a spectrum with a precomputed vector.
    """

    identifier: str
    precursor_mz: float | None
    metadata: dict[str, Any]
    vector: dict[str, float]

    def to_record(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "precursor_mz": self.precursor_mz,
            "metadata": self.metadata,
            "vector": self.vector,
        }


@dataclass
class SearchHit:
    entry: LibraryEntry
    score: float


class LocalSpectralLibrary:
    """
    Minimal persistent library backed by JSON or SQLite.
    """

    def __init__(self, path: str | Path, storage: Literal["json", "sqlite"] | None = None):
        self.path = Path(path)
        self.storage: Literal["json", "sqlite"] = storage or self._infer_storage()
        self._ensure_store()

    def add_spectrum(
        self,
        spectrum: Spectrum,
        *,
        identifier: str | None = None,
        vectorizer: Callable[[Spectrum], Mapping[str, float]] = spec2vec_vectorize,
        overwrite: bool = False,
    ) -> LibraryEntry:
        """
        Add a spectrum to the library, computing a Spec2Vec-like vector if needed.
        """
        metadata = self._spectrum_metadata(spectrum)
        spectrum_id = identifier or metadata.get("name") or f"spectrum-{uuid4().hex}"
        precursor_mz = spectrum.get("precursor_mz") or spectrum.get("parent_mass")
        vector = dict(vectorizer(spectrum))
        entry = LibraryEntry(
            identifier=spectrum_id,
            precursor_mz=float(precursor_mz) if precursor_mz is not None else None,
            metadata=metadata,
            vector=vector,
        )
        self._write_entry(entry, overwrite=overwrite)
        return entry

    def iter_entries(self) -> Iterator[LibraryEntry]:
        """
        Yield all stored entries.
        """
        if self.storage == "json":
            yield from self._iter_json_entries()
        else:
            yield from self._iter_sqlite_entries()

    def search(
        self,
        query_vector: Mapping[str, float],
        *,
        top_n: int = 5,
        min_score: float = 0.0,
        metadata_filter: Callable[[LibraryEntry], bool] | None = None,
        backend: str = "naive",
    ) -> list[SearchHit]:
        """
        Return the top entries ranked by cosine similarity to ``query_vector``.
        """
        entries = self.iter_entries()
        if metadata_filter:
            entries = (entry for entry in entries if metadata_filter(entry))
        from yogimass.similarity.backends import create_index_backend

        index = create_index_backend(backend, entries=entries)
        return index.query(query_vector, top_n=top_n, min_score=min_score)

    def build_index(self, *, backend: str = "naive"):
        """
        Build and return a populated search backend for this library.
        """
        from yogimass.similarity.backends import create_index_backend

        return create_index_backend(backend, entries=self.iter_entries())

    def __len__(self) -> int:
        if self.storage == "json":
            data = self._read_json_data()
            return len(self._json_entries(data))
        with sqlite3.connect(self.path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM spectra")
            (count,) = cursor.fetchone()
            return int(count)

    def write_entries(self, entries: Iterable[LibraryEntry]) -> None:
        """
        Replace the stored spectra with the provided ``entries``.
        """
        if self.storage == "json":
            self._write_json([entry.to_record() for entry in entries])
            return
        with sqlite3.connect(self.path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM spectra")
            cursor.executemany(
                "INSERT INTO spectra(identifier, precursor_mz, metadata, vector) VALUES (?, ?, ?, ?)",
                [
                    (
                        entry.identifier,
                        entry.precursor_mz,
                        json.dumps(entry.metadata),
                        json.dumps(entry.vector),
                    )
                    for entry in entries
                ],
            )
            conn.commit()

    def _infer_storage(self) -> Literal["json", "sqlite"]:
        if self.path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            return "sqlite"
        return "json"

    def _ensure_store(self) -> None:
        if self.storage == "json":
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._write_json([])
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS spectra (
                        identifier TEXT PRIMARY KEY,
                        precursor_mz REAL,
                        metadata TEXT NOT NULL,
                        vector TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                    ("schema_version", SCHEMA_VERSION),
                )
                conn.commit()

    def _write_entry(self, entry: LibraryEntry, *, overwrite: bool) -> None:
        if self.storage == "json":
            data = self._read_json_data()
            records = self._json_entries(data)
            existing = {item["identifier"]: idx for idx, item in enumerate(records)}
            if entry.identifier in existing and not overwrite:
                raise ValueError(f"Entry '{entry.identifier}' already exists in {self.path}")
            entry_record = entry.to_record()
            if entry.identifier in existing:
                records[existing[entry.identifier]] = entry_record
            else:
                records.append(entry_record)
            self._write_json(records)
        else:
            self._write_sqlite(entry, overwrite=overwrite)

    def _iter_json_entries(self) -> Iterator[LibraryEntry]:
        data = self._read_json_data()
        for record in self._json_entries(data):
            yield LibraryEntry(
                identifier=record["identifier"],
                precursor_mz=record.get("precursor_mz"),
                metadata=record.get("metadata", {}),
                vector=record.get("vector", {}),
            )

    def _iter_sqlite_entries(self) -> Iterator[LibraryEntry]:
        with sqlite3.connect(self.path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT identifier, precursor_mz, metadata, vector FROM spectra")
            for identifier, precursor_mz, metadata_json, vector_json in cursor.fetchall():
                yield LibraryEntry(
                    identifier=identifier,
                    precursor_mz=precursor_mz,
                    metadata=json.loads(metadata_json),
                    vector=json.loads(vector_json),
                )

    def _write_sqlite(self, entry: LibraryEntry, *, overwrite: bool) -> None:
        with sqlite3.connect(self.path) as conn:
            cursor = conn.cursor()
            sql = (
                "INSERT OR REPLACE INTO spectra(identifier, precursor_mz, metadata, vector) VALUES (?, ?, ?, ?)"
                if overwrite
                else "INSERT INTO spectra(identifier, precursor_mz, metadata, vector) VALUES (?, ?, ?, ?)"
            )
            cursor.execute(
                sql,
                (
                    entry.identifier,
                    entry.precursor_mz,
                    json.dumps(entry.metadata),
                    json.dumps(entry.vector),
                ),
            )
            conn.commit()

    def _read_json_data(self) -> Any:
        text = self.path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON library %s, resetting to empty list.", self.path)
            return []

    def _json_entries(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            entries = data.get("entries", [])
            return entries if isinstance(entries, list) else []
        return []

    def _write_json(self, records: Iterable[dict[str, Any]]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "entries": list(records),
        }
        data = json.dumps(payload, indent=2)
        self.path.write_text(data, encoding="utf-8")

    def _spectrum_metadata(self, spectrum: Spectrum) -> dict[str, Any]:
        metadata = dict(spectrum.metadata or {})
        if "name" not in metadata and "compound_name" in metadata:
            metadata["name"] = metadata["compound_name"]
        return {key: value for key, value in metadata.items() if value is not None}
