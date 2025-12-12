"""
Interoperability helpers for Yogimass libraries (export, schema version checks).
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from yogimass.similarity.library import LocalSpectralLibrary, SCHEMA_VERSION


def read_schema_version(library_path: str | Path) -> str | None:
    path = Path(library_path)
    if path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            version = data.get("schema_version")
            return str(version) if version is not None else None
        return None
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT value FROM metadata WHERE key='schema_version'")
            except sqlite3.Error:
                return None
            row = cursor.fetchone()
            if row:
                version = row[0]
                return str(version) if version is not None else None
        return None
    return None


def assert_schema_compatible(library_path: str | Path, minimum_version: str = SCHEMA_VERSION) -> None:
    version = read_schema_version(library_path)
    if version is None:
        return  # legacy/unknown; accept for now
    if _parse_version(version) < _parse_version(minimum_version):
        raise ValueError(f"Library schema version {version} is older than supported minimum {minimum_version}.")


def export_library_summary_csv(library_path: str | Path, output_path: str | Path) -> Path:
    lib = LocalSpectralLibrary(library_path)
    rows = []
    for entry in lib.iter_entries():
        metadata = entry.metadata or {}
        rows.append(
            {
                "id": entry.identifier,
                "precursor_mz": entry.precursor_mz if entry.precursor_mz is not None else "",
                "name": metadata.get("name") or metadata.get("compound_name") or "",
                "retention_time": metadata.get("retention_time") or "",
            }
        )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "precursor_mz", "name", "retention_time"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def import_library_from_mgf(*_: Any, **__: Any) -> None:
    raise NotImplementedError("Importing entire libraries from MGF is not implemented.")


def _parse_version(text: str) -> tuple[int, ...]:
    parts = []
    for part in text.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)
