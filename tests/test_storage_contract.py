"""
Storage backend contract tests.

The annotation layer must not care whether the underlying library is
SQLite, Zarr, or hybrid.  These tests assert the unified
``MassFlow.storage.SpectralStore`` interface contract:

- interface conformance of every backend (metadata lookup,
  precursor-range filtering, sequential iteration, batched access,
  spectrum count, backend provenance);
- the SAME search run against SQLite-built and Zarr-built libraries
  returns identical spectrum IDs, precursor metadata, peak arrays,
  similarity scores, and ranking order;
- ``storage_backend: zarr`` has one unambiguous meaning: the library
  store built for the run is a Zarr store, and run provenance records
  the effective backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.library import (
    LibrarySpec,
    RawFileLibraryStore,
    open_library,
    prepare_library,
)
from MassFlow.storage import SpectralStore, create_spectral_store


def _write_library(tmp_path: Path, filename: str = "library.msp") -> Path:
    """A small library with overlapping peaks (multiple ranked hits per
    query)."""
    entries = [
        ("CmpdA", 200.0, [(100.0, 999.0), (200.0, 500.0), (300.0, 100.0)]),
        ("CmpdB", 201.0, [(100.0, 950.0), (200.0, 480.0), (310.0, 90.0)]),
        ("CmpdC", 202.0, [(100.0, 900.0), (205.0, 450.0), (300.0, 80.0)]),
        ("CmpdD", 500.0, [(400.0, 800.0), (500.0, 300.0)]),
    ]
    lines: list[str] = []
    for name, precursor_mz, peaks in entries:
        lines += [
            f"NAME: {name}",
            f"PRECURSORMZ: {precursor_mz}",
            "IONMODE: Positive",
            "CHARGE: 1",
            f"Num Peaks: {len(peaks)}",
        ]
        lines += [f"{mz} {intensity}" for mz, intensity in peaks]
        lines.append("")
    path = tmp_path / filename
    path.write_text("\n".join(lines) + "\n")
    return path


def _config(tmp_path: Path, library: Path, storage_backend: str) -> MassFlowConfig:
    return MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "out"),
        input=InputConfig(
            input_path=tmp_path / "query.mgf",
            library_path=library,
            storage_backend=storage_backend,  # type: ignore[arg-type]
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(
            algorithm="cosine",
            min_score=0.0,
            min_matched_peaks=1,
            fdr_threshold=1.0,
            ms1_tolerance=100.0,
            ms2_tolerance=0.5,
        ),
    )


def _query_spectrum() -> Spectrum:
    # Overlaps ref_1 (100/200/300), ref_2 (150), and ref_3 (100/300) so the
    # ranking contains multiple hits with distinct scores.
    return Spectrum(
        mz=np.array([100.0, 150.0, 200.0, 300.0], dtype=np.float64),
        intensities=np.array([999.0, 400.0, 500.0, 100.0], dtype=np.float64),
        metadata={"id": "query_1", "precursor_mz": 200.0, "charge": 1},
    )


def _make_backends(tmp_path: Path) -> dict[str, SpectralStore]:
    """Build one store per backend (sqlite / zarr / hybrid) with the same
    spectra."""
    spectra = [
        Spectrum(
            mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
            intensities=np.array([999.0, 500.0, 100.0], dtype=np.float64),
            metadata={
                "id": "ref_1",
                "compound_name": "Ref One",
                "precursor_mz": 200.0,
                "charge": 1,
                "ionmode": "positive",
                "adduct": "[M+H]+",
            },
        ),
        Spectrum(
            mz=np.array([150.0, 250.0], dtype=np.float64),
            intensities=np.array([800.0, 400.0], dtype=np.float64),
            metadata={
                "id": "ref_2",
                "compound_name": "Ref Two",
                "precursor_mz": 250.0,
                "charge": 1,
                "ionmode": "positive",
                "adduct": "[M+H]+",
            },
        ),
        Spectrum(
            mz=np.array([100.0, 300.0], dtype=np.float64),
            intensities=np.array([900.0, 80.0], dtype=np.float64),
            metadata={
                "id": "ref_3",
                "compound_name": "Ref Three",
                "precursor_mz": 202.0,
                "charge": 1,
                "ionmode": "positive",
                "adduct": "[M+H]+",
            },
        ),
    ]
    stores: dict[str, SpectralStore] = {}
    for backend, suffix in (
        ("sqlite", "lib.db"),
        ("zarr", "lib.zarr"),
        ("hybrid", "lib.db"),
    ):
        path = tmp_path / f"{backend}_{suffix}"
        store = create_spectral_store(path, backend=backend)
        store.add_spectra(iter(spectra), category="library")
        stores[backend] = store
    return stores


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["sqlite", "zarr", "hybrid"])
def test_interface_conformance(tmp_path: Path, backend: str) -> None:
    """Every backend implements the full SpectralStore interface with the
    same observable behavior."""
    stores = _make_backends(tmp_path)
    store = stores[backend]
    try:
        # Count + provenance.
        assert store.get_total_spectra_count() == 3
        assert store.spectrum_count() == 3
        provenance = store.backend_provenance()
        assert provenance["backend"] == backend
        assert provenance["spectrum_count"] == 3
        assert provenance["path"]

        # Sequential iteration.
        ids = [s.get("id") for s in store.get_spectra()]
        assert ids == ["ref_1", "ref_2", "ref_3"]
        assert [s.get("id") for s in store.iter_spectra()] == ids

        # Chunked iteration.
        chunks = list(store.iter_processed_chunks(chunk_size=1))
        assert [len(c) for c in chunks] == [1, 1, 1]
        assert [c[0].get("id") for c in chunks] == ["ref_1", "ref_2", "ref_3"]

        # Single-spectrum lookup.
        assert store.get_spectrum_by_id("ref_1") is not None
        assert store.get_spectrum_by_id("missing") is None

        # Precursor-range filtering primitives.
        lo, hi = store.get_precursor_mz_range()
        assert (lo, hi) == (200.0, 250.0)

        # Batched array access.
        mz_arrays, intensity_arrays = store.batch_get_arrays(
            ["ref_1", "ref_2", "ref_3"]
        )
        assert len(mz_arrays) == 3
        assert np.array_equal(mz_arrays[0], np.array([100.0, 200.0, 300.0]))
        assert np.array_equal(intensity_arrays[0], np.array([999.0, 500.0, 100.0]))

        # Metadata lookup.
        metadata = store.metadata_query(["id", "precursor_mz", "name"])
        assert list(metadata["id"]) == ["ref_1", "ref_2", "ref_3"]
        assert np.allclose(metadata["precursor_mz"], [200.0, 250.0, 202.0])
        assert list(metadata["name"]) == ["Ref One", "Ref Two", "Ref Three"]
        with pytest.raises(ValueError, match="Unknown metadata field"):
            store.metadata_query(["nonexistent_field"])

        # Category filter in metadata lookup.
        by_category = store.metadata_query(["id"], category="library")
        assert list(by_category["id"]) == ["ref_1", "ref_2", "ref_3"]
        empty = store.metadata_query(["id"], category="nope")
        assert len(empty["id"]) == 0

        # Category counts.
        assert store.get_category_counts().get("library") == 3
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The DoD: the annotation layer does not care about the backend
# ---------------------------------------------------------------------------


def _run_search(backend: SpectralStore) -> list[dict]:
    """Run the same search through the annotation path (engine consuming the
    backend stream) and return the normalized hit ranking."""
    from MassFlow.similarity import SimilarityEngine

    config = SimilarityConfig(
        algorithm="cosine",
        min_score=0.0,
        min_matched_peaks=1,
        fdr_threshold=1.0,
        ms1_tolerance=100.0,
        ms2_tolerance=0.5,
    )
    engine = SimilarityEngine(config)
    results = engine.search(
        [_query_spectrum()],
        backend.iter_spectra(),
        min_score=0.0,
        include_decoys=False,
    )
    normalized = []
    for result in results:
        normalized.append(
            {
                "query_id": result["query_id"],
                "reference_id": result["reference_id"],
                "reference_name": result["reference_name"],
                "reference_precursor_mz": result["reference_precursor_mz"],
                "score": round(float(result["score"]), 12),
                "matched_peaks": result["matched_peaks"],
            }
        )
    return normalized


def _backend_stream_snapshot(backend: SpectralStore) -> dict:
    spectra = list(backend.iter_spectra())
    return {
        "ids": [s.get("id") for s in spectra],
        "precursor_mz": [s.get("precursor_mz") for s in spectra],
        "names": [s.get("compound_name") for s in spectra],
        "peaks": [
            (
                s.peaks.mz.astype(np.float64).tolist(),
                s.peaks.intensities.astype(np.float64).tolist(),
            )
            for s in spectra
        ],
    }


@pytest.mark.parametrize("backend_a", ["sqlite", "zarr", "hybrid"])
@pytest.mark.parametrize("backend_b", ["sqlite", "zarr", "hybrid"])
def test_contract_same_search_across_backends(
    tmp_path: Path, backend_a: str, backend_b: str
) -> None:
    """The same library and query, searched through the annotation path,
    produce identical results regardless of the storage backend."""
    stores = _make_backends(tmp_path)
    try:
        stream_a = _backend_stream_snapshot(stores[backend_a])
        stream_b = _backend_stream_snapshot(stores[backend_b])
        assert stream_a == stream_b

        results_a = _run_search(stores[backend_a])
        results_b = _run_search(stores[backend_b])
        assert results_a == results_b, (
            f"search({backend_a}) != search({backend_b}): {results_a} vs {results_b}"
        )
        # The ranking must be non-trivial (multiple ranked hits).
        assert len(results_a) >= 2
        scores = [r["score"] for r in results_a]
        assert scores == sorted(scores, reverse=True)
    finally:
        for store in stores.values():
            store.close()


def test_contract_prepared_libraries_identical_sqlite_vs_zarr(
    tmp_path: Path,
) -> None:
    """prepare_library-built SQLite and Zarr libraries (the actual pipeline
    path) yield identical streams and identical search results."""
    library = _write_library(tmp_path)

    sqlite_spec, sqlite_count = prepare_library(
        _config(tmp_path, library, "sqlite"), tmp_path / "out_sqlite"
    )
    zarr_spec, zarr_count = prepare_library(
        _config(tmp_path, library, "zarr"), tmp_path / "out_zarr"
    )
    assert sqlite_count == zarr_count == 4

    sqlite_backend = open_library(sqlite_spec, ProcessingConfig(min_peaks=1))
    zarr_backend = open_library(zarr_spec, ProcessingConfig(min_peaks=1))
    try:
        assert sqlite_backend.backend_provenance()["backend"] == "sqlite"
        assert zarr_backend.backend_provenance()["backend"] == "zarr"
        assert _backend_stream_snapshot(sqlite_backend) == _backend_stream_snapshot(
            zarr_backend
        )
        assert _run_search(sqlite_backend) == _run_search(zarr_backend)
    finally:
        sqlite_backend.close()
        zarr_backend.close()


# ---------------------------------------------------------------------------
# storage_backend has one unambiguous meaning
# ---------------------------------------------------------------------------


def test_storage_backend_config_selects_the_store_backend(tmp_path: Path) -> None:
    """``storage_backend: zarr`` builds a Zarr library store; ``sqlite``
    builds a SQLite store; the store on disk matches the setting."""
    library = _write_library(tmp_path)

    zarr_spec, _ = prepare_library(
        _config(tmp_path, library, "zarr"), tmp_path / "out_zarr"
    )
    assert zarr_spec.path.suffix == ".zarr"
    assert zarr_spec.storage_backend == "zarr"
    assert zarr_spec.path.is_dir()  # Zarr stores are directories

    sqlite_spec, _ = prepare_library(
        _config(tmp_path, library, "sqlite"), tmp_path / "out_sqlite"
    )
    assert sqlite_spec.path.suffix == ".db"
    assert sqlite_spec.storage_backend == "sqlite"
    assert sqlite_spec.path.is_file()

    hybrid_spec, _ = prepare_library(
        _config(tmp_path, library, "hybrid"), tmp_path / "out_hybrid"
    )
    assert hybrid_spec.path.suffix == ".db"
    assert hybrid_spec.storage_backend == "hybrid"
    assert hybrid_spec.path.with_suffix(".zarr").is_dir()


def test_store_cache_not_crossed_between_backends(tmp_path: Path) -> None:
    """Rebuilding the same source with a different backend produces a fresh
    store (the cache key includes the backend)."""
    library = _write_library(tmp_path)

    sqlite_spec, sqlite_count = prepare_library(
        _config(tmp_path, library, "sqlite"), tmp_path / "out"
    )
    zarr_spec, zarr_count = prepare_library(
        _config(tmp_path, library, "zarr"), tmp_path / "out"
    )
    assert sqlite_count == zarr_count == 4
    assert sqlite_spec.path != zarr_spec.path
    # Both stores exist and are readable.
    sqlite_backend = open_library(sqlite_spec, ProcessingConfig(min_peaks=1))
    zarr_backend = open_library(zarr_spec, ProcessingConfig(min_peaks=1))
    try:
        assert sqlite_backend.get_total_spectra_count() == 4
        assert zarr_backend.get_total_spectra_count() == 4
    finally:
        sqlite_backend.close()
        zarr_backend.close()


def test_run_provenance_records_effective_backend(tmp_path: Path) -> None:
    """The run-level provenance ``backend`` field records the backend the
    library store was actually built with (not merely the configured value)."""
    from MassFlow.workflow import run_annotation_pipeline

    library = _write_library(tmp_path)
    query = tmp_path / "query.mgf"
    query.write_text(
        "BEGIN IONS\n"
        "TITLE=query_1\n"
        "PEPMASS=200.0\n"
        "CHARGE=1+\n"
        "100.0 999.0\n"
        "200.0 500.0\n"
        "300.0 100.0\n"
        "END IONS\n"
    )
    config = _config(tmp_path, library, "zarr")
    config.input.input_path = query
    config.similarity.min_score = 0.0

    results = run_annotation_pipeline(config)
    assert results and results[0].status == "success"

    provenance_files = sorted(tmp_path.glob("out/run_provenance*.json"))
    assert len(provenance_files) == 1
    provenance = json.loads(provenance_files[0].read_text())
    assert provenance["backend"] == "zarr"
    # The source library is recorded; the effective Zarr store was built in
    # the output directory.
    assert provenance["reference_library_path"].endswith("library.msp")
    assert (tmp_path / "out" / "library_library.zarr").is_dir()


# ---------------------------------------------------------------------------
# Raw-file adapter
# ---------------------------------------------------------------------------


def test_raw_file_store_read_only_interface(tmp_path: Path) -> None:
    """The raw-file adapter implements the same interface for reads and
    rejects writes explicitly."""
    library = _write_library(tmp_path)
    spec = LibrarySpec(path=library, kind="file")
    backend = open_library(spec, ProcessingConfig(min_peaks=1))
    try:
        assert isinstance(backend, RawFileLibraryStore)
        assert isinstance(backend, SpectralStore)
        assert backend.get_total_spectra_count() == 4
        assert backend.backend_provenance()["backend"] == "raw-file"
        ids = [s.get("id") for s in backend.iter_spectra()]
        assert len(ids) == 4
        with pytest.raises(NotImplementedError, match="read-only"):
            backend.add_spectra(iter([]))
    finally:
        backend.close()


def test_open_library_returns_spectral_store_for_store_inputs(
    tmp_path: Path,
) -> None:
    """Store inputs open through the unified interface, in their own
    backend."""
    stores = _make_backends(tmp_path)
    for backend_name, store in stores.items():
        store.close()
    for backend_name, filename in (
        ("sqlite", "sqlite_lib.db"),
        ("zarr", "zarr_lib.zarr"),
    ):
        spec = LibrarySpec(path=tmp_path / filename, kind="store")
        backend = open_library(spec, ProcessingConfig(min_peaks=1))
        try:
            assert isinstance(backend, SpectralStore)
            assert backend.backend_provenance()["backend"] == backend_name
            assert backend.get_total_spectra_count() == 3
        finally:
            backend.close()
