"""
Tests for the worker-owned library backend architecture (MassFlow.library).

Covers the memory-model contract:

* raw libraries are normalized into a worker-openable store exactly once;
* the store round-trips spectra byte-for-byte (float64 peaks + full metadata),
  which is what makes worker results identical to in-memory results;
* the cached store is invalidated when the source or the processing pipeline
  changes;
* the compact LibrarySpec is the only object that crosses the process
  boundary;
* results reproduce the pre-refactor golden outputs captured from the previous
  in-memory design (tests/data/golden_multiprocessing), compared row-for-row
  with a tight float tolerance (byte equality is not portable across platforms).
"""

import csv
import math
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

GOLDEN_DIR = Path(__file__).parent / "data" / "golden_multiprocessing"


def _as_float(value: str | None) -> float | None:
    """Return *value* as a float, or ``None`` when it is not numeric."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def assert_csv_rows_equivalent(actual_path: Path, expected_path: Path) -> None:
    """Compare two result CSVs row-for-row, tolerating float round-off.

    Byte-for-byte equality is not portable: scores are exported at full float64
    precision, and the low-order bits differ between platforms (macOS
    Accelerate vs Linux OpenBLAS, FMA contraction, ...), so identical
    scientific results can serialize to different bytes. Categorical and integer
    columns are compared exactly; numeric columns are compared with a tight
    tolerance (still orders of magnitude below any real scoring change).
    """
    with open(actual_path, newline="") as handle:
        actual_reader = csv.DictReader(handle)
        actual_fieldnames = actual_reader.fieldnames
        actual_rows = list(actual_reader)
    with open(expected_path, newline="") as handle:
        expected_reader = csv.DictReader(handle)
        expected_fieldnames = expected_reader.fieldnames
        expected_rows = list(expected_reader)

    assert actual_fieldnames == expected_fieldnames, (
        f"Result CSV columns changed: {actual_fieldnames} != {expected_fieldnames}"
    )
    assert len(actual_rows) == len(expected_rows), (
        f"{len(actual_rows)} rows exported, {len(expected_rows)} expected"
    )

    for index, (actual, expected) in enumerate(zip(actual_rows, expected_rows)):
        for column, expected_value in expected.items():
            actual_value = actual[column]
            actual_number = _as_float(actual_value)
            expected_number = _as_float(expected_value)
            if actual_number is None or expected_number is None:
                assert actual_value == expected_value, (
                    f"row {index} column {column!r}: {actual_value!r} != "
                    f"{expected_value!r}"
                )
                continue
            if math.isnan(actual_number) and math.isnan(expected_number):
                continue
            assert actual_number == pytest.approx(
                expected_number, rel=1e-9, abs=1e-12
            ), f"row {index} column {column!r}: {actual_value!r} != {expected_value!r}"


def make_spectrum(spec_id: str, precursor_mz: float = 100.0) -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensities=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        metadata={
            "id": spec_id,
            "precursor_mz": float(precursor_mz),
            "charge": 1,
            "adduct": "[M+H]+",
            "compound_name": spec_id,
            "ionmode": "positive",
            "smiles": "CCO",
        },
    )


def write_msp(path: Path, spectra) -> None:
    lines = []
    for spectrum in spectra:
        lines.append(f"NAME: {spectrum.get('compound_name')}")
        lines.append(f"PRECURSOR_MZ: {spectrum.get('precursor_mz')}")
        lines.append("CHARGE: 1")
        lines.append(f"NUM PEAKS: {len(spectrum.peaks.mz)}")
        for mz, intensity in zip(spectrum.peaks.mz, spectrum.peaks.intensities):
            lines.append(f"{mz}\t{intensity}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_config(tmp_path: Path, library_path: Path) -> MassFlowConfig:
    return MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=tmp_path / "q.mgf", library_path=library_path, format="mgf"
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(fdr_threshold=1.0, min_score=0.0),
    )


class TestPrepareLibrary:
    def test_raw_file_is_normalized_to_store(self, tmp_path):
        library_path = tmp_path / "lib.msp"
        write_msp(library_path, [make_spectrum("ref_1", 200.0)])

        spec, count = prepare_library(
            make_config(tmp_path, library_path), tmp_path / "results"
        )

        assert spec.kind == "store"
        assert spec.path.name == "lib_library.db"
        assert count == 1
        assert spec.path.exists()

    @pytest.mark.parametrize("backend_name", ["sqlite", "zarr", "hybrid"])
    def test_store_input_is_used_directly_without_modification(
        self, tmp_path, backend_name
    ):
        from MassFlow.storage import create_spectral_store

        store_path = tmp_path / ("lib.zarr" if backend_name == "zarr" else "lib.db")
        store = create_spectral_store(store_path, backend=backend_name)
        store.add_spectra(iter([make_spectrum("ref_1", 200.0)]), category="library")
        store.close()

        store_paths = [store_path]
        if backend_name == "hybrid":
            store_paths.append(store_path.with_suffix(".zarr"))

        def snapshot(paths):
            files = [
                file_path
                for path in paths
                for file_path in ([path] if path.is_file() else path.rglob("*"))
                if file_path.is_file()
            ]
            return sorted(
                (
                    str(file_path.relative_to(tmp_path)),
                    file_path.stat().st_size,
                    file_path.stat().st_mtime_ns,
                )
                for file_path in files
            )

        before = snapshot(store_paths)

        config = make_config(tmp_path, store_path)
        spec, count = prepare_library(config, tmp_path / "results")

        assert spec.kind == "store"
        assert spec.path == store_path
        assert count == 1
        assert snapshot(store_paths) == before

    def test_cache_reused_when_source_and_config_unchanged(self, tmp_path):
        library_path = tmp_path / "lib.msp"
        write_msp(library_path, [make_spectrum("ref_1", 200.0)])
        output_dir = tmp_path / "results"

        spec1, _ = prepare_library(make_config(tmp_path, library_path), output_dir)
        first_mtime = spec1.path.stat().st_mtime_ns

        spec2, count2 = prepare_library(make_config(tmp_path, library_path), output_dir)
        assert spec2.path == spec1.path
        assert count2 == 1
        assert spec2.path.stat().st_mtime_ns == first_mtime

    def test_cache_invalidated_when_processing_changes(self, tmp_path):
        library_path = tmp_path / "lib.msp"
        write_msp(library_path, [make_spectrum("ref_1", 200.0)])
        output_dir = tmp_path / "results"

        config_a = make_config(tmp_path, library_path)
        spec_a, _ = prepare_library(config_a, output_dir)

        config_b = make_config(tmp_path, library_path)
        config_b.processing.min_peaks = 2  # different pipeline fingerprint
        spec_b, _ = prepare_library(config_b, output_dir)

        assert spec_a.path == spec_b.path  # same location, rebuilt
        assert spec_b.path.stat().st_mtime_ns >= spec_a.path.stat().st_mtime_ns

    @pytest.mark.parametrize("backend_name", ["sqlite", "zarr", "hybrid"])
    def test_invalidated_cache_replaces_old_spectra(self, tmp_path, backend_name):
        library_path = tmp_path / "lib.msp"
        write_msp(library_path, [make_spectrum("ref_1", 200.0)])
        output_dir = tmp_path / "results"
        config = make_config(tmp_path, library_path)
        config.input.storage_backend = backend_name

        first_spec, _ = prepare_library(config, output_dir)
        config.processing.min_peaks = 2
        rebuilt_spec, rebuilt_count = prepare_library(config, output_dir)

        backend = open_library(rebuilt_spec, config.processing)
        try:
            names = [s.get("compound_name") for s in backend.iter_spectra()]
        finally:
            backend.close()

        assert rebuilt_spec.path == first_spec.path
        assert rebuilt_count == 1
        assert names == ["ref_1"]

    @pytest.mark.parametrize("backend_name", ["sqlite", "zarr", "hybrid"])
    def test_failed_rebuild_preserves_previous_cache(
        self, tmp_path, backend_name, monkeypatch
    ):
        from MassFlow import library as library_module

        library_path = tmp_path / "lib.msp"
        write_msp(library_path, [make_spectrum("ref_1", 200.0)])
        output_dir = tmp_path / "results"
        original_config = make_config(tmp_path, library_path)
        original_config.input.storage_backend = backend_name
        original_spec, _ = prepare_library(original_config, output_dir)

        changed_config = make_config(tmp_path, library_path)
        changed_config.input.storage_backend = backend_name
        changed_config.processing.min_peaks = 2
        original_factory = library_module.create_spectral_store

        def create_failing_store(store_path, backend):
            store = original_factory(store_path, backend=backend)
            original_add_spectra = store.add_spectra

            def add_one_then_fail(spectra, category="default", batch_size=5000):
                original_add_spectra(iter([next(iter(spectra))]), category, batch_size)
                raise RuntimeError("simulated interrupted rebuild")

            store.add_spectra = add_one_then_fail
            return store

        monkeypatch.setattr(
            library_module, "create_spectral_store", create_failing_store
        )
        with pytest.raises(RuntimeError, match="simulated interrupted rebuild"):
            prepare_library(changed_config, output_dir)

        backend = open_library(original_spec, original_config.processing)
        try:
            names = [s.get("compound_name") for s in backend.iter_spectra()]
        finally:
            backend.close()

        assert names == ["ref_1"]
        monkeypatch.setattr(library_module, "create_spectral_store", original_factory)
        cached_spec, cached_count = prepare_library(original_config, output_dir)
        assert cached_spec.path == original_spec.path
        assert cached_count == 1

    @pytest.mark.parametrize("backend_name", ["sqlite", "zarr", "hybrid"])
    def test_failed_cache_promotion_rolls_back_all_backend_paths(
        self, tmp_path, backend_name, monkeypatch
    ):
        import os

        from MassFlow import library as library_module

        store_path = tmp_path / (
            "library.zarr" if backend_name == "zarr" else "library.db"
        )
        staged_directory = tmp_path / "staging"
        staged_directory.mkdir()
        staged_store_path = staged_directory / store_path.name
        meta_path = store_path.with_suffix(store_path.suffix + ".meta.json")
        staged_meta_path = staged_store_path.with_suffix(
            staged_store_path.suffix + ".meta.json"
        )

        def write_resource(path, value):
            if path.suffix == ".zarr":
                path.mkdir(parents=True)
                (path / "marker").write_text(value)
            else:
                path.write_text(value)

        related_paths = [store_path, meta_path]
        if backend_name == "hybrid":
            related_paths.append(store_path.with_suffix(".zarr"))
        staged_paths = [staged_store_path, staged_meta_path]
        if backend_name == "hybrid":
            staged_paths.append(staged_store_path.with_suffix(".zarr"))
        for path in related_paths:
            write_resource(path, f"old:{path.name}")
        for path in staged_paths:
            write_resource(path, f"new:{path.name}")

        original_replace = os.replace

        def fail_on_metadata_promotion(source, destination):
            if Path(source) == staged_meta_path:
                raise OSError("simulated metadata promotion failure")
            return original_replace(source, destination)

        monkeypatch.setattr(library_module.os, "replace", fail_on_metadata_promotion)
        with pytest.raises(OSError, match="simulated metadata promotion failure"):
            library_module._promote_prepared_store(
                staged_store_path, staged_meta_path, store_path, backend_name
            )

        for path in related_paths:
            if path.is_dir():
                assert (path / "marker").read_text() == f"old:{path.name}"
            else:
                assert path.read_text() == f"old:{path.name}"

    def test_cache_invalidated_when_source_changes(self, tmp_path):
        library_path = tmp_path / "lib.msp"
        write_msp(library_path, [make_spectrum("ref_1", 200.0)])
        output_dir = tmp_path / "results"

        spec_a, count_a = prepare_library(
            make_config(tmp_path, library_path), output_dir
        )
        assert count_a == 1

        # Modify the source: the cache must be rebuilt with the new content.
        write_msp(
            library_path,
            [make_spectrum("ref_1", 200.0), make_spectrum("ref_2", 250.0)],
        )
        spec_b, count_b = prepare_library(
            make_config(tmp_path, library_path), output_dir
        )
        assert count_b == 2
        backend = open_library(spec_b, make_config(tmp_path, library_path).processing)
        try:
            names = [s.get("compound_name") for s in backend.iter_spectra()]
        finally:
            backend.close()
        assert "ref_2" in names


class TestBackendFidelity:
    def test_store_round_trip_is_byte_identical(self, tmp_path):
        """The store must round-trip peaks and metadata exactly, because the
        whole determinism argument rests on it."""
        from MassFlow.storage import create_spectral_store

        original = make_spectrum("ref_1", 200.0)
        store_path = tmp_path / "lib.db"
        store = create_spectral_store(store_path, backend="sqlite")
        store.add_spectra(iter([original]), category="library")
        store.close()

        backend = open_library(
            LibrarySpec(path=store_path, kind="store", storage_backend="sqlite"),
            ProcessingConfig(min_peaks=1),
        )
        try:
            read_back = list(backend.iter_spectra())[0]
        finally:
            backend.close()

        assert np.array_equal(read_back.peaks.mz, original.peaks.mz)
        assert np.array_equal(read_back.peaks.intensities, original.peaks.intensities)
        for key in (
            "id",
            "precursor_mz",
            "charge",
            "adduct",
            "compound_name",
            "ionmode",
            "smiles",
        ):
            assert read_back.get(key) == original.get(key), key

    def test_library_spec_is_compact(self, tmp_path):
        """The only object crossing the process boundary is a path plus two
        strings — constant size regardless of library size."""
        import pickle

        spec = LibrarySpec(
            path=tmp_path / "lib.db", kind="store", storage_backend="sqlite"
        )
        payload = pickle.dumps(spec)
        assert len(payload) < 512

    def test_file_backend_applies_processing(self, tmp_path):
        library_path = tmp_path / "lib.msp"
        write_msp(library_path, [make_spectrum("ref_1", 200.0)])
        spec = LibrarySpec(path=library_path, kind="file")
        backend = open_library(spec, ProcessingConfig(min_peaks=1))
        assert isinstance(backend, RawFileLibraryStore)
        try:
            spectra = list(backend.iter_spectra())
        finally:
            backend.close()
        assert len(spectra) == 1
        assert spectra[0].get("compound_name") == "ref_1"


class TestGoldenDeterminism:
    """Determinism against outputs captured from the PRE-REFACTOR in-memory
    design (tests/data/golden_multiprocessing).

    The comparison is row-for-row with a tight float tolerance rather than byte
    equality: scores are exported at full float64 precision, so identical
    scientific results can serialize differently across platforms. Column order,
    row count, categorical and integer fields are compared exactly.
    """

    @pytest.mark.parametrize("query_index", [0, 1, 2])
    def test_worker_path_matches_pre_refactor_golden(self, tmp_path, query_index):
        """The new backend architecture must reproduce the pre-refactor golden
        results (row-for-row; numeric columns within round-off tolerance)."""
        from MassFlow.workflow import run_annotation_pipeline

        config = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path / "results"),
            input=InputConfig(
                input_path=GOLDEN_DIR / f"queries_{query_index}.mgf",
                library_path=GOLDEN_DIR / "golden_library.msp",
                format="mgf",
            ),
            processing=ProcessingConfig(min_peaks=1),
            similarity=SimilarityConfig(
                algorithm="cosine",
                min_score=0.0,
                fdr_threshold=1.0,
                ms1_tolerance=100.0,
                ms2_tolerance=0.5,
            ),
        )

        results = run_annotation_pipeline(config)
        assert len(results) == 1
        assert results[0].status == "success"

        out_file = tmp_path / "results" / f"queries_{query_index}_results.csv"
        golden_file = GOLDEN_DIR / "results" / f"queries_{query_index}_results.csv"
        assert_csv_rows_equivalent(out_file, golden_file)
