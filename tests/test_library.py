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

    def test_store_input_is_used_directly(self, tmp_path):
        from MassFlow.storage import create_spectral_store

        store_path = tmp_path / "lib.db"
        store = create_spectral_store(store_path, backend="sqlite")
        store.add_spectra(iter([make_spectrum("ref_1", 200.0)]), category="library")
        store.close()

        config = make_config(tmp_path, store_path)
        spec, count = prepare_library(config, tmp_path / "results")

        assert spec.kind == "store"
        assert spec.path == store_path
        assert count == 1

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
