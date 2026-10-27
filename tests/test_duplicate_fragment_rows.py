"""Exact duplicate rows must not manufacture distinct fragment evidence.

These are synthetic caffeine-formula spectra, not measured fragmentation data.
Each ion mass is a hypothetical protonated formula subcomposition. Splitting
each intensity across identical rows conserves per-ion signal while making a
single shared fragment satisfy matchms's row-based three-match gate.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import yaml
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp
from pyteomics import mass

from MassFlow.config import InputConfig, MassFlowConfig, ProjectConfig, SimilarityConfig
from MassFlow.io import _validate_spectra_iterator, load_spectra
from MassFlow.library import open_library, prepare_library
from MassFlow.models import MolecularStructure, SpectrumMetadata
from MassFlow.processing import physical_integrity_reason
from MassFlow.similarity import SimilarityEngine
from MassFlow.storage import create_spectral_store
from MassFlow.workflow import run_annotation_pipeline

pytestmark = [pytest.mark.core, pytest.mark.scientific]

_FORMULA = "C8H10N4O2"
_PRECURSOR_MZ = float(mass.calculate_mass(formula=_FORMULA, charge=1))
_SHARED_MZ = float(mass.calculate_mass(formula="C6H7N3O", charge=1))
_REFERENCE_ONLY_MZ = float(mass.calculate_mass(formula="C5H7N3", charge=1))
_QUERY_ONLY_MZ = float(mass.calculate_mass(formula="C4H6N2", charge=1))


def _spectrum(identifier: str, copies: int = 1, *, query: bool = False) -> Spectrum:
    """Build two distinct ions with conserved intensity under exact row splitting."""
    second_mz = _QUERY_ONLY_MZ if query else _REFERENCE_ONLY_MZ
    spectrum = Spectrum(
        mz=np.repeat(np.array([second_mz, _SHARED_MZ], dtype=np.float64), copies),
        intensities=np.repeat(np.array([0.1, 1.0], dtype=np.float64) / copies, copies),
        metadata={
            "id": identifier,
            "compound_name": identifier,
            "precursor_mz": _PRECURSOR_MZ,
            "formula": _FORMULA,
            "charge": 1,
            "adduct": "[M+H]+",
            "ionmode": "positive",
        },
        metadata_harmonization=False,
    )
    molecule = MolecularStructure(formula=_FORMULA)
    metadata = SpectrumMetadata(
        spectrum_id=identifier,
        precursor_mz=_PRECURSOR_MZ,
        charge=1,
        adduct="[M+H]+",
        molecule=molecule,
    )
    assert molecule.is_physically_valid and metadata.is_physically_valid
    assert physical_integrity_reason(spectrum) is None
    return spectrum


def _write(path: Path, spectra: list[Spectrum]) -> None:
    """Write actual open-format records without mocking the importers."""
    if path.suffix == ".mgf":
        save_as_mgf(spectra, str(path))
    else:
        save_as_msp(spectra, str(path))


@pytest.mark.parametrize("extension", ["mgf", "msp"])
@pytest.mark.parametrize("copies", [2, 3, 5])
def test_duplicate_rows_are_quarantined(
    tmp_path: Path, extension: str, copies: int
) -> None:
    """Reject a malformed spectrum once, with an observable duplicate-mass reason."""
    path = tmp_path / f"duplicate.{extension}"
    _write(path, [_spectrum("duplicate", copies)])
    reasons: list[str] = []
    assert list(load_spectra(path, rejection_reporter=reasons.append)) == []
    assert reasons == ["Duplicate fragment m/z values"]


@pytest.mark.parametrize("extension", ["mgf", "msp"])
@pytest.mark.parametrize("separation", [0.001, 0.02])
def test_close_distinct_masses_are_preserved(
    tmp_path: Path, extension: str, separation: float
) -> None:
    """Do not turn scoring tolerance into an implicit peak-merging tolerance."""
    original = _spectrum("close")
    original.peaks = type(original.peaks)(
        mz=np.array([_SHARED_MZ, _SHARED_MZ + separation], dtype=np.float64),
        intensities=np.array([0.1, 1.0], dtype=np.float64),
    )
    path = tmp_path / f"close.{extension}"
    _write(path, [original])
    reasons: list[str] = []
    loaded = list(load_spectra(path, rejection_reporter=reasons.append))
    assert not reasons
    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0].peaks.mz, original.peaks.mz, atol=1e-8, rtol=0)
    assert loaded[0].peaks.mz.dtype == np.float64


@pytest.mark.parametrize("single_peak", [False, True])
def test_exact_equality_boundary(single_peak: bool) -> None:
    """A single row or one-ULP mass separation must not count as a duplicate."""
    original = _spectrum("boundary")
    mz_values = (
        [_SHARED_MZ] if single_peak else [_SHARED_MZ, np.nextafter(_SHARED_MZ, np.inf)]
    )
    original.peaks = type(original.peaks)(
        mz=np.array(mz_values, dtype=np.float64),
        intensities=np.ones(len(mz_values), dtype=np.float64),
    )
    reasons: list[str] = []
    loaded = list(
        _validate_spectra_iterator([original], Path("synthetic.mgf"), reasons.append)
    )
    assert len(loaded) == 1
    assert not reasons
    np.testing.assert_array_equal(loaded[0].peaks.mz, original.peaks.mz)


@pytest.mark.parametrize("extension", ["mgf", "msp"])
@pytest.mark.parametrize("algorithm", ["cosine", "modified_cosine"])
def test_duplicate_rows_cannot_enter_raw_file_candidate_pool(
    tmp_path: Path, extension: str, algorithm: str
) -> None:
    """The uncorrected raw files score 100/101 with three row matches, one ion."""
    reference = _spectrum("reference", 3)
    query = _spectrum("query", 3, query=True)
    engine = SimilarityEngine(SimilarityConfig(algorithm=algorithm))
    raw_score = engine.similarity_function.pair(reference, query)
    assert float(raw_score["score"]) == pytest.approx(100 / 101, abs=1e-12)
    assert int(raw_score["matches"]) == 3
    reference_path = tmp_path / f"reference.{extension}"
    query_path = tmp_path / f"query.{extension}"
    _write(reference_path, [reference])
    _write(query_path, [query])
    hits = engine.search(
        list(load_spectra(query_path)),
        list(load_spectra(reference_path)),
        include_decoys=False,
    )
    assert hits == []


def test_mixed_query_rejection_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A rejected duplicate query remains counted in the real workflow sidecar."""
    good = _spectrum("good")
    query_path = tmp_path / "queries.mgf"
    library_path = tmp_path / "library.msp"
    _write(query_path, [good, _spectrum("duplicate", 3)])
    _write(library_path, [_spectrum("reference")])
    output_directory = tmp_path / "results"
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_directory),
        input=InputConfig(input_path=query_path, library_path=library_path),
        similarity=SimilarityConfig(min_matched_peaks=1, fdr_threshold=1.0),
    )
    quarantine_logger = logging.getLogger("quarantine")
    quarantine_logger.addHandler(caplog.handler)
    try:
        results = run_annotation_pipeline(config)
    finally:
        quarantine_logger.removeHandler(caplog.handler)
    assert len(results) == 1
    assert results[0].spectra_loaded == 1
    assert results[0].spectra_rejected == 1
    assert "duplicate" not in (output_directory / "queries_results.csv").read_text()
    report = yaml.safe_load(
        (output_directory / "queries_results.report.yaml").read_text()
    )
    assert report["spectra_rejected"] == 1
    assert "Reason: Duplicate fragment m/z values" in caplog.text


def test_all_duplicate_queries_fail_explicitly(tmp_path: Path) -> None:
    """An all-quarantined input must produce a failure report rather than a CSV."""
    query_path = tmp_path / "queries.mgf"
    library_path = tmp_path / "library.msp"
    _write(query_path, [_spectrum("duplicate", 3)])
    _write(library_path, [_spectrum("reference")])
    output_directory = tmp_path / "results"
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_directory),
        input=InputConfig(input_path=query_path, library_path=library_path),
    )
    result = run_annotation_pipeline(config)[0]
    assert result.status == "failed"
    assert result.spectra_rejected == 1
    assert not (output_directory / "queries_results.csv").exists()
    report = yaml.safe_load(
        (output_directory / "queries_failed.report.yaml").read_text()
    )
    assert report["status"] == "failed"
    assert report["spectra_rejected"] == 1
    assert any("Duplicate fragment m/z" in error for error in report["fatal_errors"])


@pytest.mark.parametrize("backend_name", ["sqlite", "zarr", "hybrid"])
def test_legacy_cache_rebuilt_under_strict_row_validation(
    tmp_path: Path, backend_name: Literal["sqlite", "zarr", "hybrid"]
) -> None:
    """Legacy managed caches must not bypass the corrected raw-ingestion policy."""
    library_path = tmp_path / "library.msp"
    spectra = [_spectrum("good"), _spectrum("duplicate", 3)]
    _write(library_path, spectra)
    output_directory = tmp_path / "results"
    output_directory.mkdir()
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_directory),
        input=InputConfig(
            input_path=tmp_path / "query.mgf",
            library_path=library_path,
            storage_backend=backend_name,
        ),
    )
    suffix = ".zarr" if backend_name == "zarr" else ".db"
    cache_path = output_directory / f"library_library{suffix}"
    legacy_store = create_spectral_store(cache_path, backend=backend_name)
    try:
        assert legacy_store.add_spectra(iter(spectra), category="library") == 2
    finally:
        legacy_store.close()
    source_stat = library_path.stat()
    legacy_metadata = {
        "source": str(library_path),
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_size": source_stat.st_size,
        "backend": backend_name,
        "fingerprint": hashlib.sha256(
            config.processing.model_dump_json().encode()
        ).hexdigest(),
    }
    cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text(
        json.dumps(legacy_metadata)
    )
    spec, count = prepare_library(config, output_directory)
    assert count == 1
    backend = open_library(spec, config.processing)
    try:
        assert [s.get("compound_name") for s in backend.iter_spectra()] == ["good"]
    finally:
        backend.close()
