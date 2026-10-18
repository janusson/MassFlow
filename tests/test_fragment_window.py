"""Fragment filtering and migration regressions using explicitly synthetic peaks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_msp
from pyteomics import mass

from MassFlow import library, processing
from MassFlow.config import InputConfig, MassFlowConfig, ProcessingConfig, ProjectConfig
from MassFlow.storage import create_spectral_store


def _spectrum(precursor_mz: float | None = 800.0) -> Spectrum:
    """Create a claim-free spectrum; these peaks are not experimental data."""
    return Spectrum(
        mz=np.array([49.0, 50.0, 100.0, 500.0, 501.0], dtype=np.float64),
        intensities=np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64),
        metadata={
            "id": "query",
            "compound_name": "synthetic",
            "precursor_mz": precursor_mz,
        },
        metadata_harmonization=False,
    )


def _config(**overrides: object) -> ProcessingConfig:
    """Enable only the fragment filter unless a test overrides it."""
    values: dict[str, object] = dict(
        filter_by_mz=True,
        mz_min=50.0,
        mz_max=500.0,
        filter_by_intensity=False,
        filter_min_peaks=False,
        normalize_intensity=False,
    )
    values.update(overrides)
    return ProcessingConfig.model_validate(values)


@pytest.mark.parametrize("precursor_mz", [800.0, 20.0, None])
def test_fragment_window_does_not_gate_precursor(precursor_mz: float | None) -> None:
    """A fragment window retains eligible peaks regardless of precursor metadata."""
    result = processing.process_spectra_batch([_spectrum(precursor_mz)], _config())
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].peaks.mz, [50.0, 100.0, 500.0])
    np.testing.assert_array_equal(result[0].peaks.intensities, [2.0, 4.0, 8.0])
    assert result[0].peaks.mz.dtype == np.float64
    assert result[0].get("precursor_mz") == precursor_mz


def test_fragment_window_toggle_disabled_preserves_peaks() -> None:
    result = processing.process_spectra_batch(
        [_spectrum()], _config(filter_by_mz=False)
    )
    np.testing.assert_array_equal(result[0].peaks.mz, [49.0, 50.0, 100.0, 500.0, 501.0])


def test_fragment_filter_preserves_noise_then_minimum_count_order() -> None:
    result = processing.process_spectra_batch(
        [_spectrum()],
        _config(
            filter_by_intensity=True,
            noise_threshold=9.0,
            filter_min_peaks=True,
            min_peaks=2,
        ),
    )
    assert result == []


def test_minimum_count_remains_before_fragment_truncation() -> None:
    result = processing.process_spectra_batch(
        [_spectrum()],
        _config(filter_min_peaks=True, min_peaks=5),
    )
    assert len(result) == 1
    assert len(result[0].peaks) == 3


@pytest.mark.parametrize("normalize", [True, False])
def test_empty_after_truncation_preserves_normalization_policy(
    normalize: bool,
) -> None:
    result = processing.process_spectra_batch(
        [_spectrum()],
        _config(mz_min=600.0, mz_max=700.0, normalize_intensity=normalize),
    )
    assert len(result) == 1
    assert len(result[0].peaks) == 0


def test_fragment_window_still_enforces_physical_integrity() -> None:
    theoretical_mz = mass.calculate_mass(formula="C9H8O4") + mass.nist_mass["H+"][0][0]
    spectrum = _spectrum(theoretical_mz * (1.0 + 6.0e-6))
    spectrum.set("formula", "C9H8O4")
    spectrum.set("adduct", "[M+H]+")
    spectrum.set("charge", 1)
    reasons: list[str] = []
    assert processing.process_spectra_batch([spectrum], _config(), reasons.append) == []
    assert len(reasons) == 1
    assert "ppm" in reasons[0]


@pytest.mark.parametrize("backend", ["sqlite", "zarr", "hybrid"])
def test_fragment_revision_rebuilds_managed_cache_once(
    tmp_path: Path,
    backend: Literal["sqlite", "zarr", "hybrid"],
) -> None:
    """Revision-1 caches cannot hide spectra discarded by the old precursor gate."""
    source = tmp_path / "library.msp"
    save_as_msp([_spectrum()], str(source))
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=tmp_path / "query.mgf",
            library_path=source,
            storage_backend=backend,
        ),
        processing=_config(),
    )
    output = config.project.output_directory
    output.mkdir()
    destination = output / (
        "library_library.zarr" if backend == "zarr" else "library_library.db"
    )
    store = create_spectral_store(destination, backend=backend)
    old_spectrum = _spectrum(100.0)
    old_spectrum.set("compound_name", "legacy-only")
    try:
        store.add_spectra(iter([old_spectrum]), category="library")
    finally:
        store.close()
    revision_one_payload = {
        "processing_semantics_revision": 1,
        "processing": config.processing.model_dump(mode="json"),
    }
    old_fingerprint = hashlib.sha256(
        json.dumps(revision_one_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata_path = destination.with_suffix(destination.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(
            dict(
                source=str(source),
                source_mtime_ns=source.stat().st_mtime_ns,
                source_size=source.stat().st_size,
                fingerprint=old_fingerprint,
                backend=backend,
            )
        )
    )
    spec, count = library.prepare_library(config, output)
    assert count == 1
    store = library.open_library(spec, config.processing)
    try:
        spectra = list(store.iter_spectra())
        assert [s.get("compound_name") for s in spectra] == ["synthetic"]
        np.testing.assert_array_equal(spectra[0].peaks.mz, [50.0, 100.0, 500.0])
    finally:
        store.close()
    assert json.loads(metadata_path.read_text())["fingerprint"] != old_fingerprint
    first_mtime_ns = metadata_path.stat().st_mtime_ns
    assert library.prepare_library(config, output) == (spec, 1)
    assert metadata_path.stat().st_mtime_ns == first_mtime_ns
