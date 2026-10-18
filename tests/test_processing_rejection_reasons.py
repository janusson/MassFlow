"""Config-driven drops must report their causes without altering library physics policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp

from MassFlow import processing
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.workflow import run_annotation_pipeline


def _spectrum(name: str, intensity: float) -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 150.0, 200.0], dtype=np.float64),
        intensities=np.full(3, intensity, dtype=np.float64),
        metadata={
            "id": name,
            "compound_name": name,
            "precursor_mz": 250.0,
            "charge": 1,
        },
    )


@pytest.mark.parametrize("stage", ["minimum_peak_count", "intensity_filter"])
def test_processing_drop_reports_stage(stage: str) -> None:
    config = ProcessingConfig(
        filter_min_peaks=True,
        min_peaks=4 if stage == "minimum_peak_count" else 3,
        filter_by_intensity=True,
        noise_threshold=10.0,
    )
    reasons: list[str] = []
    physics_reasons: list[str] = []
    result = processing.process_spectra_batch(
        [_spectrum("discard", 1.0)],
        config,
        rejection_reporter=physics_reasons.append,
        filter_rejection_reporter=reasons.append,
    )
    assert result == []
    assert len(reasons) == 1
    assert stage in reasons[0]
    assert physics_reasons == []


@pytest.mark.parametrize("all_filtered", [True, False])
def test_processing_reasons_reach_exported_reports(
    tmp_path: Path, all_filtered: bool
) -> None:
    query = tmp_path / "query.mgf"
    spectra = [_spectrum("low", 1.0)]
    if not all_filtered:
        spectra.append(_spectrum("high", 100.0))
    save_as_mgf(spectra, str(query))
    source = tmp_path / "library.msp"
    # The low reference is intentionally removed by configured filters. This
    # must not be treated as a physical-integrity failure of the raw library.
    save_as_msp([_spectrum("ref-low", 1.0), _spectrum("ref-high", 100.0)], str(source))
    output = tmp_path / "results"
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output),
        input=InputConfig(input_path=query, library_path=source, format="mgf"),
        processing=ProcessingConfig(
            filter_by_intensity=True,
            noise_threshold=10.0,
            filter_min_peaks=True,
            min_peaks=3,
        ),
        similarity=SimilarityConfig(
            algorithm="cosine", fdr_threshold=1.0, min_score=0.1
        ),
    )
    result = run_annotation_pipeline(config)[0]
    assert result.spectra_rejected == 1
    suffix = "failed" if all_filtered else "results"
    report_path = output / f"query_{suffix}.report.yaml"
    report = yaml.safe_load(report_path.read_text())
    assert "intensity_filter" in str(report)
    if all_filtered:
        assert result.status == "failed"
        assert "intensity_filter" in str(result.fatal_errors)
        assert not (output / "query_results.csv").exists()
    else:
        assert result.status != "failed"
        assert "intensity_filter" in str(result.warnings)
        assert (output / "query_results.csv").exists()
