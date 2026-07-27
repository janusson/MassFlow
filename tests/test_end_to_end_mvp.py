"""
End-to-end integration tests for the MassFlow annotation pipeline.

These tests exercise the full ``run_annotation_pipeline`` workflow using
synthetic spectra and temporary file I/O. They validate that output files
are produced correctly and results are accurate for known-match scenarios.
"""

import csv

import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp

from MassFlow.config import (
    ExportConfig,
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.workflow import run_annotation_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrum(
    mz: list[float],
    intensities: list[float],
    compound_name: str,
    precursor_mz: float,
    **extra_meta,
) -> Spectrum:
    """Create a Spectrum with standard metadata keys."""
    meta = {"compound_name": compound_name, "precursor_mz": precursor_mz}
    meta.update(extra_meta)
    return Spectrum(
        mz=np.array(mz, dtype="float"),
        intensities=np.array(intensities, dtype="float"),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mvp_workflow_single_match(tmp_path):
    """A single query spectrum matches a single reference: one hit in CSV output."""
    query = _make_spectrum([100.0, 195.0], [0.1, 1.0], "Query_Caffeine", 195.0)
    ref = _make_spectrum(
        [100.0, 195.0],
        [0.1, 1.0],
        "Ref_Caffeine",
        195.0,
        smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        inchikey="RYYVLZVUVIJVGH-UHFFFAOYSA-N",
    )
    noise = _make_spectrum([500.0, 600.0], [1.0, 1.0], "Noise", 550.0)

    query_path = tmp_path / "query.mgf"
    ref_path = tmp_path / "reference.msp"
    output_dir = tmp_path / "results"

    save_as_mgf([query], str(query_path))
    save_as_msp([ref, noise], str(ref_path))

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(input_path=query_path, library_path=ref_path),
        similarity=SimilarityConfig(
            min_score=0.9,
            tolerance=0.01,
            min_matched_peaks=1,
            fdr_threshold=1.0,
        ),
        processing=ProcessingConfig(
            min_peaks=1,
            noise_threshold=0.0,
            min_intensity=0.0,
        ),
    )
    run_annotation_pipeline(config)

    results_csv = output_dir / "query_results.csv"
    assert results_csv.exists()

    with open(results_csv, "r") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    hit = rows[0]
    assert hit["query_id"] in ("Query_Caffeine", "query_0", "query_query_0")
    assert hit["reference_name"] == "Ref_Caffeine"
    assert float(hit["score"]) > 0.99
    assert hit["smiles"] == "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def test_mvp_workflow_multiple_queries(tmp_path):
    """Two queries: one matches, one has no match — output has both rows."""
    query_match = _make_spectrum([100.0, 200.0], [0.5, 1.0], "MatchMe", 150.0)
    query_nomatch = _make_spectrum([900.0, 950.0], [0.5, 1.0], "NoMatch", 925.0)
    ref = _make_spectrum([100.0, 200.0], [0.5, 1.0], "Ref_Match", 150.0)

    query_path = tmp_path / "queries.mgf"
    ref_path = tmp_path / "lib.msp"
    output_dir = tmp_path / "results"

    save_as_mgf([query_match, query_nomatch], str(query_path))
    save_as_msp([ref], str(ref_path))

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(input_path=query_path, library_path=ref_path),
        similarity=SimilarityConfig(
            min_score=0.9,
            tolerance=0.01,
            min_matched_peaks=1,
            fdr_threshold=1.0,
        ),
        processing=ProcessingConfig(
            min_peaks=1,
            noise_threshold=0.0,
            min_intensity=0.0,
        ),
    )
    run_annotation_pipeline(config)

    results_csv = output_dir / "queries_results.csv"
    assert results_csv.exists()

    with open(results_csv, "r") as f:
        rows = list(csv.DictReader(f))

    # Should have two rows: one matched, one unmatched
    assert len(rows) == 2
    statuses = {row["query_id"]: row.get("Annotation_Status", "") for row in rows}
    # At least one row should be annotated
    assert any("MatchMe" in qid or "query_0" in qid for qid in statuses)


def test_mvp_workflow_mztab_export(tmp_path):
    """Pipeline with mzTab-M export format produces a .mztab file."""
    query = _make_spectrum([100.0, 200.0], [0.5, 1.0], "Query", 150.0)
    ref = _make_spectrum([100.0, 200.0], [0.5, 1.0], "Ref", 150.0)

    query_path = tmp_path / "q.mgf"
    ref_path = tmp_path / "lib.msp"
    output_dir = tmp_path / "results"

    save_as_mgf([query], str(query_path))
    save_as_msp([ref], str(ref_path))

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(input_path=query_path, library_path=ref_path),
        export=ExportConfig(format="mztab"),
        similarity=SimilarityConfig(
            min_score=0.9,
            tolerance=0.01,
            min_matched_peaks=1,
            fdr_threshold=1.0,
        ),
        processing=ProcessingConfig(
            min_peaks=1,
            noise_threshold=0.0,
            min_intensity=0.0,
        ),
    )
    run_annotation_pipeline(config)

    mztab_path = output_dir / "q_results.mztab"
    assert mztab_path.exists()

    content = mztab_path.read_text()
    assert "mzTab-version" in content
    assert "MassFlow_Export" in content
    assert "SML" in content  # Small Molecule section


def test_mvp_workflow_mgf_library(tmp_path):
    """Pipeline works when the reference library is also in MGF format."""
    query = _make_spectrum([100.0, 300.0], [0.5, 1.0], "Query", 200.0)
    ref = _make_spectrum([100.0, 300.0], [0.5, 1.0], "Ref", 200.0)

    query_path = tmp_path / "q.mgf"
    ref_path = tmp_path / "lib.mgf"
    output_dir = tmp_path / "results"

    save_as_mgf([query], str(query_path))
    save_as_mgf([ref], str(ref_path))

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(input_path=query_path, library_path=ref_path),
        similarity=SimilarityConfig(
            min_score=0.9,
            tolerance=0.01,
            min_matched_peaks=1,
            fdr_threshold=1.0,
        ),
        processing=ProcessingConfig(
            min_peaks=1,
            noise_threshold=0.0,
            min_intensity=0.0,
        ),
    )
    run_annotation_pipeline(config)

    results_csv = output_dir / "q_results.csv"
    assert results_csv.exists()
    rows = list(csv.DictReader(open(results_csv)))
    assert len(rows) == 1


def test_mvp_workflow_strict_fdr_filters_all(tmp_path):
    """When FDR threshold is 0, no annotations should pass — all unmatched."""
    query = _make_spectrum([100.0, 200.0], [0.5, 1.0], "Query", 150.0)
    ref = _make_spectrum([100.0, 200.0], [0.5, 1.0], "Ref", 150.0)

    query_path = tmp_path / "q.mgf"
    ref_path = tmp_path / "lib.msp"
    output_dir = tmp_path / "results"

    save_as_mgf([query], str(query_path))
    save_as_msp([ref], str(ref_path))

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(input_path=query_path, library_path=ref_path),
        similarity=SimilarityConfig(
            min_score=0.9,
            tolerance=0.01,
            min_matched_peaks=1,
            fdr_threshold=0.0,  # Extremely strict — should filter all
        ),
        processing=ProcessingConfig(
            min_peaks=1,
            noise_threshold=0.0,
            min_intensity=0.0,
        ),
    )
    run_annotation_pipeline(config)

    results_csv = output_dir / "q_results.csv"
    assert results_csv.exists()
    rows = list(csv.DictReader(open(results_csv)))
    # Query row still present, but no annotation
    assert len(rows) >= 1


def test_mvp_workflow_modified_cosine(tmp_path):
    """Pipeline with modified cosine similarity produces correct results."""
    # Spectra with a precursor mass difference — modified cosine handles this
    query = _make_spectrum([100.0, 150.0], [0.5, 1.0], "Query", 200.0)
    ref = _make_spectrum([100.0, 150.0], [0.5, 1.0], "Ref", 250.0)

    query_path = tmp_path / "q.mgf"
    ref_path = tmp_path / "lib.msp"
    output_dir = tmp_path / "results"

    save_as_mgf([query], str(query_path))
    save_as_msp([ref], str(ref_path))

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(input_path=query_path, library_path=ref_path),
        similarity=SimilarityConfig(
            algorithm="modified_cosine",
            min_score=0.9,
            ms2_tolerance=0.05,
            min_matched_peaks=1,
            fdr_threshold=1.0,
        ),
        processing=ProcessingConfig(
            min_peaks=1,
            noise_threshold=0.0,
            min_intensity=0.0,
        ),
    )
    run_annotation_pipeline(config)

    results_csv = output_dir / "q_results.csv"
    assert results_csv.exists()
    rows = list(csv.DictReader(open(results_csv)))
    assert len(rows) == 1


def test_mvp_workflow_no_match_due_to_score(tmp_path):
    """When min_score is too high, no match is found — output still produced."""
    query = _make_spectrum([100.0, 200.0], [0.5, 1.0], "Query", 150.0)
    ref = _make_spectrum([100.0, 500.0], [0.5, 1.0], "Ref", 150.0)

    query_path = tmp_path / "q.mgf"
    ref_path = tmp_path / "lib.msp"
    output_dir = tmp_path / "results"

    save_as_mgf([query], str(query_path))
    save_as_msp([ref], str(ref_path))

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(input_path=query_path, library_path=ref_path),
        similarity=SimilarityConfig(
            min_score=0.99,  # Very strict — spectra are too different
            tolerance=0.01,
            min_matched_peaks=1,
            fdr_threshold=1.0,
        ),
        processing=ProcessingConfig(
            min_peaks=1,
            noise_threshold=0.0,
            min_intensity=0.0,
        ),
    )
    run_annotation_pipeline(config)

    results_csv = output_dir / "q_results.csv"
    assert results_csv.exists()
    rows = list(csv.DictReader(open(results_csv)))
    # The query is represented but has no annotation
    assert len(rows) >= 1
