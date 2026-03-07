import csv

import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.workflow import run_annotation_pipeline


def test_mvp_workflow(tmp_path):
    # 1. Create Dummy Data

    # Query Spectrum (Target: Caffeine-like)
    query_spectrum = Spectrum(
        mz=np.array([100.0, 195.0], dtype="float"),
        intensities=np.array([0.1, 1.0], dtype="float"),
        metadata={"compound_name": "Query_Caffeine", "precursor_mz": 195.0},
    )

    # Reference Spectrum (Match: Caffeine)
    ref_spectrum = Spectrum(
        mz=np.array([100.0, 195.0], dtype="float"),
        intensities=np.array([0.1, 1.0], dtype="float"),
        metadata={
            "compound_name": "Ref_Caffeine",
            "precursor_mz": 195.0,
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
        },
    )

    # Noise Spectrum (No Match)
    noise_spectrum = Spectrum(
        mz=np.array([500.0, 600.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"compound_name": "Noise", "precursor_mz": 550.0},
    )

    # Save files
    query_path = tmp_path / "query.mgf"
    save_as_mgf([query_spectrum], str(query_path))

    ref_path = tmp_path / "reference.msp"
    save_as_msp([ref_spectrum, noise_spectrum], str(ref_path))

    output_dir = tmp_path / "results"

    # 2. Run Workflow
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(file_path=query_path, reference_library=ref_path),
        similarity=SimilarityConfig(min_score=0.9, tolerance=0.01, min_matched_peaks=1),
        processing=ProcessingConfig(
            min_peaks=1, noise_threshold=0.0, min_intensity=0.0
        ),
    )
    run_annotation_pipeline(config)

    # 3. Verify Results
    # Output file is named after experimental file stem + "_results.csv"
    results_csv = output_dir / "query_results.csv"
    assert results_csv.exists()

    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    hit = rows[0]

    # Check key fields from CSV export
    # Note: Column names depend on io.save_match_results implementation
    # Assuming it writes the keys from the result dict in similarity.py
    assert hit["query_id"] == "Query_Caffeine" or hit.get("query_id") == "query_0"
    assert hit["reference_name"] == "Ref_Caffeine"
    assert float(hit["score"]) > 0.99
    assert hit["smiles"] == "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
