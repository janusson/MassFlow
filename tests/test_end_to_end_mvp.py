import csv
import os
from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp

from MassFlow import workflow
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)


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

    # 2. Create Config
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir, name="MVP_Test"),
        input=InputConfig(
            file_path=query_path, format="mgf", reference_library=ref_path
        ),
        processing=ProcessingConfig(
            min_peaks=1,
            min_intensity=0.0,
            normalize_intensity=True,
            noise_threshold=0.0,
        ),
        similarity=SimilarityConfig(
            algorithm="cosine", min_score=0.9, min_matched_peaks=1
        ),
    )

    # 3. Write config to file (to test loading logic implicitly via run_workflow if it took a path,
    # but run_workflow currently takes a path string/object to load.
    # However, workflow.run_workflow takes a path.
    # We should save this config to yaml and pass the path.

    config_path = tmp_path / "config.yaml"
    # Pydantic v2 has model_dump, v1 has dict(). Assuming v2 or compat.
    # If using PyYAML dump, we need dict.
    import yaml

    # Convert pydantic model to dict (using .dict() for compatibility or model_dump() for v2)
    # Since we are inside the test, let's just dump the dict representation.
    # Note: Pydantic models require correct serialization of Path objects.
    # We can rely on MassFlowConfig.from_yaml inside workflow to parse it back.

    # We need to manually construct the dict because json/yaml dumpers might struggle with Path objects directly without custom encoders
    # unless we use pydantic's json/dump methods.
    # Let's try to pass the config object logic if workflow supported it, but workflow.run_workflow takes a path.
    # So we must write the yaml.

    config_dict = config.dict()

    # Recursively convert Path to str for yaml dump
    def _convert_paths(d):
        if isinstance(d, dict):
            return {k: _convert_paths(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [_convert_paths(v) for v in d]
        elif isinstance(d, Path):
            return str(d)
        else:
            return d

    clean_config_dict = _convert_paths(config_dict)

    with open(config_path, "w") as f:
        yaml.dump(clean_config_dict, f)

    # 3. Run Workflow
    workflow.run_workflow(config_path)

    # 4. Verify Results
    # Based on ProjectConfig name="MVP_Test" and default export format "csv"
    results_csv = output_dir / "MVP_Test_results.csv"
    assert results_csv.exists()

    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    hit = rows[0]
    assert hit["Query_Name"] == "Query_Caffeine"
    assert hit["Match_Name"] == "Ref_Caffeine"
    assert float(hit["Score"]) > 0.99
    assert hit["Smiles"] == "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
