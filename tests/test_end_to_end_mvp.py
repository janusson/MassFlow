
import os
import csv
import pytest
from pathlib import Path
from matchms import Spectrum
from matchms.exporting import save_as_msp, save_as_mgf
import numpy as np
from MassFlow import workflow
from MassFlow.config import MassFlowConfig, InputConfig, ProcessingConfig, SimilarityConfig

def test_mvp_workflow(tmp_path):
    # 1. Create Dummy Data
    
    # Query Spectrum (Target: Caffeine-like)
    query_spectrum = Spectrum(
        mz=np.array([100.0, 195.0], dtype="float"),
        intensities=np.array([0.1, 1.0], dtype="float"),
        metadata={"compound_name": "Query_Caffeine", "precursor_mz": 195.0}
    )
    
    # Reference Spectrum (Match: Caffeine)
    ref_spectrum = Spectrum(
        mz=np.array([100.0, 195.0], dtype="float"),
        intensities=np.array([0.1, 1.0], dtype="float"),
        metadata={"compound_name": "Ref_Caffeine", "precursor_mz": 195.0, "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N"}
    )
    
    # Noise Spectrum (No Match)
    noise_spectrum = Spectrum(
        mz=np.array([500.0, 600.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"compound_name": "Noise", "precursor_mz": 550.0}
    )
    
    # Save files
    query_path = tmp_path / "query.mgf"
    save_as_mgf([query_spectrum], str(query_path))
    
    ref_path = tmp_path / "reference.msp"
    save_as_msp([ref_spectrum, noise_spectrum], str(ref_path))
    
    output_dir = tmp_path / "results"
    
    # 2. Create Config
    config = MassFlowConfig(
        input=InputConfig(
            file_path=query_path,
            format="mgf",
            reference_library=ref_path
        ),
        processing=ProcessingConfig(
            min_peaks=1,
            min_intensity=0.0,
            normalize_intensity=True
        ),
        similarity=SimilarityConfig(
            algorithm="cosine",
            min_score=0.9
        ),
        output_directory=output_dir
    )
    
    # 3. Run Workflow
    workflow.run_workflow(config)
    
    # 4. Verify Results
    results_csv = output_dir / "results.csv"
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
