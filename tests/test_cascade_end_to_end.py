import csv
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

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

pytestmark = pytest.mark.experimental


@pytest.fixture(autouse=True)
def reset_worker_engine(monkeypatch):
    monkeypatch.setattr("MassFlow.workflow._worker_engine", None)


def make_result(
    query: Spectrum, reference: Spectrum, score: float
) -> dict[str, object]:
    return {
        "query_id": str(query.get("id")),
        "query_precursor_mz": float(query.get("precursor_mz")),
        "reference_id": str(reference.get("id")),
        "reference_name": str(
            reference.get("compound_name")
            or reference.get("name")
            or reference.get("id")
        ),
        "reference_precursor_mz": float(reference.get("precursor_mz")),
        "score": score,
        "matched_peaks": 3,
        "smiles": reference.get("smiles"),
        "inchikey": reference.get("inchikey"),
        "is_decoy": False,
        "q_value": 1.0,
        "annotation_tier": None,
    }


@pytest.fixture(autouse=True)
def mock_similarity_engines():
    class FakeSimilarityEngine:
        def __init__(self, config):
            self.config = config

        def search(self, query_spectra, reference_spectra, min_score=None, top_n=None):
            def find_reference(label: str) -> Spectrum:
                for ref in reference_spectra:
                    candidates = {
                        str(ref.get("id")),
                        str(ref.get("compound_name")),
                        str(ref.get("name")),
                    }
                    if label in candidates:
                        return ref
                raise KeyError(f"Reference {label} not found in test fixture")

            results = []

            for query in query_spectra:
                query_id = str(query.get("id"))

                if self.config.algorithm == "cosine":
                    if query_id == "High_Conf_Query":
                        results.append(
                            make_result(query, find_reference("Ref_HighConf"), 0.96)
                        )
                    elif query_id == "Gray_Zone_Query":
                        results.append(
                            make_result(query, find_reference("Ref_GrayZone"), 0.6)
                        )
                    elif query_id == "Noise_Query":
                        results.append(
                            make_result(query, find_reference("Ref_Noise"), 0.2)
                        )

                elif self.config.algorithm == "ms2deepscore":
                    if query_id == "Gray_Zone_Query":
                        results.append(
                            make_result(query, find_reference("Ref_GrayZone"), 0.95)
                        )
                    elif query_id == "High_Conf_Query":
                        results.append(
                            make_result(query, find_reference("Ref_HighConf"), 0.6)
                        )

            return results

    with patch("MassFlow.similarity.SimilarityEngine", FakeSimilarityEngine):
        yield


@patch("MassFlow.workflow.ProcessPoolExecutor")
def test_cascade_e2e_workflow(mock_executor, tmp_path):
    mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
        max_workers=1
    )

    # 1. Create Synthetic Data for Cascade Routing
    # Reference spectra
    ref_spec_high_conf = Spectrum(
        mz=np.array([100.0, 195.0, 290.0], dtype="float"),
        intensities=np.array([0.1, 1.0, 0.5], dtype="float"),
        metadata={
            "id": "Ref_HighConf",
            "compound_name": "Ref_HighConf",
            "precursor_mz": 195.0,
            "smiles": "SMILES1",
        },
    )
    ref_spec_gray_zone = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([0.1, 1.0, 0.5], dtype="float"),
        metadata={
            "id": "Ref_GrayZone",
            "compound_name": "Ref_GrayZone",
            "precursor_mz": 200.0,
            "smiles": "SMILES2",
        },
    )
    ref_spec_noise = Spectrum(
        mz=np.array([50.0, 150.0], dtype="float"),
        intensities=np.array([0.2, 0.8], dtype="float"),
        metadata={
            "id": "Ref_Noise",
            "compound_name": "Ref_Noise",
            "precursor_mz": 150.0,
        },
    )
    ref_spec_decoy = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([0.1, 1.0, 0.5], dtype="float"),
        metadata={
            "id": "Ref_Decoy",
            "compound_name": "Ref_Decoy",
            "is_decoy": True,
            "precursor_mz": 200.0,
        },
    )

    # Query spectra
    high_conf_query = Spectrum(
        mz=np.array([100.0, 195.0, 290.0], dtype="float"),
        intensities=np.array([0.1, 1.0, 0.5], dtype="float"),
        metadata={
            "id": "High_Conf_Query",
            "compound_name": "High_Conf_Query",
            "precursor_mz": 195.0,
        },
    )  # High Cosine score with Ref_HighConf

    gray_zone_query = Spectrum(
        mz=np.array([100.1, 200.1, 300.1], dtype="float"),
        intensities=np.array([0.05, 0.9, 0.45], dtype="float"),
        metadata={
            "id": "Gray_Zone_Query",
            "compound_name": "Gray_Zone_Query",
            "precursor_mz": 200.0,
        },
    )  # Moderate Cosine with Ref_GrayZone (will be in gray zone), but high MS2DeepScore

    noise_query = Spectrum(
        mz=np.array([900.0, 950.0], dtype="float"),
        intensities=np.array([0.3, 0.7], dtype="float"),
        metadata={
            "id": "Noise_Query",
            "compound_name": "Noise_Query",
            "precursor_mz": 925.0,
        },
    )  # Low Cosine score with all references

    query_path = tmp_path / "cascade_queries.mgf"
    save_as_mgf([high_conf_query, gray_zone_query, noise_query], str(query_path))

    ref_path = tmp_path / "cascade_references.msp"
    save_as_msp(
        [ref_spec_high_conf, ref_spec_gray_zone, ref_spec_noise, ref_spec_decoy],
        str(ref_path),
    )

    output_dir = tmp_path / "cascade_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Configure MassFlow for CascadeEngine
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(
            file_path=query_path,
            reference_library=ref_path,
            format="mgf",  # Assuming MGF for queries
        ),
        processing=ProcessingConfig(
            min_peaks=1, noise_threshold=0.0, min_intensity=0.0
        ),
        similarity=SimilarityConfig(
            algorithm="cascade",
            cascade_tier1="cosine",
            cascade_tier2="ms2deepscore",
            cascade_lower_bound=0.4,  # Queries below this are discarded by Tier 1
            cascade_upper_bound=0.85,  # Queries above this are annotated by Tier 1
            min_score=0.1,  # Overall min_score
            fdr_threshold=1.0,  # Disable FDR for easier testing of routing
            tolerance=0.02,  # For cosine
            ms2_tolerance=0.02,  # For cosine
            model_path=tmp_path / "dummy_ms2deepscore_model.hdf5",  # Mocked path
        ),
    )

    # Create a dummy model file to satisfy Path.exists() check
    config.similarity.model_path.touch()

    # 3. Run the workflow
    run_annotation_pipeline(config)

    # 4. Verify Results
    results_csv = output_dir / "cascade_queries_results.csv"
    assert results_csv.exists()

    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Reports now include unmatched queries, so the noise spectrum remains with an Unknown status.
    assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}: {rows}"

    high_conf_result = next(
        (r for r in rows if r["query_id"] == "High_Conf_Query"), None
    )
    gray_zone_result = next(
        (r for r in rows if r["query_id"] == "Gray_Zone_Query"), None
    )
    noise_result = next((r for r in rows if r["query_id"] == "Noise_Query"), None)

    assert high_conf_result is not None
    assert gray_zone_result is not None
    assert noise_result is not None
    assert noise_result["Annotation_Status"] == "Unknown"
    assert noise_result["score"] == ""

    # Verify High-Confidence Query (Tier 1)
    assert high_conf_result["reference_name"] == "Ref_HighConf"
    assert float(high_conf_result["score"]) > config.similarity.cascade_upper_bound
    assert high_conf_result["annotation_tier"] == "Tier 1 (cosine)"
    assert high_conf_result["Annotation_Status"] == "Matched"

    # Verify Gray-Zone Query (Tier 2)
    assert gray_zone_result["reference_name"] == "Ref_GrayZone"
    # MS2DeepScore score should be high, while cosine was in gray zone
    assert (
        float(gray_zone_result["score"]) > config.similarity.cascade_upper_bound
    )  # Mocked score for MS2DeepScore is 0.95
    assert gray_zone_result["annotation_tier"] == "Tier 2 (ms2deepscore)"
    assert gray_zone_result["Annotation_Status"] == "Matched"
