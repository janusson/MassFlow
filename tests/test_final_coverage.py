"""
Final coverage tests for MassFlow core annotation logic.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import (
    MassFlowConfig,
    ProcessingConfig,
    SimilarityConfig,
    WorkflowConfig,
)
from MassFlow.similarity import CascadeEngine, SimilarityEngine


@pytest.fixture
def mock_config():
    return MassFlowConfig(
        project={"output_directory": "results"},
        input={"input_path": "data", "library_path": "library.msp"},
        similarity=SimilarityConfig(algorithm="cosine", min_score=0.7),
        processing=ProcessingConfig(),
        workflow=WorkflowConfig(),
        export={"format": "csv"},
    )


def test_cascade_engine_search(mock_config):
    # This test covers the gray-zone logic in the CascadeEngine
    config = mock_config.similarity
    config.algorithm = "cascade"
    config.cascade_tier1 = "cosine"
    config.cascade_tier2 = "modified_cosine"
    config.cascade_lower_bound = 0.5
    config.cascade_upper_bound = 0.9

    engine = CascadeEngine(config)

    # Mock tier1 and tier2 engines
    mock_tier1_config = MagicMock()
    mock_tier1_config.min_score = 0.0
    engine.tier1_engine = MagicMock(spec=SimilarityEngine)
    engine.tier1_engine.config = mock_tier1_config
    engine.tier2_engine = MagicMock(spec=SimilarityEngine)

    query_spec = Spectrum(
        mz=np.array([100.0]), intensities=np.array([1.0]), metadata={"id": "q1"}
    )
    ref_spec = Spectrum(
        mz=np.array([100.0]), intensities=np.array([1.0]), metadata={"id": "r1"}
    )

    # Case 1: Score is in gray zone (between lower and upper bounds) -> should trigger tier 2
    engine.tier1_engine.search.return_value = [
        {"query_id": "q1", "score": 0.7, "is_decoy": False}
    ]
    engine.tier2_engine.search.return_value = [
        {"query_id": "q1", "score": 0.95, "annotation_tier": "Tier 2"}
    ]

    results = engine.search([query_spec], [ref_spec], include_decoys=False)

    engine.tier1_engine.search.assert_called_once()
    engine.tier2_engine.search.assert_called_once()
    assert len(results) == 1
    assert results[0]["annotation_tier"].startswith("Tier 2")

    # Case 2: No results from tier 1 (noise) -> should not trigger tier 2
    engine.tier1_engine.reset_mock()
    engine.tier2_engine.reset_mock()
    engine.tier1_engine.search.return_value = []

    results = engine.search([query_spec], [ref_spec], include_decoys=False)
    engine.tier2_engine.search.assert_not_called()
    assert len(results) == 0


def test_workflow_networking_and_fbmn(mock_config):
    import tempfile

    from MassFlow.workflow import run_annotation_pipeline

    mock_config.workflow.perform_networking = True
    mock_config.export.format = "fbmn"

    # Create a more realistic mock spectrum to satisfy generate_decoys
    mock_spec_for_decoys = MagicMock(spec=Spectrum)
    mock_spec_for_decoys.metadata = {"id": "mock_spec", "precursor_mz": 200.0}
    mock_spec_for_decoys.peaks.mz = np.array([100.0, 200.0])
    mock_spec_for_decoys.peaks.intensities = np.array([0.5, 1.0])

    from pathlib import Path

    # We use a real temporary directory to avoid Path.mkdir() errors (FileExistsError/TypeError)
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_config.project.output_directory = Path(tmpdir)

        # Mock dependencies to prevent actual execution while allowing Path operations
        with patch("MassFlow.workflow.io.load_spectra"):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                return_value=[mock_spec_for_decoys],
            ):
                with patch("MassFlow.workflow.ProcessPoolExecutor"):
                    with patch("MassFlow.workflow.as_completed", return_value=[]):
                        with patch(
                            "MassFlow.networking.generate_molecular_network"
                        ) as mock_network:
                            with patch(
                                "MassFlow.workflow.io.save_spectra_to_mgf"
                            ) as mock_mgf:
                                # Only mock the input checks so we bypass the file existence check but allow the rest
                                with patch("pathlib.Path.exists", return_value=True):
                                    with patch(
                                        "pathlib.Path.is_file", return_value=True
                                    ):
                                        with patch(
                                            "pathlib.Path.stat",
                                            return_value=MagicMock(
                                                st_size=1024, st_mode=16877
                                            ),
                                        ):
                                            run_annotation_pipeline(mock_config)

                                # Verify that the correct export paths were triggered
                                mock_network.assert_called_once()
                                mock_mgf.assert_called_once()


def test_run_annotation_pipeline_no_library(mock_config):
    from MassFlow.workflow import run_annotation_pipeline

    mock_config.input.library_path = None  # Critical condition
    with pytest.raises(
        ValueError, match="Library path not specified in configuration."
    ):
        run_annotation_pipeline(mock_config)
