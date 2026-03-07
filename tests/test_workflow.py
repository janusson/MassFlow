"""
Tests for MassFlow annotation workflow.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from MassFlow.config import InputConfig, MassFlowConfig, ProjectConfig
from MassFlow.workflow import run_annotation_pipeline


@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.save_match_results")
@patch("MassFlow.workflow.SimilarityEngine")
@patch("MassFlow.workflow.io.load_spectra")
def test_run_annotation_pipeline_success(
    mock_load, mock_engine_cls, mock_save, mock_process, tmp_path
):
    """Test successful execution of the annotation pipeline."""

    # Setup Config
    exp_path = Path("experimental.mgf")
    ref_path = Path("reference.msp")
    out_dir = tmp_path / "results"

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=out_dir),
        input=InputConfig(file_path=exp_path, reference_library=ref_path),
    )

    # Mock Data
    mock_query = MagicMock()
    mock_query.get.return_value = "Query1"

    mock_ref = MagicMock()
    mock_ref.get.return_value = "Ref1"

    # Mock load_spectra to return iterables
    # First call for reference (workflow logic change), second for query
    mock_load.side_effect = [[mock_ref], [mock_query]]

    # Mock process_spectra to pass through
    mock_process.side_effect = lambda s, c: s

    # Mock Engine
    mock_engine_instance = mock_engine_cls.return_value
    mock_results = [{"query_id": "Query1", "score": 0.9}]
    mock_engine_instance.search.return_value = mock_results

    # Run
    run_annotation_pipeline(config)

    # Verify
    assert mock_load.call_count == 2
    mock_engine_cls.assert_called_with(config.similarity)
    mock_engine_instance.search.assert_called_with([mock_query], [mock_ref])

    expected_out_file = out_dir / "experimental_results.csv"
    mock_save.assert_called_with(mock_results, expected_out_file)


@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
def test_run_annotation_pipeline_no_query_spectra(mock_load, mock_process, tmp_path):
    """Test warning when no query spectra are found (should not raise)."""
    # Ref loaded first (valid), Query loaded second (empty)
    mock_ref = MagicMock()
    mock_load.side_effect = [[mock_ref], []]
    mock_process.side_effect = lambda s, c: s

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path),
        input=InputConfig(file_path=Path("exp.mgf"), reference_library=Path("ref.msp")),
    )

    # Should not raise exception, just log warning
    run_annotation_pipeline(config)


@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
def test_run_annotation_pipeline_no_reference_spectra(
    mock_load, mock_process, tmp_path
):
    """Test error when no reference spectra are found."""
    # First call returns empty (ref)
    mock_load.side_effect = [[]]
    mock_process.side_effect = lambda s, c: s

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path),
        input=InputConfig(file_path=Path("exp.mgf"), reference_library=Path("ref.msp")),
    )

    with pytest.raises(ValueError, match="No valid spectra found in reference library"):
        run_annotation_pipeline(config)
