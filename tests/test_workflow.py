"""
Tests for MassFlow workflow module.
"""

from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from MassFlow import workflow
from MassFlow.config import (
    ExportConfig,
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
)


@pytest.fixture
def mock_config():
    return MassFlowConfig(
        project=ProjectConfig(output_directory=Path("out"), name="TestProject"),
        input=InputConfig(
            file_path=Path("test.mgf"), reference_library=Path("ref.msp")
        ),
        processing=ProcessingConfig(),
        export=ExportConfig(format="csv"),
    )


@patch("MassFlow.workflow.io.load_spectra")
@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.get_similarity_calculator")
@patch("MassFlow.workflow.io.save_match_results")
def test_run_workflow_file_path(
    mock_save, mock_calc, mock_proc, mock_load, mock_config
):
    """Test run_workflow with file_path input."""
    # Setup mocks
    mock_spectrum = MagicMock()
    mock_spectrum.get.return_value = "TestSpec"
    mock_load.return_value = [mock_spectrum]
    mock_proc.return_value = [mock_spectrum]

    # Mock calculator
    mock_scores = MagicMock()
    mock_scores.scores_by_query.return_value = [
        (mock_spectrum, {"CosineGreedy_score": 0.9, "CosineGreedy_matches": 5})
    ]
    mock_calc_instance = MagicMock()
    mock_calc_instance.calculate.return_value = mock_scores
    mock_calc.return_value = mock_calc_instance

    # Mock Config loading
    with patch("MassFlow.config.MassFlowConfig.from_yaml", return_value=mock_config):
        workflow.run_workflow("dummy_config.yaml")

    mock_load.assert_called_with(Path("test.mgf"), "mgf")
    mock_proc.assert_called()
    mock_save.assert_called_once()

    # Verify save path
    args, _ = mock_save.call_args
    # output directory is "out", project name is "TestProject", format is "csv"
    # Expected: out/TestProject_results.csv
    assert args[1] == Path("out/TestProject_results.csv")


@patch("MassFlow.workflow.io.load_spectra")
@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.get_similarity_calculator")
def test_run_workflow_directory_input(mock_calc, mock_proc, mock_load):
    """Test run_workflow with data_directory input."""
    config = MassFlowConfig(
        input=InputConfig(data_directory=Path("data_dir"), format="mgf")
    )

    with (
        patch("MassFlow.config.MassFlowConfig.from_yaml", return_value=config),
        patch("pathlib.Path.glob") as mock_glob,
    ):
        # Mock glob to return a file
        mock_glob.return_value = [Path("data_dir/file1.mgf")]

        # We need mock_load to return something to avoid iteration error in process_spectra
        mock_load.return_value = []

        workflow.run_workflow("dummy.yaml")

        mock_load.assert_called_with(Path("data_dir/file1.mgf"), "mgf")


def test_run_workflow_no_input():
    """Test run_workflow raises error when no input specified."""
    config = MassFlowConfig(
        input=InputConfig()  # No file_path or data_directory
    )

    with patch("MassFlow.config.MassFlowConfig.from_yaml", return_value=config):
        with pytest.raises(ValueError, match="No input file or directory"):
            workflow.run_workflow("dummy.yaml")
