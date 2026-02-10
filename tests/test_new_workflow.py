import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matchms import Scores, Spectrum
from pydantic import ValidationError

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    OutputConfig,
    ProcessingConfig,
    SimilarityConfig,
)
from MassFlow.workflow import load_spectra, process_spectrum, run_workflow, save_results

# Strict tolerance for floating-point comparisons
ABS_TOL = 1e-6


@pytest.fixture
def mock_config_path(tmp_path: Path) -> Path:
    """Fixture for a mock configuration file path."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("dummy_content: true")  # Placeholder content
    return config_file


@pytest.fixture
def mock_output_dir(tmp_path: Path) -> Path:
    """Fixture for a mock output directory."""
    return tmp_path / "output"


@pytest.fixture
def basic_spectrum() -> Spectrum:
    """Fixture for a basic spectrum."""
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float32"),
        intensities=np.array([100.0, 500.0, 999.0], dtype="float32"),
        metadata={"precursor_mz": 250.0, "id": "spec1", "compound_name": "Compound A"},
    )


@pytest.fixture
def empty_spectrum() -> Spectrum:
    """Fixture for an empty spectrum."""
    return Spectrum(
        mz=np.array([], dtype="float32"),
        intensities=np.array([], dtype="float32"),
        metadata={"precursor_mz": 0.0, "id": "empty_spec", "compound_name": "Empty"},
    )


@pytest.fixture
def noise_spectrum() -> Spectrum:
    """Fixture for a noise-only spectrum (low intensities)."""
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float32"),
        intensities=np.array([0.01, 0.02, 0.03], dtype="float32"),
        metadata={"precursor_mz": 200.0, "id": "noise_spec", "compound_name": "Noise"},
    )


@pytest.fixture
def mock_massflow_config(mock_output_dir: Path) -> MassFlowConfig:
    """Fixture for a mock MassFlowConfig instance."""
    return MassFlowConfig(
        input=InputConfig(file_path=Path("query.mgf"), format="mgf"),
        processing=ProcessingConfig(
            clean_metadata=True, min_intensity=0.01, normalize_intensity=True
        ),
        similarity=SimilarityConfig(
            algorithm="cosine", tolerance=0.1, min_matched_peaks=3
        ),
        output_directory=mock_output_dir,
    )


@pytest.fixture
def mock_massflow_config_modified_cosine(mock_output_dir: Path) -> MassFlowConfig:
    """Fixture for a mock MassFlowConfig instance using Modified Cosine."""
    return MassFlowConfig(
        input=InputConfig(file_path=Path("query.mgf"), format="mgf"),
        processing=ProcessingConfig(
            clean_metadata=True, min_intensity=0.01, normalize_intensity=True
        ),
        similarity=SimilarityConfig(
            algorithm="modified_cosine", tolerance=0.1, min_matched_peaks=3
        ),
        output_directory=mock_output_dir,
    )


@pytest.fixture
def mock_massflow_config_with_reference(mock_output_dir: Path) -> MassFlowConfig:
    """Fixture for a mock MassFlowConfig with a reference library."""
    return MassFlowConfig(
        input=InputConfig(
            file_path=Path("query.mgf"),
            format="mgf",
            reference_library=Path("reference.msp"),
        ),
        processing=ProcessingConfig(
            clean_metadata=True, min_intensity=0.01, normalize_intensity=True
        ),
        similarity=SimilarityConfig(
            algorithm="cosine", tolerance=0.1, min_matched_peaks=3
        ),
        output_directory=mock_output_dir,
    )


# --- Test load_spectra ---
@patch("MassFlow.workflow.load_from_mgf")
@patch("MassFlow.workflow.load_from_msp")
def test_load_spectra_mgf(mock_load_msp: MagicMock, mock_load_mgf: MagicMock):
    mock_load_mgf.return_value = [MagicMock(spec=Spectrum), MagicMock(spec=Spectrum)]
    file_path = Path("test.mgf")
    spectra = load_spectra(file_path, "mgf")
    mock_load_mgf.assert_called_once_with(str(file_path))
    assert len(spectra) == 2


@patch("MassFlow.workflow.load_from_mgf")
@patch("MassFlow.workflow.load_from_msp")
def test_load_spectra_msp(mock_load_msp: MagicMock, mock_load_mgf: MagicMock):
    mock_load_msp.return_value = [MagicMock(spec=Spectrum)]
    file_path = Path("test.msp")
    spectra = load_spectra(file_path, "msp")
    mock_load_msp.assert_called_once_with(str(file_path))
    assert len(spectra) == 1


def test_load_spectra_unsupported_format():
    file_path = Path("test.mzml")
    with pytest.raises(ValueError, match="Unsupported format: mzml"):
        load_spectra(file_path, "mzml")


@patch("MassFlow.workflow.load_from_mgf")
def test_load_spectra_filters_none(mock_load_mgf: MagicMock):
    mock_load_mgf.return_value = [
        MagicMock(spec=Spectrum),
        None,
        MagicMock(spec=Spectrum),
    ]
    file_path = Path("test.mgf")
    spectra = load_spectra(file_path, "mgf")
    assert len(spectra) == 2
    assert all(s is not None for s in spectra)


# --- Test process_spectrum ---
@patch("MassFlow.workflow.metadata_processing")
@patch("MassFlow.workflow.peak_processing")
def test_process_spectrum_full_processing(
    mock_peak_processing: MagicMock,
    mock_metadata_processing: MagicMock,
    basic_spectrum: Spectrum,
    mock_massflow_config: MassFlowConfig,
):
    mock_metadata_processing.return_value = (
        basic_spectrum.clone()
    )  # Simulate metadata_processing returning a spectrum
    mock_peak_processing.return_value = (
        basic_spectrum.clone()
    )  # Simulate peak_processing returning a spectrum

    processed_spectrum = process_spectrum(basic_spectrum, mock_massflow_config)

    mock_metadata_processing.assert_called_once_with(basic_spectrum)
    mock_peak_processing.assert_called_once_with(
        mock_metadata_processing.return_value,
        min_intensity=mock_massflow_config.processing.min_intensity,
        normalize=mock_massflow_config.processing.normalize_intensity,
    )
    assert processed_spectrum is not None
    assert isinstance(processed_spectrum, Spectrum)


@patch("MassFlow.workflow.metadata_processing")
@patch("MassFlow.workflow.peak_processing")
def test_process_spectrum_no_metadata_cleaning(
    mock_peak_processing: MagicMock,
    mock_metadata_processing: MagicMock,
    basic_spectrum: Spectrum,
    mock_massflow_config: MassFlowConfig,
):
    mock_massflow_config.processing.clean_metadata = False
    mock_peak_processing.return_value = basic_spectrum.clone()

    processed_spectrum = process_spectrum(basic_spectrum, mock_massflow_config)

    mock_metadata_processing.assert_not_called()
    mock_peak_processing.assert_called_once_with(
        basic_spectrum,
        min_intensity=mock_massflow_config.processing.min_intensity,
        normalize=mock_massflow_config.processing.normalize_intensity,
    )
    assert processed_spectrum is not None


@patch("MassFlow.workflow.metadata_processing")
@patch("MassFlow.workflow.peak_processing")
def test_process_spectrum_metadata_returns_none(
    mock_peak_processing: MagicMock,
    mock_metadata_processing: MagicMock,
    basic_spectrum: Spectrum,
    mock_massflow_config: MassFlowConfig,
):
    mock_metadata_processing.return_value = None

    processed_spectrum = process_spectrum(basic_spectrum, mock_massflow_config)

    mock_metadata_processing.assert_called_once_with(basic_spectrum)
    mock_peak_processing.assert_not_called()
    assert processed_spectrum is None


def test_process_spectrum_empty_input_spectrum(
    empty_spectrum: Spectrum, mock_massflow_config: MassFlowConfig
):
    # If the input spectrum is already empty, peak_processing should handle it
    # We expect it to return the same empty spectrum or a new empty one
    processed_spectrum = process_spectrum(empty_spectrum, mock_massflow_config)
    assert processed_spectrum is not None
    assert isinstance(processed_spectrum, Spectrum)
    assert len(processed_spectrum.mz) == 0
    assert len(processed_spectrum.intensities) == 0


# --- Test save_results ---
def test_save_results_creates_directory_and_file(mock_output_dir: Path):
    results = [
        {
            "Query_ID": "Q1",
            "Query_Name": "CompQ1",
            "Match_Name": "CompR1",
            "Score": "0.9",
            "Matches": 5,
            "Smiles": "CCO",
            "InChIKey": "AAA",
        },
    ]

    assert not mock_output_dir.exists()

    saved_file = save_results(results, mock_output_dir)

    assert mock_output_dir.exists()
    assert saved_file.exists()
    assert saved_file == mock_output_dir / "results.csv"

    with open(saved_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = list(reader)

        assert headers == [
            "Query_ID",
            "Query_Name",
            "Match_Name",
            "Score",
            "Matches",
            "Smiles",
            "InChIKey",
        ]
        assert len(data) == 1
        assert data[0] == {
            "Query_ID": "Q1",
            "Query_Name": "CompQ1",
            "Match_Name": "CompR1",
            "Score": "0.9",
            "Matches": "5",
            "Smiles": "CCO",
            "InChIKey": "AAA",
        }


def test_save_results_empty_list(mock_output_dir: Path):
    results: List[dict[str, Any]] = []

    saved_file = save_results(results, mock_output_dir)

    assert mock_output_dir.exists()
    assert saved_file.exists()

    with open(saved_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = list(reader)

        assert headers == [
            "Query_ID",
            "Query_Name",
            "Match_Name",
            "Score",
            "Matches",
            "Smiles",
            "InChIKey",
        ]
        assert len(data) == 0


# --- Test run_workflow ---
@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
@patch("MassFlow.workflow.get_similarity_calculator")
@patch("MassFlow.workflow.load_spectra")
@patch("MassFlow.workflow.process_spectrum")
@patch("MassFlow.workflow.save_results")
def test_run_workflow_successful_cosine_search(
    mock_save_results: MagicMock,
    mock_process_spectrum: MagicMock,
    mock_load_spectra: MagicMock,
    mock_get_similarity_calculator: MagicMock,
    mock_from_yaml: MagicMock,
    mock_config_path: Path,
    mock_massflow_config: MassFlowConfig,
    basic_spectrum: Spectrum,
):
    # Mock config loading
    mock_from_yaml.return_value = mock_massflow_config_with_reference = (
        mock_massflow_config
    )
    mock_massflow_config_with_reference.input.reference_library = Path("reference.msp")

    # Mock spectra loading
    mock_load_spectra.side_effect = [
        [basic_spectrum.clone()],  # Reference spectra
        [basic_spectrum.clone()],  # Query spectra
    ]

    # Mock spectrum processing
    mock_processed_ref_spectrum = basic_spectrum.clone()
    mock_processed_query_spectrum = basic_spectrum.clone()
    mock_process_spectrum.side_effect = [
        mock_processed_ref_spectrum,  # For reference
        mock_processed_query_spectrum,  # For query
    ]

    # Mock similarity calculation
    mock_calculator = MagicMock()
    mock_get_similarity_calculator.return_value = mock_calculator

    mock_scores = MagicMock(spec=Scores)
    mock_calculator.calculate.return_value = mock_scores

    # Simulate scores_by_query returning a match
    mock_scores.scores_by_query.return_value = [
        (
            basic_spectrum.clone(),
            {"CosineGreedy_score": 0.95, "CosineGreedy_matches": 3},
        )
    ]

    run_workflow(mock_config_path)

    mock_from_yaml.assert_called_once_with(mock_config_path)
    mock_load_spectra.assert_any_call(
        mock_massflow_config_with_reference.input.reference_library, "msp"
    )
    mock_load_spectra.assert_any_call(
        mock_massflow_config_with_reference.input.file_path, "mgf"
    )
    assert mock_process_spectrum.call_count == 2
    mock_get_similarity_calculator.assert_called_once_with(
        mock_massflow_config_with_reference.similarity
    )
    mock_calculator.calculate.assert_called_once_with(
        [mock_processed_ref_spectrum], [mock_processed_query_spectrum]
    )
    mock_scores.scores_by_query.assert_called_once()
    mock_save_results.assert_called_once()

    # Verify the content passed to save_results
    args, kwargs = mock_save_results.call_args
    results = args[0]
    assert len(results) == 1
    assert results[0]["Query_ID"] == "spec1"
    assert results[0]["Score"] == "0.9500"


@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
@patch("MassFlow.workflow.get_similarity_calculator")
@patch("MassFlow.workflow.load_spectra")
@patch("MassFlow.workflow.process_spectrum")
@patch("MassFlow.workflow.save_results")
def test_run_workflow_successful_modified_cosine_search(
    mock_save_results: MagicMock,
    mock_process_spectrum: MagicMock,
    mock_load_spectra: MagicMock,
    mock_get_similarity_calculator: MagicMock,
    mock_from_yaml: MagicMock,
    mock_config_path: Path,
    mock_massflow_config_modified_cosine: MassFlowConfig,
    basic_spectrum: Spectrum,
):
    # Mock config loading
    mock_from_yaml.return_value = mock_massflow_config_modified_cosine
    mock_massflow_config_modified_cosine.input.reference_library = Path("reference.msp")

    # Mock spectra loading
    mock_load_spectra.side_effect = [
        [basic_spectrum.clone()],  # Reference spectra
        [basic_spectrum.clone()],  # Query spectra
    ]

    # Mock spectrum processing
    mock_processed_ref_spectrum = basic_spectrum.clone()
    mock_processed_query_spectrum = basic_spectrum.clone()
    mock_process_spectrum.side_effect = [
        mock_processed_ref_spectrum,  # For reference
        mock_processed_query_spectrum,  # For query
    ]

    # Mock similarity calculation
    mock_calculator = MagicMock()
    mock_get_similarity_calculator.return_value = mock_calculator

    mock_scores = MagicMock(spec=Scores)
    mock_calculator.calculate.return_value = mock_scores

    # Simulate scores_by_query returning a match with ModifiedCosine_score
    mock_scores.scores_by_query.return_value = [
        (
            basic_spectrum.clone(),
            {"ModifiedCosine_score": 0.88, "ModifiedCosine_matches": 2},
        )
    ]

    run_workflow(mock_config_path)

    mock_from_yaml.assert_called_once_with(mock_config_path)
    mock_get_similarity_calculator.assert_called_once_with(
        mock_massflow_config_modified_cosine.similarity
    )

    args, kwargs = mock_save_results.call_args
    results = args[0]
    assert len(results) == 1
    assert results[0]["Query_ID"] == "spec1"
    assert results[0]["Score"] == "0.8800"
    assert results[0]["Matches"] == 2


@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
@patch("MassFlow.workflow.get_similarity_calculator")
@patch("MassFlow.workflow.load_spectra")
@patch("MassFlow.workflow.process_spectrum")
@patch("MassFlow.workflow.save_results")
def test_run_workflow_no_reference_library(
    mock_save_results: MagicMock,
    mock_process_spectrum: MagicMock,
    mock_load_spectra: MagicMock,
    mock_get_similarity_calculator: MagicMock,
    mock_from_yaml: MagicMock,
    mock_config_path: Path,
    mock_massflow_config: MassFlowConfig,
    basic_spectrum: Spectrum,
):
    # Mock config loading, ensure no reference library is set
    mock_from_yaml.return_value = mock_massflow_config
    mock_massflow_config.input.reference_library = None

    # Mock spectra loading for query only
    mock_load_spectra.return_value = [basic_spectrum]
    mock_process_spectrum.return_value = basic_spectrum.clone()

    run_workflow(mock_config_path)

    # load_spectra should only be called once for the query file
    mock_load_spectra.assert_called_once_with(
        mock_massflow_config.input.file_path, "mgf"
    )
    mock_process_spectrum.assert_called_once()
    mock_get_similarity_calculator.assert_not_called()
    mock_save_results.assert_not_called()  # No reference, no results to save


@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
@patch("MassFlow.workflow.get_similarity_calculator")
@patch("MassFlow.workflow.load_spectra")
@patch("MassFlow.workflow.process_spectrum")
@patch("MassFlow.workflow.save_results")
def test_run_workflow_no_query_matches_above_threshold(
    mock_save_results: MagicMock,
    mock_process_spectrum: MagicMock,
    mock_load_spectra: MagicMock,
    mock_get_similarity_calculator: MagicMock,
    mock_from_yaml: MagicMock,
    mock_config_path: Path,
    mock_massflow_config_with_reference: MassFlowConfig,
    basic_spectrum: Spectrum,
):
    mock_from_yaml.return_value = mock_massflow_config_with_reference
    mock_load_spectra.side_effect = [
        [basic_spectrum.clone()],
        [basic_spectrum.clone()],
    ]
    mock_process_spectrum.side_effect = [
        basic_spectrum.clone(),
        basic_spectrum.clone(),
    ]

    mock_calculator = MagicMock()
    mock_get_similarity_calculator.return_value = mock_calculator
    mock_scores = MagicMock(spec=Scores)
    mock_calculator.calculate.return_value = mock_scores

    # Simulate score below threshold (e.g., min_score = 0.5, actual score = 0.4)
    mock_scores.scores_by_query.return_value = [
        (basic_spectrum.clone(), {"CosineGreedy_score": 0.4, "CosineGreedy_matches": 1})
    ]
    mock_massflow_config_with_reference.similarity.min_score = 0.5

    run_workflow(mock_config_path)

    mock_save_results.assert_not_called()


@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
@patch("MassFlow.workflow.get_similarity_calculator")
@patch("MassFlow.workflow.load_spectra")
@patch("MassFlow.workflow.process_spectrum")
@patch("MassFlow.workflow.save_results")
def test_run_workflow_empty_query_spectra(
    mock_save_results: MagicMock,
    mock_process_spectrum: MagicMock,
    mock_load_spectra: MagicMock,
    mock_get_similarity_calculator: MagicMock,
    mock_from_yaml: MagicMock,
    mock_config_path: Path,
    mock_massflow_config_with_reference: MassFlowConfig,
    empty_spectrum: Spectrum,
):
    mock_from_yaml.return_value = mock_massflow_config_with_reference
    mock_load_spectra.side_effect = [
        [
            empty_spectrum
        ],  # Reference spectra (can be empty, though not typical for this test)
        [empty_spectrum],  # Query spectra is empty
    ]
    mock_process_spectrum.return_value = (
        empty_spectrum  # Processed empty spectrum is empty
    )

    run_workflow(mock_config_path)

    assert mock_process_spectrum.call_count == 2  # Once for ref, once for query
    mock_get_similarity_calculator.assert_called_once()  # Should still initialize calculator
    mock_save_results.assert_not_called()  # No actual query spectra, so no results


@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
def test_run_workflow_config_validation_error(
    mock_from_yaml: MagicMock, mock_config_path: Path
):
    mock_from_yaml.side_effect = ValidationError(
        "Test validation error", model=MassFlowConfig
    )
    with pytest.raises(ValidationError, match="Test validation error"):
        run_workflow(mock_config_path)


@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
def test_run_workflow_file_not_found_error(
    mock_from_yaml: MagicMock, mock_config_path: Path
):
    mock_from_yaml.side_effect = FileNotFoundError("Config not found")
    with pytest.raises(FileNotFoundError, match="Config not found"):
        run_workflow(mock_config_path)


@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
def test_run_workflow_generic_exception(
    mock_from_yaml: MagicMock, mock_config_path: Path
):
    mock_from_yaml.side_effect = Exception("Generic error occurred")
    with pytest.raises(Exception, match="Generic error occurred"):
        run_workflow(mock_config_path)


@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
@patch("MassFlow.workflow.get_similarity_calculator")
@patch("MassFlow.workflow.load_spectra")
@patch("MassFlow.workflow.process_spectrum")
@patch("MassFlow.workflow.save_results")
def test_run_workflow_logs_no_matches_found_when_no_results(
    mock_save_results: MagicMock,
    mock_process_spectrum: MagicMock,
    mock_load_spectra: MagicMock,
    mock_get_similarity_calculator: MagicMock,
    mock_from_yaml: MagicMock,
    mock_config_path: Path,
    mock_massflow_config_with_reference: MassFlowConfig,
    caplog: pytest.LogCaptureFixture,  # Capture logs
    basic_spectrum: Spectrum,
):
    mock_from_yaml.return_value = mock_massflow_config_with_reference
    mock_load_spectra.side_effect = [
        [basic_spectrum.clone()],
        [basic_spectrum.clone()],
    ]
    mock_process_spectrum.side_effect = [
        basic_spectrum.clone(),
        basic_spectrum.clone(),
    ]

    mock_calculator = MagicMock()
    mock_get_similarity_calculator.return_value = mock_calculator
    mock_scores = MagicMock(spec=Scores)
    mock_calculator.calculate.return_value = mock_scores

    # Simulate no matches returned by scores_by_query
    mock_scores.scores_by_query.return_value = []
    mock_massflow_config_with_reference.similarity.min_score = (
        0.0  # Even with low threshold, no matches means empty list
    )

    with caplog.at_level(20):  # INFO level
        run_workflow(mock_config_path)

    mock_save_results.assert_not_called()
    assert "No matches found above threshold." in caplog.text


@patch("MassFlow.workflow.MassFlowConfig.from_yaml")
@patch("MassFlow.workflow.get_similarity_calculator")
@patch("MassFlow.workflow.load_spectra")
@patch("MassFlow.workflow.process_spectrum")
@patch("MassFlow.workflow.save_results")
def test_run_workflow_with_multiple_query_spectra(
    mock_save_results: MagicMock,
    mock_process_spectrum: MagicMock,
    mock_load_spectra: MagicMock,
    mock_get_similarity_calculator: MagicMock,
    mock_from_yaml: MagicMock,
    mock_config_path: Path,
    mock_massflow_config_with_reference: MassFlowConfig,
    basic_spectrum: Spectrum,
):
    mock_from_yaml.return_value = mock_massflow_config_with_reference
    ref_spectrum = basic_spectrum.clone()
    query_spectrum_1 = basic_spectrum.clone()
    query_spectrum_1.set("id", "query1")
    query_spectrum_2 = basic_spectrum.clone()
    query_spectrum_2.set("id", "query2")

    mock_load_spectra.side_effect = [
        [ref_spectrum],  # Reference spectra
        [query_spectrum_1, query_spectrum_2],  # Query spectra
    ]

    mock_process_spectrum.side_effect = [
        ref_spectrum.clone(),  # Processed ref
        query_spectrum_1.clone(),  # Processed query 1
        query_spectrum_2.clone(),  # Processed query 2
    ]

    mock_calculator = MagicMock()
    mock_get_similarity_calculator.return_value = mock_calculator
    mock_scores = MagicMock(spec=Scores)
    mock_calculator.calculate.return_value = mock_scores

    # Simulate scores_by_query for each query
    # Query 1 matches
    mock_scores.scores_by_query.side_effect = [
        [
            (
                ref_spectrum.clone(),
                {"CosineGreedy_score": 0.9, "CosineGreedy_matches": 3},
            )
        ],
        [
            (
                ref_spectrum.clone(),
                {"CosineGreedy_score": 0.8, "CosineGreedy_matches": 2},
            )
        ],
    ]
