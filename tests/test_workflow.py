"""
Tests for MassFlow annotation workflow.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import InputConfig, MassFlowConfig, ProjectConfig, SimilarityConfig
from MassFlow.workflow import _process_single_file, run_annotation_pipeline


@pytest.fixture(autouse=True)
def reset_worker_engine(monkeypatch):
    monkeypatch.setattr("MassFlow.workflow._worker_engine", None)


def make_spectrum(spec_id: str, precursor_mz: float = 100.0) -> Spectrum:
    return Spectrum(
        mz=np.array([precursor_mz], dtype="float"),
        intensities=np.array([1.0], dtype="float"),
        metadata={"id": spec_id, "precursor_mz": precursor_mz, "charge": 1},
    )


@patch("MassFlow.workflow.ProcessPoolExecutor")
@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.save_analysis_report")
@patch("MassFlow.workflow.io.save_match_results")
@patch("MassFlow.workflow.get_similarity_engine")
@patch("MassFlow.workflow.io.load_spectra")
def test_run_annotation_pipeline_success(
    mock_load,
    mock_engine_cls,
    mock_save,
    mock_save_report,
    mock_process,
    mock_executor,
    tmp_path,
):
    """Test successful execution of the annotation pipeline."""

    from concurrent.futures import ThreadPoolExecutor

    mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
        max_workers=1
    )

    # Setup Config
    exp_path = tmp_path / "experimental.mgf"
    exp_path.touch()
    ref_path = tmp_path / "reference.msp"
    ref_path.touch()
    out_dir = tmp_path / "results"

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=out_dir),
        input=InputConfig(input_path=exp_path, library_path=ref_path),
        similarity=SimilarityConfig(fdr_threshold=1.0),
    )
    # Mock Data
    mock_query = MagicMock()
    mock_query.get.side_effect = lambda k, d=None: "Query1" if k == "id" else d

    mock_ref = MagicMock()
    mock_ref.get.return_value = "Ref1"
    mock_ref.metadata = {"compound_name": "Ref1_MOCK", "id": "Ref1"}
    mock_ref.peaks.mz = np.array([100.0, 200.0])
    mock_ref.peaks.intensities = np.array([1.0, 1.0])

    # Mock load_spectra to return iterables
    # First call for reference (run_annotation_pipeline),
    # Second call for query (_process_single_file),
    # Third call for reference again (_process_single_file)
    mock_load.side_effect = [[mock_ref], [mock_query], [mock_ref]]

    # Mock process_spectra to pass through
    mock_process.side_effect = lambda s, c: s

    # Mock Engine
    mock_engine_instance = mock_engine_cls.return_value
    mock_results = [{"query_id": "Query1", "score": 0.9}]

    # Save a copy of arguments to avoid issues with list.clear()
    search_calls = []

    def mock_search(q, r, **kwargs):
        search_calls.append((list(q), list(r)))
        return mock_results

    mock_engine_instance.search.side_effect = mock_search

    # Run
    run_annotation_pipeline(config)

    # Verify
    assert mock_load.call_count == 3
    mock_engine_cls.assert_called_with(config.similarity)
    assert search_calls == [([mock_query], [mock_ref])]

    expected_out_file = out_dir / "experimental_results.csv"
    expected_report_file = out_dir / "experimental_results.report.yaml"
    mock_save.assert_called_with(
        mock_results, expected_out_file, query_spectra=[mock_query]
    )
    mock_save_report.assert_called_once()
    report_args, report_kwargs = mock_save_report.call_args
    assert report_args[0] == expected_report_file
    assert report_kwargs == {}
    report_payload = report_args[1]
    assert report_payload["query_file"] == str(exp_path)
    assert report_payload["results_csv"] == str(expected_out_file)
    assert report_payload["library_path"] == str(ref_path)
    assert report_payload["processing"] == config.processing.model_dump(mode="json")
    assert report_payload["similarity"] == config.similarity.model_dump(mode="json")


@patch("MassFlow.workflow.ProcessPoolExecutor")
@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
def test_run_annotation_pipeline_no_query_spectra(
    mock_load, mock_process, mock_executor, tmp_path
):
    """Test warning when no query spectra are found (should not raise)."""
    from concurrent.futures import ThreadPoolExecutor

    mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
        max_workers=1
    )

    # Ref loaded first (valid), Query loaded second (empty)
    mock_ref = MagicMock()
    mock_ref.get.return_value = "Ref1"
    mock_ref.metadata = {"compound_name": "Ref1_MOCK", "id": "Ref1"}
    mock_ref.peaks.mz = np.array([100.0, 200.0])
    mock_ref.peaks.intensities = np.array([1.0, 1.0])

    mock_load.side_effect = [[mock_ref], []]
    mock_process.side_effect = lambda s, c: s

    exp_path = tmp_path / "exp.mgf"
    exp_path.touch()
    ref_path = tmp_path / "ref.msp"
    ref_path.touch()

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path),
        input=InputConfig(input_path=exp_path, library_path=ref_path),
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

    exp_path = tmp_path / "exp.mgf"
    exp_path.touch()
    ref_path = tmp_path / "ref.msp"
    ref_path.touch()

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path),
        input=InputConfig(input_path=exp_path, library_path=ref_path),
    )

    with pytest.raises(ValueError, match="No valid spectra found in library"):
        run_annotation_pipeline(config)


@pytest.mark.parametrize(
    ("reference_count", "should_warn"), [(1999, True), (2000, False)]
)
@patch("MassFlow.workflow.ProcessPoolExecutor")
@patch("MassFlow.workflow._process_single_file")
@patch("MassFlow.workflow.io.save_analysis_report")
@patch("MassFlow.workflow.io.save_match_results")
@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
def test_run_annotation_pipeline_small_library_warning_threshold(
    mock_load,
    mock_process,
    mock_save,
    mock_save_report,
    mock_process_single_file,
    mock_executor,
    reference_count,
    should_warn,
    tmp_path,
    caplog,
):
    """Warn below the scientific threshold, but not at the threshold."""
    from concurrent.futures import ThreadPoolExecutor

    mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
        max_workers=1
    )

    reference_spectra = [
        make_spectrum(f"ref_{i}", precursor_mz=100.0 + i)
        for i in range(reference_count)
    ]
    query_spectrum = make_spectrum("query_1", precursor_mz=250.0)

    mock_load.return_value = reference_spectra
    mock_process.side_effect = lambda spectra, config: spectra
    exp_path = tmp_path / "experimental.mgf"
    exp_path.touch()
    ref_path = tmp_path / "reference.msp"
    ref_path.touch()

    mock_process_single_file.return_value = (
        exp_path,
        [query_spectrum],
        [],
    )

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=exp_path,
            library_path=ref_path,
        ),
    )

    # Capture standard output for the rich console
    from unittest.mock import patch
    import MassFlow.cli

    with patch.object(MassFlow.cli.console, 'print') as mock_print:
        run_annotation_pipeline(config)

        output = ""
        for call_args in mock_print.call_args_list:
            args, kwargs = call_args
            if args:
                # The first argument is typically the Panel or string
                panel_or_str = args[0]
                # If it's a Panel, it has a renderable attribute which is the warning string
                if hasattr(panel_or_str, "renderable"):
                    output += str(panel_or_str.renderable)
                else:
                    output += str(panel_or_str)

    found_warning = "CRITICAL SCIENTIFIC WARNING: SMALL LIBRARY DETECTED" in output
    assert found_warning is should_warn
    mock_save.assert_called_once()
    mock_save_report.assert_called_once()


@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
def test_run_annotation_pipeline_empty_data_directory(
    mock_load, mock_process, tmp_path
):
    """Fail fast when the configured input directory is empty."""
    input_dir = tmp_path / "empty_inputs"
    input_dir.mkdir()

    mock_load.return_value = [make_spectrum("ref_1")]
    mock_process.side_effect = lambda spectra, config: spectra

    ref_path = tmp_path / "reference.msp"
    ref_path.touch()

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=input_dir,
            library_path=ref_path,
        ),
    )

    with pytest.raises(ValueError, match="No supported spectral files found"):
        run_annotation_pipeline(config)


@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
def test_run_annotation_pipeline_non_spectral_files_only(
    mock_load, mock_process, tmp_path
):
    """Ignore unrelated files and fail when no supported spectra remain."""
    input_dir = tmp_path / "mixed_inputs"
    input_dir.mkdir()
    (input_dir / "notes.txt").write_text("not a spectrum")
    (input_dir / "table.csv").write_text("col1,col2\n1,2\n")

    mock_load.return_value = [make_spectrum("ref_1")]
    mock_process.side_effect = lambda spectra, config: spectra

    ref_path = tmp_path / "reference.msp"
    ref_path.touch()

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=input_dir,
            library_path=ref_path,
        ),
    )

    with pytest.raises(ValueError, match="No supported spectral files found"):
        run_annotation_pipeline(config)


@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
def test_process_single_file_logs_and_returns_empty_on_malformed_input(
    mock_load, mock_process, tmp_path, caplog
):
    """Malformed query files should be logged and skipped by the worker helper."""
    mock_load.side_effect = ValueError("Malformed spectral file")
    mock_process.side_effect = lambda spectra, config: spectra

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path),
        input=InputConfig(
            input_path=Path("bad.mgf"),
            library_path=Path("reference.msp"),
            format="mgf",
        ),
    )

    with caplog.at_level(logging.ERROR):
        processed_file, query_spectra, results = _process_single_file(
            Path("bad.mgf"), config
        )

    assert processed_file == Path("bad.mgf")
    assert query_spectra == []
    assert results == []
    assert "Failed to process bad.mgf" in caplog.text


@pytest.mark.parametrize(
    ("fdr_threshold", "expected_result_count", "expected_q_value"),
    [(0.01, 0, None), (1.0, 1, 1.0)],
)
@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
def test_process_single_file_tiny_library_fdr_sensitivity(
    mock_load,
    mock_process,
    fdr_threshold,
    expected_result_count,
    expected_q_value,
    tmp_path,
):
    """Tiny target/decoy sets should show the expected strict-vs-relaxed FDR behavior."""
    query = make_spectrum("query_1", precursor_mz=100.0)
    reference = make_spectrum("ref_1", precursor_mz=100.0)

    mock_load.side_effect = [[query], [reference]]
    mock_process.side_effect = lambda spectra, config: spectra

    fake_engine = MagicMock()
    fake_engine.search.return_value = [
        {
            "query_id": "query_1",
            "query_precursor_mz": 100.0,
            "reference_id": "ref_1",
            "reference_name": "Target",
            "reference_precursor_mz": 100.0,
            "score": 0.9,
            "matched_peaks": 5,
            "smiles": None,
            "inchikey": None,
            "is_decoy": False,
            "q_value": 1.0,
            "annotation_tier": None,
        },
        {
            "query_id": "query_1",
            "query_precursor_mz": 100.0,
            "reference_id": "ref_1_decoy",
            "reference_name": "Target_decoy",
            "reference_precursor_mz": 100.0,
            "score": 0.95,
            "matched_peaks": 5,
            "smiles": None,
            "inchikey": None,
            "is_decoy": True,
            "q_value": 1.0,
            "annotation_tier": None,
        },
    ]

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path),
        input=InputConfig(
            input_path=Path("query.mgf"),
            library_path=Path("reference.msp"),
            format="mgf",
        ),
        similarity=SimilarityConfig(fdr_threshold=fdr_threshold),
    )

    with patch("MassFlow.workflow._worker_engine", fake_engine):
        processed_file, query_spectra, results = _process_single_file(
            Path("query.mgf"), config
        )

    assert processed_file == Path("query.mgf")
    assert query_spectra == [query]
    assert len(results) == expected_result_count
    if expected_q_value is not None:
        assert results[0]["q_value"] == pytest.approx(expected_q_value)


@pytest.mark.parametrize("export_format", ["csv", "mztab"])
@patch("MassFlow.workflow.ProcessPoolExecutor")
@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.save_analysis_report")
@patch("MassFlow.workflow.io.save_match_results_to_mztab")
@patch("MassFlow.workflow.io.save_match_results")
@patch("MassFlow.workflow.get_similarity_engine")
@patch("MassFlow.workflow.io.load_spectra")
def test_run_annotation_pipeline_export_routing(
    mock_load,
    mock_engine_cls,
    mock_save_csv,
    mock_save_mztab,
    mock_save_report,
    mock_process,
    mock_executor,
    export_format,
    tmp_path,
):
    """Test that pipeline routes output to the correct export function based on config."""
    from concurrent.futures import ThreadPoolExecutor

    from MassFlow.config import ExportConfig

    mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
        max_workers=1
    )

    exp_path = tmp_path / "exp.mgf"
    exp_path.touch()
    ref_path = tmp_path / "ref.msp"
    ref_path.touch()
    out_dir = tmp_path / "results"

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=out_dir),
        input=InputConfig(input_path=exp_path, library_path=ref_path),
        export=ExportConfig(format=export_format),
        similarity=SimilarityConfig(fdr_threshold=1.0),
    )
    mock_query = MagicMock()
    mock_query.get.side_effect = lambda k, d=None: "Query1" if k == "id" else d

    mock_ref = MagicMock()
    mock_ref.get.return_value = "Ref1"
    mock_ref.metadata = {"compound_name": "Ref1_MOCK", "id": "Ref1"}
    mock_ref.peaks.mz = np.array([100.0, 200.0])
    mock_ref.peaks.intensities = np.array([1.0, 1.0])

    mock_load.side_effect = [[mock_ref], [mock_query], [mock_ref]]
    mock_process.side_effect = lambda s, c: s

    mock_engine_instance = mock_engine_cls.return_value
    mock_results = [{"query_id": "Query1", "score": 0.9}]
    mock_engine_instance.search.return_value = mock_results

    run_annotation_pipeline(config)

    expected_ext = export_format
    expected_out_file = out_dir / f"exp_results.{expected_ext}"

    if export_format == "csv":
        mock_save_csv.assert_called_once_with(
            mock_results, expected_out_file, query_spectra=[mock_query]
        )
    elif export_format == "mztab":
        mock_save_mztab.assert_called_once_with(
            mock_results, expected_out_file, query_spectra=[mock_query]
        )

    mock_save_report.assert_called_once()
    report_args = mock_save_report.call_args[0]
    assert report_args[0] == out_dir / "exp_results.report.yaml"
    assert report_args[1]["results_csv"] == str(expected_out_file)


def test_run_annotation_pipeline_missing_library_path(tmp_path):
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(input_path=tmp_path, library_path=None),
    )
    with pytest.raises(ValueError, match="Library path not specified in configuration"):
        run_annotation_pipeline(config)


def test_run_annotation_pipeline_library_path_not_exists(tmp_path):
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=tmp_path, library_path=tmp_path / "nonexistent.msp"
        ),
    )
    with pytest.raises(ValueError, match="Library path does not exist"):
        run_annotation_pipeline(config)


@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
@patch("MassFlow.workflow.ProcessPoolExecutor")
def test_run_annotation_pipeline_nested_input_dirs(
    mock_executor, mock_load, mock_process, tmp_path
):
    from concurrent.futures import ThreadPoolExecutor

    mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
        max_workers=1
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    # Supported files
    (input_dir / "valid.mgf").touch()

    # .d directory
    d_dir = input_dir / "sample.d"
    d_dir.mkdir()

    ref_path = tmp_path / "ref.msp"
    ref_path.touch()

    mock_load.return_value = [make_spectrum("ref_1")]
    mock_process.side_effect = lambda spectra, config: spectra

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(input_path=input_dir, library_path=ref_path),
    )

    # Ignore the rest
    with patch("MassFlow.workflow._process_single_file") as mock_psf:
        mock_psf.return_value = (Path("mock"), [], [])
        run_annotation_pipeline(config)

        # Check that both valid.mgf and sample.d were submitted
        calls = mock_psf.call_args_list
        files_submitted = [call[0][0] for call in calls]
        assert len(files_submitted) == 2


@patch("MassFlow.workflow.processing.process_spectra")
@patch("MassFlow.workflow.io.load_spectra")
def test_process_single_file_deduplicates_query_ids(mock_load, mock_process, tmp_path):
    """Test that query IDs are deduplicated to prevent Cartesian explosions during result export."""
    import numpy as np
    from matchms import Spectrum

    q1 = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "duplicate_id", "precursor_mz": 100.0},
    )
    q2 = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "duplicate_id", "precursor_mz": 100.0},
    )
    q3 = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": None, "precursor_mz": 100.0},
    )

    query_spectra = [q1, q2, q3]
    reference = make_spectrum("ref_1", precursor_mz=100.0)

    mock_load.side_effect = [query_spectra, [reference]]
    mock_process.side_effect = lambda spectra, config: spectra

    fake_engine = MagicMock()
    fake_engine.search.return_value = []

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path),
        input=InputConfig(
            input_path=Path("query.mgf"),
            library_path=Path("reference.msp"),
            format="mgf",
        ),
    )

    with patch("MassFlow.workflow._worker_engine", fake_engine):
        processed_file, processed_queries, results = _process_single_file(
            Path("query.mgf"), config
        )

    # Validate that the IDs have been rewritten to be unique
    extracted_ids = [q.get("id") for q in processed_queries]

    assert extracted_ids[0] == "duplicate_id"
    assert extracted_ids[1] == "duplicate_id_1"
    assert extracted_ids[2] == "query_query_2"  # It uses stem "query"

    # Assert they are unique
    assert len(set(extracted_ids)) == 3
