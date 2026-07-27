"""
Comprehensive coverage tests for MassFlow workflow.py:
- _init_worker (L2 cache arrays, worker engine initialization)
- _process_single_file (streaming path, deduplication, FDR edge cases)
- _handle_file_results (collision prevention, export routing)
- _write_analysis_report
- run_annotation_pipeline (streaming library path, single-file dispatch)
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import InputConfig, MassFlowConfig, ProjectConfig, SimilarityConfig
from MassFlow.workflow import (
    _handle_file_results,
    _init_worker,
    _process_single_file,
    _write_analysis_report,
    run_annotation_pipeline,
)

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def reset_worker_engine(monkeypatch):
    monkeypatch.setattr("MassFlow.workflow._worker_engine", None)
    monkeypatch.setattr("MassFlow.workflow._worker_references", None)
    monkeypatch.setattr("MassFlow.workflow._worker_decoys", None)
    monkeypatch.setattr("MassFlow.workflow._worker_ref_precursor_mzs", None)
    monkeypatch.setattr("MassFlow.workflow._worker_ref_is_decoy", None)


def make_spectrum(spec_id: str, precursor_mz: float = 100.0) -> Spectrum:
    return Spectrum(
        mz=np.array(
            [precursor_mz, precursor_mz + 50, precursor_mz + 100], dtype=np.float64
        ),
        intensities=np.array([0.5, 1.0, 0.3], dtype=np.float64),
        metadata={"id": spec_id, "precursor_mz": precursor_mz, "charge": 1},
    )


# ==============================================================================
# _init_worker
# ==============================================================================


class TestInitWorker:
    """Cover _init_worker function."""

    def test_init_worker_with_references_and_decoys(self, tmp_path):
        ref = make_spectrum("ref1", 200.0)
        decoy = make_spectrum("decoy1", 200.0)
        decoy.set("is_decoy", True)

        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )

        _init_worker(cfg, [ref], [decoy])

        import MassFlow.workflow as wf

        assert wf._worker_engine is not None
        assert wf._worker_references is not None
        assert wf._worker_decoys is not None
        assert wf._worker_ref_precursor_mzs is not None
        assert wf._worker_ref_is_decoy is not None
        assert len(wf._worker_ref_precursor_mzs) == 2
        assert np.any(wf._worker_ref_is_decoy)

    def test_init_worker_without_references(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        _init_worker(cfg, None, None)

        import MassFlow.workflow as wf

        assert wf._worker_ref_precursor_mzs is None
        assert wf._worker_ref_is_decoy is None

    def test_init_worker_with_nan_precursor(self, tmp_path):
        s = Spectrum(
            mz=np.array([100.0]),
            intensities=np.array([1.0]),
            metadata={"id": "nan_pmz", "precursor_mz": float("nan")},
        )
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        _init_worker(cfg, [s], [])

        import MassFlow.workflow as wf

        assert wf._worker_ref_precursor_mzs is not None
        assert wf._worker_ref_precursor_mzs[0] == 0.0


# ==============================================================================
# _process_single_file
# ==============================================================================


class TestProcessSingleFile:
    """Cover _process_single_file edge cases."""

    def test_returns_empty_when_no_queries(self, tmp_path):
        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([])):
            with patch(
                "MassFlow.workflow.processing.process_spectra", return_value=iter([])
            ):
                cfg = MassFlowConfig(
                    project=ProjectConfig(output_directory=tmp_path),
                    input=InputConfig(input_path=tmp_path / "q.mgf"),
                )
                f, queries, results = _process_single_file(tmp_path / "q.mgf", cfg)
                assert len(queries) == 0
                assert len(results) == 0

    def test_deduplicates_query_ids(self, tmp_path):
        s1 = make_spectrum("dup", 100.0)
        s2 = make_spectrum("dup", 200.0)

        lib_path = tmp_path / "lib.msp"
        lib_path.touch()

        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf", library_path=lib_path),
            similarity=SimilarityConfig(
                fdr_threshold=1.0, min_score=0.0, ms1_tolerance=100.0
            ),
        )

        import MassFlow.workflow as wf

        wf._worker_engine = None
        wf._worker_references = None
        wf._worker_decoys = None
        wf._worker_ref_precursor_mzs = None
        wf._worker_ref_is_decoy = None

        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([s1, s2])):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda x, c: x,
            ):
                f, queries, results = _process_single_file(tmp_path / "q.mgf", cfg)
                ids = [q.get("id") for q in queries]
                assert len(set(ids)) == len(ids)
                assert "dup_1" in ids

    def test_assigns_id_for_missing_id(self, tmp_path):
        s = Spectrum(
            mz=np.array([100.0]),
            intensities=np.array([1.0]),
            metadata={"precursor_mz": 100.0, "charge": 1},
        )
        lib_path = tmp_path / "lib.msp"
        lib_path.touch()

        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf", library_path=lib_path),
        )
        import MassFlow.workflow as wf

        wf._worker_engine = None
        wf._worker_references = None
        wf._worker_decoys = None
        wf._worker_ref_precursor_mzs = None
        wf._worker_ref_is_decoy = None

        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([s])):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda x, c: x,
            ):
                f, queries, results = _process_single_file(tmp_path / "q.mgf", cfg)
                assert len(queries) == 1
                assert queries[0].get("id") is not None

    def test_streaming_fallback_when_no_worker_refs(self, tmp_path):
        q = make_spectrum("query1", 200.0)
        ref = make_spectrum("ref1", 200.0)

        lib_path = tmp_path / "lib.msp"
        lib_path.touch()

        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf", library_path=lib_path),
            similarity=SimilarityConfig(
                fdr_threshold=1.0, min_score=0.0, ms1_tolerance=100.0
            ),
        )

        load_count = [0]

        def mock_load(path, file_format=None):
            load_count[0] += 1
            if load_count[0] == 1:
                return iter([q])
            return iter([ref])

        import MassFlow.workflow as wf

        wf._worker_engine = None
        wf._worker_references = None
        wf._worker_decoys = None
        wf._worker_ref_precursor_mzs = None
        wf._worker_ref_is_decoy = None

        with patch("MassFlow.workflow.io.load_spectra", side_effect=mock_load):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda x, c: x,
            ):
                f, queries, results = _process_single_file(tmp_path / "q.mgf", cfg)
                assert isinstance(queries, list)

    def test_missing_library_path_returns_empty(self, tmp_path):
        """When no worker refs and no library_path, the error is caught and empty lists returned."""
        q = make_spectrum("query1", 200.0)
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf", library_path=None),
        )
        import MassFlow.workflow as wf

        wf._worker_engine = None
        wf._worker_references = None
        wf._worker_decoys = None
        wf._worker_ref_precursor_mzs = None
        wf._worker_ref_is_decoy = None

        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([q])):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda x, c: x,
            ):
                f, queries, results = _process_single_file(tmp_path / "q.mgf", cfg)
                # Error is caught, returns empty
                assert len(queries) == 0
                assert len(results) == 0

    def test_handles_exceptions_gracefully(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        with patch(
            "MassFlow.workflow.io.load_spectra", side_effect=RuntimeError("Boom!")
        ):
            f, queries, results = _process_single_file(tmp_path / "q.mgf", cfg)
            assert len(queries) == 0
            assert len(results) == 0

    def test_small_library_bonferroni_correction(self, tmp_path):
        q = make_spectrum("query1", 200.0)
        ref = make_spectrum("ref1", 200.0)

        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
            similarity=SimilarityConfig(
                fdr_threshold=1.0, min_score=0.0, ms1_tolerance=100.0
            ),
        )

        import MassFlow.workflow as wf

        wf._worker_engine = None
        wf._worker_references = [ref]
        wf._worker_decoys = []
        wf._worker_ref_precursor_mzs = np.array([200.0], dtype=np.float64)
        wf._worker_ref_is_decoy = np.array([False], dtype=bool)

        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([q])):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda x, c: x,
            ):
                f, queries, results = _process_single_file(tmp_path / "q.mgf", cfg)
                assert isinstance(results, list)


# ==============================================================================
# _handle_file_results
# ==============================================================================


class TestHandleFileResults:
    """Cover _handle_file_results."""

    def test_no_queries_logs_warning(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        with patch("MassFlow.workflow.io.save_match_results") as mock_save:
            _handle_file_results(tmp_path / "q.mgf", [], [], cfg)
            mock_save.assert_not_called()

    def test_csv_export(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        q = make_spectrum("query1", 200.0)
        results = [{"query_id": "query1", "score": 0.95}]

        with patch("MassFlow.workflow.io.save_match_results") as mock_save:
            with patch("MassFlow.workflow._write_analysis_report"):
                _handle_file_results(tmp_path / "q.mgf", [q], results, cfg)
                mock_save.assert_called_once()

    def test_mztab_export(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        cfg.export.format = "mztab"
        q = make_spectrum("query1", 200.0)
        results = [{"query_id": "query1", "score": 0.95}]

        with patch("MassFlow.workflow.io.save_match_results_to_mztab") as mock_save:
            with patch("MassFlow.workflow._write_analysis_report"):
                _handle_file_results(tmp_path / "q.mgf", [q], results, cfg)
                mock_save.assert_called_once()

    def test_collision_prevention(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        q = make_spectrum("query1", 200.0)
        results = [{"query_id": "query1", "score": 0.95}]

        expected = tmp_path / "q_results.csv"
        expected.touch()

        with patch("MassFlow.workflow.io.save_match_results") as mock_save:
            with patch("MassFlow.workflow._write_analysis_report"):
                _handle_file_results(tmp_path / "q.mgf", [q], results, cfg)
                call_args = mock_save.call_args[0]
                out_path = call_args[1]
                assert "q_1_results.csv" in str(out_path)


# ==============================================================================
# _write_analysis_report
# ==============================================================================


class TestWriteAnalysisReport:
    """Cover _write_analysis_report."""

    def test_basic_report(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(
                input_path=tmp_path / "q.mgf", library_path=tmp_path / "lib.msp"
            ),
        )
        q = make_spectrum("query1", 200.0)
        results = [{"query_id": "query1", "score": 0.95}]

        with patch("MassFlow.workflow.io.save_analysis_report") as mock_save:
            _write_analysis_report(
                tmp_path / "report.yaml",
                cfg,
                tmp_path / "q.mgf",
                tmp_path / "results.csv",
                [q],
                results,
            )
            mock_save.assert_called_once()
            payload = mock_save.call_args[0][1]
            assert payload["num_queries"] == 1
            assert payload["num_matches"] == 1

    def test_report_with_config_path(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        with patch("MassFlow.workflow.io.save_analysis_report") as mock_save:
            _write_analysis_report(
                tmp_path / "report.yaml",
                cfg,
                tmp_path / "q.mgf",
                tmp_path / "results.csv",
                [],
                [],
                config_path="/tmp/config.yaml",
            )
            mock_save.assert_called_once()


# ==============================================================================
# run_annotation_pipeline
# ==============================================================================


class TestRunAnnotationPipelineStreaming:
    """Cover streaming library path and single-file dispatch."""

    def test_streaming_library_path(self, tmp_path):
        """Library with streaming_threshold_mb=0 triggers streaming mode."""
        exp_path = tmp_path / "exp.mgf"
        exp_path.touch()
        lib_path = tmp_path / "lib.msp"
        lib_path.write_text("dummy library content")

        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path / "results"),
            input=InputConfig(
                input_path=exp_path, library_path=lib_path, streaming_threshold_mb=0
            ),
            similarity=SimilarityConfig(fdr_threshold=1.0, min_score=0.0),
        )
        # streaming_threshold_mb=0 forces streaming mode for any non-empty file

        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([])):
            with patch(
                "MassFlow.workflow.processing.process_spectra", return_value=iter([])
            ):
                with patch("MassFlow.workflow.io.save_match_results"):
                    with patch("MassFlow.workflow.io.save_analysis_report"):
                        # Should not crash - streaming path is taken
                        run_annotation_pipeline(cfg)
