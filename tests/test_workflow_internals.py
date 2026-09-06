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

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.workflow import (
    FileExecutionResult,
    _handle_file_results,
    _init_worker,
    _process_single_file,
    _write_analysis_report,
    run_annotation_pipeline,
)

DATA_DIR = Path(__file__).parent / "data"


def write_msp(path: Path, spectra) -> None:
    """Minimal MSP writer for fixtures."""
    lines = []
    for spectrum in spectra:
        lines.append(f"NAME: {spectrum.get('compound_name') or spectrum.get('id')}")
        lines.append(f"PRECURSOR_MZ: {spectrum.get('precursor_mz')}")
        lines.append("CHARGE: 1")
        lines.append(f"NUM PEAKS: {len(spectrum.peaks.mz)}")
        for mz, intensity in zip(spectrum.peaks.mz, spectrum.peaks.intensities):
            lines.append(f"{mz}\t{intensity}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(autouse=True)
def reset_worker_engine(monkeypatch):
    monkeypatch.setattr("MassFlow.workflow._worker_engine", None)
    monkeypatch.setattr("MassFlow.workflow._worker_backend", None)
    monkeypatch.setattr("MassFlow.workflow._worker_library_spec", None)


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
    """Cover _init_worker (backend-opening contract)."""

    def test_init_worker_opens_backend(self, tmp_path):
        from MassFlow.library import LibrarySpec

        ref = make_spectrum("ref1", 200.0)
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )

        # A store spec is opened by the worker itself; the payload is never
        # passed between processes.
        store_path = tmp_path / "lib.db"
        from MassFlow.storage import create_spectral_store

        store = create_spectral_store(store_path, backend="sqlite")
        store.add_spectra(iter([ref]), category="library")
        store.close()
        spec = LibrarySpec(path=store_path, kind="store", storage_backend="sqlite")

        _init_worker(cfg, spec)

        import MassFlow.workflow as wf

        assert wf._worker_engine is not None
        assert wf._worker_library_spec is spec
        assert wf._worker_backend is not None
        assert wf._worker_backend.spectrum_count() == 1

    def test_init_worker_without_spec(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        _init_worker(cfg, None)

        import MassFlow.workflow as wf

        assert wf._worker_backend is None
        assert wf._worker_library_spec is None


# ==============================================================================
# _process_single_file
# ==============================================================================


class TestProcessSingleFile:
    """Cover _process_single_file edge cases (FileExecutionResult contract)."""

    def test_returns_failed_when_no_queries(self, tmp_path):
        """A file with no analyzable spectra is an explicit failure, never
        an empty success."""
        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([])):
            with patch(
                "MassFlow.workflow.processing.process_spectra", return_value=iter([])
            ):
                cfg = MassFlowConfig(
                    project=ProjectConfig(output_directory=tmp_path),
                    input=InputConfig(input_path=tmp_path / "q.mgf"),
                )
                result = _process_single_file(tmp_path / "q.mgf", cfg)
                assert result.status == "failed"
                assert len(result.fatal_errors) == 1
                assert result.query_spectra == []
                assert result.results == []

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

        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([s1, s2])):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda x, c: x,
            ):
                result = _process_single_file(tmp_path / "q.mgf", cfg)
                ids = [q.get("id") for q in result.query_spectra]
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

        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([s])):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda x, c: x,
            ):
                result = _process_single_file(tmp_path / "q.mgf", cfg)
                assert len(result.query_spectra) == 1
                assert result.query_spectra[0].get("id") is not None

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

        def mock_load(path, file_format=None, rejection_reporter=None):
            load_count[0] += 1
            if load_count[0] == 1:
                return iter([q])
            return iter([ref])

        with patch("MassFlow.workflow.io.load_spectra", side_effect=mock_load):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda x, c: x,
            ):
                result = _process_single_file(tmp_path / "q.mgf", cfg)
                assert isinstance(result.query_spectra, list)

    def test_missing_library_path_returns_failed(self, tmp_path):
        """When no backend spec and no library_path, the file fails explicitly."""
        q = make_spectrum("query1", 200.0)
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf", library_path=None),
        )

        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([q])):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda x, c: x,
            ):
                result = _process_single_file(tmp_path / "q.mgf", cfg)
                assert result.status == "failed"
                assert any("Library path" in e for e in result.fatal_errors)

    def test_handles_exceptions_as_explicit_failure(self, tmp_path):
        """An exception during processing is a failed result, not an empty
        success."""
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        with patch(
            "MassFlow.workflow.io.load_spectra", side_effect=RuntimeError("Boom!")
        ):
            result = _process_single_file(tmp_path / "q.mgf", cfg)
            assert result.status == "failed"
            assert len(result.fatal_errors) == 1
            assert "Boom!" in result.fatal_errors[0]

    def test_worker_backend_streaming_path(self, tmp_path):
        """With a worker backend set (store), the file is searched against
        spectra streamed from the store."""
        from MassFlow.library import LibrarySpec, open_library
        from MassFlow.storage import create_spectral_store

        q = make_spectrum("query1", 200.0)
        ref = make_spectrum("ref1", 200.0)
        store_path = tmp_path / "lib.db"
        store = create_spectral_store(store_path, backend="sqlite")
        store.add_spectra(iter([ref]), category="library")
        store.close()
        spec = LibrarySpec(path=store_path, kind="store", storage_backend="sqlite")

        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf", library_path=store_path),
            similarity=SimilarityConfig(
                fdr_threshold=1.0, min_score=0.0, ms1_tolerance=100.0
            ),
        )

        import MassFlow.workflow as wf

        wf._worker_backend = open_library(spec, cfg.processing)
        wf._worker_library_spec = spec
        try:
            with patch("MassFlow.workflow.io.load_spectra", return_value=iter([q])):
                with patch(
                    "MassFlow.workflow.processing.process_spectra",
                    side_effect=lambda x, c: x,
                ):
                    result = _process_single_file(
                        tmp_path / "q.mgf", cfg, library_size=1
                    )
        finally:
            wf._worker_backend = None
            wf._worker_library_spec = None

        assert isinstance(result.results, list)
        assert result.status in ("success", "degraded")


# ==============================================================================
# _handle_file_results
# ==============================================================================


class TestHandleFileResults:
    """Cover _handle_file_results (FileExecutionResult contract)."""

    @staticmethod
    def _result(
        tmp_path: Path,
        status: str = "success",
        queries=None,
        results=None,
    ) -> "FileExecutionResult":
        return FileExecutionResult(
            status=status,  # type: ignore[arg-type]
            input_path=tmp_path / "q.mgf",
            query_spectra=queries or [],
            results=results or [],
        )

    def test_failed_file_writes_failure_report_and_no_csv(self, tmp_path):
        """A failed file never produces a results CSV; it produces an
        explicit failure provenance report."""
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        result = self._result(tmp_path, status="failed")
        result.fatal_errors.append("UnsupportedVendorFormatError: vendor raw")

        with patch("MassFlow.workflow.io.save_match_results") as mock_save:
            with patch("MassFlow.workflow.io.save_analysis_report") as mock_report:
                _handle_file_results(result, cfg)
        mock_save.assert_not_called()
        mock_report.assert_called_once()
        report_path = mock_report.call_args[0][0]
        assert "q_failed.report.yaml" in str(report_path)

    def test_no_queries_logs_warning(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        with patch("MassFlow.workflow.io.save_match_results") as mock_save:
            _handle_file_results(self._result(tmp_path), cfg)
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
                _handle_file_results(
                    self._result(tmp_path, queries=[q], results=results), cfg
                )
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
                _handle_file_results(
                    self._result(tmp_path, queries=[q], results=results), cfg
                )
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
                _handle_file_results(
                    self._result(tmp_path, queries=[q], results=results), cfg
                )
                call_args = mock_save.call_args[0]
                out_path = call_args[1]
                assert "q_1_results.csv" in str(out_path)


# ==============================================================================
# _write_analysis_report
# ==============================================================================


class TestWriteAnalysisReport:
    """Cover _write_analysis_report (status/provenance contract)."""

    @staticmethod
    def _result(tmp_path: Path, status: str = "success"):
        q = make_spectrum("query1", 200.0)
        results = [{"query_id": "query1", "score": 0.95}]
        return FileExecutionResult(
            status=status,  # type: ignore[arg-type]
            input_path=tmp_path / "q.mgf",
            spectra_loaded=3,
            spectra_rejected=1,
            hits_produced=1,
            query_spectra=[q],
            results=results,  # type: ignore[arg-type]
            fdr_summary={"n_competing_queries": 1, "library_size": 100},
        )

    def test_basic_report(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(
                input_path=tmp_path / "q.mgf", library_path=tmp_path / "lib.msp"
            ),
        )

        with patch("MassFlow.workflow.io.save_analysis_report") as mock_save:
            _write_analysis_report(
                tmp_path / "report.yaml", cfg, self._result(tmp_path)
            )
            mock_save.assert_called_once()
            payload = mock_save.call_args[0][1]
            assert payload["num_queries"] == 1
            assert payload["num_matches"] == 1
            assert payload["status"] == "success"
            assert payload["spectra_loaded"] == 3
            assert payload["spectra_rejected"] == 1
            assert payload["hits_produced"] == 1
            assert payload["fatal_errors"] == []
            assert payload["degraded_mode_flags"] == []
            assert payload["fdr"]["library_size"] == 100

    def test_failed_report_records_fatal_errors(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        result = self._result(tmp_path, status="failed")
        result.fatal_errors.append("ValueError: boom")
        with patch("MassFlow.workflow.io.save_analysis_report") as mock_save:
            _write_analysis_report(tmp_path / "report.yaml", cfg, result)
            payload = mock_save.call_args[0][1]
            assert payload["status"] == "failed"
            assert payload["fatal_errors"] == ["ValueError: boom"]

    def test_degraded_report_records_flags(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        result = self._result(tmp_path, status="degraded")
        result.degraded_mode_flags.append("engine_fallback:spec2vec")
        result.warnings.append("fell back")
        with patch("MassFlow.workflow.io.save_analysis_report") as mock_save:
            _write_analysis_report(tmp_path / "report.yaml", cfg, result)
            payload = mock_save.call_args[0][1]
            assert payload["status"] == "degraded"
            assert payload["degraded_mode_flags"] == ["engine_fallback:spec2vec"]
            assert payload["warnings"] == ["fell back"]

    def test_report_with_config_path(self, tmp_path):
        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path),
            input=InputConfig(input_path=tmp_path / "q.mgf"),
        )
        with patch("MassFlow.workflow.io.save_analysis_report") as mock_save:
            _write_analysis_report(
                tmp_path / "report.yaml",
                cfg,
                self._result(tmp_path),
                config_path="/tmp/config.yaml",
            )
            mock_save.assert_called_once()


# ==============================================================================
# run_annotation_pipeline
# ==============================================================================


class TestRunAnnotationPipelineStreaming:
    """Cover the worker-library-store path and single-file dispatch."""

    def test_empty_library_fails_fast(self, tmp_path):
        """The parent normalizes the library into a store; an empty library
        fails fast instead of silently succeeding with no data."""
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

        with patch("MassFlow.workflow.io.load_spectra", return_value=iter([])):
            with patch(
                "MassFlow.workflow.processing.process_spectra", return_value=iter([])
            ):
                with pytest.raises(ValueError, match="No valid spectra"):
                    run_annotation_pipeline(cfg)

    def test_library_store_built_and_reused(self, tmp_path):
        """A raw library file is normalized into a worker-openable store in
        the output directory, and a second run reuses it."""
        from MassFlow.library import prepare_library

        exp_path = tmp_path / "exp.mgf"
        exp_path.touch()
        lib_path = tmp_path / "lib.msp"
        write_msp(lib_path, [make_spectrum("ref1", 200.0)])

        cfg = MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path / "results"),
            input=InputConfig(
                input_path=exp_path, library_path=lib_path, streaming_threshold_mb=0
            ),
            processing=ProcessingConfig(min_peaks=1),
            similarity=SimilarityConfig(fdr_threshold=1.0, min_score=0.0),
        )

        spec, count = prepare_library(cfg, tmp_path / "results")
        assert spec.kind == "store"
        assert count == 1
        assert spec.path.exists()
        first_mtime = spec.path.stat().st_mtime_ns

        # Second run reuses the cached store (no rebuild).
        spec2, count2 = prepare_library(cfg, tmp_path / "results")
        assert spec2.path == spec.path
        assert count2 == 1
        assert spec2.path.stat().st_mtime_ns == first_mtime
