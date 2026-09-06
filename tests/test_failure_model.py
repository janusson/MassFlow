"""
Integration tests for the MassFlow failure model.

Contract under test (see docs/user-guide/scoring_logic.md and
``MassFlow.workflow.FileExecutionResult``):

* A scientific analysis must never silently succeed when required data were
  not processed.
* File-level failures are explicit: status="failed" with fatal_errors, a
  ``<stem>_failed.report.yaml`` sidecar, and NO results CSV (an empty CSV
  would be mistaken for a successful annotation).
* Batch processing continues across files without pretending failed files
  succeeded, and the CLI exit status reflects partial failure.
* Unsupported vendor formats are reported, never silently dropped.
* Degraded execution (engine fallback, uncalibrated FDR) is recorded in the
  output provenance.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from MassFlow import cli
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.workflow import run_annotation_pipeline


# ---------------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------------


def write_mgf(path: Path, spectra: list[dict]) -> None:
    """Write a minimal MGF file. Each spectrum dict: id, precursor_mz,
    peaks (list of (mz, intensity) tuples). Omitting ``precursor_mz``
    produces a spectrum the validation layer rejects."""
    lines = []
    for spec in spectra:
        lines.append("BEGIN IONS")
        if spec.get("id"):
            lines.append(f"TITLE={spec['id']}")
        if spec.get("precursor_mz") is not None:
            lines.append(f"PEPMASS={spec['precursor_mz']}")
        lines.append("CHARGE=1+")
        for mz, intensity in spec["peaks"]:
            lines.append(f"{mz} {intensity}")
        lines.append("END IONS")
    path.write_text("\n".join(lines) + "\n")


def write_msp(path: Path, spectra: list[dict]) -> None:
    """Write a minimal MSP library file."""
    lines = []
    for spec in spectra:
        lines.append(f"NAME: {spec['id']}")
        lines.append(f"PRECURSOR_MZ: {spec['precursor_mz']}")
        lines.append("CHARGE: 1")
        lines.append("NUM PEAKS: %d" % len(spec["peaks"]))
        for mz, intensity in spec["peaks"]:
            lines.append(f"{mz}\t{intensity}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def query_spectrum(query_id: str, precursor_mz: float = 100.0) -> dict:
    return {
        "id": query_id,
        "precursor_mz": precursor_mz,
        "peaks": [(100.0, 1.0), (150.0, 2.0), (200.0, 1.0)],
    }


def reference_spectrum(ref_id: str, precursor_mz: float = 100.0) -> dict:
    return {
        "id": ref_id,
        "precursor_mz": precursor_mz,
        "peaks": [(100.0, 1.0), (150.0, 2.0), (200.0, 1.0)],
    }


class _StubEngine:
    """Deterministic engine stub: every query gets one target hit."""

    def search(self, query_spectra, reference_spectra, **kwargs):
        list(reference_spectra)  # consume the (possibly counted) library
        hits = []
        for i, query in enumerate(query_spectra):
            hits.append(
                {
                    "query_id": str(query.get("id", f"query_{i}")),
                    "query_precursor_mz": float(query.get("precursor_mz") or 0.0),
                    "reference_id": "ref_1",
                    "reference_name": "ref_1",
                    "reference_precursor_mz": 100.0,
                    "score": 0.95,
                    "matched_peaks": 3,
                    "smiles": None,
                    "inchikey": None,
                    "is_decoy": False,
                    "q_value": 1.0,
                    "p_value": None,
                    "annotation_tier": None,
                    "structural_similarity": None,
                    "mass_error_ppm": None,
                    "score_breakdown": None,
                }
            )
            hits.append(
                {
                    "query_id": str(query.get("id", f"query_{i}")),
                    "query_precursor_mz": float(query.get("precursor_mz") or 0.0),
                    "reference_id": "ref_1_decoy",
                    "reference_name": "ref_1_decoy",
                    "reference_precursor_mz": 100.0,
                    "score": 0.3,
                    "matched_peaks": 1,
                    "smiles": None,
                    "inchikey": None,
                    "is_decoy": True,
                    "q_value": 1.0,
                    "p_value": None,
                    "annotation_tier": None,
                    "structural_similarity": None,
                    "mass_error_ppm": None,
                    "score_breakdown": None,
                }
            )
        return hits


@pytest.fixture()
def pipeline_config(tmp_path: Path) -> MassFlowConfig:
    """Config pointing at a real library and a per-test input directory.

    ``min_peaks=1`` disables the batch-level peak-count filter so the small
    synthetic fixtures are analyzable.
    """
    library_path = tmp_path / "library.msp"
    write_msp(library_path, [reference_spectrum("ref_1")])
    return MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=tmp_path / "inputs",
            library_path=library_path,
            format="mgf",
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(fdr_threshold=1.0, min_score=0.0),
    )


@pytest.fixture()
def run_pipeline():
    """Run the real pipeline with the stub engine (worker pool stubbed to
    threads so tests run in-process and deterministically)."""
    from concurrent.futures import ThreadPoolExecutor

    def _run(config: MassFlowConfig):
        with patch(
            "MassFlow.workflow.get_similarity_engine", return_value=_StubEngine()
        ):
            with patch("MassFlow.workflow.ProcessPoolExecutor") as mock_executor:
                mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
                    max_workers=4
                )
                return run_annotation_pipeline(config)

    return _run


# ---------------------------------------------------------------------------
# 1. One bad file among five: the other four still succeed
# ---------------------------------------------------------------------------


class TestBatchPartialFailure:
    def test_one_bad_file_among_five(self, tmp_path, pipeline_config, run_pipeline):
        """Four good MGF files + one vendor .raw file: the four are
        processed and exported; the vendor file is an explicit failure and
        does not stop the batch."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        for index in range(4):
            write_mgf(
                inputs / f"good_{index}.mgf",
                [query_spectrum(f"q_{index}")],
            )
        (inputs / "bad_vendor.raw").write_bytes(b"\x00\x01\x02vendor")

        results = run_pipeline(pipeline_config)

        by_path = {str(r.input_path.name): r for r in results}
        assert len(results) == 5
        for index in range(4):
            assert by_path[f"good_{index}.mgf"].status == "success"
            assert by_path[f"good_{index}.mgf"].hits_produced == 1
            out = pipeline_config.project.output_directory / f"good_{index}_results.csv"
            assert out.exists(), "good files must still be exported"

        bad = by_path["bad_vendor.raw"]
        assert bad.status == "failed"
        assert len(bad.fatal_errors) >= 1
        assert (
            "vendor" in bad.fatal_errors[0].lower()
            or "convert" in bad.fatal_errors[0].lower()
        )

    def test_failed_file_has_no_csv_and_explicit_failure_report(
        self, tmp_path, pipeline_config, run_pipeline
    ):
        """No empty CSV may be mistaken for a successful annotation: a failed
        file writes no results CSV and produces a _failed.report.yaml
        carrying the fatal error."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "vendor.raw").write_bytes(b"raw")
        write_mgf(inputs / "good.mgf", [query_spectrum("q1")])

        run_pipeline(pipeline_config)

        out_dir = pipeline_config.project.output_directory
        assert not (out_dir / "vendor_results.csv").exists()
        assert (out_dir / "good_results.csv").exists()

        failure_report = out_dir / "vendor_failed.report.yaml"
        assert failure_report.exists()
        import yaml

        payload = yaml.safe_load(failure_report.read_text())
        assert payload["status"] == "failed"
        assert payload["fatal_errors"]
        assert payload["results_csv"] is None
        assert payload["hits_produced"] == 0

    def test_empty_input_file_is_an_explicit_failure(
        self, tmp_path, pipeline_config, run_pipeline
    ):
        """A file that loads but yields zero analyzable spectra must not be
        reported as a successful zero-hit annotation."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "empty.mgf").write_text("")  # no BEGIN IONS blocks

        results = run_pipeline(pipeline_config)

        assert len(results) == 1
        assert results[0].status == "failed"
        assert "No analyzable spectra" in results[0].fatal_errors[0]
        out_dir = pipeline_config.project.output_directory
        assert not (out_dir / "empty_results.csv").exists()
        assert (out_dir / "empty_failed.report.yaml").exists()

    def test_spectrum_rejections_are_counted(
        self, tmp_path, pipeline_config, run_pipeline
    ):
        """Recoverable spectrum-level issues are explicit: one invalid
        spectrum (missing precursor) among two valid ones is counted in
        spectra_rejected and recorded in the provenance."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        write_mgf(
            inputs / "mixed.mgf",
            [
                query_spectrum("good_1"),
                query_spectrum("good_2"),
                {
                    "id": "no_precursor",
                    "precursor_mz": None,  # rejected by the validation layer
                    "peaks": [(100.0, 1.0)],
                },
            ],
        )

        results = run_pipeline(pipeline_config)

        assert len(results) == 1
        assert results[0].status == "success"
        assert results[0].spectra_loaded == 2
        assert results[0].spectra_rejected == 1
        assert results[0].hits_produced == 2

        import yaml

        report = pipeline_config.project.output_directory / "mixed_results.report.yaml"
        payload = yaml.safe_load(report.read_text())
        assert payload["spectra_loaded"] == 2
        assert payload["spectra_rejected"] == 1
        assert payload["status"] == "success"


# ---------------------------------------------------------------------------
# 2. Vendor-only input: everything is reported, nothing vanishes
# ---------------------------------------------------------------------------


class TestVendorFormats:
    def test_vendor_only_directory_fails_explicitly(
        self, tmp_path, pipeline_config, run_pipeline
    ):
        """A directory containing only vendor files produces explicit failed
        results for every file (previously they were logged and the run
        'succeeded' with no outputs)."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "a.raw").write_bytes(b"raw-a")
        (inputs / "b.wiff").write_bytes(b"raw-b")

        results = run_pipeline(pipeline_config)

        assert len(results) == 2
        assert all(r.status == "failed" for r in results)
        for result in results:
            assert any("vendor" in e.lower() for e in result.fatal_errors), (
                result.fatal_errors
            )
        out_dir = pipeline_config.project.output_directory
        assert not list(out_dir.glob("*_results.csv"))


# ---------------------------------------------------------------------------
# 3. Degraded execution is recorded in provenance
# ---------------------------------------------------------------------------


class TestDegradedExecution:
    def test_engine_fallback_is_recorded_as_degraded(
        self, tmp_path, pipeline_config, run_pipeline
    ):
        """When the configured engine fails and the workflow falls back to
        modified_cosine, the file is DEGRADED and the provenance sidecar
        records the degradation flag."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        write_mgf(inputs / "exp.mgf", [query_spectrum("q1")])

        class _FailingEngine:
            def search(self, *args, **kwargs):
                raise ConnectionError("ML service unreachable")

        from concurrent.futures import ThreadPoolExecutor

        with patch(
            "MassFlow.workflow.get_similarity_engine", return_value=_FailingEngine()
        ):
            with patch("MassFlow.workflow.ProcessPoolExecutor") as mock_executor:
                mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
                    max_workers=1
                )
                results = run_annotation_pipeline(pipeline_config)

        assert len(results) == 1
        assert results[0].status == "degraded"
        assert any(
            flag.startswith("engine_fallback:")
            for flag in results[0].degraded_mode_flags
        )

        import yaml

        report = pipeline_config.project.output_directory / "exp_results.report.yaml"
        payload = yaml.safe_load(report.read_text())
        assert payload["status"] == "degraded"
        assert any(
            flag.startswith("engine_fallback:")
            for flag in payload["degraded_mode_flags"]
        )

    def test_successful_file_has_empty_failure_fields(
        self, tmp_path, pipeline_config, run_pipeline
    ):
        """A clean run has no fatal errors and no degradation flags — the
        provenance distinguishes clean from degraded from failed."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        write_mgf(inputs / "clean.mgf", [query_spectrum("q1")])

        results = run_pipeline(pipeline_config)

        assert results[0].status == "success"
        assert results[0].fatal_errors == []
        assert results[0].degraded_mode_flags == []
        assert results[0].output_path is not None


# ---------------------------------------------------------------------------
# 4. CLI exit status reflects partial failure
# ---------------------------------------------------------------------------


class TestCliExitStatus:
    def _cli_config(self, tmp_path: Path, input_path: Path) -> Path:
        library_path = tmp_path / "library.msp"
        write_msp(library_path, [reference_spectrum("ref_1")])
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""
project:
  name: "failure_model_test"
  output_directory: "{tmp_path / "results"}"

input:
  input_path: "{input_path}"
  library_path: "{library_path}"
  format: "mgf"

processing:
  min_peaks: 1

similarity:
  algorithm: "cosine"
  min_score: 0.0
  fdr_threshold: 1.0
"""
        )
        return config_path

    def test_all_good_files_exit_zero(self, tmp_path):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        write_mgf(inputs / "ok.mgf", [query_spectrum("q1")])
        config_path = self._cli_config(tmp_path, inputs)

        runner = CliRunner()
        with patch(
            "MassFlow.workflow.get_similarity_engine", return_value=_StubEngine()
        ):
            result = runner.invoke(cli.app, ["annotate", "--config", str(config_path)])

        assert result.exit_code == 0
        assert "Annotation complete" in result.output

    def test_single_vendor_file_exits_nonzero(self, tmp_path):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "vendor.raw").write_bytes(b"raw")
        config_path = self._cli_config(tmp_path, inputs)

        runner = CliRunner()
        result = runner.invoke(cli.app, ["annotate", "--config", str(config_path)])

        assert result.exit_code == 1
        assert "FAILED" in result.output
        assert "vendor" in result.output.lower()

    def test_partial_batch_failure_exits_nonzero(self, tmp_path):
        """Four good files + one vendor file: the run completes the batch
        but exits 1 and reports the failed file explicitly."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        for index in range(4):
            write_mgf(inputs / f"good_{index}.mgf", [query_spectrum(f"q{index}")])
        (inputs / "vendor.raw").write_bytes(b"raw")
        config_path = self._cli_config(tmp_path, inputs)

        runner = CliRunner()
        from concurrent.futures import ThreadPoolExecutor

        with patch(
            "MassFlow.workflow.get_similarity_engine", return_value=_StubEngine()
        ):
            with patch("MassFlow.workflow.ProcessPoolExecutor") as mock_executor:
                mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
                    max_workers=4
                )
                result = runner.invoke(
                    cli.app, ["annotate", "--config", str(config_path)]
                )

        assert result.exit_code == 1
        assert "FAILED" in result.output
        assert "vendor.raw" in result.output
        # The good files are still reported as processed.
        assert "good_0.mgf" in result.output
        # And their CSVs exist.
        results_dir = tmp_path / "results"
        assert (results_dir / "good_0_results.csv").exists()
        assert (results_dir / "good_3_results.csv").exists()


# ---------------------------------------------------------------------------
# 5. Worker-crash synthesis (batch robustness without silent success)
# ---------------------------------------------------------------------------


class TestWorkerCrash:
    def test_crashed_worker_produces_failed_result_and_batch_continues(
        self, tmp_path, pipeline_config
    ):
        """If a worker raises outside _process_single_file, the batch records
        an explicit failed result for that file and keeps going."""
        from concurrent.futures import ThreadPoolExecutor

        inputs = tmp_path / "inputs"
        inputs.mkdir()
        write_mgf(inputs / "a.mgf", [query_spectrum("qa")])
        write_mgf(inputs / "b.mgf", [query_spectrum("qb")])

        def _boom(query_file, config, library_size=None, library_spec=None):
            raise RuntimeError("worker died")

        with patch(
            "MassFlow.workflow.get_similarity_engine", return_value=_StubEngine()
        ):
            with patch("MassFlow.workflow.ProcessPoolExecutor") as mock_executor:
                mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
                    max_workers=2
                )
                with patch("MassFlow.workflow._process_single_file", side_effect=_boom):
                    results = run_annotation_pipeline(pipeline_config)

        assert len(results) == 2
        assert all(r.status == "failed" for r in results)
        assert all(any("worker_crash" in e for e in r.fatal_errors) for r in results)
        out_dir = pipeline_config.project.output_directory
        assert not list(out_dir.glob("*_results.csv"))
        assert (out_dir / "a_failed.report.yaml").exists()
        assert (out_dir / "b_failed.report.yaml").exists()
