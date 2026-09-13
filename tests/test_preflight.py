"""
Pre-flight sanity checks: MassFlow must fail fast with clear, human-readable
errors BEFORE any expensive library-store build or search starts.

Contract under test (see docs/user-guide/annotation.md, "Pre-flight
validation", and ``MassFlow.workflow.preflight_annotation_run``):

* A run whose similarity surface needs missing optional extras (e.g.
  ``spec2vec`` without ``massflow[ml]``) aborts at pre-flight with the
  install instruction — never as a late per-file failure or an opaque worker
  crash after the library store was built.
* The reference library must be configured, exist, and be a loadable
  open-format/store input; vendor raw formats and unknown extensions abort
  before any store or output directory is created.
* Query inputs must exist and be dispatchable. Vendor raw files inside an
  *input directory* keep the documented batch semantics (explicit per-file
  failed results, batch continues) and are announced up-front; a vendor raw
  *direct* input (file or ``.d`` directory) aborts.
* Missing optional extras that degrade gracefully (``consensus``/``cascade``
  without ``[ml]``, ``cascade`` + ``hnsw_enabled`` without ``[hnsw]``) are
  reported as run-start warnings with the plain-English install fix.
* The CLI prints ``fix:`` hints (MassFlow.tui.diagnostics) instead of raw
  tracebacks, and ``db build``/``db merge`` validate their inputs before the
  output store file is created.
"""

from concurrent.futures import ThreadPoolExecutor
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
from MassFlow.workflow import (
    optional_extra_warnings,
    preflight_annotation_run,
    run_annotation_pipeline,
)


# ---------------------------------------------------------------------------
# Fixture writers (mirrors tests/test_failure_model.py)
# ---------------------------------------------------------------------------


def write_mgf(path: Path, spectra: list[dict]) -> None:
    """Write a minimal MGF file."""
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
    """Config pointing at a real library and a per-test input directory."""
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
    """Run the real pipeline in-process with the stub engine."""

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


def _write_cli_config(tmp_path: Path, **overrides) -> Path:
    """Write a minimal annotate config YAML with absolute paths."""
    values = {
        "output_directory": str(tmp_path / "results"),
        "input_path": str(tmp_path / "inputs"),
        "library_path": str(tmp_path / "library.msp"),
        "format": "mgf",
        "algorithm": "cosine",
    }
    values.update(overrides)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
project:
  name: "preflight_test"
  output_directory: "{values["output_directory"]}"

input:
  input_path: "{values["input_path"]}"
  library_path: "{values["library_path"]}"
  format: "{values["format"]}"

processing:
  min_peaks: 1

similarity:
  algorithm: "{values["algorithm"]}"
  fdr_threshold: 1.0
  min_score: 0.0
"""
    )
    return config_path


# ---------------------------------------------------------------------------
# 1. Reference library problems abort before any store/output exists
# ---------------------------------------------------------------------------


def test_vendor_raw_library_aborts_with_no_artifacts(tmp_path):
    """A vendor raw reference library fails pre-flight: nothing is built and
    not even the output directory is created."""
    library_path = tmp_path / "vendor_library.raw"
    library_path.write_bytes(b"\x00vendor")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_mgf(inputs / "q.mgf", [query_spectrum("q1")])

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=inputs,
            library_path=library_path,
            format="mgf",
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )

    with pytest.raises(Exception) as excinfo:
        run_annotation_pipeline(config)

    from MassFlow.io import UnsupportedVendorFormatError

    assert isinstance(excinfo.value, UnsupportedVendorFormatError)
    assert "vendor" in str(excinfo.value).lower()
    # No output directory, no temporary store: the failure happened before
    # any expensive work or side effect.
    assert not (tmp_path / "results").exists()


def test_missing_library_path_aborts_before_input_checks(tmp_path):
    """The library is validated before the input path, matching the previous
    prepare_library-first error ordering — but now with zero artifacts."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_mgf(inputs / "q.mgf", [query_spectrum("q1")])

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=inputs,
            library_path=tmp_path / "does_not_exist.msp",
            format="mgf",
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )

    with pytest.raises(ValueError, match="Library path does not exist"):
        run_annotation_pipeline(config)
    assert not (tmp_path / "results").exists()


def test_unsupported_library_extension_aborts_with_no_artifacts(tmp_path):
    """A library the loader cannot dispatch (e.g. .txt) fails before a temp
    store is created."""
    (tmp_path / "library.txt").write_text("not a library")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_mgf(inputs / "q.mgf", [query_spectrum("q1")])

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=inputs,
            library_path=tmp_path / "library.txt",
            format="mgf",
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )

    with pytest.raises(ValueError, match="not supported by MassFlow"):
        run_annotation_pipeline(config)
    assert not (tmp_path / "results").exists()


# ---------------------------------------------------------------------------
# 2. Query input problems
# ---------------------------------------------------------------------------


def test_missing_input_path_fails_before_any_store(tmp_path):
    """A missing input path aborts pre-flight (same message as before, but
    before the library store is built and with no output artifacts)."""
    library_path = tmp_path / "library.msp"
    write_msp(library_path, [reference_spectrum("ref_1")])

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=tmp_path / "missing_inputs",
            library_path=library_path,
            format="mgf",
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )

    with pytest.raises(ValueError, match="Input path does not exist"):
        run_annotation_pipeline(config)
    assert not (tmp_path / "results").exists()


def test_single_vendor_query_file_aborts_preflight(tmp_path):
    """A direct single-file vendor query input aborts the run (documented
    semantics) before the library store is built."""
    vendor_file = tmp_path / "query.raw"
    vendor_file.write_bytes(b"\x00raw")
    library_path = tmp_path / "library.msp"
    write_msp(library_path, [reference_spectrum("ref_1")])

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=vendor_file,
            library_path=library_path,
            format="mgf",
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )

    from MassFlow.io import UnsupportedVendorFormatError

    with pytest.raises(UnsupportedVendorFormatError):
        run_annotation_pipeline(config)
    assert not (tmp_path / "results").exists()


def test_vendor_dot_d_directory_input_aborts_preflight(tmp_path):
    """Pointing the query input at a Bruker/Agilent ``.d`` directory directly
    is a vendor raw input, not a scan root."""
    vendor_dir = tmp_path / "acquisition.d"
    vendor_dir.mkdir()
    library_path = tmp_path / "library.msp"
    write_msp(library_path, [reference_spectrum("ref_1")])

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=vendor_dir,
            library_path=library_path,
            format="mgf",
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )

    from MassFlow.io import UnsupportedVendorFormatError

    with pytest.raises(UnsupportedVendorFormatError):
        run_annotation_pipeline(config)
    assert not (tmp_path / "results").exists()


def test_directory_vendor_files_are_announced_but_batch_continues(
    tmp_path, pipeline_config, run_pipeline, caplog
):
    """Vendor raw files inside an input directory keep the documented batch
    semantics (explicit per-file failure, good files still processed) — but
    the pre-flight announces them up-front with the conversion hint."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_mgf(inputs / "good.mgf", [query_spectrum("q1")])
    (inputs / "vendor.raw").write_bytes(b"\x00raw")

    with caplog.at_level("WARNING", logger="MassFlow.workflow"):
        results = run_pipeline(pipeline_config)

    by_path = {r.input_path.name: r for r in results}
    assert set(by_path) == {"good.mgf", "vendor.raw"}
    assert by_path["good.mgf"].status == "success"
    assert by_path["vendor.raw"].status == "failed"
    assert "PRE-FLIGHT" in caplog.text
    assert "vendor" in caplog.text.lower()
    assert "convert" in caplog.text.lower()


# ---------------------------------------------------------------------------
# 3. Missing optional extras: hard failures at pre-flight, warnings otherwise
# ---------------------------------------------------------------------------


def test_missing_ml_engine_aborts_preflight(tmp_path, monkeypatch):
    """A pure ML engine without the [ml] extra aborts BEFORE any library
    store is built, with the plain-English install message."""
    from MassFlow import similarity as similarity_module

    monkeypatch.setattr(similarity_module, "_HAS_ML", False)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_mgf(inputs / "q.mgf", [query_spectrum("q1")])
    write_msp(tmp_path / "library.msp", [reference_spectrum("ref_1")])

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=inputs,
            library_path=tmp_path / "library.msp",
            format="mgf",
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(algorithm="spec2vec"),
    )

    with pytest.raises(RuntimeError, match="machine-learning extras") as excinfo:
        run_annotation_pipeline(config)
    assert "massflow[ml]" in str(excinfo.value)
    assert not (tmp_path / "results").exists()


def test_optional_extra_warnings_meta_engines_without_ml(monkeypatch):
    """consensus/cascade without [ml] degrade to classical sub-engines; the
    warning carries the install fix."""
    from MassFlow import similarity as similarity_module

    monkeypatch.setattr(similarity_module, "_HAS_ML", False)
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=Path("out")),
        input=InputConfig(input_path=Path("q.mgf"), library_path=Path("lib.msp")),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(algorithm="consensus"),
    )
    warnings = optional_extra_warnings(config)
    assert any("massflow[ml]" in w for w in warnings)


def test_optional_extra_warnings_cascade_hnsw_without_hnswlib(monkeypatch, tmp_path):
    """cascade + hnsw_enabled without the [hnsw] extra warns with the fix;
    exact scoring still runs (graceful degradation)."""
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def _no_hnswlib(name):
        if name == "hnswlib":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", _no_hnswlib)
    # Pre-flight requires real paths for the library and query input.
    (tmp_path / "q.mgf").touch()
    (tmp_path / "lib.msp").touch()
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "out"),
        input=InputConfig(
            input_path=tmp_path / "q.mgf", library_path=tmp_path / "lib.msp"
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(algorithm="cascade", hnsw_enabled=True),
    )
    warnings = optional_extra_warnings(config)
    assert any("massflow[hnsw]" in w for w in warnings)

    # Pre-flight itself does not raise for the graceful-degradation case.
    preflight_annotation_run(config)


def test_no_optional_extra_warnings_for_stable_config(tmp_path):
    """A pure stable configuration produces no optional-extra warnings."""
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "out"),
        input=InputConfig(
            input_path=tmp_path / "q.mgf", library_path=tmp_path / "lib.msp"
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(algorithm="cosine"),
    )
    assert optional_extra_warnings(config) == []


# ---------------------------------------------------------------------------
# 4. CLI surfaces: human-readable failures with fix hints, no tracebacks
# ---------------------------------------------------------------------------


def test_cli_annotate_vendor_library_fails_with_fix_hint(tmp_path):
    """A vendor raw library in the config fails at the CLI with the
    plain-English conversion fix, not a traceback."""
    (tmp_path / "library.raw").write_bytes(b"\x00vendor")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_mgf(inputs / "q.mgf", [query_spectrum("q1")])
    config_path = _write_cli_config(
        tmp_path, library_path=str(tmp_path / "library.raw")
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["annotate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Annotation failed" in result.output
    assert "vendor" in result.output.lower()
    assert "convert" in result.output.lower()
    assert "fix:" in result.output
    assert "Traceback" not in result.output
    # Nothing was written: the failure is pre-flight.
    assert not (tmp_path / "results").exists()


def test_cli_annotate_missing_input_path_fails_fast(tmp_path):
    """A nonexistent input path fails before the library store is built."""
    write_msp(tmp_path / "library.msp", [reference_spectrum("ref_1")])
    config_path = _write_cli_config(
        tmp_path, input_path=str(tmp_path / "missing_inputs")
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["annotate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Input path does not exist" in result.output
    assert not (tmp_path / "results").exists()


def test_cli_annotate_missing_ml_extra_prints_install_fix(tmp_path, monkeypatch):
    """Requesting spec2vec without [ml] prints the install fix at the CLI."""
    from MassFlow import similarity as similarity_module

    monkeypatch.setattr(similarity_module, "_HAS_ML", False)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_mgf(inputs / "q.mgf", [query_spectrum("q1")])
    write_msp(tmp_path / "library.msp", [reference_spectrum("ref_1")])
    config_path = _write_cli_config(tmp_path, algorithm="spec2vec")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["annotate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "machine-learning" in result.output
    assert "massflow[ml]" in result.output
    assert "fix:" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "results").exists()


def test_cli_db_build_vendor_input_fails_before_store_creation(tmp_path):
    """`db build` on a vendor raw input fails before the output .db exists."""
    (tmp_path / "library.raw").write_bytes(b"\x00vendor")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'input:\n  input_path: "{tmp_path}"\n')
    output_db = tmp_path / "output.db"

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "build",
            "--input",
            str(tmp_path / "library.raw"),
            "--output",
            str(output_db),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 1
    assert "vendor" in result.output.lower()
    assert "convert" in result.output.lower()
    assert "fix:" in result.output
    assert not output_db.exists()


def test_cli_db_build_unsupported_input_fails_before_store_creation(tmp_path):
    """`db build` on an unloadable input fails before the output store
    exists (previously an empty .db was created first)."""
    (tmp_path / "library.txt").write_text("not spectra")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'input:\n  input_path: "{tmp_path}"\n')
    output_db = tmp_path / "output.db"

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "build",
            "--input",
            str(tmp_path / "library.txt"),
            "--output",
            str(output_db),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 1
    assert "unsupported" in result.output.lower()
    assert not output_db.exists()


def test_cli_db_merge_missing_input_fails_before_store_creation(tmp_path):
    """`db merge` with a missing input fails before the output store exists."""
    output_db = tmp_path / "merged.db"

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "merge",
            "--inputs",
            str(tmp_path / "missing.db"),
            "--output",
            str(output_db),
        ],
    )

    assert result.exit_code == 1
    assert "Input database does not exist" in result.output
    assert "fix:" in result.output
    assert not output_db.exists()


def test_cli_db_merge_text_input_rejected(tmp_path):
    """`db merge` accepts only MassFlow stores as inputs."""
    (tmp_path / "library.msp").write_text("NAME: x\n")
    output_db = tmp_path / "merged.db"

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "merge",
            "--inputs",
            str(tmp_path / "library.msp"),
            "--output",
            str(output_db),
        ],
    )

    assert result.exit_code == 1
    assert "not a MassFlow database" in result.output
    assert not output_db.exists()
