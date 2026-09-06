"""
Configuration-strictness tests.

Covers the configuration audit contract:

- misspelled / unknown YAML keys are rejected with human-readable errors
  (line number + spelling suggestion), never silently ignored
- relative paths resolve deterministically against the YAML file's
  directory (not the caller's CWD), with a documented compatibility mode
- defaults, aliases, and optional fields keep their documented behavior
- invalid engine/processing combinations fail at validation time
- the normalized configuration representation (with digest) is written into
  run-level and per-file provenance
- the batch processing path honors the documented filter toggles (no
  silent spectrum drops when a toggle is disabled)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    SimilarityConfig,
)

# ---------------------------------------------------------------------------
# Unknown keys are rejected
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


def test_misspelled_key_rejected_with_suggestion(tmp_path: Path) -> None:
    """``ms2_tolerence`` fails immediately with a spelling suggestion."""
    config_file = _write_config(
        tmp_path,
        "input:\n  input_path: data.mgf\nsimilarity:\n  ms2_tolerence: 0.02\n",
    )
    with pytest.raises(ValueError) as excinfo:
        MassFlowConfig.from_yaml(config_file)
    message = str(excinfo.value)
    assert "ms2_tolerence" in message
    assert "ms2_tolerance" in message  # the suggestion
    assert "Line 4" in message  # the exact YAML line


def test_unknown_root_key_rejected_with_allowed_keys(tmp_path: Path) -> None:
    config_file = _write_config(
        tmp_path,
        "input:\n"
        "  input_path: data.mgf\n"
        "output_directory: results\n",  # legacy root alias does not exist
    )
    with pytest.raises(ValueError) as excinfo:
        MassFlowConfig.from_yaml(config_file)
    message = str(excinfo.value)
    assert "output_directory" in message
    assert "Allowed keys" in message
    assert "input" in message


def test_unknown_nested_key_in_processing_rejected(tmp_path: Path) -> None:
    config_file = _write_config(
        tmp_path,
        "input:\n  input_path: data.mgf\n"
        "processing:\n  min_peak: 5\n",  # typo for min_peaks
    )
    with pytest.raises(ValueError) as excinfo:
        MassFlowConfig.from_yaml(config_file)
    message = str(excinfo.value)
    assert "min_peak" in message
    assert "min_peaks" in message  # suggestion


def test_programmatic_construction_rejects_unknown_keys() -> None:
    """Strictness applies to programmatic construction too."""
    with pytest.raises(Exception):
        MassFlowConfig(
            input=InputConfig(input_path=Path("data.mgf")),
            similarity={"algorithm": "cosine", "min_scor": 0.5},
        )


def test_line_number_loader_marker_does_not_leak(tmp_path: Path) -> None:
    """The internal ``__lines__`` bookkeeping key is never treated as a
    configuration field."""
    config_file = _write_config(
        tmp_path,
        "input:\n  input_path: data.mgf\n"
        "similarity:\n  algorithm: cosine\n  min_score: 0.5\n",
    )
    config = MassFlowConfig.from_yaml(config_file)
    assert config.similarity.min_score == 0.5


# ---------------------------------------------------------------------------
# Relative path handling
# ---------------------------------------------------------------------------


def test_relative_paths_resolve_against_config_dir(tmp_path: Path) -> None:
    """Relative paths resolve against the YAML file's directory, regardless
    of the caller's CWD."""
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    library = tmp_path / "lib" / "library.msp"
    library.parent.mkdir()
    library.write_text("NAME: x\nPRECURSORMZ: 100.0\nNum Peaks: 1\n50.0 1.0\n")
    config_file = _write_config(
        config_dir,
        "project:\n"
        "  output_directory: out\n"
        "input:\n"
        "  input_path: data.mgf\n"
        "  library_path: ../lib/library.msp\n",
    )

    # Run from a completely unrelated CWD.
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    env = dict(os.environ)
    env.pop("MASSFLOW_COMPAT_CWD_PATHS", None)
    script = (
        "from pathlib import Path\n"
        "from MassFlow.config import MassFlowConfig\n"
        f"c = MassFlowConfig.from_yaml({str(config_file)!r})\n"
        "print(c.project.output_directory)\n"
        "print(c.input.input_path)\n"
        "print(c.input.library_path)\n"
        "print(c.config_path)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.strip().splitlines()
    assert Path(lines[0]) == config_dir / "out"
    assert Path(lines[1]) == config_dir / "data.mgf"
    assert Path(lines[2]) == library
    assert Path(lines[3]) == config_file.resolve()


def test_absolute_paths_untouched(tmp_path: Path) -> None:
    config_file = _write_config(
        tmp_path,
        "project:\n"
        f"  output_directory: {tmp_path / 'out'}\n"
        "input:\n"
        f"  input_path: {tmp_path / 'data.mgf'}\n",
    )
    config = MassFlowConfig.from_yaml(config_file)
    assert config.project.output_directory == (tmp_path / "out").resolve()
    assert config.input.input_path == (tmp_path / "data.mgf").resolve()


def test_compat_mode_keeps_cwd_relative_paths(tmp_path: Path) -> None:
    """MASSFLOW_COMPAT_CWD_PATHS=1 restores the legacy CWD-relative behavior."""
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    config_file = _write_config(
        config_dir,
        "project:\n  output_directory: out\ninput:\n  input_path: data.mgf\n",
    )
    env = dict(os.environ)
    env["MASSFLOW_COMPAT_CWD_PATHS"] = "1"
    script = (
        "from pathlib import Path\n"
        "from MassFlow.config import MassFlowConfig\n"
        f"c = MassFlowConfig.from_yaml({str(config_file)!r})\n"
        "print(c.project.output_directory)\n"
        "print(c.input.input_path)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,  # CWD == tmp_path, NOT config_dir
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.strip().splitlines()
    # Legacy behavior: paths stay relative and resolve against the CWD at
    # use time (unchanged from the pre-audit behavior).
    assert Path(lines[0]) == Path("out")
    assert Path(lines[1]) == Path("data.mgf")


def test_programmatic_config_paths_left_untouched() -> None:
    """Direct construction does not silently rewrite paths."""
    config = MassFlowConfig(input=InputConfig(input_path=Path("data.mgf")))
    assert config.input.input_path == Path("data.mgf")
    assert config.project.output_directory == Path("results")
    assert config.config_path is None


# ---------------------------------------------------------------------------
# Defaults, aliases, optional fields
# ---------------------------------------------------------------------------


def test_defaults_unchanged() -> None:
    config = MassFlowConfig(input=InputConfig(input_path=Path("x.mgf")))
    assert config.project.name == "MassFlow_Project"
    assert config.processing.min_peaks == 5
    assert config.processing.filter_by_mz is False
    assert config.processing.filter_min_peaks is False
    assert config.similarity.algorithm == "cosine"
    assert config.similarity.fdr_threshold == 0.01
    assert config.export.format == "csv"
    assert config.input.storage_backend == "sqlite"


def test_alias_reference_library_still_accepted(tmp_path: Path) -> None:
    config_file = _write_config(
        tmp_path,
        "input:\n"
        "  input_path: data.mgf\n"
        "  reference_library: lib.msp\n",  # legacy alias
    )
    config = MassFlowConfig.from_yaml(config_file)
    assert config.input.library_path == (tmp_path / "lib.msp").resolve()


def test_optional_fields_accept_none(tmp_path: Path) -> None:
    config_file = _write_config(
        tmp_path,
        "input:\n  input_path: data.mgf\n  format: null\n"
        "similarity:\n  resolution_ppm: null\n  rt_tolerance: null\n"
        "processing:\n  n_max: null\n",
    )
    config = MassFlowConfig.from_yaml(config_file)
    assert config.input.format is None
    assert config.similarity.resolution_ppm is None
    assert config.similarity.rt_tolerance is None
    assert config.processing.n_max is None


def test_modifications_key_is_a_valid_field(tmp_path: Path) -> None:
    """The documented ``modifications`` section is a real field, not an
    unknown key."""
    config_file = _write_config(
        tmp_path,
        "input:\n  input_path: data.mgf\n"
        "modifications:\n"
        "  pS:\n    formula: HO3P\n    type: aa\n",
    )
    config = MassFlowConfig.from_yaml(config_file)
    assert "pS" in config.modifications


# ---------------------------------------------------------------------------
# Invalid combinations fail at validation
# ---------------------------------------------------------------------------


def test_hnsw_requires_cascade_engine() -> None:
    with pytest.raises(ValueError, match="algorithm='cascade'"):
        SimilarityConfig(algorithm="cosine", hnsw_enabled=True)


def test_top_n_reduction_requires_n_max() -> None:
    with pytest.raises(ValueError, match="n_max"):
        ProcessingConfig(reduce_to_top_n_peaks=True)


def test_consensus_weights_unknown_engine_rejected() -> None:
    with pytest.raises(ValueError, match="consensus_weights"):
        SimilarityConfig(consensus_weights={"cosin": 1.0})


def test_consensus_weights_non_positive_rejected() -> None:
    with pytest.raises(ValueError, match="positive weight"):
        SimilarityConfig(consensus_weights={"cosine": 0.0})


def test_cascade_stages_empty_rejected() -> None:
    with pytest.raises(ValueError, match="cascade_stages"):
        SimilarityConfig(cascade_stages=[])


def test_cascade_stages_unknown_engine_rejected() -> None:
    with pytest.raises(ValueError, match="cascade_stages"):
        SimilarityConfig(cascade_stages=["cosine", "hnsw"])


def test_cascade_zero_upper_bound_is_valid() -> None:
    """cascade_upper_bound=0.0 is the documented 'no final threshold'
    sentinel and may be below cascade_lower_bound."""
    config = SimilarityConfig(
        algorithm="cascade",
        cascade_lower_bound=0.3,
        cascade_upper_bound=0.0,
        cascade_stages=["cosine"],
    )
    assert config.cascade_upper_bound == 0.0


def test_fdr_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="fdr_threshold"):
        SimilarityConfig(fdr_threshold=1.5)


def test_misspelled_engine_name_rejected() -> None:
    with pytest.raises(Exception):
        SimilarityConfig(algorithm="cosin")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Normalized configuration representation
# ---------------------------------------------------------------------------


def test_normalized_config_shape_and_digest(tmp_path: Path) -> None:
    config_file = _write_config(
        tmp_path,
        "input:\n"
        "  input_path: data.mgf\n"
        "  library_path: lib.msp\n"
        "similarity:\n"
        "  algorithm: modified_cosine\n"
        "  fdr_threshold: 0.05\n",
    )
    first = MassFlowConfig.from_yaml(config_file)
    second = MassFlowConfig.from_yaml(config_file)

    norm = first.normalized_config()
    assert norm["schema_version"] == 1
    assert norm["config_file"] == str(config_file.resolve())
    effective = norm["effective_config"]
    # Paths are absolute (post-resolution).
    assert effective["input"]["input_path"] == str((tmp_path / "data.mgf").resolve())
    assert effective["similarity"]["algorithm"] == "modified_cosine"
    # Digest is stable across reloads of the same file.
    assert (
        norm["config_digest_sha256"]
        == second.normalized_config()["config_digest_sha256"]
    )
    # Digest is sensitive to a real change.
    third = first.model_copy(deep=True)
    third.similarity.min_score = 0.9
    assert (
        third.normalized_config()["config_digest_sha256"]
        != norm["config_digest_sha256"]
    )


# ---------------------------------------------------------------------------
# Provenance integration (run-level + per-file)
# ---------------------------------------------------------------------------


def test_run_provenance_written_and_referenced_from_reports(tmp_path: Path) -> None:
    """A real (small) pipeline run writes the normalized configuration into
    the run-level provenance file and the per-file report."""
    from MassFlow.config import MassFlowConfig
    from MassFlow.workflow import run_annotation_pipeline

    # Tiny library (1 spectrum) + a matching query.
    library = tmp_path / "lib.msp"
    library.write_text(
        "NAME: CmpdA\n"
        "PRECURSORMZ: 200.0\n"
        "IONMODE: Positive\n"
        "CHARGE: 1\n"
        "Num Peaks: 3\n"
        "100.0 999.0\n"
        "200.0 500.0\n"
        "300.0 100.0\n"
    )
    query = tmp_path / "query.mgf"
    query.write_text(
        "BEGIN IONS\n"
        "TITLE=query_1\n"
        "PEPMASS=200.0\n"
        "CHARGE=1+\n"
        "100.0 999.0\n"
        "200.0 500.0\n"
        "300.0 100.0\n"
        "END IONS\n"
    )
    config = MassFlowConfig(
        project={"name": "prov", "output_directory": tmp_path / "out"},
        input={"input_path": query, "library_path": library, "format": "mgf"},
        processing={"min_peaks": 1, "filter_min_peaks": False},
        similarity={
            "algorithm": "cosine",
            "min_score": 0.0,
            "fdr_threshold": 1.0,
            "ms1_tolerance": 100.0,
            "ms2_tolerance": 0.5,
            "min_matched_peaks": 1,
        },
    )
    results = run_annotation_pipeline(config)

    assert results and results[0].status == "success"

    # Run-level provenance: environment + normalized config + hashes,
    # written before processing, finalized after.
    provenance_files = sorted(tmp_path.glob("out/run_provenance*.json"))
    assert len(provenance_files) == 1
    run_provenance = json.loads(provenance_files[0].read_text())
    assert run_provenance["schema_version"] == 2
    assert run_provenance["massflow_version"]
    assert run_provenance["python_version"]
    assert run_provenance["effective_config"]["similarity"]["algorithm"] == "cosine"
    assert (
        run_provenance["config_digest_sha256"]
        == config.normalized_config()["config_digest_sha256"]
    )
    # Finalized with the completion summary.
    assert "completed_at" in run_provenance
    assert run_provenance["results"]["files_total"] == 1
    assert run_provenance["results"]["files_succeeded"] == 1

    # Per-file report carries the same normalized config.
    report = tmp_path / "out" / "query_results.report.yaml"
    assert report.exists()
    report_text = report.read_text()
    assert "config:" in report_text
    assert run_provenance["config_digest_sha256"] in report_text

    # A second run in the same output directory does not overwrite.
    run_annotation_pipeline(config)
    assert len(list(tmp_path.glob("out/run_provenance*.json"))) == 2


# ---------------------------------------------------------------------------
# Batch processing honors the documented toggles (no silent drops)
# ---------------------------------------------------------------------------


def _single_peak_spectrum(spec_id: str, precursor_mz: float) -> Spectrum:
    return Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": spec_id, "precursor_mz": precursor_mz, "charge": 1},
    )


def _six_peak_spectrum(spec_id: str, precursor_mz: float) -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0]),
        intensities=np.array([1.0] * 6),
        metadata={"id": spec_id, "precursor_mz": precursor_mz, "charge": 1},
    )


def test_batch_path_keeps_spectra_when_toggles_disabled() -> None:
    """Default toggles (filter_min_peaks=False, filter_by_mz=False) must NOT
    drop spectra — the batch path honors the documented semantics."""
    from MassFlow import processing

    spectra = [
        _single_peak_spectrum("few_peaks", 400.0),  # 1 peak < min_peaks(5)
        _single_peak_spectrum("high_precursor", 1200.0),  # > mz_max(1000)
    ]
    config = ProcessingConfig(min_peaks=5)  # defaults: toggles False
    processed = list(processing.process_spectra(iter(spectra), config))
    assert {s.get("id") for s in processed} == {"few_peaks", "high_precursor"}


def test_batch_path_filters_when_toggles_enabled() -> None:
    """With the toggles on, the batch path applies the same gates as the
    per-spectrum path."""
    from MassFlow import processing

    spectra = [
        _single_peak_spectrum("few_peaks", 400.0),
        _single_peak_spectrum("high_precursor", 1200.0),
        _six_peak_spectrum("ok", 500.0),
    ]
    config = ProcessingConfig(
        min_peaks=5,
        filter_min_peaks=True,
        filter_by_mz=True,
        mz_min=0.0,
        mz_max=1000.0,
        noise_threshold=0.0,
    )
    processed = list(processing.process_spectra(iter(spectra), config))
    assert [s.get("id") for s in processed] == ["ok"]
