"""
First-class run provenance tests.

Every completed annotation run records: MassFlow version, git SHA, Python
version, dependency versions, the committed lockfile digest, the normalized
configuration, input file hashes, the reference-library digest, the decoy
seed, engine/processing configuration, the storage backend, timestamps, and
the aggregated warnings / degraded modes.

Provenance is deterministic: two runs from the same checkout and environment
over the same inputs produce byte-identical records except for the
explicitly time-varying fields (``run_started_at``, ``completed_at``).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from MassFlow.config import MassFlowConfig


def _write_mini_fixture(tmp_path: Path) -> tuple[MassFlowConfig, Path, Path]:
    """A tiny library + query, returning (config, library_path, query_path)."""
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
    return config, library, query


def _read_run_provenance(tmp_path: Path) -> dict:
    from MassFlow.workflow import run_annotation_pipeline

    config, _, _ = _write_mini_fixture(tmp_path)
    results = run_annotation_pipeline(config)
    assert results and results[0].status == "success"
    provenance_files = sorted(tmp_path.glob("out/run_provenance*.json"))
    assert len(provenance_files) == 1
    return json.loads(provenance_files[0].read_text())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


def test_run_provenance_records_all_required_fields(tmp_path: Path) -> None:
    """The run provenance contains every field required by the contract."""
    _, library, query = _write_mini_fixture(tmp_path)
    provenance = _read_run_provenance(tmp_path)

    # Version / environment.
    assert provenance["schema_version"] == 2
    assert provenance["massflow_version"]  # e.g. "0.1.0"
    assert provenance["python_version"]
    assert provenance["python_implementation"]
    assert provenance["platform"]
    assert isinstance(provenance["dependencies"], dict) and provenance["dependencies"]
    assert provenance["lockfile_digest_sha256"]

    # git identity.
    assert provenance["git_sha"]
    assert provenance["git_dirty"] in (True, False)

    # Configuration (normalized + digest + engine/processing/backend).
    assert provenance["effective_config"]["similarity"]["algorithm"] == "cosine"
    assert provenance["config_digest_sha256"]
    assert provenance["engine"]["algorithm"] == "cosine"
    assert provenance["processing"]["min_peaks"] == 1
    assert provenance["backend"] == "sqlite"

    # Decoy seed + decoy configuration.
    assert provenance["decoy_seed"] == 42
    assert provenance["decoy_config"]["min_relative_intensity"] == 0.01
    assert provenance["decoy_config"]["mz_shift_da"] == 1.0

    # Input hashes: one entry for the query file, matching its content.
    assert provenance["input_file_hashes"] == {
        str(query): f"file:{_sha256_file(query)}"
    }

    # Reference-library digest.
    assert provenance["reference_library_path"] == str(library)
    assert provenance["reference_library_kind"] == "file"
    assert provenance["reference_library_sha256"] == _sha256_file(library)

    # Timestamps + completion summary (warnings / degraded modes).
    assert provenance["run_started_at"]
    assert provenance["completed_at"]
    assert provenance["results"]["files_total"] == 1
    assert provenance["results"]["files_succeeded"] == 1
    assert provenance["results"]["files_failed"] == 0
    assert provenance["results"]["spectra_loaded_total"] >= 1
    assert provenance["results"]["hits_produced_total"] >= 0
    assert isinstance(provenance["results"]["warnings"], list)
    assert isinstance(provenance["results"]["degraded_mode_flags"], list)
    assert provenance["results"]["failed_files"] == []


def test_input_hash_matches_file_content(tmp_path: Path) -> None:
    config, _, query = _write_mini_fixture(tmp_path)
    from MassFlow.workflow import _file_sha256, _path_digest

    assert _file_sha256(query) == _sha256_file(query)
    kind, digest = _path_digest(query)
    assert kind == "file"
    assert digest == _sha256_file(query)

    # Directory manifest digest is stable for identical contents.
    dir_a = tmp_path / "dira"
    dir_b = tmp_path / "dirb"
    for directory in (dir_a, dir_b):
        directory.mkdir()
        (directory / "a.mgf").write_text("x")
        (directory / "sub").mkdir()
        (directory / "sub" / "b.msp").write_text("y" * 100)
    kind_a, digest_a = _path_digest(dir_a)
    kind_b, digest_b = _path_digest(dir_b)
    assert kind_a == kind_b == "directory"
    assert digest_a == digest_b

    # Content change changes the digest.
    (dir_a / "a.mgf").write_text("changed")
    _, digest_a2 = _path_digest(dir_a)
    assert digest_a2 != digest_a


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_git_sha_matches_checkout(tmp_path: Path) -> None:
    """The recorded git SHA is the actual checkout HEAD."""
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    provenance = _read_run_provenance(tmp_path)
    assert provenance["git_sha"] == git_sha


def test_lockfile_digest_matches_committed_lockfile(tmp_path: Path) -> None:
    from MassFlow.workflow import _repo_root

    root = _repo_root()
    assert root is not None
    lockfile = root / "uv.lock"
    assert lockfile.is_file()
    provenance = _read_run_provenance(tmp_path)
    assert provenance["lockfile_digest_sha256"] == _sha256_file(lockfile)


def test_provenance_deterministic_except_time_fields(tmp_path: Path) -> None:
    """Two identical runs produce identical provenance except for the
    explicitly time-varying fields."""
    from MassFlow.workflow import run_annotation_pipeline

    config, _, _ = _write_mini_fixture(tmp_path)

    def _run() -> dict:
        results = run_annotation_pipeline(config)
        assert results and results[0].status == "success"
        files = sorted((tmp_path / "out").glob("run_provenance*.json"))
        return json.loads(files[-1].read_text())

    first_provenance = _run()
    second_provenance = _run()

    # The time-varying fields exist.
    assert first_provenance["run_started_at"]
    assert first_provenance["completed_at"]

    def _static(payload: dict) -> dict:
        return {
            key: value
            for key, value in payload.items()
            if key not in ("run_started_at", "completed_at")
        }

    # Identical inputs + configuration + environment → identical records
    # (including input hashes, digests, and the results summary).
    assert _static(first_provenance) == _static(second_provenance)
    assert (
        first_provenance["config_digest_sha256"]
        == second_provenance["config_digest_sha256"]
    )
    assert first_provenance["results"] == second_provenance["results"]


def test_provenance_results_summary_records_warnings_and_degraded(
    tmp_path: Path,
) -> None:
    """A degraded file's warnings and flags are aggregated into the run
    record."""
    from MassFlow.workflow import run_annotation_pipeline

    # Library lives OUTSIDE the scanned data directory so discovery only
    # sees the query files.
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    library = library_dir / "lib.msp"
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
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    query = data_dir / "query.mgf"
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
    bad_file = data_dir / "vendor.raw"
    bad_file.write_bytes(b"\x00\x01\x02vendor raw data")

    config = MassFlowConfig(
        project={"name": "prov", "output_directory": tmp_path / "out"},
        input={"input_path": data_dir, "library_path": library, "format": None},
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
    statuses = {r.status for r in results}
    assert "failed" in statuses

    provenance_files = sorted(tmp_path.glob("out/run_provenance*.json"))
    assert len(provenance_files) == 1
    provenance = json.loads(provenance_files[0].read_text())
    summary = provenance["results"]
    assert summary["files_total"] == 2
    assert summary["files_failed"] == 1
    assert summary["files_succeeded"] == 1
    assert len(summary["failed_files"]) == 1
    assert summary["failed_files"][0]["input_path"].endswith("vendor.raw")
    assert summary["failed_files"][0]["fatal_errors"]
    # The successful file's hash is still recorded.
    assert any(str(query) in p for p in provenance["input_file_hashes"])
