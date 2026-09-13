"""
Tests for library-build provenance and result lineage.

Contract under test:

* SQLite-backed stores record a lightweight provenance history
  (``store_meta`` + ``library_builds``): input files used (with content
  hashes), the effective config hash, processing parameters, similarity /
  target-decoy configuration, and exact build timestamps. The schema is
  additive and self-upgrading on open (``PRAGMA user_version``).
* ``massflow db build`` / ``db merge`` / ``db inspect`` record and surface
  that history as a user-visible query surface.
* The YAML provenance sidecar produced by ``massflow annotate`` links each
  result back to the exact database store and its build-history row
  (``library.build.id`` / ``config_digest_sha256``), so lineage is persisted
  and queryable end to end.
* Recording provenance never alters scientific payload bytes: plain
  ``add_spectra`` writes create no history rows and Zarr stores are
  untouched.
"""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp
from typer.testing import CliRunner

from MassFlow import cli
from MassFlow.config import MassFlowConfig
from MassFlow.database import (
    LIBRARY_BUILDS_TABLE,
    PROVENANCE_SCHEMA_VERSION,
    STORE_META_TABLE,
    SpectralDatabase,
)
from MassFlow.library import processing_fingerprint
from MassFlow.workflow import run_annotation_pipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _spectrum(spec_id: str, precursor_mz: float) -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, precursor_mz], dtype=np.float64),
        intensities=np.array([0.5, 1.0], dtype=np.float64),
        metadata={
            "id": spec_id,
            "compound_name": spec_id,
            "precursor_mz": precursor_mz,
            "charge": 1,
        },
    )


def _write_msp(path: Path, count: int = 4, prefix: str = "ref") -> None:
    save_as_msp(
        [_spectrum(f"{prefix}_{i}", 195.0 + i) for i in range(count)], str(path)
    )


def _write_mgf(path: Path, count: int = 2, prefix: str = "query") -> None:
    save_as_mgf(
        [_spectrum(f"{prefix}_{i}", 195.0 + i) for i in range(count)], str(path)
    )


def _write_config_yaml(
    directory: Path, input_path: Path, library_path: Path, output_directory: Path
) -> Path:
    """Write ``config.yaml`` inside ``directory`` and return its path."""
    config_path = directory / "config.yaml"
    payload = {
        "project": {"name": "lineage_test", "output_directory": str(output_directory)},
        "input": {
            "input_path": str(input_path),
            "library_path": str(library_path),
            "format": "mgf",
        },
        "processing": {
            "min_peaks": 1,
            "noise_threshold": 0.0,
            "decoy_min_relative_intensity": 0.02,
            "decoy_mz_shift_da": 1.5,
        },
        "similarity": {
            "algorithm": "cosine",
            "min_score": 0.0,
            "fdr_threshold": 0.05,
            "min_matched_peaks": 1,
            "ms1_tolerance": 100.0,
        },
    }
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Database-level: provenance schema, recording, querying, self-upgrade
# ---------------------------------------------------------------------------


class TestDatabaseProvenanceSchema:
    def test_fresh_store_records_schema_and_created_at(self, tmp_path):
        db = SpectralDatabase(tmp_path / "lib.db")
        try:
            assert db.get_schema_version() == PROVENANCE_SCHEMA_VERSION
            created_at = db.get_store_created_at()
            assert created_at is not None
            # Exact first-open timestamp, ISO-8601 with timezone.
            datetime.fromisoformat(created_at)
            assert db.get_library_builds() == []
            assert db.get_latest_library_build() is None
        finally:
            db.close()

        # Reopening is idempotent: no duplicate meta rows, no version bumps.
        db2 = SpectralDatabase(tmp_path / "lib.db")
        try:
            assert db2.get_schema_version() == PROVENANCE_SCHEMA_VERSION
            assert db2.get_store_created_at() == created_at
        finally:
            db2.close()

    def test_record_library_build_stores_full_lineage(self, tmp_path):
        source = tmp_path / "library.msp"
        _write_msp(source)
        db = SpectralDatabase(tmp_path / "lib.db")
        try:
            build_id = db.record_library_build(
                source_path=source,
                spectrum_count=4,
                category="personal",
                storage_backend="sqlite",
                processing_config={"min_peaks": 5, "decoy_mz_shift_da": 1.5},
                similarity_config={"algorithm": "cosine", "fdr_threshold": 0.01},
                processing_fingerprint="fp-abc",
                config_digest_sha256="d" * 64,
                details={"note": "unit test"},
            )
            assert build_id == 1
            second_id = db.record_library_build(
                source_path=None,
                spectrum_count=4,
                category="merged",
                storage_backend="sqlite",
                details={"merged_sources": [{"path": "a.db", "spectra_added": 2}]},
            )
            assert second_id == 2
        finally:
            db.close()

        db = SpectralDatabase(tmp_path / "lib.db")
        try:
            builds = db.get_library_builds()
            # Newest first.
            assert [b["id"] for b in builds] == [2, 1]
            assert db.get_latest_library_build()["id"] == 2
            assert db.get_latest_library_build()["category"] == "merged"

            first = builds[1]
            assert first["source_path"] == str(source)
            assert first["source_sha256"] == _file_sha256(source)
            assert first["source_size"] == source.stat().st_size
            assert first["category"] == "personal"
            assert first["spectrum_count"] == 4
            assert first["storage_backend"] == "sqlite"
            # Configurations round-trip as parsed objects.
            assert first["processing_config"] == {
                "min_peaks": 5,
                "decoy_mz_shift_da": 1.5,
            }
            assert first["similarity_config"] == {
                "algorithm": "cosine",
                "fdr_threshold": 0.01,
            }
            assert first["processing_fingerprint"] == "fp-abc"
            assert first["config_digest_sha256"] == "d" * 64
            assert first["details"] == {"note": "unit test"}
            # Exact timestamp.
            datetime.fromisoformat(first["built_at"])
            assert first["massflow_version"]  # resolved installed version

            # Merge-style rows carry no source file but keep details.
            assert builds[0]["source_path"] is None
            assert builds[0]["processing_config"] is None
            assert builds[0]["details"] == {
                "merged_sources": [{"path": "a.db", "spectra_added": 2}]
            }
        finally:
            db.close()

    def test_pre_provenance_database_upgrades_in_place(self, tmp_path):
        """A database created before the provenance schema existed is
        upgraded on its next open without touching stored spectra."""
        db_path = tmp_path / "pre.db"
        db = SpectralDatabase(db_path)
        db.add_spectra(iter([_spectrum("keep_me", 200.0)]), category="library")
        db.close()

        # Simulate a pre-provenance database: drop the provenance tables and
        # reset user_version (spectra data stays untouched).
        connection = sqlite3.connect(db_path)
        connection.execute(f"DROP TABLE IF EXISTS {LIBRARY_BUILDS_TABLE}")
        connection.execute(f"DROP TABLE IF EXISTS {STORE_META_TABLE}")
        connection.execute("PRAGMA user_version = 0")
        connection.commit()
        connection.close()

        db = SpectralDatabase(db_path)
        try:
            assert db.get_schema_version() == PROVENANCE_SCHEMA_VERSION
            assert db.get_store_created_at() is not None
            assert db.get_total_spectra_count() == 1
            spectrum = db.get_spectrum_by_id("keep_me")
            assert spectrum is not None
            assert spectrum.get("precursor_mz") == 200.0
        finally:
            db.close()

    def test_plain_add_spectra_creates_no_history_rows(self, tmp_path):
        """Provenance is explicit: plain spectrum writes never fabricate a
        build event (scientific payloads are unaffected)."""
        db = SpectralDatabase(tmp_path / "lib.db")
        try:
            db.add_spectra(iter([_spectrum("s1", 200.0)]), category="library")
            assert db.get_library_builds() == []
        finally:
            db.close()


# ---------------------------------------------------------------------------
# CLI-level: db build / db merge / db inspect record and query lineage
# ---------------------------------------------------------------------------


class TestCliBuildLineage:
    def _build_via_cli(
        self, directory: Path, config_yaml: Path, *, library_count: int = 4
    ) -> Path:
        """Write ``library.msp`` in ``directory`` and build it via the CLI."""
        library = directory / "library.msp"
        _write_msp(library, count=library_count)
        db_path = directory / "user_library.db"
        runner = CliRunner()
        result = runner.invoke(
            cli.app,
            [
                "db",
                "build",
                "--input",
                str(library),
                "--output",
                str(db_path),
                "--config",
                str(config_yaml),
                "--category",
                "personal",
            ],
        )
        assert result.exit_code == 0, result.output
        return db_path

    def test_db_build_records_input_config_and_timestamp(self, tmp_path):
        config_yaml = _write_config_yaml(
            tmp_path,
            input_path=tmp_path / "q.mgf",
            library_path=tmp_path / "library.msp",
            output_directory=tmp_path / "results",
        )
        library = tmp_path / "library.msp"
        db_path = self._build_via_cli(tmp_path, config_yaml)

        config = MassFlowConfig.from_yaml(config_yaml)
        db = SpectralDatabase(db_path)
        try:
            builds = db.get_library_builds()
            assert len(builds) == 1
            build = builds[0]
            assert build["source_path"] == str(library)
            assert build["source_sha256"] == _file_sha256(library)
            assert build["category"] == "personal"
            assert build["spectrum_count"] == 4
            assert build["storage_backend"] == "sqlite"
            # Config hash + processing parameters recorded exactly.
            assert (
                build["config_digest_sha256"]
                == config.normalized_config()["config_digest_sha256"]
            )
            assert build["processing_fingerprint"] == processing_fingerprint(
                config.processing
            )
            processing = build["processing_config"]
            assert processing["min_peaks"] == 1
            assert processing["decoy_min_relative_intensity"] == 0.02
            assert processing["decoy_mz_shift_da"] == 1.5
            # Target-decoy / similarity configuration recorded.
            assert build["similarity_config"]["algorithm"] == "cosine"
            assert build["similarity_config"]["fdr_threshold"] == 0.05
            datetime.fromisoformat(build["built_at"])
        finally:
            db.close()

    def test_db_inspect_queries_build_history(self, tmp_path):
        config_yaml = _write_config_yaml(
            tmp_path,
            input_path=tmp_path / "q.mgf",
            library_path=tmp_path / "library.msp",
            output_directory=tmp_path / "results",
        )
        db_path = self._build_via_cli(tmp_path, config_yaml)

        runner = CliRunner()
        # Wide COLUMNS keeps rich from folding long cells mid-word, so the
        # full digest and headers render on single lines.
        result = runner.invoke(
            cli.app, ["db", "inspect", str(db_path)], env={"COLUMNS": "160"}
        )
        assert result.exit_code == 0
        output = result.output
        assert "Total Spectra" in output
        assert "Store Metadata" in output
        assert "Schema Version" in output
        assert "Library Build History" in output
        assert "personal" in output
        assert "Config Digest (sha256)" in output
        assert "Processing Parameters" in output
        assert "decoy_min_relative_intensity" in output
        assert "Target-Decoy Configuration" in output
        assert "fdr_threshold" in output
        # The recorded config digest is visible (the 64-char value folds
        # across lines at any width; the leading chunk is enough to query).
        config = MassFlowConfig.from_yaml(config_yaml)
        digest = config.normalized_config()["config_digest_sha256"]
        assert digest[:16] in output

    def test_db_merge_records_input_databases(self, tmp_path):
        config_yaml = _write_config_yaml(
            tmp_path,
            input_path=tmp_path / "q.mgf",
            library_path=tmp_path / "library.msp",
            output_directory=tmp_path / "results",
        )
        first = self._build_via_cli(tmp_path, config_yaml)
        second_dir = tmp_path / "second"
        second_dir.mkdir(exist_ok=True)
        second_cfg = _write_config_yaml(
            second_dir,
            input_path=tmp_path / "q.mgf",
            library_path=second_dir / "library.msp",
            output_directory=second_dir / "results",
        )
        second = self._build_via_cli(second_dir, second_cfg)

        merged = tmp_path / "merged.db"
        runner = CliRunner()
        result = runner.invoke(
            cli.app,
            [
                "db",
                "merge",
                "--inputs",
                str(first),
                "--inputs",
                str(second),
                "--output",
                str(merged),
            ],
        )
        assert result.exit_code == 0, result.output

        db = SpectralDatabase(merged)
        try:
            assert db.get_total_spectra_count() == 8
            builds = db.get_library_builds()
            assert len(builds) == 1
            build = builds[0]
            assert build["category"] == "merged"
            assert build["spectrum_count"] == 8
            assert build["source_path"] is None
            sources = build["details"]["merged_sources"]
            assert [s["path"] for s in sources] == [str(first), str(second)]
            assert [s["spectra_added"] for s in sources] == [4, 4]
            datetime.fromisoformat(build["built_at"])
        finally:
            db.close()

        # Inspect surfaces merges as build history too (source listed as a
        # merge of the input databases, with a dedicated inputs table).
        # 240 columns keeps the (long) absolute input paths on single lines.
        inspect = runner.invoke(
            cli.app, ["db", "inspect", str(merged)], env={"COLUMNS": "240"}
        )
        assert inspect.exit_code == 0
        assert "merged" in inspect.output
        assert "Library Build History" in inspect.output
        assert "Merge Inputs" in inspect.output
        assert str(first) in inspect.output
        assert str(second) in inspect.output

    def test_db_inspect_empty_database_still_reports_store_meta(self, tmp_path):
        db_path = tmp_path / "empty.db"
        SpectralDatabase(db_path).close()
        result = CliRunner().invoke(
            cli.app, ["db", "inspect", str(db_path)], env={"COLUMNS": "160"}
        )
        assert result.exit_code == 0
        assert "Empty (0 spectra)" in result.output
        assert "Store Metadata" in result.output
        assert "No recorded library build history" in result.output


# ---------------------------------------------------------------------------
# Annotate sidecar: explicit link to the exact database build & config hash
# ---------------------------------------------------------------------------


class TestAnnotateLineageSidecar:
    def _annotate(self, tmp_path: Path, config_yaml: Path) -> Path:
        """Run annotate and return the just-written per-file YAML sidecar.

        Repeated runs in one output directory use counter-suffixed result
        files, so the newest ``*_results.report.yaml`` is selected by mtime.
        """
        config = MassFlowConfig.from_yaml(config_yaml)
        results = run_annotation_pipeline(config, config_path=config_yaml)
        assert results[0].status in ("success", "degraded"), results[0].fatal_errors
        candidates = list(config.project.output_directory.glob("*_results.report.yaml"))
        assert candidates, "no sidecar written"
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def test_annotate_raw_library_links_prepared_store_build(self, tmp_path):
        """Annotate over a raw library builds a store that carries a
        history row; the sidecar points at that exact row and digest."""
        library = tmp_path / "library.msp"
        _write_msp(library)
        query = tmp_path / "queries.mgf"
        _write_mgf(query)
        config_yaml = _write_config_yaml(
            tmp_path, query, library, output_directory=tmp_path / "results"
        )

        sidecar_path = self._annotate(tmp_path, config_yaml)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        library_section = sidecar["library"]

        # The store actually searched is the prepared <stem>_library.db.
        store_path = Path(library_section["store"]["path"])
        assert store_path.name == "library_library.db"
        assert store_path.exists()
        assert library_section["store"]["kind"] == "store"
        assert library_section["store"]["storage_backend"] == "sqlite"
        assert library_section["store"]["schema_version"] == PROVENANCE_SCHEMA_VERSION
        assert library_section["store"]["created_at"] is not None
        assert library_section["configured_library_path"] == str(library)

        # The build row exists in the prepared store with matching digests.
        db = SpectralDatabase(store_path)
        try:
            build = db.get_latest_library_build()
            assert build is not None
            assert library_section["build"]["id"] == build["id"]
            assert library_section["build"]["built_at"] == build["built_at"]
            assert library_section["build"]["source_path"] == str(library)
            assert library_section["build"]["category"] == "library"
            assert (
                library_section["build"]["config_digest_sha256"]
                == build["config_digest_sha256"]
            )
            assert (
                library_section["build"]["processing_fingerprint"]
                == build["processing_fingerprint"]
            )
        finally:
            db.close()

        # The sidecar's run-config digest equals the digest recorded at build
        # time: same YAML, same effective config, closed lineage loop.
        assert (
            library_section["build"]["config_digest_sha256"]
            == sidecar["config"]["config_digest_sha256"]
        )

    def test_annotate_repeated_runs_keep_stable_lineage(self, tmp_path):
        """A cached prepared store is reused: identical runs link to the
        same build id / timestamp (lineage is deterministic)."""
        library = tmp_path / "library.msp"
        _write_msp(library)
        query = tmp_path / "queries.mgf"
        _write_mgf(query)
        config_yaml = _write_config_yaml(
            tmp_path, query, library, output_directory=tmp_path / "results"
        )

        first = yaml.safe_load(self._annotate(tmp_path, config_yaml).read_text())
        second = yaml.safe_load(self._annotate(tmp_path, config_yaml).read_text())

        assert first["library"]["build"]["id"] == second["library"]["build"]["id"]
        assert (
            first["library"]["build"]["built_at"]
            == second["library"]["build"]["built_at"]
        )
        assert (
            first["library"]["build"]["config_digest_sha256"]
            == second["library"]["build"]["config_digest_sha256"]
        )
        # Only one build row exists for the cached store.
        db = SpectralDatabase(Path(first["library"]["store"]["path"]))
        try:
            assert len(db.get_library_builds()) == 1
        finally:
            db.close()

    def test_annotate_against_db_links_the_database_build(self, tmp_path):
        """Annotate against a `massflow db build` output links the sidecar
        to that database's recorded build row."""
        library = tmp_path / "library.msp"
        _write_msp(library)
        query = tmp_path / "queries.mgf"
        _write_mgf(query)
        config_yaml = _write_config_yaml(
            tmp_path, query, library, output_directory=tmp_path / "results"
        )
        db_path = tmp_path / "user_library.db"
        result = CliRunner().invoke(
            cli.app,
            [
                "db",
                "build",
                "--input",
                str(library),
                "--output",
                str(db_path),
                "--config",
                str(config_yaml),
                "--category",
                "personal",
            ],
        )
        assert result.exit_code == 0, result.output

        # Annotate with the database as the library (same YAML config so the
        # recorded digest matches the run digest).
        db_config_yaml = tmp_path / "annotate_db.yaml"
        payload = yaml.safe_load(config_yaml.read_text())
        payload["input"]["library_path"] = str(db_path)
        db_config_yaml.write_text(yaml.safe_dump(payload))
        sidecar_path = self._annotate(tmp_path, db_config_yaml)

        sidecar = yaml.safe_load(sidecar_path.read_text())
        library_section = sidecar["library"]
        assert library_section["configured_library_path"] == str(db_path)
        assert library_section["store"]["path"] == str(db_path)

        db = SpectralDatabase(db_path)
        try:
            build = db.get_latest_library_build()
            assert build is not None
            assert library_section["build"]["id"] == build["id"]
            assert library_section["build"]["category"] == "personal"
            assert (
                library_section["build"]["config_digest_sha256"]
                == build["config_digest_sha256"]
            )
        finally:
            db.close()

        # The sidecar also records the *run's* own config digest, so a
        # consumer can compare run config vs. the config the database was
        # built with (they differ here because the annotate config points at
        # the database instead of the raw library).
        assert sidecar["config"]["config_digest_sha256"]
        assert library_section["build"]["config_digest_sha256"]

    def test_sidecar_reports_no_build_for_pre_provenance_store(self, tmp_path):
        """A store without history yields build=None in the sidecar — the
        lineage section says what it knows and never guesses."""
        library = tmp_path / "library.msp"
        _write_msp(library)
        query = tmp_path / "queries.mgf"
        _write_mgf(query)
        config_yaml = _write_config_yaml(
            tmp_path, query, library, output_directory=tmp_path / "results"
        )

        # Build the DB, then erase its history to simulate a pre-provenance
        # database (spectra untouched).
        db_path = tmp_path / "pre.db"
        result = CliRunner().invoke(
            cli.app,
            [
                "db",
                "build",
                "--input",
                str(library),
                "--output",
                str(db_path),
                "--config",
                str(config_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        connection = sqlite3.connect(db_path)
        connection.execute(f"DROP TABLE {LIBRARY_BUILDS_TABLE}")
        connection.execute(f"DROP TABLE {STORE_META_TABLE}")
        connection.execute("PRAGMA user_version = 0")
        connection.commit()
        connection.close()

        db_config_yaml = tmp_path / "pre_annotate.yaml"
        payload = yaml.safe_load(config_yaml.read_text())
        payload["input"]["library_path"] = str(db_path)
        payload["project"]["output_directory"] = str(tmp_path / "results_pre")
        db_config_yaml.write_text(yaml.safe_dump(payload))

        sidecar_path = self._annotate(tmp_path, db_config_yaml)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        library_section = sidecar["library"]
        assert library_section["store"]["path"] == str(db_path)
        assert library_section["store"]["schema_version"] == 1
        # No recorded build row: reported explicitly as None.
        assert library_section["build"] is None
