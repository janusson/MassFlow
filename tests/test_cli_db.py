"""
Tests for the MassFlow CLI db subcommands.
"""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from MassFlow import cli


@patch("MassFlow.database.SpectralDatabase")
def test_run_db_inspect(mock_db_class):
    mock_db = mock_db_class.return_value
    mock_db.get_total_spectra_count.return_value = 100
    mock_db.get_category_counts.return_value = {"ref": 50, "test": 50}
    mock_db.get_precursor_mz_range.return_value = (100.0, 200.0)
    mock_db.backend_provenance.return_value = {
        "backend": "sqlite",
        "path": "dummy.db",
        "spectrum_count": 100,
    }
    mock_db.get_schema_version.return_value = 1
    mock_db.get_store_created_at.return_value = "2026-09-06T00:00:00+00:00"
    mock_db.get_library_builds.return_value = [
        {
            "id": 3,
            "built_at": "2026-09-06T10:00:00+00:00",
            "massflow_version": "0.1.0",
            "source_path": "library.msp",
            "category": "personal",
            "spectrum_count": 100,
            "storage_backend": "sqlite",
            "processing_config": {
                "min_peaks": 5,
                "decoy_min_relative_intensity": 0.01,
                "decoy_mz_shift_da": 1.0,
            },
            "similarity_config": {
                "algorithm": "cosine",
                "fdr_threshold": 0.01,
            },
            "config_digest_sha256": "a" * 64,
            "details": None,
        }
    ]

    runner = CliRunner()
    result = runner.invoke(cli.app, ["db", "inspect", "dummy.db"])

    assert result.exit_code == 0
    assert "Total Spectra" in result.output
    assert "100" in result.output
    # Lineage sections are part of the inspect surface. Rich folds long
    # headers/titles across lines at narrow widths, so assert on fragments
    # that survive wrapping.
    assert "Store Metadata" in result.output
    assert "Library Build History" in result.output
    assert "sha256" in result.output
    assert "Target-Decoy" in result.output
    assert "fdr_threshold" in result.output
    assert "0.01" in result.output
    assert "personal" in result.output


@patch("MassFlow.database.SpectralDatabase", side_effect=Exception("DB Error"))
def test_db_inspect_error(mock_db):
    runner = CliRunner()
    result = runner.invoke(cli.app, ["db", "inspect", "fake.sqlite"])
    assert result.exit_code == 1


@patch("MassFlow.storage.create_spectral_store")
def test_run_db_merge(mock_create_store, tmp_path):
    """run_db_merge should merge from two input databases and exit 0."""
    mock_in_db = MagicMock()
    mock_out_db = MagicMock()

    # Make the SQLite fast-path fall through to the iterator path.
    mock_out_db.merge_from_sqlite.side_effect = NotImplementedError("fallback")
    mock_out_db.add_spectra.return_value = 1
    mock_in_db.get_spectra.return_value = iter([MagicMock()])

    # First call creates the output store; subsequent calls open input stores.
    mock_create_store.side_effect = [mock_out_db, mock_in_db, mock_in_db]

    # The merge pre-flight validates --inputs on the real filesystem.
    input_one = tmp_path / "in1.db"
    input_two = tmp_path / "in2.db"
    input_one.touch()
    input_two.touch()

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "merge",
            "--inputs",
            str(input_one),
            "--inputs",
            str(input_two),
            "--output",
            str(tmp_path / "out.db"),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert mock_out_db.add_spectra.call_count == 2
    mock_in_db.close.assert_called()
    mock_out_db.close.assert_called_once()


@patch("MassFlow.database.SpectralDatabase")
@patch("MassFlow.cli.logger")
def test_run_db_merge_empty(mock_logger, mock_db_class, tmp_path):
    mock_in_db = MagicMock()
    mock_out_db = MagicMock()

    mock_db_class.side_effect = [mock_out_db, mock_in_db]

    mock_out_db.add_spectra.return_value = 0

    input_one = tmp_path / "in1.db"
    input_one.touch()

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "merge",
            "--inputs",
            str(input_one),
            "--output",
            str(tmp_path / "out.db"),
        ],
    )

    assert result.exit_code == 1


@patch("MassFlow.config.MassFlowConfig.from_yaml")
@patch("MassFlow.storage.create_spectral_store")
@patch("MassFlow.io.load_spectra")
@patch("MassFlow.processing.process_spectra")
def test_run_db_build(
    mock_process, mock_load, mock_store_factory, mock_config, tmp_path
):
    mock_store = mock_store_factory.return_value
    mock_store.add_spectra.return_value = 10

    mock_process.return_value = iter([MagicMock()])
    mock_load.return_value = iter([MagicMock()])

    # The provenance recording path consumes the real config models (to
    # serialize processing/similarity JSON and the config digest).
    from MassFlow.config import ProcessingConfig, SimilarityConfig

    mock_cfg = mock_config.return_value
    mock_cfg.input.storage_backend = "sqlite"
    mock_cfg.processing = ProcessingConfig(min_peaks=1)
    mock_cfg.similarity = SimilarityConfig()
    mock_cfg.normalized_config.return_value = {"config_digest_sha256": "d" * 64}

    # The db-build pre-flight validates the --input source on the real
    # filesystem before the (mocked) store is created.
    source = tmp_path / "dummy.mgf"
    source.touch()

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "build",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "dummy.db"),
            "--config",
            str(tmp_path / "dummy.yaml"),
            "--category",
            "test",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    mock_store.add_spectra.assert_called_once()
    assert mock_store.add_spectra.call_args[1]["category"] == "test"
    # The build is recorded in the store's provenance history.
    mock_store.record_library_build.assert_called_once()
    recorded = mock_store.record_library_build.call_args[1]
    assert recorded["category"] == "test"
    assert recorded["spectrum_count"] == 10
    assert recorded["config_digest_sha256"] == "d" * 64


@patch("MassFlow.config.MassFlowConfig.from_yaml")
@patch("MassFlow.storage.create_spectral_store")
@patch("MassFlow.io.load_spectra")
@patch("MassFlow.processing.process_spectra")
@patch("MassFlow.cli.logger")
def test_run_db_build_empty(
    mock_logger, mock_process, mock_load, mock_store_factory, mock_config, tmp_path
):
    mock_store = mock_store_factory.return_value
    mock_store.add_spectra.return_value = 0

    source = tmp_path / "dummy.mgf"
    source.touch()

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "build",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "dummy.db"),
            "--config",
            str(tmp_path / "dummy.yaml"),
            "--category",
            "test",
        ],
    )

    assert result.exit_code == 1


def test_db_inspect_empty(tmp_path):
    runner = CliRunner()
    from MassFlow.database import SpectralDatabase

    db_path = tmp_path / "empty.sqlite"
    SpectralDatabase(db_path)  # create empty

    result = runner.invoke(cli.app, ["db", "inspect", str(db_path)])
    assert result.exit_code == 0
    assert "Empty (0 spectra)" in result.output
