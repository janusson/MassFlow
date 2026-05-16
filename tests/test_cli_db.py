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

    runner = CliRunner()
    result = runner.invoke(cli.app, ["db", "inspect", "dummy.db"])

    assert result.exit_code == 0
    assert "Total Spectra" in result.output
    assert "100" in result.output


@patch("MassFlow.database.SpectralDatabase", side_effect=Exception("DB Error"))
def test_db_inspect_error(mock_db):
    runner = CliRunner()
    result = runner.invoke(cli.app, ["db", "inspect", "fake.sqlite"])
    assert result.exit_code == 1


@patch("MassFlow.database.SpectralDatabase")
def test_run_db_merge(mock_db_class):
    mock_in_db = MagicMock()
    mock_out_db = MagicMock()

    # Return out_db for the first call (output), then in_db for inputs
    mock_db_class.side_effect = [mock_out_db, mock_in_db, mock_in_db]

    mock_in_db.get_spectra.return_value = iter([MagicMock()])
    mock_out_db.add_spectra.return_value = 1

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "merge",
            "--inputs",
            "in1.db",
            "--inputs",
            "in2.db",
            "--output",
            "out.db",
        ],
    )

    assert result.exit_code == 0
    assert mock_out_db.add_spectra.call_count == 2
    mock_in_db.close.assert_called()
    mock_out_db.close.assert_called_once()


@patch("MassFlow.database.SpectralDatabase")
@patch("MassFlow.cli.logger")
def test_run_db_merge_empty(mock_logger, mock_db_class):
    mock_in_db = MagicMock()
    mock_out_db = MagicMock()

    mock_db_class.side_effect = [mock_out_db, mock_in_db]

    mock_out_db.add_spectra.return_value = 0

    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["db", "merge", "--inputs", "in1.db", "--output", "out.db"]
    )

    assert result.exit_code == 1


@patch("MassFlow.config.MassFlowConfig.from_yaml")
@patch("MassFlow.database.SpectralDatabase")
@patch("MassFlow.io.load_spectra")
@patch("MassFlow.processing.process_spectra")
def test_run_db_build(mock_process, mock_load, mock_db_class, mock_config):
    mock_db = mock_db_class.return_value
    mock_db.add_spectra.return_value = 10

    mock_process.return_value = iter([MagicMock()])
    mock_load.return_value = iter([MagicMock()])

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "build",
            "--input",
            "dummy.mgf",
            "--output",
            "dummy.db",
            "--config",
            "dummy.yaml",
            "--category",
            "test",
        ],
    )

    assert result.exit_code == 0
    mock_db.add_spectra.assert_called_once()
    assert mock_db.add_spectra.call_args[1]["category"] == "test"


@patch("MassFlow.config.MassFlowConfig.from_yaml")
@patch("MassFlow.database.SpectralDatabase")
@patch("MassFlow.io.load_spectra")
@patch("MassFlow.processing.process_spectra")
@patch("MassFlow.cli.logger")
def test_run_db_build_empty(
    mock_logger, mock_process, mock_load, mock_db_class, mock_config
):
    mock_db = mock_db_class.return_value
    mock_db.add_spectra.return_value = 0

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "db",
            "build",
            "--input",
            "dummy.mgf",
            "--output",
            "dummy.db",
            "--config",
            "dummy.yaml",
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
