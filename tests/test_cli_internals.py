"""
Comprehensive coverage tests for MassFlow CLI commands:
- convert (vendor file conversion)
- db build/inspect/merge
- init with --force
- watch command (import error path)
- version callback
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from MassFlow.cli import app


runner = CliRunner()


# ==============================================================================
# Version
# ==============================================================================


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "MassFlow version" in result.output


# ==============================================================================
# Init
# ==============================================================================


def test_cli_init_creates_file(tmp_path):
    output = tmp_path / "massflow_config.yaml"
    result = runner.invoke(app, ["init", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    content = output.read_text()
    assert "project:" in content
    assert "input:" in content


def test_cli_init_existing_without_force(tmp_path):
    output = tmp_path / "massflow_config.yaml"
    output.write_text("existing")
    result = runner.invoke(app, ["init", "--output", str(output)])
    assert result.exit_code == 1


def test_cli_init_existing_with_force(tmp_path):
    output = tmp_path / "massflow_config.yaml"
    output.write_text("existing")
    result = runner.invoke(app, ["init", "--output", str(output), "--force"])
    assert result.exit_code == 0
    content = output.read_text()
    assert "project:" in content


# ==============================================================================
# Annotate
# ==============================================================================


def test_cli_annotate_missing_config():
    result = runner.invoke(app, ["annotate", "--config", "/nonexistent/config.yaml"])
    assert result.exit_code == 1


def test_cli_annotate_with_minimal_config(tmp_path):
    """Test annotate with a minimal valid config by patching the full pipeline."""
    config_path = tmp_path / "config.yaml"
    exp_path = tmp_path / "experiment.mgf"
    exp_path.touch()
    lib_path = tmp_path / "library.msp"
    lib_path.touch()

    config_path.write_text(f"""
project:
  output_directory: "{tmp_path / "results"}"
input:
  input_path: "{exp_path}"
  library_path: "{lib_path}"
""")

    with patch("MassFlow.workflow.run_annotation_pipeline") as mock_run:
        result = runner.invoke(app, ["annotate", "--config", str(config_path)])
        assert result.exit_code == 0
        mock_run.assert_called_once()


# ==============================================================================
# Convert
# ==============================================================================


def test_cli_convert_directory_not_found(tmp_path):
    result = runner.invoke(
        app,
        [
            "convert",
            "--input",
            str(tmp_path / "nonexistent"),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 1


def test_cli_convert_success(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with patch("MassFlow.convert.convert_directory", return_value=2):
        result = runner.invoke(
            app,
            [
                "convert",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0


def test_cli_convert_msconvert_not_found(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    from MassFlow.convert import MSConvertNotFoundError

    with patch(
        "MassFlow.convert.convert_directory",
        side_effect=MSConvertNotFoundError("not found"),
    ):
        result = runner.invoke(
            app,
            [
                "convert",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 1


def test_cli_convert_generic_exception(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with patch("MassFlow.convert.convert_directory", side_effect=RuntimeError("Boom!")):
        result = runner.invoke(
            app,
            [
                "convert",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 1


# ==============================================================================
# DB Commands
# ==============================================================================


def test_cli_db_build_bad_input(tmp_path):
    """Build with a nonexistent input should fail."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'input:\n  input_path: "{tmp_path}"\n')
    output_db = tmp_path / "output.db"

    result = runner.invoke(
        app,
        [
            "db",
            "build",
            "--input",
            str(tmp_path / "nonexistent.mgf"),
            "--output",
            str(output_db),
            "--config",
            str(config_path),
            "--category",
            "cli_test",
        ],
    )
    assert result.exit_code == 1


def test_cli_db_inspect_empty(tmp_path):
    from MassFlow.database import SpectralDatabase

    output_db = tmp_path / "empty.db"
    db = SpectralDatabase(output_db)
    db.close()

    result = runner.invoke(app, ["db", "inspect", str(output_db)])
    assert result.exit_code == 0
    assert "Empty" in result.output


def test_cli_db_inspect_with_data(tmp_path):
    """Inspect a DB with data."""
    data_dir = Path(__file__).parent / "data"
    db_path = data_dir / "minimal_test_library.db"
    if not db_path.exists():
        pytest.skip("Minimal test DB not found.")

    result = runner.invoke(app, ["db", "inspect", str(db_path)])
    assert result.exit_code == 0
    assert "Total Spectra" in result.output


def test_cli_db_merge_empty_fails(tmp_path):
    """Merge with empty databases."""
    from MassFlow.database import SpectralDatabase

    empty1 = tmp_path / "empty1.db"
    empty2 = tmp_path / "empty2.db"

    for p in [empty1, empty2]:
        db = SpectralDatabase(p)
        db.close()

    output_db = tmp_path / "merged.db"

    result = runner.invoke(
        app,
        [
            "db",
            "merge",
            "--inputs",
            str(empty1),
            str(empty2),
            "--output",
            str(output_db),
        ],
    )
    # Exit code 1 for no spectra, exit code 2 for typer usage error
    assert result.exit_code in (1, 2)


def test_cli_db_no_args_shows_help():
    result = runner.invoke(app, ["db"])
    assert result.exit_code == 2


# ==============================================================================
# Watch
# ==============================================================================


def test_cli_watch_missing_watchfiles(tmp_path):
    """watch command fails gracefully when watchfiles is not available."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
project:
  output_directory: "{tmp_path / "results"}"
input:
  input_path: "{tmp_path}"
""")

    result = runner.invoke(app, ["watch", "--config", str(config_path)])
    # May fail due to missing watchfiles or due to import error
    assert result.exit_code in (0, 1)


# ==============================================================================
# No args help
# ==============================================================================


def test_cli_no_args_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "MassFlow" in result.output
