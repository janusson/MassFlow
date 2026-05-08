"""
Tests for MassFlow CLI module.
"""

from unittest.mock import patch

from typer.testing import CliRunner

from MassFlow import cli
from MassFlow.config import MassFlowConfig


def test_run_annotate_success():
    """Test successful execution of run_annotate."""
    runner = CliRunner()
    with (
        patch("MassFlow.workflow.run_annotation_pipeline") as mock_pipeline,
        patch("MassFlow.config.MassFlowConfig.from_yaml") as mock_config,
    ):
        result = runner.invoke(cli.app, ["annotate", "--config", "config.yaml"])

        assert result.exit_code == 0
        mock_config.assert_called_with("config.yaml")
        mock_pipeline.assert_called_once_with(
            mock_config.return_value, config_path="config.yaml"
        )


def test_run_annotate_failure():
    """Test failure execution of run_annotate."""
    runner = CliRunner()
    with (
        patch("MassFlow.config.MassFlowConfig.from_yaml"),
        patch(
            "MassFlow.workflow.run_annotation_pipeline",
            side_effect=Exception("Pipeline Error"),
        ),
    ):
        with patch("MassFlow.cli.logger") as mock_logger:
            result = runner.invoke(cli.app, ["annotate", "--config", "config.yaml"])

            assert result.exit_code == 1
            mock_logger.error.assert_called_once()


def test_run_init_success(tmp_path):
    """Test successful execution of run_init."""
    output_file = tmp_path / "test_config.yaml"
    runner = CliRunner()

    with patch("MassFlow.cli.logger") as mock_logger:
        result = runner.invoke(cli.app, ["init", "--output", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "project:" in content
        assert "My_MassFlow_Analysis" in content
        mock_logger.info.assert_called_once()

        # Verify the generated template is a valid MassFlowConfig
        config = MassFlowConfig.from_yaml(output_file)
        assert config.project.name == "My_MassFlow_Analysis"


def test_run_init_exists_no_force(tmp_path):
    """Test run_init when file exists and force is False."""
    output_file = tmp_path / "test_config.yaml"
    output_file.touch()
    runner = CliRunner()

    with patch("MassFlow.cli.logger") as mock_logger:
        result = runner.invoke(cli.app, ["init", "--output", str(output_file)])

        assert result.exit_code == 1
        mock_logger.error.assert_called_once()


def test_run_init_exists_with_force(tmp_path):
    """Test run_init when file exists and force is True."""
    output_file = tmp_path / "test_config.yaml"
    output_file.write_text("old content")
    runner = CliRunner()

    with patch("MassFlow.cli.logger") as mock_logger:
        result = runner.invoke(
            cli.app, ["init", "--output", str(output_file), "--force"]
        )

        assert result.exit_code == 0
        assert "project:" in output_file.read_text()
        mock_logger.info.assert_called_once()

        # Verify the generated template is a valid MassFlowConfig
        config = MassFlowConfig.from_yaml(output_file)
        assert config.project.name == "My_MassFlow_Analysis"


def test_main_no_args():
    """Test main function with no arguments (prints help)."""
    runner = CliRunner()
    result = runner.invoke(cli.app, [])
    assert result.exit_code in (0, 2)
    assert "Usage:" in result.output


def test_main_version():
    """Test main function with version argument."""
    runner = CliRunner()
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert "MassFlow" in result.output


@patch("MassFlow.convert.convert_directory")
def test_run_convert_success(mock_convert, tmp_path):
    mock_convert.return_value = 2

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["convert", "--input", str(in_dir), "--output", str(out_dir)]
    )

    assert result.exit_code == 0
    mock_convert.assert_called_once_with(in_dir, out_dir)


def test_run_convert_invalid_input(tmp_path):
    in_file = tmp_path / "not_a_dir.txt"
    in_file.touch()
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["convert", "--input", str(in_file), "--output", str(out_dir)]
    )

    assert result.exit_code == 1


@patch("MassFlow.convert.convert_directory")
def test_run_convert_msconvert_not_found(mock_convert, tmp_path):
    from MassFlow.convert import MSConvertNotFoundError

    mock_convert.side_effect = MSConvertNotFoundError("not found")

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["convert", "--input", str(in_dir), "--output", str(out_dir)]
    )

    assert result.exit_code == 1


@patch("MassFlow.cli.open", side_effect=Exception("File system error"))
def test_init_config_error(mock_open):
    runner = CliRunner()

    result = runner.invoke(cli.app, ["init", "--force"])
    assert result.exit_code == 1
    # Check for the exit code correctly. Since typer catches it and sys.exits, it won't print "Failed to initialize configuration" to stdout sometimes, so just check exit code.


@patch("MassFlow.cli.Path.is_dir", return_value=True)
@patch(
    "MassFlow.convert.convert_directory",
    side_effect=Exception("Conversion backend failed"),
)
def test_convert_general_error(mock_convert, mock_isdir):
    runner = CliRunner()

    result = runner.invoke(
        cli.app, ["convert", "--input", "fake_dir", "--output", "out_dir"]
    )
    assert result.exit_code == 1


@patch(
    "MassFlow.cli.visualize_graphml", side_effect=Exception("Viz Error"), create=True
)
def test_visualize_error(mock_viz):
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.graphml", "w") as f:
            f.write("<graphml></graphml>")
        result = runner.invoke(cli.app, ["visualize", "test.graphml"])
        assert result.exit_code == 1
