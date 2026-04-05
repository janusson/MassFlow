"""
Tests for MassFlow CLI module.
"""

import argparse
from unittest.mock import patch

import pytest

from MassFlow import cli
from MassFlow.config import MassFlowConfig


def test_run_annotate_success():
    """Test successful execution of run_annotate."""
    args = argparse.Namespace(config="config.yaml")

    with (
        patch("MassFlow.cli.run_annotation_pipeline") as mock_pipeline,
        patch("MassFlow.cli.MassFlowConfig.from_yaml") as mock_config,
    ):
        ret = cli.run_annotate(args)

        assert ret == 0
        mock_config.assert_called_with("config.yaml")
        mock_pipeline.assert_called_once_with(
            mock_config.return_value, config_path="config.yaml"
        )


def test_run_annotate_failure():
    """Test failure execution of run_annotate."""
    args = argparse.Namespace(config="config.yaml")

    with (
        patch("MassFlow.cli.MassFlowConfig.from_yaml"),
        patch(
            "MassFlow.cli.run_annotation_pipeline",
            side_effect=Exception("Pipeline Error"),
        ),
    ):
        with patch("MassFlow.cli.logger") as mock_logger:
            ret = cli.run_annotate(args)

            assert ret == 1
            mock_logger.error.assert_called_once()


def test_run_init_success(tmp_path):
    """Test successful execution of run_init."""
    output_file = tmp_path / "test_config.yaml"
    args = argparse.Namespace(output=str(output_file), force=False)

    with patch("MassFlow.cli.logger") as mock_logger:
        ret = cli.run_init(args)

        assert ret == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "project:" in content
        assert "My_Annotation_Project" in content
        mock_logger.info.assert_called_once()

        # Verify the generated template is a valid MassFlowConfig
        config = MassFlowConfig.from_yaml(output_file)
        assert config.project.name == "My_Annotation_Project"


def test_run_init_exists_no_force(tmp_path):
    """Test run_init when file exists and force is False."""
    output_file = tmp_path / "test_config.yaml"
    output_file.touch()
    args = argparse.Namespace(output=str(output_file), force=False)

    with patch("MassFlow.cli.logger") as mock_logger:
        ret = cli.run_init(args)

        assert ret == 1
        mock_logger.error.assert_called_once()


def test_run_init_exists_with_force(tmp_path):
    """Test run_init when file exists and force is True."""
    output_file = tmp_path / "test_config.yaml"
    output_file.write_text("old content")
    args = argparse.Namespace(output=str(output_file), force=True)

    with patch("MassFlow.cli.logger") as mock_logger:
        ret = cli.run_init(args)

        assert ret == 0
        assert "project:" in output_file.read_text()
        mock_logger.info.assert_called_once()

        # Verify the generated template is a valid MassFlowConfig
        config = MassFlowConfig.from_yaml(output_file)
        assert config.project.name == "My_Annotation_Project"


def test_run_browse_success():
    """Test successful execution of run_browse."""
    args = argparse.Namespace(file="library.msp")

    with patch("MassFlow.tui.browse_file") as mock_browse_file:
        ret = cli.run_browse(args)

        assert ret == 0
        mock_browse_file.assert_called_once_with("library.msp")


def test_run_browse_import_error():
    """Test run_browse when the experimental TUI import fails."""
    args = argparse.Namespace(file="library.msp")

    with patch(
        "MassFlow.tui.browse_file",
        side_effect=ImportError("textual unavailable"),
    ):
        with patch("MassFlow.cli.logger") as mock_logger:
            ret = cli.run_browse(args)

            assert ret == 1
            mock_logger.error.assert_called_once()


def test_run_browse_cas_success():
    """Test successful execution of run_browse_cas."""
    args = argparse.Namespace(
        library="library.msp",
        cas="50-00-0",
        top_k=5,
        chem_weight=0.7,
        spec_weight=0.3,
        mz_tol=0.02,
    )

    with patch("MassFlow.tui.browse_cas_main") as mock_browse_cas_main:
        mock_browse_cas_main.return_value = 0

        ret = cli.run_browse_cas(args)

        assert ret == 0
        mock_browse_cas_main.assert_called_once_with(
            [
                "library.msp",
                "--cas",
                "50-00-0",
                "--top-k",
                "5",
                "--chem-weight",
                "0.7",
                "--spec-weight",
                "0.3",
                "--mz-tol",
                "0.02",
            ]
        )


def test_run_browse_cas_import_error():
    """Test run_browse_cas when the experimental TUI import fails."""
    args = argparse.Namespace(
        library="library.msp",
        cas="50-00-0",
        top_k=5,
        chem_weight=0.7,
        spec_weight=0.3,
        mz_tol=0.02,
    )

    with patch(
        "MassFlow.tui.browse_cas_main",
        side_effect=ImportError("tui unavailable"),
    ):
        with patch("MassFlow.cli.logger") as mock_logger:
            ret = cli.run_browse_cas(args)

            assert ret == 1
            mock_logger.error.assert_called_once()


def test_main_annotate():
    """Test main function calling annotate command."""
    with patch("MassFlow.cli.run_annotate") as mock_run:
        mock_run.return_value = 0

        ret = cli.main(["annotate", "--config", "config.yaml"])

        assert ret == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.config == "config.yaml"


def test_main_browse():
    """Test main function calling browse command."""
    with patch("MassFlow.cli.run_browse") as mock_run:
        mock_run.return_value = 0

        ret = cli.main(["browse", "library.msp"])

        assert ret == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.file == "library.msp"


def test_main_browse_cas():
    """Test main function calling browse-cas command."""
    with patch("MassFlow.cli.run_browse_cas") as mock_run:
        mock_run.return_value = 0

        ret = cli.main(
            [
                "browse-cas",
                "library.msp",
                "--cas",
                "50-00-0",
                "--top-k",
                "5",
                "--chem-weight",
                "0.7",
                "--spec-weight",
                "0.3",
                "--mz-tol",
                "0.02",
            ]
        )

        assert ret == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.library == "library.msp"
        assert args.cas == "50-00-0"
        assert args.top_k == 5
        assert args.chem_weight == 0.7
        assert args.spec_weight == 0.3
        assert args.mz_tol == 0.02


def test_main_no_args():
    """Test main function with no arguments (prints help)."""
    with patch("argparse.ArgumentParser.print_help") as mock_print:
        ret = cli.main([])
        assert ret == 0
        mock_print.assert_called_once()


def test_main_version(capsys):
    """Test main function with version argument."""
    # argparse exits when version is printed, so we catch SystemExit
    with pytest.raises(SystemExit):
        cli.main(["--version"])

    captured = capsys.readouterr()
    assert "MassFlow" in captured.out or "MassFlow" in captured.err
