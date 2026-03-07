"""
Tests for MassFlow CLI module.
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from MassFlow import cli


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
        mock_pipeline.assert_called_once_with(mock_config.return_value)


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


def test_main_annotate():
    """Test main function calling annotate command."""
    with patch("MassFlow.cli.run_annotate") as mock_run:
        mock_run.return_value = 0

        ret = cli.main(["annotate", "--config", "config.yaml"])

        assert ret == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.config == "config.yaml"


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
