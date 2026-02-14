import argparse
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from MassFlow import cli


def test_setup_logging_tty():
    # Mock sys.stderr.isatty to be True
    with patch("sys.stderr.isatty", return_value=True):
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.handlers = []
            mock_get_logger.return_value = mock_logger

            cli.setup_logging()

            assert len(mock_logger.addHandler.call_args_list) == 1


def test_run_clean_flow():
    args = argparse.Namespace(input="test.msp", output_dir="out", format="json")

    with (
        patch("MassFlow.processing.process_spectra") as mock_process,
        patch("MassFlow.io.load_spectra") as mock_load,
        patch("MassFlow.io.save_spectra_to_json") as mock_save,
    ):
        mock_load.return_value = ["raw_spec"]
        mock_process.return_value = ["proc_spec"]

        ret = cli.run_clean(args)

        assert ret == 0
        mock_load.assert_called()
        mock_process.assert_called()
        # Output path construction
        # input="test.msp" -> stem="test" -> out="out/test.json"
        # We need to verify check the Path argument roughly
        args_save, _ = mock_save.call_args
        assert args_save[0] == ["proc_spec"]
        assert str(args_save[1]).endswith("test.json")


def test_run_process_success():
    args = argparse.Namespace(config="test_config.yaml")

    with patch("MassFlow.workflow.run_workflow") as mock_workflow:
        ret = cli.run_process(args)
        assert ret == 0
        mock_workflow.assert_called_once()


def test_run_process_failure():
    args = argparse.Namespace(config="test_config.yaml")

    with (
        patch("MassFlow.workflow.run_workflow") as mock_workflow,
        patch("MassFlow.cli.logger") as mock_logger,
    ):
        mock_workflow.side_effect = Exception("Config Error")
        ret = cli.run_process(args)
        assert ret == 1
        mock_logger.error.assert_called()


def test_run_plot_success():
    args = argparse.Namespace(input="lib.msp", name="Spec1", more=False)

    with (
        patch("MassFlow.io.load_spectra") as mock_load,
        patch("builtins.print") as mock_print,
    ):
        mock_spec = MagicMock()
        mock_spec.get.return_value = "Spec1"
        mock_spec.peaks.mz = [100.0]
        mock_spec.peaks.intensities = [10.0]
        # Make intensities numpy array to support division
        import numpy as np

        mock_spec.peaks.intensities = np.array([10.0])
        mock_spec.peaks.mz = np.array([100.0])

        mock_load.return_value = iter([mock_spec])

        ret = cli.run_plot(args)
        assert ret == 0
        mock_print.assert_called()


def test_run_plot_list_more():
    args = argparse.Namespace(input="lib.msp", name=None, more=True)
    with (
        patch("MassFlow.io.load_spectra") as mock_load,
        patch("builtins.print") as mock_print,
    ):
        mock_spec = MagicMock()
        mock_spec.get.return_value = "Spec1"
        mock_load.return_value = iter([mock_spec])

        ret = cli.run_plot(args)
        assert ret == 0
        mock_print.assert_called_with("Spec1")


def test_run_convert_success():
    args = argparse.Namespace(
        input="test.mgf", output="test.mzml", input_format=None, output_format=None
    )

    with (
        patch("MassFlow.io.load_spectra") as mock_load,
        patch("MassFlow.io.save_spectra_to_mzml") as mock_save,
    ):
        mock_load.return_value = ["spec"]

        ret = cli.run_convert(args)

        assert ret == 0
        mock_load.assert_called_with(Path("test.mgf"), "mgf")
        mock_save.assert_called_with(["spec"], Path("test.mzml"))


def test_run_convert_db_success():
    args = argparse.Namespace(
        input="test.mgf", output="test.db", input_format=None, output_format="db"
    )

    with (
        patch("MassFlow.io.load_spectra") as mock_load,
        patch("MassFlow.cli.SpectralDatabase") as mock_db_cls,
    ):
        mock_load.return_value = ["spec"]
        mock_db_instance = mock_db_cls.return_value
        mock_db_instance.add_spectra.return_value = 1

        ret = cli.run_convert(args)

        assert ret == 0
        mock_db_cls.assert_called_with(Path("test.db"))
        mock_db_instance.add_spectra.assert_called_with(["spec"])
        mock_db_instance.close.assert_called()


def test_run_database_init():
    args = argparse.Namespace(action="init", db="test.db")

    with patch("MassFlow.cli.SpectralDatabase") as mock_db_cls:
        ret = cli.run_database(args)
        assert ret == 0
        mock_db_cls.assert_called_with(Path("test.db"))
        mock_db_cls.return_value.close.assert_called()


def test_run_database_add():
    args = argparse.Namespace(
        action="add", db="test.db", input="lib.msp", category="test"
    )

    with (
        patch("MassFlow.io.load_spectra") as mock_load,
        patch("MassFlow.cli.SpectralDatabase") as mock_db_cls,
    ):
        mock_load.return_value = ["spec"]
        mock_db_instance = mock_db_cls.return_value
        mock_db_instance.add_spectra.return_value = 1

        ret = cli.run_database(args)

        assert ret == 0
        mock_load.assert_called_with(Path("lib.msp"), "msp")
        mock_db_instance.add_spectra.assert_called_with(["spec"], "test")


def test_run_database_export():
    args = argparse.Namespace(
        action="export", db="test.db", output="lib.mgf", category="test"
    )

    with (
        patch("MassFlow.cli.SpectralDatabase") as mock_db_cls,
        patch("MassFlow.io.save_spectra_to_mgf") as mock_save,
    ):
        mock_db_instance = mock_db_cls.return_value
        mock_db_instance.get_spectra.return_value = ["spec"]

        ret = cli.run_database(args)

        assert ret == 0
        mock_db_instance.get_spectra.assert_called_with(category="test")
        mock_save.assert_called_with(["spec"], Path("lib.mgf"))


def test_main_clean():
    with patch("MassFlow.cli.run_clean") as mock_run:
        cli.main(["clean", "--input", "in.msp", "--output-dir", "out"])
        mock_run.assert_called_once()


def test_main_plot():
    with patch("MassFlow.cli.run_plot") as mock_run:
        cli.main(["plot", "--input", "in.msp"])
        mock_run.assert_called_once()


def test_main_process():
    with patch("MassFlow.cli.run_process") as mock_run:
        cli.main(["process", "config.yaml"])
        mock_run.assert_called_once()


def test_main_convert():
    with patch("MassFlow.cli.run_convert") as mock_run:
        cli.main(["convert", "--input", "in.mgf", "--output", "out.mzml"])
        mock_run.assert_called_once()


def test_main_database():
    with patch("MassFlow.cli.run_database") as mock_run:
        cli.main(["database", "init", "--db", "test.db"])
        mock_run.assert_called_once()


def test_main_no_args():
    # Should print help and exit 0
    with patch("argparse.ArgumentParser.print_help") as mock_print:
        ret = cli.main([])
        assert ret == 0
        mock_print.assert_called_once()


def test_run_clean_failure():
    args = argparse.Namespace(input="test.msp", output_dir="out", format="json")
    with patch("MassFlow.io.load_spectra", side_effect=Exception("Load Error")):
        ret = cli.run_clean(args)
        assert ret == 1


def test_run_convert_failure():
    args = argparse.Namespace(
        input="in.mgf", output="out.mzml", input_format=None, output_format=None
    )
    with patch("MassFlow.io.load_spectra", side_effect=Exception("Convert Error")):
        ret = cli.run_convert(args)
        assert ret == 1


def test_run_convert_unsupported_format():
    args = argparse.Namespace(
        input="in.mgf", output="out.xyz", input_format=None, output_format="xyz"
    )
    with patch("MassFlow.io.load_spectra", return_value=["spec"]):
        ret = cli.run_convert(args)
        assert ret == 1


def test_run_database_init_failure():
    args = argparse.Namespace(action="init", db="test.db")
    with patch("MassFlow.cli.SpectralDatabase", side_effect=Exception("DB Error")):
        ret = cli.run_database(args)
        assert ret == 1


def test_run_database_add_failure():
    args = argparse.Namespace(
        action="add", db="test.db", input="in.msp", category="default"
    )
    with patch("MassFlow.io.load_spectra", side_effect=Exception("Load Error")):
        ret = cli.run_database(args)
        assert ret == 1


def test_run_database_add_missing_input():
    args = argparse.Namespace(action="add", db="test.db", input=None)
    ret = cli.run_database(args)
    assert ret == 1


def test_run_database_export_failure():
    args = argparse.Namespace(
        action="export", db="test.db", output="out.mgf", category="default"
    )
    with patch("MassFlow.cli.SpectralDatabase", side_effect=Exception("DB Error")):
        ret = cli.run_database(args)
        assert ret == 1


def test_run_database_export_missing_output():
    args = argparse.Namespace(action="export", db="test.db", output=None)
    ret = cli.run_database(args)
    assert ret == 1


def test_run_database_export_unsupported_format():
    args = argparse.Namespace(
        action="export", db="test.db", output="out.xyz", category="default"
    )
    with patch("MassFlow.cli.SpectralDatabase"):
        ret = cli.run_database(args)
        assert ret == 1


def test_run_plot_load_failure():
    args = argparse.Namespace(input="test.msp")
    with patch("MassFlow.io.load_spectra", side_effect=Exception("Load Error")):
        ret = cli.run_plot(args)
        assert ret == 1


def test_run_plot_no_spectra():
    args = argparse.Namespace(input="test.msp")
    with patch("MassFlow.io.load_spectra", return_value=iter([])):
        ret = cli.run_plot(args)
        assert ret == 0


def test_run_plot_name_not_found():
    args = argparse.Namespace(input="test.msp", name="Missing", more=False)
    mock_spec = MagicMock()
    mock_spec.get.return_value = "Present"
    with patch("MassFlow.io.load_spectra", return_value=iter([mock_spec])):
        ret = cli.run_plot(args)
        assert ret == 1
