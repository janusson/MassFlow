
import pytest
import os
from unittest.mock import MagicMock, patch
from MassFlow import cli
import argparse

def test_setup_logging_tty():
    # Mock sys.stderr.isatty to be True
    with patch("sys.stderr.isatty", return_value=True):
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.handlers = []
            mock_get_logger.return_value = mock_logger
            
            cli.setup_logging()
            
            assert len(mock_logger.addHandler.call_args_list) == 1
            # Verify ColoredFormatter use is hard, but we can verify setup passed

def test_run_clean_invalid_input():
    args = argparse.Namespace(input="bad.txt", output_dir="out", format="pickle")
    
    with patch("MassFlow.cli.logger") as mock_logger:
        ret = cli.run_clean(args)
        assert ret == 1
        mock_logger.error.assert_called_with("Input must be .msp or .mgf")

def test_run_clean_msp_flow():
    args = argparse.Namespace(input="test.msp", output_dir="out", format="json")
    
    with patch("MassFlow.processing.clean_msp_library") as mock_clean:
        # Return a fake non-empty list
        mock_clean.return_value = ["spec1"]
        
        with patch("MassFlow.io.save_spectra_to_json") as mock_save:
            with patch("os.path.exists", return_value=True): 
                ret = cli.run_clean(args)
                
                assert ret == 0
                mock_clean.assert_called_with("test.msp")
                mock_save.assert_called_with(["spec1"], "out", "test")

def test_main_arg_parsing():
    with patch("MassFlow.cli.run_clean") as mock_run_clean:
        mock_run_clean.return_value = 0
        argv = ["clean", "--input", "my.msp", "--output-dir", "res"]
        
        ret = cli.main(argv)
        assert ret == 0
        mock_run_clean.assert_called_once()
        call_args = mock_run_clean.call_args[0][0]
        assert call_args.input == "my.msp"

def test_run_process_success():
    args = argparse.Namespace(config="test_config.yaml")
    
    with patch("MassFlow.config.MassFlowConfig.from_yaml") as mock_conf_load, \
         patch("MassFlow.workflow.run_workflow") as mock_workflow:
        
        mock_conf_load.return_value = MagicMock()
        ret = cli.run_process(args)
        assert ret == 0
        mock_conf_load.assert_called_with("test_config.yaml")
        mock_workflow.assert_called_once()

def test_run_process_failure():
    args = argparse.Namespace(config="test_config.yaml")
    
    with patch("MassFlow.config.MassFlowConfig.from_yaml") as mock_conf_load, \
         patch("MassFlow.workflow.run_workflow"), \
         patch("MassFlow.cli.logger") as mock_logger:
        
        mock_conf_load.side_effect = Exception("Config Error")
        ret = cli.run_process(args)
        assert ret == 1
        mock_logger.error.assert_called()

def test_run_plot_success():
    args = argparse.Namespace(input="lib.msp", name="Spec1", more=False)
    
    with patch("MassFlow.cli.load_from_msp") as mock_load, \
         patch("builtins.print") as mock_print:
        
        mock_spec = MagicMock()
        mock_spec.get.return_value = "Spec1"
        mock_spec.peaks.mz = [100.0]
        mock_spec.peaks.intensities = [10.0]
        # Make intensities numpy array to support division
        import numpy as np
        mock_spec.peaks.intensities = np.array([10.0])
        mock_spec.peaks.mz = np.array([100.0])
        
        mock_load.return_value = [mock_spec]
        
        ret = cli.run_plot(args)
        assert ret == 0
        # Check that we tried to plot (print called)
        # Note: mocking print for the plot object
        mock_print.assert_called()

def test_run_plot_list_more():
    args = argparse.Namespace(input="lib.msp", name=None, more=True)
    with patch("MassFlow.cli.load_from_msp") as mock_load, \
         patch("builtins.print") as mock_print:
        
        mock_spec = MagicMock()
        mock_spec.get.return_value = "Spec1"
        mock_load.return_value = [mock_spec]
        
        ret = cli.run_plot(args)
        assert ret == 0
        mock_print.assert_called_with("Spec1")

