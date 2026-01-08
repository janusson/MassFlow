
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
