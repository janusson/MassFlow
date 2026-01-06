"""
Simple CLI for MassFlow.
Replaces massflow_buildDB.py and massflow_pipeline.py entry points.
"""
from __future__ import annotations

import argparse
import sys
import os
from MassFlow import io, processing, similarity, __version__

# Configure logging
import logging

class ColoredFormatter(logging.Formatter):
    """
    Formatter to add colors to logging output based on log level.
    """
    grey = "\x1b[38;20m"
    green = "\x1b[32m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def __init__(self):
        super().__init__()
        self.formatters = {
            level: logging.Formatter(fmt)
            for level, fmt in self.FORMATS.items()
        }

    def format(self, record):
        formatter = self.formatters.get(record.levelno)
        if formatter is None:
            return super().format(record)
        return formatter.format(record)


def setup_logging():
    """Set up logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Check if handlers are already configured to avoid duplication
    if not logger.handlers:
        handler = logging.StreamHandler()

        # Use colored formatter only if stream is a TTY (terminal)
        if sys.stderr.isatty():
            handler.setFormatter(ColoredFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        logger.addHandler(handler)

logger = logging.getLogger(__name__)


def run_clean(args):
    """Run library cleaning (similar to massflow_buildDB.py)."""
    input_path = args.input
    output_dir = args.output_dir
    export_format = args.format
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger.info(f"Starting clean operation on {input_path}")
    
    # Detect input type (naive check)
    if input_path.endswith(".msp"):
        spectra = processing.clean_msp_library(input_path)
        lib_name = os.path.basename(input_path).replace(".msp", "")
    elif input_path.endswith(".mgf"):
        spectra = processing.clean_mgf_library(input_path)
        lib_name = os.path.basename(input_path).replace(".mgf", "")
    else:
        logger.error("Input must be .msp or .mgf")
        return 1
        
    if not spectra:
        logger.warning("No spectra found or retained.")
        return 0
        
    # Export
    if export_format == "pickle":
        io.save_spectra_to_pickle(spectra, output_dir, lib_name)
    elif export_format == "msp":
        io.save_spectra_to_msp(spectra, output_dir, lib_name)
    elif export_format == "mgf":
        io.save_spectra_to_mgf(spectra, output_dir, lib_name)
    elif export_format == "json":
        io.save_spectra_to_json(spectra, output_dir, lib_name)
        
    return 0


def run_search(args):
    """Run similarity search."""
    # This is a placeholder for the pipeline search logic.
    # To implement this fully, we'd need to load a reference library and query spectra.
    logger.info("Search functionality is available in MassFlow.similarity but CLI wrapper is pending specific requirements.")
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="MassFlow",
        description="MassFlow: Tandem MS/MS data analysis pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Clean command
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean and process a spectral library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    clean_parser.add_argument("--input", required=True, help="Input library file (.msp or .mgf)")
    clean_parser.add_argument("--output-dir", required=True, help="Directory to save processed library")
    clean_parser.add_argument("--format", choices=["pickle", "msp", "mgf", "json"], default="pickle", help="Output format")
    clean_parser.set_defaults(func=run_clean)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Run similarity search.")
    search_parser.set_defaults(func=run_search)

    args = parser.parse_args(argv)
    
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())
