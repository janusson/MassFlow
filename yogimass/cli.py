"""
Simple CLI for Yogimass.
Replaces yogimass_buildDB.py and yogimass_pipeline.py entry points.
"""
from __future__ import annotations

import argparse
import sys
import os
import logging
from yogimass import io, processing, similarity, __version__

# --- Colored Formatter ---
class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds ANSI colors to log levels.
    """
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # Custom format: Time in grey, Level in color, Message standard
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s - %(name)s - %(levelname)s" + RESET + " - %(message)s",
        logging.INFO: GREEN + "%(asctime)s - %(name)s - %(levelname)s" + RESET + " - %(message)s",
        logging.WARNING: YELLOW + "%(asctime)s - %(name)s - %(levelname)s" + RESET + " - %(message)s",
        logging.ERROR: RED + "%(asctime)s - %(name)s - %(levelname)s" + RESET + " - %(message)s",
        logging.CRITICAL: BOLD_RED + "%(asctime)s - %(name)s - %(levelname)s" + RESET + " - %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


def run_clean(args):
    """Run library cleaning (similar to yogimass_buildDB.py)."""
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
        # UX Improvement: More descriptive error
        logger.error(f"Unsupported file extension for input: '{input_path}'.")
        logger.error("Please provide a file ending in .msp or .mgf.")
        return 1
        
    if not spectra:
        logger.warning("No spectra found or retained.")
        return 0
        
    # Export
    logger.info(f"Saving results to {output_dir} in {export_format} format...")
    if export_format == "pickle":
        io.save_spectra_to_pickle(spectra, output_dir, lib_name)
    elif export_format == "msp":
        io.save_spectra_to_msp(spectra, output_dir, lib_name)
    elif export_format == "mgf":
        io.save_spectra_to_mgf(spectra, output_dir, lib_name)
    elif export_format == "json":
        io.save_spectra_to_json(spectra, output_dir, lib_name)

    logger.info("Operation completed successfully.")
    return 0


def run_search(args):
    """Run similarity search."""
    # This is a placeholder for the pipeline search logic.
    # To implement this fully, we'd need to load a reference library and query spectra.
    logger.info("Search functionality is available in yogimass.similarity but CLI wrapper is pending specific requirements.")
    return 0


def main(argv: list[str] | None = None) -> int:
    # UX Improvement: ArgumentDefaultsHelpFormatter shows default values in help
    parser = argparse.ArgumentParser(
        prog="yogimass",
        description="Yogimass: Tandem MS/MS data analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    search_parser = subparsers.add_parser(
        "search",
        help="Run similarity search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    search_parser.set_defaults(func=run_search)

    args = parser.parse_args(argv)
    
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())
