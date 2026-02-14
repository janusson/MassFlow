"""
Simple CLI for MassFlow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from MassFlow import __version__, io
from MassFlow.workflow import run_annotation_pipeline

# Setup logging (Simplified for brevity but effective)
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

logger = logging.getLogger(__name__)

def run_annotate(args: argparse.Namespace) -> int:
    """
    Run the annotation pipeline.
    """
    try:
        run_annotation_pipeline(
            experimental_path=Path(args.experimental),
            reference_path=Path(args.reference),
            output_directory=Path(args.output_dir)
        )
        return 0
    except Exception as e:
        logger.error(f"Annotation failed: {e}")
        return 1

def main(argv: list[str] | None = None) -> int:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="MassFlow",
        description="MassFlow: Tandem MS/MS data analysis pipeline.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Annotate command (New)
    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Annotate experimental spectra against a reference library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    annotate_parser.add_argument(
        "--experimental", required=True, help="Input experimental file (.mzML, .msp, .mgf)"
    )
    annotate_parser.add_argument(
        "--reference", required=True, help="Reference library file (.msp, .mgf)"
    )
    annotate_parser.add_argument(
        "--output-dir", required=True, help="Directory to save results"
    )
    annotate_parser.set_defaults(func=run_annotate)

    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())
