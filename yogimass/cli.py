"""
Command-line interface for Yogimass batch cleaning helpers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yogimass import pipeline
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yogimass",
        description="Batch clean mass spectrometry libraries using Yogimass.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_clean_subparser(subparsers)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - surfaced to users
        logger.error("%s", exc)
        return 1


def _add_clean_subparser(subparsers):
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean and export all libraries in a directory.",
    )
    clean_parser.add_argument(
        "input_dir",
        help="Directory containing MGF or MSP libraries.",
    )
    clean_parser.add_argument(
        "output_dir",
        help="Directory for cleaned exports (created if missing).",
    )
    clean_parser.add_argument(
        "--type",
        choices=["mgf", "msp"],
        default="mgf",
        help="Library format to process (default: mgf).",
    )
    clean_parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        metavar="FMT",
        help="Export formats (mgf, msp, json, pickle). Defaults to the library type.",
    )
    clean_parser.set_defaults(func=_run_clean_command)
    return clean_parser


def _run_clean_command(args) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_formats = tuple(args.formats) if args.formats else (args.type,)

    if args.type == "mgf":
        exported = pipeline.batch_clean_mgf_libraries(
            input_dir,
            output_dir,
            export_formats=export_formats,
        )
    else:
        exported = pipeline.batch_clean_msp_libraries(
            input_dir,
            output_dir,
            export_formats=export_formats,
        )

    if exported:
        logger.info("Exported %d files to %s", len(exported), output_dir)
    else:
        logger.warning("No libraries were processed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
