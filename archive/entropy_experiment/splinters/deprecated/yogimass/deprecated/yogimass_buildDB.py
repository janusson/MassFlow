#!/usr/bin/env python3
"""
Deprecated wrapper for building libraries from MSP files.

Delegates to ``yogimass.workflow.build_library``. Prefer using the unified CLI:
`yogimass library build --input <files/dirs> --format msp --library out/library.json`.
"""

from __future__ import annotations

import argparse
import sys

from yogimass import workflow
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deprecated MSP build wrapper. Use `yogimass library build` instead.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more MSP files or directories.",
    )
    parser.add_argument(
        "--library",
        required=True,
        help="Output library path (JSON or SQLite).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when loading MSP files.",
    )
    args = parser.parse_args(argv)

    logger.warning(
        "This script is deprecated; prefer `yogimass library build --format msp` (input=%s, output=%s).",
        args.input,
        args.library,
    )
    workflow.build_library(
        args.input,
        args.library,
        input_format="msp",
        recursive=args.recursive,
        storage=None,
        overwrite=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
