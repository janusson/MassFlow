#!/usr/bin/env python3
"""
Deprecated MS-DIAL fragment search script.

This wrapper only loads MS-DIAL outputs via ``yogimass.workflow.load_data`` and
reminds users to migrate to config-driven runs.
"""

from __future__ import annotations

import argparse
import sys

from yogimass import workflow
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deprecated MS-DIAL helper. Prefer `yogimass config run` with msdial input.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing MS-DIAL alignment outputs.",
    )
    parser.add_argument(
        "--msdial-output",
        help="Optional output dir for cleaned MS-DIAL tables.",
    )
    args = parser.parse_args(argv)

    logger.warning(
        "This script is deprecated; prefer using a config file with `yogimass config run` for MS-DIAL processing."
    )
    workflow.load_data(
        args.input,
        input_format="msdial",
        recursive=False,
        msdial_output_dir=args.msdial_output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
