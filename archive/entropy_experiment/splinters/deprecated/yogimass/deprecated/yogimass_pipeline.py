#!/usr/bin/env python3
"""
Deprecated legacy pipeline script.

This wrapper now delegates to ``yogimass.workflow.run_from_config``.
Prefer: ``yogimass config run --config <path>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yogimass import workflow
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deprecated pipeline wrapper. Use `yogimass config run` instead.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/simple_workflow.yaml"),
        help="Path to a YAML/JSON config file (default: examples/simple_workflow.yaml).",
    )
    args = parser.parse_args(argv)

    logger.warning(
        "This script is deprecated; prefer using `yogimass config run --config %s`.",
        args.config,
    )
    workflow.run_from_config(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
