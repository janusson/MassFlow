"""
Minimal Command-line interface (CLI) for Yogimass Core.
"""

from __future__ import annotations

import argparse
import sys
from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yogimass",
        description="Yogimass Core: Deterministic MS/MS spectrum processing.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Show help and roadmap if no arguments are provided
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        print("\nNote: Higher-level workflows and CLI features are currently in the development roadmap.")
        return 0
    if argv is not None and len(argv) == 0:
        parser.print_help()
        print("\nNote: Higher-level workflows and CLI features are currently in the development roadmap.")
        return 0

    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
