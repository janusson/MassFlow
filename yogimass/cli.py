"""
Minimal Command-line interface (CLI) for Yogimass.
"""

from __future__ import annotations

import argparse
import sys
from . import __version__

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yogimass",
        description="Yogimass: Config-first LC-MS/MS spectral similarity toolkit.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # config command group
    config_parser = subparsers.add_parser("config", help="Configuration driven workflows")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config subcommands")

    # config run
    run_parser = config_subparsers.add_parser("run", help="Run a workflow from a config file")
    run_parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file"
    )

    args = parser.parse_args(argv)

    if args.command == "config":
        if args.config_command == "run":
            print(f"Loading configuration from {args.config}...")
            print("Workflow execution is not yet implemented (Week 2 goal).")
            return 0
        else:
            config_parser.print_help()
            return 0

    # Show help if no arguments provided (or just the main command)
    if not args.command:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
