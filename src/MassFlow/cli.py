"""
Command-Line Interface (CLI) for MassFlow.

This module provides the entry point for the MassFlow application, handling
command-line argument parsing, logging configuration, and dispatching execution
to specific subcommands (e.g., annotation). It serves as the primary interface
for users interacting with the MassFlow pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from MassFlow import __version__, io
from MassFlow.config import MassFlowConfig
from MassFlow.workflow import run_annotation_pipeline


# Setup logging (Simplified for brevity but effective)
def setup_logging() -> None:
    """
    Configure the basic logging settings for the application.

    This function initializes the global logging configuration with a specific
    format string that includes the timestamp, logger name, level name, and
    message. The default logging level is set to INFO.

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


logger = logging.getLogger(__name__)


def run_annotate(args: argparse.Namespace) -> int:
    """
    Execute the annotation pipeline based on provided arguments.

    This function loads the MassFlow configuration from a YAML file specified
    in the arguments and instantiates the annotation pipeline. It encapsulates
    the execution within a try-except block to handle runtime errors gracefully.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments. It must include the ``config`` attribute,
        which specifies the file path to the configuration YAML.

    Returns
    -------
    int
        Returns 0 if the pipeline executes successfully.
        Returns 1 if an exception occurs during configuration loading or execution.

    Raises
    ------
    Exception
        Caught internally. Logs the error message and returns exit code 1.
    """
    try:
        config = MassFlowConfig.from_yaml(args.config)
        run_annotation_pipeline(config)
        return 0
    except Exception as e:
        logger.error(f"Annotation failed: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the MassFlow command-line interface.

    This function sets up the logging environment, configures the argument parser
    with supported subcommands (e.g., ``annotate``), and dispatches execution
    to the appropriate function based on the user input.

    Parameters
    ----------
    argv : list[str] or None, optional
        A list of command-line arguments to parse. If None, arguments are
        retrieved from ``sys.argv``. Default is None.

    Returns
    -------
    int
        The exit status code. Returns the result of the invoked subcommand,
        or 0 if no command was specified (displays help).
    """
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
        "--config", required=True, help="Path to configuration YAML file."
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
