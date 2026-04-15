"""
Command-line entry points for MassFlow.

This module defines the user-facing CLI for running MassFlow in a reproducible,
configuration-driven way. It wires together argument parsing, logging, and
dispatch for the major operational surfaces in the package:

- ``annotate`` for the end-to-end annotation workflow.
- ``init`` for generating a starter YAML configuration.
- ``db`` subcommands for building, inspecting, and merging SQLite libraries.

The CLI intentionally keeps domain logic out of this layer. It validates the
requested command shape, translates arguments into configuration objects or file
paths, and delegates the substantive work to workflow and database
modules.
"""

from __future__ import annotations

import argparse
import logging
import sys

from MassFlow import __version__
from MassFlow.config import MassFlowConfig
from MassFlow.workflow import run_annotation_pipeline


# Setup logging (Simplified for brevity but effective)
def setup_logging() -> None:
    """
    Configure process-wide logging for CLI execution.

    The CLI uses a single basic logging configuration shared across all
    subcommands. The formatter includes the timestamp, logger name, level, and
    message, and the default verbosity is ``INFO``.

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
    Run the annotation workflow from a YAML configuration file.

    This command is the primary batch execution path for MassFlow. It loads the
    user-supplied configuration, validates it through
    :meth:`MassFlow.config.MassFlowConfig.from_yaml`, and then hands execution
    to :func:`MassFlow.workflow.run_annotation_pipeline`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Must include ``config``, the path to the YAML
        configuration file.

    Returns
    -------
    int
        ``0`` if the workflow completes successfully, otherwise ``1``.

    Notes
    -----
    Exceptions are handled inside the command function so the CLI can return a
    stable exit code for shell scripting and CI usage.
    """
    try:
        config = MassFlowConfig.from_yaml(args.config)
        run_annotation_pipeline(config, config_path=args.config)
        return 0
    except Exception as e:
        logger.error(f"Annotation failed: {e}")
        return 1


def run_init(args: argparse.Namespace) -> int:
    """
    Write a starter MassFlow configuration file.

    The template is defined inline in this module and captures the expected
    config-first shape of a typical annotation run, including project, input,
    processing, and similarity sections.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Must include ``output`` and may include ``force``
        to allow overwriting an existing file.

    Returns
    -------
    int
        ``0`` if the template is written, otherwise ``1``.

    Notes
    -----
    If the target path already exists and ``--force`` is not set, the function
    logs an error and returns without modifying the file.
    """
    try:
        from pathlib import Path

        output_path = Path(args.output)
        if output_path.exists() and not args.force:
            logger.error(
                f"Configuration file already exists at {output_path}. Use --force to overwrite."
            )
            return 1

        # Basic template content based on standard_config.yaml
        template = """project:
  name: "Standard_Annotation_Project"
  output_directory: "results/standard_analysis"

input:
  # Mandatory paths to your spectral data
  file_path: "data/experiments/experiment.mzML"
  # data_directory: "data/experiments/" # Alternatively, use a directory of files
  library_path: "data/libraries/library.msp"
  format: "mzml"

processing:
  # ------------------------------------------------------------------
  # METADATA PROCESSING
  # These toggles map to matchms metadata filters.
  # ------------------------------------------------------------------
  clean_metadata: true
  add_retention_time: true
  repair_inchi_inchikey_smiles: true
  derive_adduct_from_name: true
  derive_formula_from_name: true
  clean_compound_name: true
  derive_ionmode: true
  make_charge_int: true

  # ------------------------------------------------------------------
  # PEAK PROCESSING
  # These settings are applied to both query and reference spectra.
  # ------------------------------------------------------------------
  filter_by_intensity: true
  noise_threshold: 1000.0
  min_intensity: 0.0

  filter_min_peaks: true
  min_peaks: 5

  filter_by_mz: true
  mz_min: 0.0
  mz_max: 1000.0

  reduce_to_top_n_peaks: true
  n_max: 100

  normalize_intensity: true

similarity:
  # ------------------------------------------------------------------
  # CORE SEARCH SETTINGS
  # Stable options: "cosine", "modified_cosine"
  # ------------------------------------------------------------------
  algorithm: "cosine"
  ms1_tolerance: 10.0
  ms2_tolerance: 0.02
  tolerance_unit: "Da"
  min_score: 0.6
  min_matched_peaks: 3
  fdr_threshold: 0.05

export:
  # The current stable annotation workflow writes CSV result tables.
  format: "csv"
"""
        with open(output_path, "w") as f:
            f.write(template)

        logger.info(f"Initialized new MassFlow configuration at: {output_path}")
        return 0
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        return 1


def run_db_build(args: argparse.Namespace) -> int:
    """
    Build a SQLite spectral library from an input file.

    Spectra are streamed from the input path through the same
    :mod:`MassFlow.processing` pipeline used during annotation and then inserted
    into a :class:`MassFlow.database.SpectralDatabase`. This creates a reusable
    local cache for faster repeated searches and lighter memory use than large
    text-based library formats.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Must include ``input``, ``output``, and
        ``config``; may include ``category``.

    Returns
    -------
    int
        ``0`` on success, otherwise ``1``.

    Notes
    -----
    Input loading is delegated to :func:`MassFlow.io.load_spectra`, so vendor
    raw formats still require pre-conversion before database ingestion.
    """
    try:
        from MassFlow import io, processing
        from MassFlow.database import SpectralDatabase

        logger.info(f"Loading configuration from {args.config}")
        config = MassFlowConfig.from_yaml(args.config)

        logger.info(f"Initializing database at {args.output}")
        db = SpectralDatabase(args.output)

        logger.info(f"Streaming and processing spectra from {args.input}")
        raw_spectra = io.load_spectra(args.input)
        cleaned_spectra = processing.process_spectra(raw_spectra, config.processing)

        added = db.add_spectra(cleaned_spectra, category=args.category)

        if added == 0:
            raise ValueError(f"No valid spectra were extracted from {args.input}.")

        logger.info(
            f"Successfully processed and added {added} spectra to the database."
        )

        return 0
    except Exception as e:
        logger.error(f"Database build failed: {e}", exc_info=True)
        return 1


def run_db_inspect(args: argparse.Namespace) -> int:
    """
    Print summary statistics for a local spectral database.

    The inspection path uses dedicated lightweight database queries instead of
    materializing full ``matchms.Spectrum`` objects.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Must include ``file``, the path to the SQLite
        database to inspect.

    Returns
    -------
    int
        ``0`` on success, otherwise ``1``.
    """
    try:
        from MassFlow.database import SpectralDatabase

        db = SpectralDatabase(args.file)

        total = db.get_total_spectra_count()
        if total == 0:
            print("\n" + "=" * 50)
            print(f"DATABASE INSPECTION: {args.file}")
            print("=" * 50)
            print("Database is empty (0 spectra).")
            print("=" * 50 + "\n")
            return 0

        cat_counts = db.get_category_counts()
        mz_min, mz_max = db.get_precursor_mz_range()

        print("\n" + "=" * 50)
        print(f"DATABASE INSPECTION: {args.file}")
        print("=" * 50)
        print(f"Total Spectra: {total}")
        print(f"Precursor m/z Range: {mz_min:.4f} to {mz_max:.4f}")
        print("\nCategories:")
        if cat_counts:
            for cat, count in cat_counts.items():
                print(f"  - {cat}: {count} spectra")
        else:
            print("  - (None)")
        print("=" * 50 + "\n")

        return 0
    except Exception as e:
        logger.error(f"Database inspection failed: {e}", exc_info=True)
        return 1


def run_db_merge(args: argparse.Namespace) -> int:
    """
    Merge multiple spectral databases into a new SQLite database.

    Each input database is streamed via ``get_spectra()`` and inserted into the
    destination database with the category label ``merged``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Must include ``inputs`` and ``output``.

    Returns
    -------
    int
        ``0`` on success, otherwise ``1``.

    Notes
    -----
    The destination database is created up front. Existing input databases are
    read sequentially to keep the merge path simple and memory bounded.
    """
    try:
        from MassFlow.database import SpectralDatabase

        logger.info(f"Initializing merged database at {args.output}")
        out_db = SpectralDatabase(args.output)

        total_added = 0
        for input_db_path in args.inputs:
            logger.info(f"Streaming from input database: {input_db_path}")
            in_db = SpectralDatabase(input_db_path)

            # Since get_spectra returns an iterator, add_spectra will batch insert smoothly
            added = out_db.add_spectra(in_db.get_spectra(), category="merged")
            total_added += added

            if added == 0:
                logger.warning(f"No valid spectra were found in {input_db_path}")
            else:
                logger.info(f"Added {added} spectra from {input_db_path}")

            in_db.close()

        if total_added == 0:
            raise ValueError("No valid spectra were merged from the input databases.")

        logger.info(
            f"Successfully merged {total_added} total spectra into {args.output}."
        )
        out_db.close()

        return 0
    except Exception as e:
        logger.error(f"Database merge failed: {e}", exc_info=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for MassFlow.

    This function configures the top-level parser, registers all supported
    subcommands, parses ``argv``, and dispatches to the selected command
    handler. If no command is supplied, it prints help and exits successfully.

    Parameters
    ----------
    argv : list[str] or None, optional
        A list of command-line arguments to parse. If None, arguments are
        retrieved from ``sys.argv``. Default is None.

    Returns
    -------
    int
        Exit status code from the invoked subcommand, or ``0`` if help is
        displayed instead of running a command.
    """
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="MassFlow",
        description="""MassFlow: A robust, config-first Python toolkit for local tandem mass spectrometry (MS/MS) annotation. It streamlines the process of loading experimental spectral data, applying matchms filters, scoring against reference libraries, and generating reproducible CSV results, all managed through a comprehensive YAML configuration. Some advanced features (e.g., ML engines, networking) are considered experimental.""",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Generate a standardized MassFlow configuration template. (Includes experimental flags)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    init_parser.add_argument(
        "--output",
        default="massflow_config.yaml",
        help="Path where the configuration YAML should be created.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration if it exists.",
    )
    init_parser.set_defaults(func=run_init)

    # Annotate command
    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Run the stable MassFlow annotation pipeline using a YAML configuration file. This is the primary command for reproducible MS/MS data analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    annotate_parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file."
    )
    annotate_parser.set_defaults(func=run_annotate)

    # Database Subcommands
    db_parser = subparsers.add_parser(
        "db",
        help="Manage local SQLite spectral libraries: build, inspect, or merge databases for efficient reuse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    db_subparsers = db_parser.add_subparsers(
        dest="db_command", help="Database command to run"
    )

    # DB Build command
    db_build_parser = db_subparsers.add_parser(
        "build",
        help="Process a raw library file and store it in an optimized SQLite database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    db_build_parser.add_argument(
        "--input",
        required=True,
        help="Path to input spectral library file (e.g., .msp, .mgf).",
    )
    db_build_parser.add_argument(
        "--output",
        required=True,
        help="Path to output SQLite database file (e.g., my_library.db).",
    )
    db_build_parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration YAML file for processing parameters.",
    )
    db_build_parser.add_argument(
        "--category",
        default="default",
        help="Tag for categorization inside the database.",
    )
    db_build_parser.set_defaults(func=run_db_build)

    # DB Inspect command
    db_inspect_parser = db_subparsers.add_parser(
        "inspect",
        help="Inspect a local spectral database to view statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    db_inspect_parser.add_argument(
        "file", help="Path to the SQLite database file to inspect."
    )
    db_inspect_parser.set_defaults(func=run_db_inspect)

    # DB Merge command
    db_merge_parser = db_subparsers.add_parser(
        "merge",
        help="Merge multiple local spectral databases into a single new database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    db_merge_parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to input SQLite database files.",
    )
    db_merge_parser.add_argument(
        "--output", required=True, help="Path to output merged SQLite database file."
    )
    db_merge_parser.set_defaults(func=run_db_merge)

    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        # We also need to handle the case where someone types `massflow db` but no subcommand
        if args.command == "db" and getattr(args, "db_command", None) is None:
            db_parser.print_help()
            return 0
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
