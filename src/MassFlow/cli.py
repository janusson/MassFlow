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

import logging
import sys
from pathlib import Path

import click

from MassFlow import __version__
from MassFlow.log_config import setup_structured_logging


def setup_logging() -> None:
    """
    Configure process-wide logging for CLI execution.

    The CLI uses a single basic logging configuration shared across all
    subcommands. It relies on the fail-fast StructuredFormatter to output
    dev-friendly structured logs.

    Returns
    -------
    None
    """
    setup_structured_logging(level=logging.INFO)


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="MassFlow")
@click.pass_context
def main(ctx: click.Context) -> None:
    """
    MassFlow: A robust, config-first Python toolkit for local tandem mass spectrometry (MS/MS) annotation.

    It streamlines the process of loading experimental spectral data, applying matchms filters,
    scoring against reference libraries, and generating reproducible CSV results, all managed
    through a comprehensive YAML configuration.
    """
    setup_logging()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command("init")
@click.option(
    "--output",
    default="massflow_config.yaml",
    show_default=True,
    help="Path where the configuration YAML should be created.",
)
@click.option(
    "--force", is_flag=True, help="Overwrite existing configuration if it exists."
)
def run_init(output: str, force: bool) -> None:
    """Generate a standardized MassFlow configuration template. (Includes experimental flags)"""
    output_path = Path(output)
    if output_path.exists() and not force:
        logger.error(
            f"Configuration file already exists at {output_path}. Use --force to overwrite."
        )
        sys.exit(1)

    template = """project:
  name: "My_MassFlow_Analysis"
  output_directory: "results/run_01"

input:
  # SIBLING DIRECTORY PATTERN:
  # Keep your large experimental data and reference libraries in a sibling folder
  # (e.g., ../MassFlow_Data/) outside of your codebase to avoid bloating the Git repository.
  #
  # Use 'input_path' for a folder of files, or for a single file.
  # If 'input_path' points to a directory with mixed formats (e.g., .mzML and .msp),
  # set `format: null` below to let MassFlow automatically infer the format per file.
  #
  # Have vendor files (.d, .raw)? Run: `massflow convert --input ../MassFlow_Data/raw --output ../MassFlow_Data/experiments` first.
  input_path: "../MassFlow_Data/experiments/"
  library_path: "../MassFlow_Data/libraries/example_library.msp"

  # Format hint. Options: "mzml", "mzxml", "mgf", "msp", "sqlite", or `null` for auto-inference.
  format: "mzml"

processing:
  # ------------------------------------------------------------------
  # METADATA & PEAK PROCESSING
  # These cleaning steps are applied identically to both experimental data and reference libraries to ensure scientific integrity.
  # ------------------------------------------------------------------
  clean_metadata: true
  filter_by_intensity: true
  noise_threshold: 1000.0  # Absolute intensity threshold

  reduce_to_top_n_peaks: true
  n_max: 100               # Keep only top 100 most intense peaks per spectrum

  normalize_intensity: true

similarity:
  # ------------------------------------------------------------------
  # SEARCH & SCORING SETTINGS
  # ------------------------------------------------------------------
  algorithm: "cosine" # Available: "cosine", "modified_cosine", "spec2vec", "ms2deepscore", "consensus", "cascade"
  ms1_tolerance: 0.02      # Precursor tolerance in Da
  # resolution_ppm: 10.0   # Optional: Precursor resolution in ppm (overrides ms1_tolerance)
  ms2_tolerance: 0.02      # Fragment tolerance in Da

  min_score: 0.7           # Minimum similarity score (0.0 to 1.0)
  min_matched_peaks: 3     # Minimum number of matching fragments
  fdr_threshold: 0.05      # Target-Decoy FDR limit

workflow:
  # Set to true to generate a GraphML network connecting queries and library hits
  perform_networking: false

export:
  # Preferred formats: "xlsx" (Excel), "csv", "json", or "parquet"
  format: "xlsx"
"""
    try:
        with open(output_path, "w") as f:
            f.write(template)
        logger.info(f"Initialized new MassFlow configuration at: {output_path}")
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        sys.exit(1)


@main.command("annotate")
@click.option("--config", required=True, help="Path to configuration YAML file.")
def run_annotate(config: str) -> None:
    """
    Run the stable MassFlow annotation pipeline using a YAML configuration file.

    This is the primary command for reproducible MS/MS data analysis.
    """
    from MassFlow.config import MassFlowConfig
    from MassFlow.workflow import run_annotation_pipeline

    try:
        cfg = MassFlowConfig.from_yaml(config)
        run_annotation_pipeline(cfg, config_path=config)
    except Exception as e:
        logger.error(f"Annotation failed: {e}")
        sys.exit(1)


@main.command("convert")
@click.option(
    "--input",
    required=True,
    help="Path to input directory containing vendor files (.raw, .d).",
)
@click.option(
    "--output",
    required=True,
    help="Path to output directory for converted .mzML files.",
)
def run_convert(input: str, output: str) -> None:
    """
    Convert vendor raw files to open formats (mzML) using ProteoWizard msconvert.
    Requires msconvert to be installed and available on the system PATH.
    """
    from MassFlow.convert import MSConvertNotFoundError, convert_directory

    input_path = Path(input)
    output_path = Path(output)

    if not input_path.is_dir():
        logger.error(f"Input path {input_path} is not a directory.")
        sys.exit(1)

    try:
        count = convert_directory(input_path, output_path)
        logger.info(f"Conversion complete. Successfully converted {count} files.")
    except MSConvertNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


@main.group("db", invoke_without_command=True)
@click.pass_context
def db_group(ctx: click.Context) -> None:
    """Manage local SQLite spectral libraries: build, inspect, or merge databases for efficient reuse."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@db_group.command("build")
@click.option(
    "--input",
    required=True,
    help="Path to input spectral library file (e.g., .msp, .mgf).",
)
@click.option(
    "--output",
    required=True,
    help="Path to output SQLite database file (e.g., my_library.db).",
)
@click.option(
    "--config",
    required=True,
    help="Path to configuration YAML file for processing parameters.",
)
@click.option(
    "--category",
    default="default",
    show_default=True,
    help="Tag for categorization inside the database.",
)
def run_db_build(input: str, output: str, config: str, category: str) -> None:
    """Process a raw library file and store it in an optimized SQLite database."""
    try:
        from MassFlow import io, processing
        from MassFlow.config import MassFlowConfig
        from MassFlow.database import SpectralDatabase

        logger.info(f"Loading configuration from {config}")
        cfg = MassFlowConfig.from_yaml(config)

        logger.info(f"Initializing database at {output}")
        db = SpectralDatabase(output)

        logger.info(f"Streaming and processing spectra from {input}")
        raw_spectra = io.load_spectra(Path(input))
        cleaned_spectra = processing.process_spectra(raw_spectra, cfg.processing)

        added = db.add_spectra(cleaned_spectra, category=category)

        if added == 0:
            raise ValueError(f"No valid spectra were extracted from {input}.")

        logger.info(
            f"Successfully processed and added {added} spectra to the database."
        )
    except Exception as e:
        logger.error(f"Database build failed: {e}", exc_info=True)
        sys.exit(1)


@db_group.command("inspect")
@click.argument("file", required=True)
def run_db_inspect(file: str) -> None:
    """Inspect a local spectral database to view statistics."""
    try:
        from MassFlow.database import SpectralDatabase

        db = SpectralDatabase(file)

        total = db.get_total_spectra_count()
        if total == 0:
            click.echo("\n" + "=" * 50)
            click.echo(f"DATABASE INSPECTION: {file}")
            click.echo("=" * 50)
            click.echo("Database is empty (0 spectra).")
            click.echo("=" * 50 + "\n")
            sys.exit(0)

        cat_counts = db.get_category_counts()
        mz_min, mz_max = db.get_precursor_mz_range()

        click.echo("\n" + "=" * 50)
        click.echo(f"DATABASE INSPECTION: {file}")
        click.echo("=" * 50)
        click.echo(f"Total Spectra: {total}")
        click.echo(f"Precursor m/z Range: {mz_min:.4f} to {mz_max:.4f}")
        click.echo("\nCategories:")
        if cat_counts:
            for cat, count in cat_counts.items():
                click.echo(f"  - {cat}: {count} spectra")
        else:
            click.echo("  - (None)")
        click.echo("=" * 50 + "\n")
    except Exception as e:
        logger.error(f"Database inspection failed: {e}", exc_info=True)
        sys.exit(1)


@db_group.command("merge")
@click.option(
    "--inputs",
    required=True,
    multiple=True,
    help="Paths to input SQLite database files.",
)
@click.option(
    "--output", required=True, help="Path to output merged SQLite database file."
)
def run_db_merge(inputs: tuple[str, ...], output: str) -> None:
    """Merge multiple local spectral databases into a single new database."""
    try:
        from MassFlow.database import SpectralDatabase

        logger.info(f"Initializing merged database at {output}")
        out_db = SpectralDatabase(output)

        total_added = 0
        for input_db_path in inputs:
            logger.info(f"Streaming from input database: {input_db_path}")
            in_db = SpectralDatabase(input_db_path)

            added = out_db.add_spectra(in_db.get_spectra(), category="merged")
            total_added += added

            if added == 0:
                logger.warning(f"No valid spectra were found in {input_db_path}")
            else:
                logger.info(f"Added {added} spectra from {input_db_path}")

            in_db.close()

        if total_added == 0:
            raise ValueError("No valid spectra were merged from the input databases.")

        logger.info(f"Successfully merged {total_added} total spectra into {output}.")
        out_db.close()
    except Exception as e:
        logger.error(f"Database merge failed: {e}", exc_info=True)
        sys.exit(1)


@main.command("visualize")
@click.argument("graphml_path", type=click.Path(exists=True, path_type=str))
@click.option(
    "--output",
    "-o",
    default="network.html",
    show_default=True,
    help="Path to save the output HTML visualization.",
)
def run_visualize(graphml_path: str, output: str) -> None:
    """
    Generate an interactive HTML visualization from a GraphML network.

    GRAPHML_PATH is the path to the .graphml file generated by the annotation workflow.
    """
    setup_logging()

    try:
        from MassFlow.visualization import visualize_graphml
    except ImportError as e:
        logger.error(f"Failed to load visualization module: {e}")
        click.echo(
            "Visualization dependencies are missing. Install them with: "
            "uv pip install pyvis networkx",
            err=True,
        )
        sys.exit(1)

    try:
        visualize_graphml(graphml_path=graphml_path, output_html=output)
        click.echo(f"Interactive visualization successfully created at: {output}")
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
