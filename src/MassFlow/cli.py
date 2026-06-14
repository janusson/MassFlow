"""
Command-line entry points for MassFlow.

This module defines the user-facing CLI for running MassFlow in a reproducible,
configuration-driven way. It wires together argument parsing, logging, and
dispatch for the major operational surfaces in the package:

- ``annotate`` for the end-to-end annotation workflow.
- ``watch`` for interactive, live-updating spectral annotation.
- ``init`` for generating a starter YAML configuration.
- ``db`` subcommands for building, inspecting, and merging SQLite libraries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from watchfiles import watch

from MassFlow import __version__
from MassFlow.log_config import setup_structured_logging

app = typer.Typer(
    help="MassFlow: A robust, config-first Python toolkit for local tandem mass spectrometry (MS/MS) annotation.",
    no_args_is_help=True,
    add_completion=False,
)

db_app = typer.Typer(help="Manage local SQLite spectral libraries.")
app.add_typer(db_app, name="db")

logger = logging.getLogger(__name__)
console = Console()


def setup_logging() -> None:
    """Configure process-wide logging for CLI execution."""
    setup_structured_logging(level=logging.INFO)


def version_callback(value: bool):
    if value:
        typer.echo(f"MassFlow version: {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, help="Show version."
    ),
):
    """
    MassFlow: A robust, config-first Python toolkit for local tandem mass spectrometry (MS/MS) annotation.
    """
    setup_logging()


@app.command("init")
def run_init(
    output: str = typer.Option(
        "massflow_config.yaml",
        "--output",
        help="Path where the configuration YAML should be created.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing configuration if it exists."
    ),
):
    """Generate a standardized MassFlow configuration template."""
    output_path = Path(output)
    if output_path.exists() and not force:
        logger.error(
            f"Configuration file already exists at {output_path}. Use --force to overwrite."
        )
        raise typer.Exit(1)

    template = """project:
  name: "My_MassFlow_Analysis"
  output_directory: "results/run_01"

input:
  input_path: "../MassFlow_Data/experiments/"
  library_path: "../MassFlow_Data/libraries/example_library.msp"
  format: "mzml"

processing:
  clean_metadata: true
  filter_by_intensity: true
  noise_threshold: 1000.0
  reduce_to_top_n_peaks: true
  n_max: 100
  normalize_intensity: true

similarity:
  algorithm: "cosine"
  ms1_tolerance: 0.02
  ms2_tolerance: 0.02
  min_score: 0.7
  min_matched_peaks: 3
  fdr_threshold: 0.05

export:
  format: "csv"
"""
    try:
        with open(output_path, "w") as f:
            f.write(template)
        logger.info(f"Initialized new MassFlow configuration at: {output_path}")
        console.print(
            f"[bold green]✓ Initialized new configuration at {output_path}[/bold green]"
        )
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        raise typer.Exit(1)


@app.command("annotate")
def run_annotate(
    config: str = typer.Option(
        ..., "--config", help="Path to configuration YAML file."
    ),
):
    """Run the stable MassFlow annotation pipeline using a YAML configuration file."""
    from MassFlow.config import MassFlowConfig
    from MassFlow.workflow import run_annotation_pipeline

    try:
        cfg = MassFlowConfig.from_yaml(config)
        run_annotation_pipeline(cfg, config_path=config)
        console.print(
            f"[bold green]✓ Annotation complete![/bold green] Results saved to {cfg.project.output_directory}"
        )
    except Exception as e:
        logger.error(f"Annotation failed: {e}")
        raise typer.Exit(1)


@app.command("convert")
def run_convert(
    input: str = typer.Option(
        ...,
        "--input",
        help="Path to input directory containing vendor files (.raw, .d).",
    ),
    output: str = typer.Option(
        ..., "--output", help="Path to output directory for converted .mzML files."
    ),
):
    """Convert vendor raw files to open formats (mzML) using ProteoWizard msconvert."""
    from MassFlow.convert import MSConvertNotFoundError, convert_directory

    input_path = Path(input)
    output_path = Path(output)

    if not input_path.is_dir():
        logger.error(f"Input path {input_path} is not a directory.")
        raise typer.Exit(1)

    try:
        count = convert_directory(input_path, output_path)
        console.print(
            f"[bold green]✓ Successfully converted {count} files.[/bold green]"
        )
    except MSConvertNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise typer.Exit(1)


@db_app.command("build")
def run_db_build(
    input: str = typer.Option(
        ..., "--input", help="Path to input spectral library file."
    ),
    output: str = typer.Option(
        ..., "--output", help="Path to output SQLite database file."
    ),
    config: str = typer.Option(
        ...,
        "--config",
        help="Path to configuration YAML file for processing parameters.",
    ),
    category: str = typer.Option(
        "default", "--category", help="Tag for categorization inside the database."
    ),
):
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

        console.print(
            f"[bold green]✓ Successfully processed and added {added} spectra to {output}.[/bold green]"
        )
    except Exception as e:
        logger.error(f"Database build failed: {e}", exc_info=True)
        raise typer.Exit(1)


@db_app.command("inspect")
def run_db_inspect(
    file: str = typer.Argument(..., help="SQLite spectral database file."),
):
    """Inspect a local spectral database to view statistics."""
    try:
        from MassFlow.database import SpectralDatabase

        db = SpectralDatabase(file)
        total = db.get_total_spectra_count()

        table = Table(title=f"Database Inspection: {file}", show_header=False)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        if total == 0:
            table.add_row("Status", "Empty (0 spectra)")
            console.print(table)
            return

        mz_min, mz_max = db.get_precursor_mz_range()
        cat_counts = db.get_category_counts()

        table.add_row("Total Spectra", str(total))
        table.add_row("Precursor m/z Range", f"{mz_min:.4f} to {mz_max:.4f}")

        console.print(table)

        cat_table = Table(title="Categories")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right", style="green")

        if cat_counts:
            for cat, count in cat_counts.items():
                cat_table.add_row(cat, str(count))
        else:
            cat_table.add_row("(None)", "0")

        console.print(cat_table)

    except Exception as e:
        logger.error(f"Database inspection failed: {e}", exc_info=True)
        raise typer.Exit(1)


@db_app.command("merge")
def run_db_merge(
    inputs: List[str] = typer.Option(
        ..., "--inputs", help="Paths to input SQLite database files."
    ),
    output: str = typer.Option(
        ..., "--output", help="Path to output merged SQLite database file."
    ),
):
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

        console.print(
            f"[bold green]✓ Successfully merged {total_added} spectra into {output}.[/bold green]"
        )
        out_db.close()
    except Exception as e:
        logger.error(f"Database merge failed: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command("watch")
def run_watch(
    config: str = typer.Option(
        ..., "--config", help="Path to configuration YAML file."
    ),
):
    """
    Watch the workspace for file changes and re-run the annotation pipeline dynamically.
    Outputs high-tech Rich tables that gracefully handle pane resizing in Tmux/Zellij.
    """
    import glob

    import polars as pl

    from MassFlow.config import MassFlowConfig
    from MassFlow.workflow import run_annotation_pipeline

    try:
        cfg = MassFlowConfig.from_yaml(config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise typer.Exit(1)

    input_path = Path(cfg.input.input_path)
    output_dir = Path(cfg.project.output_directory)

    # Render layout builder
    def generate_results_table() -> Table:
        """Reads recent CSV results and formats them via Rich."""
        table = Table(
            title="🛰️ MassFlow Interactive Annotation Results",
            caption="Listening for file changes... (Press Ctrl+C to exit)",
            expand=True,
            show_lines=False,
            header_style="bold cyan",
        )
        table.add_column("Query ID", style="dim", no_wrap=True)
        table.add_column("Reference Match", style="bold white")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Status", justify="center")

        # Find the latest results file
        ext = cfg.export.format.lower()
        search_ext = ext

        # Simplified find latest
        files = glob.glob(str(output_dir / f"*_results.{search_ext}"))
        if not files:
            table.add_row("...", "Waiting for first run to complete...", "...", "...")
            return table

        latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)

        try:
            if search_ext == "csv":
                df = pl.read_csv(latest_file)
            else:
                # mzTab — treat as tab-delimited for preview
                df = pl.read_csv(latest_file, separator="\t")

            # Limit preview to top 15 results
            df_preview = df.head(15)

            for row in df_preview.iter_rows(named=True):
                q_id = str(row.get("query_id", "Unknown"))
                ref = str(row.get("reference_name", "Unknown"))

                score = row.get("score")
                score_str = f"{score:.3f}" if score is not None else "N/A"

                status = str(row.get("Annotation_Status", "Unknown"))
                status_color = (
                    "green"
                    if status == "Matched"
                    else "yellow"
                    if status == "Putative"
                    else "red"
                )
                status_text = Text(status, style=f"bold {status_color}")

                table.add_row(q_id, ref, score_str, status_text)

            if len(df) > 15:
                table.add_row("...", f"+ {len(df) - 15} more rows", "...", "...")

        except Exception as e:
            table.add_row("Error reading results", str(e), "...", "...")

        return table

    def trigger_run():
        """Executes the pipeline and catches exceptions quietly."""
        try:
            # Silence core logger during interactive watch to prevent table corruption
            logging.getLogger("MassFlow").setLevel(logging.CRITICAL)
            run_annotation_pipeline(cfg, config_path=config)
        except Exception as e:
            console.print(f"[bold red]Pipeline Error:[/bold red] {e}")

    # Initial Run
    trigger_run()

    # Watch loop with Rich Live updating Table
    try:
        with Live(generate_results_table(), refresh_per_second=4, screen=False) as live:
            for changes in watch(input_path):
                # Trigger pipeline on file change
                trigger_run()
                # Update UI
                live.update(generate_results_table())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stopping watch mode.[/bold yellow]")


def main():
    app()


if __name__ == "__main__":
    main()
