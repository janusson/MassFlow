"""
Simple CLI for MassFlow.
Run `python -m MassFlow.cli` to start the CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal, getcontext
from pathlib import Path

import pandas as pd
from plotnine import aes, geom_segment, ggplot, labs, theme_bw

from MassFlow import __version__, io, processing
from MassFlow.config import ProcessingConfig


class ColoredFormatter(logging.Formatter):
    """
    Formatter to add colors to logging output based on log level.
    """

    grey = "\x1b[38;20m"
    green = "\x1b[32m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def __init__(self) -> None:
        super().__init__()
        self.formatters = {
            level: logging.Formatter(fmt) for level, fmt in self.FORMATS.items()
        }

    def format(self, record: logging.LogRecord) -> str:
        formatter = self.formatters.get(record.levelno)
        if formatter is None:
            return super().format(record)
        return formatter.format(record)


def setup_logging() -> None:
    """Set up logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        if sys.stderr.isatty():
            handler.setFormatter(ColoredFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        logger.addHandler(handler)


logger = logging.getLogger(__name__)


def run_process(args: argparse.Namespace) -> int:
    """
    Run the MassFlow processing pipeline from a config file.
    """
    from MassFlow.workflow import run_workflow

    try:
        run_workflow(Path(args.config))
        return 0
    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1


def run_clean(args: argparse.Namespace) -> int:
    """
    Run library cleaning operation.
    """
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    export_format = args.format

    logger.info(f"Starting clean operation on {input_path}")

    try:
        # 1. Load data
        file_format = input_path.suffix.lstrip(".")
        raw_spectra = io.load_spectra(input_path, file_format)

        # 2. Setup processing config (default values as this is a standalone CLI command)
        config = ProcessingConfig()

        # 3. Process spectra
        processed_spectra = processing.process_spectra(raw_spectra, config)

        # 4. Save results
        lib_name = input_path.stem
        export_path = output_dir / f"{lib_name}.{export_format}"

        if export_format == "pickle":
            io.save_spectra_to_pickle(processed_spectra, export_path)
        elif export_format == "msp":
            io.save_spectra_to_msp(processed_spectra, export_path)
        elif export_format == "mgf":
            io.save_spectra_to_mgf(processed_spectra, export_path)
        elif export_format == "json":
            io.save_spectra_to_json(processed_spectra, export_path)

        return 0
    except Exception as e:
        logger.error(f"Clean operation failed: {e}")
        return 1


def run_plot(args: argparse.Namespace) -> int:
    """
    Plot a mass spectrum from a spectral file.
    """
    input_path = Path(args.input)
    file_format = input_path.suffix.lstrip(".")

    logger.info(f"Loading spectra from {input_path}... please wait.")
    try:
        # Consume iterator for searching
        spectra = list(io.load_spectra(input_path, file_format))
    except Exception as e:
        logger.error(f"Failed to load spectra: {e}")
        return 1

    if not spectra:
        logger.warning("No spectra found in the file.")
        return 0

    names = [spec.get("name", spec.get("compound_name", "Unknown")) for spec in spectra]

    if args.more:
        for name in names:
            print(name)
        return 0

    if args.name is None:
        logger.info("Top 20 compounds:")
        for i, name in enumerate(names[:20]):
            print(f"{i + 1}. {name}")
        logger.info("\nTo plot a spectrum, run the command with the --name flag.")
        logger.info("To see all compound names, run with the --more flag.")
        return 0

    # Find the selected spectrum
    selected_spectrum = None
    target_name = args.name.lower()
    for spec in spectra:
        spec_name = (spec.get("name") or spec.get("compound_name") or "").lower()
        if spec_name == target_name:
            selected_spectrum = spec
            break

    if selected_spectrum:
        getcontext().prec = 50

        mz_decimal = [Decimal(str(m)) for m in selected_spectrum.peaks.mz]
        raw_intensity_decimal = [
            Decimal(str(i)) for i in selected_spectrum.peaks.intensities
        ]

        max_intensity_decimal = (
            max(raw_intensity_decimal) if raw_intensity_decimal else Decimal("0")
        )
        if max_intensity_decimal > Decimal("0"):
            intensity_decimal = [
                (i / max_intensity_decimal) * Decimal("100")
                for i in raw_intensity_decimal
            ]
        else:
            intensity_decimal = [Decimal("0")] * len(raw_intensity_decimal)

        df = pd.DataFrame({"mz": mz_decimal, "intensity": intensity_decimal})

        p = (
            ggplot(df, aes(x="mz", y="intensity"))
            + geom_segment(aes(x="mz", xend="mz", yend="intensity"), y=0)
            + theme_bw()
            + labs(
                title=selected_spectrum.get("name")
                or selected_spectrum.get("compound_name"),
                x="m/z",
                y="Relative Intensity",
            )
        )

        print(p)
        return 0
    else:
        logger.error(f"Spectrum with name '{args.name}' not found.")
        return 1


def main(argv: list[str] | None = None) -> int:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="MassFlow",
        description="MassFlow: Tandem MS/MS data analysis pipeline.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Clean command
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean and process a spectral library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    clean_parser.add_argument(
        "--input", required=True, help="Input library file (.msp or .mgf)"
    )
    clean_parser.add_argument(
        "--output-dir", required=True, help="Directory to save processed library"
    )
    clean_parser.add_argument(
        "--format",
        choices=["pickle", "msp", "mgf", "json"],
        default="pickle",
        help="Output format",
    )
    clean_parser.set_defaults(func=run_clean)

    # Plot command
    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot a spectrum from a spectral library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plot_parser.add_argument(
        "--input", required=True, help="Input library file (.msp, .mgf, .mzml)"
    )
    plot_parser.add_argument("--name", help="Name of the spectrum to plot.")
    plot_parser.add_argument(
        "--more", action="store_true", help="List all spectrum names."
    )
    plot_parser.set_defaults(func=run_plot)

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Run the MassFlow processing pipeline from a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    process_parser.add_argument("config", help="Path to config.yaml")
    process_parser.set_defaults(func=run_process)

    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
