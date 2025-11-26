"""
Unified command-line interface for Yogimass workflows.

Responsibilities:
- Parse user commands/subcommands and dispatch to ``yogimass.workflow`` helpers.
- Provide entrypoints for config runs, library build/search/curate, network build, and legacy clean.
- Emit user-facing logs/errors.

Depends on:
- ``yogimass.workflow`` for core operations.
- ``yogimass.reporting`` for optional output writing.

Avoids:
- Business logic (delegates to workflow/reporting).
- Deep coupling to specific file formats beyond flags passed through to workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yogimass import pipeline, workflow
from yogimass.reporting import summarize_network, write_network_summary, write_search_results
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yogimass",
        description="End-to-end LC-MS/MS workbench and utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_config_subparser(subparsers)
    _add_library_subparser(subparsers)
    _add_network_subparser(subparsers)
    _add_clean_subparser(subparsers)  # legacy support

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - surfaced to users
        logger.error("%s", exc)
        return 1


def _add_config_subparser(subparsers):
    config_parser = subparsers.add_parser("config", help="Run workflows from a config file.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    run_parser = config_subparsers.add_parser("run", help="Execute a YAML/JSON config.")
    run_parser.add_argument("--config", required=True, help="Path to the configuration file.")
    run_parser.set_defaults(func=_run_config_command)
    return run_parser


def _add_library_subparser(subparsers):
    library_parser = subparsers.add_parser("library", help="Build or search spectral libraries.")
    library_subparsers = library_parser.add_subparsers(dest="library_command", required=True)

    build_parser = library_subparsers.add_parser("build", help="Build or update a local library.")
    build_parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more spectrum files or directories.",
    )
    build_parser.add_argument(
        "--format",
        choices=["mgf", "msp"],
        default="mgf",
        help="Input format (default: mgf).",
    )
    build_parser.add_argument(
        "--library",
        required=True,
        help="Output path for the library (JSON or SQLite).",
    )
    build_parser.add_argument(
        "--storage",
        choices=["json", "sqlite"],
        default=None,
        help="Force storage type (defaults to extension).",
    )
    build_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when loading spectra.",
    )
    build_parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing entries in the library.",
    )
    build_parser.set_defaults(func=_run_library_build_command, overwrite=True)

    search_parser = library_subparsers.add_parser("search", help="Search queries against a library.")
    search_parser.add_argument(
        "--queries",
        nargs="+",
        required=True,
        help="Query spectrum files or directories.",
    )
    search_parser.add_argument(
        "--library",
        required=True,
        help="Path to the library JSON/SQLite file.",
    )
    search_parser.add_argument(
        "--format",
        choices=["mgf", "msp"],
        default="mgf",
        help="Query spectrum format (default: mgf).",
    )
    search_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of hits to return per query.",
    )
    search_parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum cosine score to report.",
    )
    search_parser.add_argument(
        "--output",
        help="Optional CSV/JSON path to write search results.",
    )
    search_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when loading queries.",
    )
    search_parser.set_defaults(func=_run_library_search_command)

    curate_parser = library_subparsers.add_parser("curate", help="Curate and QC an existing library.")
    curate_parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw library (JSON or SQLite).",
    )
    curate_parser.add_argument(
        "--output",
        required=True,
        help="Path for the curated library (same type as input is recommended).",
    )
    curate_parser.add_argument(
        "--qc-report",
        help="Optional path for the QC/curation report (defaults next to --output).",
    )
    curate_parser.add_argument(
        "--min-peaks",
        type=int,
        default=3,
        help="Minimum number of peaks required to keep a spectrum (default: 3).",
    )
    curate_parser.add_argument(
        "--min-tic",
        type=float,
        default=0.05,
        help="Minimum total ion current (TIC) to keep a spectrum (default: 0.05).",
    )
    curate_parser.add_argument(
        "--max-single-peak-fraction",
        type=float,
        default=0.9,
        help="Drop spectra where a single peak exceeds this fraction of TIC (default: 0.9).",
    )
    curate_parser.add_argument(
        "--precursor-tolerance",
        type=float,
        default=0.01,
        help="Precursor m/z tolerance for duplicate detection (default: 0.01).",
    )
    curate_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for merging duplicates (default: 0.95).",
    )
    curate_parser.add_argument(
        "--allow-missing-precursor",
        action="store_true",
        help="Keep spectra even if precursor m/z is missing.",
    )
    curate_parser.set_defaults(func=_run_library_curate_command)
    return library_parser


def _add_network_subparser(subparsers):
    network_parser = subparsers.add_parser(
        "network",
        help="Build MS/MS similarity networks.",
    )
    network_subparsers = network_parser.add_subparsers(dest="network_command", required=True)

    build_parser = network_subparsers.add_parser(
        "build",
        help="Build a similarity network from spectra or a library.",
    )
    source_group = build_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        help="Directory containing MGF spectra to include in the network.",
    )
    source_group.add_argument(
        "--library",
        help="Path to a LocalSpectralLibrary (JSON or SQLite).",
    )
    build_parser.add_argument(
        "--metric",
        choices=["cosine", "modified_cosine", "spec2vec", "modified-cosine"],
        default="cosine",
        help="Similarity metric to use (default: cosine).",
    )
    build_parser.add_argument(
        "--threshold",
        type=float,
        help="Similarity threshold for edge creation (threshold mode).",
    )
    build_parser.add_argument(
        "--knn",
        type=int,
        help="Number of nearest neighbors per node (k-NN mode).",
    )
    build_parser.add_argument(
        "--output",
        required=True,
        help="Output graph file (.csv, .graphml, .gexf, or .gml).",
    )
    build_parser.add_argument(
        "--summary",
        help="Optional JSON/text file for a brief network summary.",
    )
    build_parser.add_argument(
        "--directed",
        action="store_true",
        help="Keep edges directed (default: undirected).",
    )
    build_parser.set_defaults(func=_run_network_build_command)
    return build_parser


def _add_clean_subparser(subparsers):
    clean_parser = subparsers.add_parser(
        "clean",
        help="(Legacy) clean and export all libraries in a directory.",
    )
    clean_parser.add_argument(
        "input_dir",
        help="Directory containing MGF or MSP libraries.",
    )
    clean_parser.add_argument(
        "output_dir",
        help="Directory for cleaned exports (created if missing).",
    )
    clean_parser.add_argument(
        "--type",
        choices=["mgf", "msp"],
        default="mgf",
        help="Library format to process (default: mgf).",
    )
    clean_parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        metavar="FMT",
        help="Export formats (mgf, msp, json, pickle). Defaults to the library type.",
    )
    clean_parser.set_defaults(func=_run_clean_command)
    return clean_parser


def _run_config_command(args) -> int:
    workflow.run_from_config(args.config)
    return 0


def _run_library_build_command(args) -> int:
    library = workflow.build_library(
        args.input,
        args.library,
        input_format=args.format,
        recursive=args.recursive,
        storage=args.storage,
        overwrite=args.overwrite,
    )
    logger.info("Library ready at %s (%d entries)", library.path, len(library))
    return 0


def _run_library_search_command(args) -> int:
    results = workflow.search_library(
        args.queries,
        args.library,
        input_format=args.format,
        recursive=args.recursive,
        top_n=args.top_n,
        min_score=args.min_score,
    )
    if args.output:
        write_search_results(results, args.output)
    logger.info("Generated %d search hits", len(results))
    return 0


def _run_library_curate_command(args) -> int:
    config = {
        "min_peaks": args.min_peaks,
        "min_total_ion_current": args.min_tic,
        "max_single_peak_fraction": args.max_single_peak_fraction,
        "precursor_tolerance": args.precursor_tolerance,
        "similarity_threshold": args.similarity_threshold,
        "require_precursor_mz": not args.allow_missing_precursor,
    }
    result = workflow.curate_library(
        args.input,
        args.output,
        config=config,
        qc_report_path=args.qc_report,
    )
    summary = result.summary
    logger.info(
        "Curation complete: kept=%d, dropped=%d, merged_groups=%d (output=%s, report=%s)",
        summary.get("kept", 0),
        summary.get("dropped", 0),
        summary.get("merged_groups", 0),
        summary.get("output_library", args.output),
        summary.get("qc_report", args.qc_report),
    )
    return 0


def _run_network_build_command(args) -> int:
    undirected = not args.directed
    nodes, edges = workflow.build_network(
        input_dir=args.input,
        library_path=args.library,
        metric=args.metric,
        threshold=args.threshold,
        knn=args.knn,
        undirected=undirected,
        output_path=args.output,
    )
    if args.summary:
        summary = summarize_network(nodes, edges)
        write_network_summary(summary, args.summary)
    logger.info("Built network with %d nodes and %d edges", len(nodes), len(edges))
    return 0


def _run_clean_command(args) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_formats = tuple(args.formats) if args.formats else (args.type,)

    if args.type == "mgf":
        exported = pipeline.batch_clean_mgf_libraries(
            input_dir,
            output_dir,
            export_formats=export_formats,
        )
    else:
        exported = pipeline.batch_clean_msp_libraries(
            input_dir,
            output_dir,
            export_formats=export_formats,
        )

    if exported:
        logger.info("Exported %d files to %s", len(exported), output_dir)
    else:
        logger.warning("No libraries were processed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
