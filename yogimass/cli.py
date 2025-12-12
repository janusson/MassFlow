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
import csv

import numpy as np
from yogimass import pipeline, workflow
from yogimass.config import ConfigError, load_config
from yogimass.networking.exporters import export_network_for_cytoscape
from yogimass.networking.network import SimilarityEdge, SpectrumNode
from yogimass.reporting import summarize_network, write_network_summary, write_search_results
from yogimass.similarity.library import LocalSpectralLibrary
from yogimass.similarity.processing import SpectrumProcessor
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)
_UNSET = object()


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(f"Expected a number, got '{value}'.") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"Value must be non-negative, got {value}.")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(f"Expected an integer, got '{value}'.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"Value must be greater than zero, got {value}.")
    return number


def _positive_int_or_none(value: str) -> int | None:
    if str(value).lower() == "none":
        return None
    return _positive_int(value)


def _add_processor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--similarity.processor.normalization",
        dest="processor_normalization",
        choices=["tic", "basepeak", "none"],
        help="Intensity normalization applied before filtering (tic/basepeak/none).",
    )
    parser.add_argument(
        "--similarity.processor.min-relative-intensity",
        dest="processor_min_relative_intensity",
        type=_non_negative_float,
        help="Relative intensity threshold (fraction of base peak) for filtering.",
    )
    parser.add_argument(
        "--similarity.processor.min-absolute-intensity",
        dest="processor_min_absolute_intensity",
        type=_non_negative_float,
        help="Absolute intensity cutoff for filtering.",
    )
    parser.add_argument(
        "--similarity.processor.max-peaks",
        dest="processor_max_peaks",
        type=_positive_int_or_none,
        default=_UNSET,
        help="Keep at most this many peaks after filtering (use 'None' to disable).",
    )
    parser.add_argument(
        "--similarity.processor.mz-dedup-tolerance",
        dest="processor_mz_dedup_tolerance",
        type=_non_negative_float,
        help="m/z window (in Da) for deduplicating peaks by apex intensity.",
    )
    parser.add_argument(
        "--similarity.processor.float-dtype",
        dest="processor_float_dtype",
        choices=["float32", "float64"],
        help="Force output dtype for processed peaks (float64 default).",
    )


def _processor_from_args(args) -> SpectrumProcessor | None:
    provided = False
    kwargs: dict[str, object] = {}

    normalization = getattr(args, "processor_normalization", None)
    if normalization is not None:
        provided = True
        kwargs["normalization"] = None if normalization == "none" else normalization

    min_rel = getattr(args, "processor_min_relative_intensity", None)
    if min_rel is not None:
        provided = True
        kwargs["min_relative_intensity"] = min_rel

    min_abs = getattr(args, "processor_min_absolute_intensity", None)
    if min_abs is not None:
        provided = True
        kwargs["min_absolute_intensity"] = min_abs

    max_peaks = getattr(args, "processor_max_peaks", _UNSET)
    if max_peaks is not _UNSET:
        provided = True
        kwargs["max_peaks"] = max_peaks

    mz_dedup_tol = getattr(args, "processor_mz_dedup_tolerance", None)
    if mz_dedup_tol is not None:
        provided = True
        kwargs["mz_dedup_tolerance"] = mz_dedup_tol

    float_dtype = getattr(args, "processor_float_dtype", None)
    if float_dtype is not None:
        provided = True
        kwargs["float_dtype"] = np.float64 if float_dtype == "float64" else np.float32

    if not provided:
        return None
    return SpectrumProcessor(**kwargs)  # type: ignore[arg-type]


def _edges_from_csv(path: Path) -> list[SimilarityEdge]:
    edges: list[SimilarityEdge] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edges.append(
                SimilarityEdge(
                    source=row["source"],
                    target=row["target"],
                    similarity=float(row.get("similarity", 0.0)),
                    metric=row.get("metric", "cosine"),
                )
            )
    return edges


def _nodes_from_library(library_path: Path) -> list[SpectrumNode]:
    nodes: list[SpectrumNode] = []
    if not library_path.exists():
        return nodes
    lib = LocalSpectralLibrary(library_path)
    for entry in lib.iter_entries():
        nodes.append(
            SpectrumNode(
                identifier=entry.identifier,
                precursor_mz=entry.precursor_mz,
                metadata=entry.metadata,
                spectrum=None,
                vector=None,
            )
        )
    return nodes


def _nodes_from_edges(edges: list[SimilarityEdge]) -> list[SpectrumNode]:
    identifiers = {edge.source for edge in edges} | {edge.target for edge in edges}
    return [
        SpectrumNode(
            identifier=identifier,
            precursor_mz=None,
            metadata={},
            spectrum=None,
            vector=None,
        )
        for identifier in sorted(identifiers)
    ]


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
    except ConfigError as exc:  # pragma: no cover - validated in config tests
        logger.error("Config error at %s: %s", exc.path, exc.message)
        return 1
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
    _add_processor_args(build_parser)
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
    search_parser.add_argument(
        "--backend",
        choices=["naive", "annoy", "faiss"],
        default="naive",
        help="Search backend to use (default: naive).",
    )
    _add_processor_args(search_parser)
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
    _add_processor_args(build_parser)
    build_parser.set_defaults(func=_run_network_build_command)

    export_parser = network_subparsers.add_parser(
        "export-cytoscape",
        help="Export an existing network to Cytoscape-friendly node/edge tables.",
    )
    export_parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML/JSON config describing the network output.",
    )
    export_parser.add_argument(
        "--out-dir",
        default="out/cytoscape_export",
        help="Directory to write Cytoscape node/edge tables (default: out/cytoscape_export).",
    )
    export_parser.add_argument(
        "--network",
        help="Optional override for the network edge CSV path (defaults to config network.output).",
    )
    export_parser.add_argument(
        "--library",
        help="Optional library path to enrich nodes (defaults to config library).",
    )
    export_parser.set_defaults(func=_run_network_export_cytoscape_command)
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
    processor = _processor_from_args(args)
    library = workflow.build_library(
        args.input,
        args.library,
        input_format=args.format,
        recursive=args.recursive,
        storage=args.storage,
        overwrite=args.overwrite,
        processor=processor,
    )
    logger.info("Library ready at %s (%d entries)", library.path, len(library))
    return 0


def _run_library_search_command(args) -> int:
    processor = _processor_from_args(args)
    results = workflow.search_library(
        args.queries,
        args.library,
        input_format=args.format,
        recursive=args.recursive,
        top_n=args.top_n,
        min_score=args.min_score,
        backend=args.backend,
        processor=processor,
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
    processor = _processor_from_args(args)
    undirected = not args.directed
    nodes, edges = workflow.build_network(
        input_dir=args.input,
        library_path=args.library,
        metric=args.metric,
        threshold=args.threshold,
        knn=args.knn,
        processor=processor,
        undirected=undirected,
        output_path=args.output,
    )
    if args.summary:
        summary = summarize_network(nodes, edges)
        write_network_summary(summary, args.summary)
    logger.info("Built network with %d nodes and %d edges", len(nodes), len(edges))
    return 0


def _run_network_export_cytoscape_command(args) -> int:
    cfg = load_config(args.config)
    network_path_value = args.network or cfg.network.output or cfg.outputs.network
    if not network_path_value:
        raise ConfigError("network.output", "Network output path is required to export Cytoscape tables.")
    network_path = Path(network_path_value)
    if not network_path.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    library_path_value = args.library or cfg.network.library or cfg.library.path
    library_path = Path(library_path_value) if library_path_value else None
    edges = _edges_from_csv(network_path)
    nodes = _nodes_from_library(library_path) if library_path else []
    if not nodes:
        nodes = _nodes_from_edges(edges)

    export_network_for_cytoscape(nodes, edges, args.out_dir)
    logger.info("Exported Cytoscape tables to %s", args.out_dir)
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
