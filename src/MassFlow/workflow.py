"""
High-level orchestration for MassFlow annotation runs.

This module coordinates the end-to-end execution path used by the CLI: loading
and validating the reference library, discovering experimental inputs,
processing spectra, dispatching per-file similarity searches across worker
processes, exporting result tables, and optionally generating a molecular
network. It is the integration layer that turns the config, I/O, processing,
similarity, and networking modules into a reproducible pipeline.

The workflow is designed to stay memory-aware. Worker processes build their own
similarity engines, reference libraries are searched in chunks rather than as a
single monolithic matrix, and false discovery rate filtering is applied after
aggregating all chunk results for each experimental file.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from matchms import Spectrum

from MassFlow import io, processing
from MassFlow.config import MassFlowConfig
from MassFlow.similarity import (
    CascadeEngine,
    ConsensusEngine,
    SearchResult,
    SimilarityEngine,
    calculate_fdr,
    get_similarity_engine,
)

logger = logging.getLogger(__name__)

_worker_engine: SimilarityEngine | ConsensusEngine | CascadeEngine | None = None
_worker_references: List[Spectrum] | None = None
_worker_decoys: List[Spectrum] | None = None
_worker_tier2_engine: SimilarityEngine | None = None


def _get_tier2_engine(config: MassFlowConfig) -> SimilarityEngine:
    global _worker_tier2_engine
    if _worker_tier2_engine is not None:
        return _worker_tier2_engine

    tier2_config = config.similarity.model_copy(
        update={"algorithm": config.similarity.cascade_tier2}
    )
    _worker_tier2_engine = SimilarityEngine(tier2_config)
    return _worker_tier2_engine


def _init_worker(
    config: MassFlowConfig,
    references: List[Spectrum] | None,
    decoys: List[Spectrum] | None,
) -> None:
    """
    Initialize a worker-local similarity engine and share pre-processed libraries.

    Each subprocess instantiates its own engine from the shared configuration so
    large model state is not serialized. References and decoys are passed once
    at initialization to avoid repetitive disk I/O.
    """
    global _worker_engine, _worker_references, _worker_decoys
    _worker_engine = get_similarity_engine(config.similarity)
    _worker_references = references
    _worker_decoys = decoys


def _process_single_file(
    query_file: Path,
    config: MassFlowConfig,
) -> Tuple[Path, List[Spectrum], List[SearchResult]]:
    """
    Process one experimental file against the configured reference library.

    The worker loads and processes the query spectra, ensures query IDs are
    stable, streams the reference library in chunks, runs similarity search on
    each chunk, and then applies FDR filtering across the aggregated results for
    that one file.

    Parameters
    ----------
    query_file : Path
        Experimental spectral file to process.
    config : MassFlowConfig
        Full pipeline configuration used for loading, processing, and scoring.

    Returns
    -------
    tuple[Path, list[matchms.Spectrum], list[SearchResult]]
        The input file path, processed query spectra, and the final filtered
        matches for that file. On failure, the function logs the exception and
        returns the file path with empty lists.
    """
    try:
        query_gen = io.load_spectra(query_file, file_format=config.input.format)
        query_spectra = list(processing.process_spectra(query_gen, config.processing))

        if not query_spectra:
            return query_file, [], []

        # Ensure unique IDs for nodes
        seen_ids = set()
        for i, q in enumerate(query_spectra):
            base_id = q.get("id")
            if base_id is None:
                new_id = f"{query_file.stem}_query_{i}"
            else:
                new_id = str(base_id)
                counter = 1
                while new_id in seen_ids:
                    new_id = f"{base_id}_{counter}"
                    counter += 1

            q.set("id", new_id)
            seen_ids.add(new_id)

        global _worker_engine, _worker_references, _worker_decoys
        engine = (
            _worker_engine
            if _worker_engine is not None
            else get_similarity_engine(config.similarity)
        )

        standard_queries = []
        triage_queries = []
        for q in query_spectra:
            if q.get("triage_flags") or q.get("triage_flag"):
                triage_queries.append(q)
            else:
                standard_queries.append(q)

        tier2_engine = None
        if triage_queries:
            tier2_engine = _get_tier2_engine(config)

        # Search against shared memory libraries if available (from Pool initializer)
        if _worker_references is not None and _worker_decoys is not None:
            all_references = _worker_references + _worker_decoys
            all_results = []
            if standard_queries:
                all_results.extend(
                    engine.search(
                        standard_queries, all_references, include_decoys=False
                    )
                )
            if triage_queries and tier2_engine:
                t2_results = tier2_engine.search(
                    triage_queries, all_references, include_decoys=False
                )
                for res in t2_results:
                    res["annotation_tier"] = (
                        f"Triage ({config.similarity.cascade_tier2})"
                    )
                all_results.extend(t2_results)
        else:
            # Fallback for single-process testing or direct invocation
            all_results = []
            if config.input.library_path is None:
                raise ValueError("Library path is not configured.")
            ref_gen = io.load_spectra(config.input.library_path)
            ref_iterator = processing.process_spectra(ref_gen, config.processing)

            chunk_size = 2000
            ref_chunk = []

            for ref_spec in ref_iterator:
                ref_chunk.append(ref_spec)
                if len(ref_chunk) >= chunk_size:
                    if standard_queries:
                        all_results.extend(
                            engine.search(
                                standard_queries, ref_chunk, include_decoys=True
                            )
                        )
                    if triage_queries and tier2_engine:
                        t2_results = tier2_engine.search(
                            triage_queries, ref_chunk, include_decoys=True
                        )
                        for res in t2_results:
                            res["annotation_tier"] = (
                                f"Triage ({config.similarity.cascade_tier2})"
                            )
                        all_results.extend(t2_results)
                    ref_chunk.clear()

            if ref_chunk:
                if standard_queries:
                    all_results.extend(
                        engine.search(standard_queries, ref_chunk, include_decoys=True)
                    )
                if triage_queries and tier2_engine:
                    t2_results = tier2_engine.search(
                        triage_queries, ref_chunk, include_decoys=True
                    )
                    for res in t2_results:
                        res["annotation_tier"] = (
                            f"Triage ({config.similarity.cascade_tier2})"
                        )
                    all_results.extend(t2_results)
                ref_chunk.clear()

        # Global FDR calculation across all chunks for this experimental file
        target_scores = []
        decoy_scores = []
        for res in all_results:
            if res.get("is_decoy", False):
                decoy_scores.append(res["score"])
            else:
                target_scores.append(res["score"])

        target_scores_arr = np.array(target_scores, dtype=float)
        decoy_scores_arr = np.array(decoy_scores, dtype=float)

        sorted_scores, q_values, _ = calculate_fdr(target_scores_arr, decoy_scores_arr)

        # For fast interpolation (needs ascending x)
        asc_scores = sorted_scores[::-1]
        asc_q = q_values[::-1]

        fdr_threshold = getattr(config.similarity, "fdr_threshold", 0.01)

        fdr_filtered_results = []
        for res in all_results:
            score = res["score"]
            if len(asc_scores) > 0:
                q_val = float(np.interp(score, asc_scores, asc_q))
            else:
                q_val = 1.0

            res["q_value"] = q_val

            # Filter by FDR and remove decoys
            if q_val <= fdr_threshold and not res.get("is_decoy", False):
                fdr_filtered_results.append(res)

        # Sort results descending by score
        fdr_filtered_results.sort(key=lambda x: x["score"], reverse=True)

        return query_file, query_spectra, fdr_filtered_results

    except Exception as e:
        logger.error(f"Failed to process {query_file}: {e}", exc_info=True)
        return query_file, [], []


def _write_analysis_report(
    report_path: Path,
    config: MassFlowConfig,
    query_file: Path,
    results_file: Path,
    query_spectra: List[Spectrum],
    results: List[SearchResult],
    config_path: Path | str | None = None,
) -> None:
    """
    Write a provenance report alongside an annotation results file.

    Parameters
    ----------
    report_path : Path
        Destination path for the provenance report.
    config : MassFlowConfig
        Full configuration used for the annotation run.
    query_file : Path
        Experimental input file processed for this result set.
    results_file : Path
        CSV results file written for this query file.
    query_spectra : list[matchms.Spectrum]
        Processed query spectra used in the search.
    results : list[SearchResult]
        Final filtered search results written to the CSV file.
    config_path : Path or str or None, optional
        Original YAML configuration path when available.

    Returns
    -------
    None
    """
    original_config_yaml = None
    if config_path is not None and Path(config_path).exists():
        original_config_yaml = Path(config_path).read_text()

    report_data = {
        "config_path": str(config_path) if config_path is not None else None,
        "original_config_yaml": original_config_yaml,
        "query_file": str(query_file),
        "results_csv": str(results_file),
        "library_path": (
            str(config.input.library_path)
            if config.input.library_path is not None
            else None
        ),
        "input_format": config.input.format,
        "query_spectra_count": len(query_spectra),
        "retained_result_count": len(results),
        "project": config.project.model_dump(mode="json"),
        "input": config.input.model_dump(mode="json"),
        "processing": config.processing.model_dump(mode="json"),
        "similarity": config.similarity.model_dump(mode="json"),
        "workflow": config.workflow.model_dump(mode="json"),
        "export": config.export.model_dump(mode="json"),
    }

    io.save_analysis_report(report_path, report_data)


def run_annotation_pipeline(
    config: MassFlowConfig, config_path: Path | str | None = None
) -> None:
    """
    Execute the full MassFlow annotation analysis pipeline.

    The workflow performs these major stages:

    1. Load and process the reference library in the parent process.
    2. Discover query inputs from either a single file or a data directory.
    3. Dispatch one task per experimental file to a process pool.
    4. Within each worker, search the processed queries against chunked
       reference spectra and apply per-file FDR filtering.
    5. Save a CSV result file for each processed experimental input.
    6. Optionally generate a GraphML molecular network from the aggregate run.

    Parameters
    ----------
    config : MassFlowConfig
        The configuration object containing all settings for input/output paths,
        processing parameters, and similarity search options.
    config_path : Path or str or None, optional
        Original YAML configuration path used to create ``config``. When
        provided, it is written into the per-results provenance report.

    Returns
    -------
    None
        This function does not return a value. It writes output files directly
        to the configured output directory.

    Raises
    ------
    ValueError
        If the reference library path is missing, no valid reference spectra are found,
        or no supported input files are found.

    Notes
    -----
    Per-file worker failures are logged and skipped so that one problematic
    experimental file does not necessarily abort the full batch. The
    ``export`` configuration section is not used directly here; result tables
    are currently written as CSV files via :func:`MassFlow.io.save_match_results`.
    """
    # 1. Load Library for main process
    if not config.input.library_path:
        raise ValueError("Library path not specified in configuration.")
    if not Path(config.input.library_path).exists():
        raise ValueError(f"Library path does not exist: {config.input.library_path}")

    library_size_mb = Path(config.input.library_path).stat().st_size / (1024 * 1024)
    stream_library = library_size_mb > 500

    if stream_library:
        logger.info(
            f"Library size ({library_size_mb:.1f} MB) exceeds 500MB threshold. Using memory-efficient streaming mode."
        )
        reference_spectra = None
        decoy_spectra = None
    else:
        logger.info(f"Loading library: {config.input.library_path}")
        ref_gen = io.load_spectra(config.input.library_path)
        reference_spectra = list(processing.process_spectra(ref_gen, config.processing))

        if not reference_spectra:
            raise ValueError("No valid spectra found in library.")

        logger.info(
            f"Loaded {len(reference_spectra)} reference spectra. Generating decoys for FDR calculation..."
        )
        from MassFlow.similarity import generate_decoys

        decoy_spectra = generate_decoys(reference_spectra)

        if len(reference_spectra) < 2000:
            logger.warning(
                f"\n"
                f"================================================================================\n"
                f"CRITICAL SCIENTIFIC WARNING: SMALL LIBRARY DETECTED\n"
                f"The library contains only {len(reference_spectra)} spectra. \n"
                f"Target-Decoy False Discovery Rate (FDR) statistics are fundamentally invalid on \n"
                f"small sample sizes because the decoy null-distribution will be too sparse. \n"
                f"A strict FDR threshold (currently set to {getattr(config.similarity, 'fdr_threshold', 0.01)}) will "
                f"likely eliminate all true and putative matches as false positives.\n\n"
                f"Recommendation:\n"
                f"1. Use a comprehensive library (e.g., GNPS, MoNA, NIST) for FDR validation.\n"
                f"2. Or, if using a small specialized library, relax the `fdr_threshold` \n"
                f"   (e.g., 0.1 or 1.0) in your config to evaluate raw Cosine scores directly.\n"
                f"================================================================================\n"
            )

    # 2. Determine Input Files
    input_files = []
    input_path = Path(config.input.input_path)
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        input_files.append(input_path)
    else:
        # Recursively find all supported spectral files
        supported_exts = {
            ".mzml",
            ".mzxml",
            ".mgf",
            ".msp",
            ".raw",
            ".d",
            ".wiff",
            ".lcd",
            ".t2d",
        }
        for f in input_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported_exts:
                input_files.append(f)
            # Handle .d directories (Agilent/Bruker)
            if f.is_dir() and f.suffix.lower() == ".d":
                input_files.append(f)

        if not input_files:
            raise ValueError(f"No supported spectral files found in {input_path}")

    # 3. Process Each File in Parallel using Multiprocessing (Bypassing GIL)
    config.output_directory.mkdir(parents=True, exist_ok=True)

    all_queries: List[Spectrum] = []
    all_results: List[SearchResult] = []

    logger.info(
        f"Processing {len(input_files)} experimental files using multiprocessing..."
    )

    with ProcessPoolExecutor(
        initializer=_init_worker, initargs=(config, reference_spectra, decoy_spectra)
    ) as executor:
        futures = {
            executor.submit(_process_single_file, qf, config): qf for qf in input_files
        }

        for future in as_completed(futures):
            qf = futures[future]
            try:
                processed_file, q_spectra, results = future.result()

                if q_spectra:
                    # OOM Prevention: Only retain all queries and results in memory if needed for downstream steps
                    if config.workflow.perform_networking:
                        all_queries.extend(q_spectra)
                        all_results.extend(results)

                    # Save intermediate results for this file (Collision Prevention)
                    base_stem = processed_file.stem
                    export_format = config.export.format.lower()
                    ext_map = {
                        "csv": "csv",
                        "json": "json",
                        "xlsx": "xlsx",
                        "parquet": "parquet",
                        "pickle": "pkl",
                        "msp": "msp",
                        "mgf": "mgf",
                    }
                    ext = ext_map.get(export_format, "csv")

                    out_file = config.output_directory / f"{base_stem}_results.{ext}"
                    counter = 1
                    while out_file.exists():
                        out_file = (
                            config.output_directory
                            / f"{base_stem}_{counter}_results.{ext}"
                        )
                        counter += 1

                    results_dict = cast(List[Dict[str, Any]], results)
                    # Route to the correct exporter based on config.export.format
                    if export_format == "json":
                        io.save_match_results_to_json(
                            results_dict, out_file, query_spectra=q_spectra
                        )
                    elif export_format == "xlsx":
                        io.save_match_results_to_xlsx(
                            results_dict, out_file, query_spectra=q_spectra
                        )
                    elif export_format == "parquet":
                        io.save_match_results_to_parquet(
                            results_dict, out_file, query_spectra=q_spectra
                        )
                    elif export_format == "csv":
                        io.save_match_results(
                            results_dict, out_file, query_spectra=q_spectra
                        )
                    else:
                        # Fallback for other formats or unimplemented results exporters
                        logger.warning(
                            f"Export format '{export_format}' is not yet supported for result tables. Falling back to CSV."
                        )
                        out_file = out_file.with_suffix(".csv")
                        io.save_match_results(
                            results_dict, out_file, query_spectra=q_spectra
                        )

                    report_file = (
                        config.output_directory / f"{out_file.stem}.report.yaml"
                    )
                    _write_analysis_report(
                        report_path=report_file,
                        config=config,
                        query_file=processed_file,
                        results_file=out_file,
                        query_spectra=q_spectra,
                        results=results,
                        config_path=config_path,
                    )

                    logger.info(f"Results saved to {out_file}")
                else:
                    logger.warning(f"No valid spectra extracted from {processed_file}.")
            except Exception as e:
                logger.error(f"Process failed for {qf}: {e}")

    # 4. Perform Molecular Networking
    if config.workflow.perform_networking:
        try:
            from MassFlow.networking import generate_molecular_network

            if reference_spectra is None:
                logger.warning(
                    "Molecular networking requested with a streamed library. "
                    "Reference nodes in the output GraphML will be created from results "
                    "but will lack full spectral metadata."
                )

            network_out = config.output_directory / "molecular_network.graphml"
            generate_molecular_network(
                all_queries=all_queries,
                all_references=reference_spectra or [],
                all_results=all_results,
                config=config,
                output_path=network_out,
            )
        except ImportError:
            logger.error(
                "Networking dependencies are missing. Please install them with 'pip install massflow[network]'."
            )
        except Exception as e:
            logger.error(f"Networking failed: {e}", exc_info=True)

    logger.info("Pipeline Finished.")
