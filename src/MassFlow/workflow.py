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
from typing import List, Tuple

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


def _init_worker(config: MassFlowConfig) -> None:
    """
    Initialize a worker-local similarity engine.

    Each subprocess instantiates its own engine from the shared configuration so
    large model state is not serialized or copied through inter-process
    communication.
    """
    global _worker_engine
    _worker_engine = get_similarity_engine(config.similarity)


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
        for i, q in enumerate(query_spectra):
            if q.get("id") is None:
                q.set("id", f"{query_file.stem}_query_{i}")

        global _worker_engine
        if _worker_engine is None:
            # Fallback in case process wasn't initialized correctly
            _worker_engine = get_similarity_engine(config.similarity)

        all_results = []
        if config.input.reference_library is None:
            raise ValueError("Reference library is not configured.")
        ref_gen = io.load_spectra(config.input.reference_library)
        ref_iterator = processing.process_spectra(ref_gen, config.processing)

        chunk_size = 2000
        ref_chunk = []

        # We extract top_n if the user added it to similarity config, otherwise None
        top_n = getattr(config.similarity, "top_n", None)

        for ref_spec in ref_iterator:
            ref_chunk.append(ref_spec)
            if len(ref_chunk) >= chunk_size:
                chunk_results = _worker_engine.search(
                    query_spectra, ref_chunk, top_n=top_n
                )
                all_results.extend(chunk_results)
                ref_chunk = []

        if ref_chunk:
            chunk_results = _worker_engine.search(query_spectra, ref_chunk, top_n=top_n)
            all_results.extend(chunk_results)

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

        # Aggregate Top-N results across all chunks per query_id
        fdr_filtered_results.sort(key=lambda x: x["score"], reverse=True)

        if top_n is not None:
            final_results = []
            query_counts: dict[str, int] = {}
            for res in fdr_filtered_results:
                qid = res["query_id"]
                query_counts[qid] = query_counts.get(qid, 0) + 1
                if query_counts[qid] <= top_n:
                    final_results.append(res)
        else:
            final_results = fdr_filtered_results

        return query_file, query_spectra, final_results

    except Exception as e:
        logger.error(f"Failed to process {query_file}: {e}", exc_info=True)
        return query_file, [], []


def run_annotation_pipeline(config: MassFlowConfig) -> None:
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
    # 1. Load Reference for main process
    if not config.input.reference_library:
        raise ValueError("Reference library path not specified in configuration.")

    logger.info(f"Loading reference library: {config.input.reference_library}")
    ref_gen = io.load_spectra(config.input.reference_library)
    reference_spectra = list(processing.process_spectra(ref_gen, config.processing))

    if not reference_spectra:
        raise ValueError("No valid spectra found in reference library.")

    logger.info(f"Loaded {len(reference_spectra)} reference spectra.")

    if len(reference_spectra) < 2000:
        logger.warning(
            f"\n"
            f"================================================================================\n"
            f"CRITICAL SCIENTIFIC WARNING: SMALL REFERENCE LIBRARY DETECTED\n"
            f"The reference library contains only {len(reference_spectra)} spectra. \n"
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
    if config.input.file_path:
        input_files.append(config.input.file_path)
    elif config.input.data_directory:
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
        for f in config.input.data_directory.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported_exts:
                input_files.append(f)
            # Handle .d directories (Agilent)
            if f.is_dir() and f.suffix.lower() == ".d":
                input_files.append(f)

        if not input_files:
            raise ValueError(
                f"No supported spectral files found in {config.input.data_directory}"
            )
    else:
        raise ValueError(
            "Neither input file_path nor data_directory specified in configuration."
        )

    # 3. Process Each File in Parallel using Multiprocessing (Bypassing GIL)
    config.output_directory.mkdir(parents=True, exist_ok=True)

    all_queries: List[Spectrum] = []
    all_results: List[SearchResult] = []

    logger.info(
        f"Processing {len(input_files)} experimental files using multiprocessing..."
    )

    with ProcessPoolExecutor(initializer=_init_worker, initargs=(config,)) as executor:
        futures = {
            executor.submit(_process_single_file, qf, config): qf for qf in input_files
        }

        for future in as_completed(futures):
            qf = futures[future]
            try:
                processed_file, q_spectra, results = future.result()

                if q_spectra:
                    all_queries.extend(q_spectra)
                    all_results.extend(results)

                    # Save intermediate results for this file
                    out_file = config.output_directory / (
                        processed_file.stem + "_results.csv"
                    )
                    io.save_match_results(results, out_file, query_spectra=q_spectra)  # type: ignore
                    logger.info(f"Results saved to {out_file}")
                else:
                    logger.warning(f"No valid spectra extracted from {processed_file}.")
            except Exception as e:
                logger.error(f"Process failed for {qf}: {e}")

    # 4. Perform Molecular Networking
    if config.workflow.perform_networking:
        try:
            from MassFlow.networking import generate_molecular_network

            network_out = config.output_directory / "molecular_network.graphml"
            generate_molecular_network(
                all_queries=all_queries,
                all_references=reference_spectra,
                all_results=all_results,
                config=config,
                output_path=network_out,
            )
        except Exception as e:
            logger.error(f"Networking failed: {e}", exc_info=True)

    logger.info("Pipeline Finished.")
