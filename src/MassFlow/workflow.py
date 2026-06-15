"""
High-level orchestration for MassFlow annotation runs.

This module coordinates the end-to-end execution path used by the CLI: loading
and validating the reference library, discovering experimental inputs,
processing spectra, dispatching per-file similarity searches across worker
processes, and exporting result tables. It is the integration layer that turns
the config, I/O, processing, and similarity modules into a reproducible
pipeline.

The workflow is designed to stay memory-aware. Worker processes build their own
similarity engines, reference libraries are searched in chunks rather than as a
single monolithic matrix, and false discovery rate filtering is applied after
aggregating all chunk results for each experimental file.
"""

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from matchms import Spectrum

from MassFlow import io, processing
from MassFlow.config import MassFlowConfig
from MassFlow.similarity import (
    SearchResult,
    SimilarityEngine,
    calculate_fdr,
    get_similarity_engine,
)

logger = logging.getLogger(__name__)

_worker_engine: SimilarityEngine | None = None
_worker_references: List[Spectrum] | None = None
_worker_decoys: List[Spectrum] | None = None


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
    from MassFlow.log_config import setup_structured_logging

    setup_structured_logging(level=logging.INFO)

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
        query_iter = iter(processing.process_spectra(query_gen, config.processing))

        seen_ids = set()
        query_idx = 0

        all_query_spectra = []
        all_results = []

        global _worker_engine, _worker_references, _worker_decoys
        engine = (
            _worker_engine
            if _worker_engine is not None
            else get_similarity_engine(config.similarity)
        )

        while True:
            chunk = list(itertools.islice(query_iter, 1000))
            if not chunk:
                break

            # Ensure unique IDs for nodes in the chunk
            for q in chunk:
                base_id = q.get("id")
                if base_id is None:
                    new_id = f"{query_file.stem}_query_{query_idx}"
                else:
                    new_id = str(base_id)
                    counter = 1
                    while new_id in seen_ids:
                        new_id = f"{base_id}_{counter}"
                        counter += 1

                q.set("id", new_id)
                seen_ids.add(new_id)
                query_idx += 1

            # Accumulate query spectra (we need them for the exporter later)
            all_query_spectra.extend(chunk)

            # Search against shared memory libraries if available (from Pool initializer)
            if _worker_references is not None and _worker_decoys is not None:
                all_references = _worker_references + _worker_decoys
                chunk_results = engine.search(
                    chunk, all_references, include_decoys=False
                )
            else:
                # Fallback for single-process testing or direct invocation, streaming the library
                if config.input.library_path is None:
                    raise ValueError("Library path is not configured.")
                ref_gen = io.load_spectra(config.input.library_path)
                ref_iterator = processing.process_spectra(ref_gen, config.processing)

                chunk_results = engine.search(
                    chunk, ref_iterator, include_decoys=True
                )

            all_results.extend(chunk_results)

            # Clear the chunk variables to free up matrix memory before the next iteration
            del chunk_results
            del chunk

        if not all_query_spectra:
            return query_file, [], []

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

        from MassFlow.similarity import calculate_empirical_p_values

        # Determine if we use Small Library Statistics
        lib_size = len(_worker_references) if _worker_references else 0
        is_small_library = lib_size < 2000

        # 1. Standard FDR
        sorted_scores, q_values, _ = calculate_fdr(target_scores_arr, decoy_scores_arr)
        asc_scores = sorted_scores[::-1]
        asc_q = q_values[::-1]

        # 2. Empirical P-Value (Small Library Alternative)
        p_vals = calculate_empirical_p_values(target_scores_arr, decoy_scores_arr)
        p_val_map = dict(zip(target_scores_arr, p_vals))

        fdr_threshold = getattr(config.similarity, "fdr_threshold", 0.01)

        fdr_filtered_results = []
        for res in all_results:
            if res.get("is_decoy", False):
                continue

            score = res["score"]
            if len(asc_scores) > 0:
                q_val = float(np.interp(score, asc_scores, asc_q))
            else:
                q_val = 1.0

            p_val = float(p_val_map.get(score, 1.0))

            res["q_value"] = q_val
            res["p_value"] = p_val

            # MODIFIED FILTERING:
            # If small library, use p-value for filtering instead of sparse q-value
            filter_metric = p_val if is_small_library else q_val

            if filter_metric <= fdr_threshold:
                fdr_filtered_results.append(res)

        # Sort results descending by score
        fdr_filtered_results.sort(key=lambda x: x["score"], reverse=True)

        return query_file, all_query_spectra, fdr_filtered_results

    except Exception as e:
        logger.error(f"Failed to process {query_file}: {e}", exc_info=True)
        return query_file, [], []


def _handle_file_results(
    processed_file: Path,
    q_spectra: List[Spectrum],
    results: List[SearchResult],
    config: MassFlowConfig,
    config_path: Path | str | None = None,
) -> None:
    """Helper to save results and write reports for a processed file."""
    if not q_spectra:
        logger.warning(f"No valid spectra extracted from {processed_file}.")
        return

    # Save intermediate results for this file (Collision Prevention)
    base_stem = processed_file.stem
    export_format = config.export.format.lower()
    ext_map = {
        "csv": "csv",
        "mztab": "mztab",
    }
    ext = ext_map.get(export_format, "csv")

    out_file = config.output_directory / f"{base_stem}_results.{ext}"
    counter = 1
    while out_file.exists():
        out_file = config.output_directory / f"{base_stem}_{counter}_results.{ext}"
        counter += 1

    results_dict = cast(List[Dict[str, Any]], results)
    # Route to the correct exporter based on config.export.format
    if export_format == "mztab":
        io.save_match_results_to_mztab(results_dict, out_file, query_spectra=q_spectra)
    else:
        io.save_match_results(results_dict, out_file, query_spectra=q_spectra)

    report_file = config.output_directory / f"{out_file.stem}.report.yaml"
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
    Generate a concise provenance payload and delegate saving to the I/O layer.

    The I/O helper :func:`io.save_analysis_report` is used so tests can patch and
    assert calls instead of inspecting on-disk files.
    """
    report_payload = {
        "query_file": str(query_file),
        "results_csv": str(results_file),
        "library_path": str(config.input.library_path)
        if config.input.library_path
        else None,
        "num_queries": len(query_spectra),
        "num_matches": len(results),
        "processing": config.processing.model_dump(mode="json"),
        "similarity": config.similarity.model_dump(mode="json"),
        "workflow": config.workflow.model_dump(mode="json"),
    }

    # Delegate to IO layer for file writing so tests can patch io.save_analysis_report
    io.save_analysis_report(report_path, report_payload)


def run_annotation_pipeline(
    config: MassFlowConfig, config_path: Path | str | None = None
) -> None:
    """
    Execute the full MassFlow annotation workflow.

    The workflow performs these major stages:

    1. Load and process the reference library in the parent process.
    2. Discover query inputs from either a single file or a data directory.
    3. Dispatch one task per experimental file to a process pool.
    4. Within each worker, search the processed queries against chunked
       reference spectra and apply per-file FDR filtering.
    5. Save a result file for each processed experimental input.

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
    experimental file does not necessarily abort the full batch.
    """
    # 1. Load Library for main process
    if not config.input.library_path:
        raise ValueError("Library path not specified in configuration.")
    if not Path(config.input.library_path).exists():
        raise ValueError(f"Library path does not exist: {config.input.library_path}")

    library_size_mb = Path(config.input.library_path).stat().st_size / (1024 * 1024)
    stream_library = library_size_mb > config.input.streaming_threshold_mb

    if stream_library:
        logger.info(
            f"Library size ({library_size_mb:.1f} MB) exceeds {config.input.streaming_threshold_mb}MB threshold. Using memory-efficient streaming mode."
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

    # 3. Process Each File
    config.output_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(input_files)} experimental files...")

    if len(input_files) == 1:
        # Optimization for single file: avoid ProcessPool overhead and pickling
        qf = input_files[0]
        # For single process, do not pre-populate worker references so the single-process
        # code path exercises the same streaming and loading behavior as the multi-process
        # path (keeps test expectations aligned with integration tests).
        processed_file, q_spectra, results = _process_single_file(qf, config)
        _handle_file_results(
            processed_file, q_spectra, results, config, config_path=config_path
        )
    else:
        with ProcessPoolExecutor(
            initializer=_init_worker,
            initargs=(config, reference_spectra, decoy_spectra),
        ) as executor:
            futures = {
                executor.submit(_process_single_file, qf, config): qf
                for qf in input_files
            }

            for future in as_completed(futures):
                qf = futures[future]
                try:
                    processed_file, q_spectra, results = future.result()
                    _handle_file_results(
                        processed_file,
                        q_spectra,
                        results,
                        config,
                        config_path=config_path,
                    )
                except Exception as e:
                    logger.error(f"Worker failed for {qf}: {e}")

    logger.info("Pipeline Finished.")
