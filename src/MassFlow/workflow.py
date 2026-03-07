"""
Workflow orchestration for the MassFlow pipeline.

This module manages the high-level execution flow of the MassFlow application.
It coordinates the loading of reference libraries and experimental spectra,
applies processing steps defined in the configuration, initializes the
similarity search engine, and manages the output of results.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
from matchms import Spectrum, calculate_scores

from MassFlow import io, processing
from MassFlow.config import MassFlowConfig
from MassFlow.similarity import SearchResult, SimilarityEngine, calculate_fdr

logger = logging.getLogger(__name__)

_worker_engine: SimilarityEngine | None = None


def _init_worker(config: MassFlowConfig) -> None:
    """
    Initialize worker processes by independently loading the engine to avoid IPC memory duplication.
    """
    global _worker_engine
    _worker_engine = SimilarityEngine(config.similarity)


def _process_single_file(
    query_file: Path,
    config: MassFlowConfig,
) -> Tuple[Path, List[Spectrum], List[SearchResult]]:
    """Process a single experimental file."""
    try:
        query_gen = io.load_spectra(query_file, file_format=config.input.format)
        query_spectra = list(processing.process_spectra(query_gen, config.processing))

        if not query_spectra:
            return query_file, [], []

        # Ensure unique IDs for network nodes
        for i, q in enumerate(query_spectra):
            if q.get("id") is None:
                q.set("id", f"{query_file.stem}_query_{i}")

        global _worker_engine
        if _worker_engine is None:
            # Fallback in case process wasn't initialized correctly
            _worker_engine = SimilarityEngine(config.similarity)

        all_results = []
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
            query_counts = {}
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


def generate_molecular_network(
    all_queries: List[Spectrum],
    all_references: List[Spectrum],
    all_results: List[SearchResult],
    config: MassFlowConfig,
    output_path: Path,
) -> None:
    """Generate a molecular network (GraphML) from query and reference spectra."""
    logger.info("Generating molecular network...")

    if not all_queries:
        logger.warning("No query spectra available for networking.")
        return

    engine = SimilarityEngine(config.similarity)

    # Calculate query-to-query similarity
    scores_obj = calculate_scores(
        references=all_queries,
        queries=all_queries,
        similarity_function=engine.similarity_function,
        is_symmetric=True,
    )

    scores_data = scores_obj.scores
    if hasattr(scores_data, "to_array"):
        scores_array = scores_data.to_array()
    else:
        scores_array = np.asarray(scores_data)

    if hasattr(scores_array.dtype, "names") and scores_array.dtype.names is not None:
        score_cols = [c for c in scores_array.dtype.names if "score" in c.lower()]
        numeric_scores = scores_array[score_cols[0]]
    else:
        numeric_scores = scores_array

    G = nx.Graph()

    # Add nodes
    for i, q in enumerate(all_queries):
        node_id = str(q.get("id", f"query_{i}"))
        mz = q.get("precursor_mz")
        G.add_node(
            node_id,
            node_type="query",
            precursor_mz=float(mz) if mz is not None else 0.0,
        )

    for r in all_references:
        node_id = str(r.get("id"))
        mz = r.get("precursor_mz")
        name = str(r.get("compound_name") or r.get("name") or "Unknown")
        G.add_node(
            node_id,
            node_type="reference",
            name=name,
            precursor_mz=float(mz) if mz is not None else 0.0,
        )

    # Add query-to-reference edges
    for res in all_results:
        if res["score"] >= config.similarity.min_score and not res.get(
            "is_decoy", False
        ):
            if res.get("q_value", 1.0) <= config.similarity.fdr_threshold:
                G.add_edge(
                    res["query_id"],
                    res["reference_id"],
                    weight=float(res["score"]),
                    edge_type="query_to_ref",
                )

    # Add query-to-query edges using sparse matrix
    min_score = config.similarity.min_score
    adj_matrix = np.triu(numeric_scores, k=1)
    adj_matrix[adj_matrix < min_score] = 0.0

    sparse_adj = sp.coo_matrix(adj_matrix)

    for i, j, v in zip(sparse_adj.row, sparse_adj.col, sparse_adj.data):
        q1_id = str(all_queries[i].get("id", f"query_{i}"))
        q2_id = str(all_queries[j].get("id", f"query_{j}"))
        G.add_edge(q1_id, q2_id, weight=float(v), edge_type="query_to_query")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(output_path))
    logger.info(f"Molecular network saved to {output_path}")


def run_annotation_pipeline(config: MassFlowConfig) -> None:
    """
    Execute the full MassFlow annotation analysis pipeline.

    This function orchestrates the end-to-end workflow:
    1. Loads and processes the reference spectral library.
    2. Identifies input experimental files (single file or directory).
    3. Initializes the similarity search engine.
    4. Iterates through each experimental file:
       - Loads and sanitizes spectra.
       - Processes spectra (filtering, normalization).
       - Performs similarity searching against the reference library.
       - Exports the results to a CSV file.

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
    """
    # 1. Load Reference for main process (needed strictly for downstream networking export)
    if not config.input.reference_library:
        raise ValueError("Reference library path not specified in configuration.")

    logger.info(f"Loading reference library: {config.input.reference_library}")
    ref_gen = io.load_spectra(config.input.reference_library)
    reference_spectra = list(processing.process_spectra(ref_gen, config.processing))

    if not reference_spectra:
        raise ValueError("No valid spectra found in reference library.")

    logger.info(f"Loaded {len(reference_spectra)} reference spectra.")

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
                    io.save_match_results(results, out_file)
                    logger.info(f"Results saved to {out_file}")
                else:
                    logger.warning(f"No valid spectra extracted from {processed_file}.")
            except Exception as e:
                logger.error(f"Process failed for {qf}: {e}")

    # 4. Generate Molecular Network
    if config.workflow.perform_networking:
        network_out = config.output_directory / "molecular_network.graphml"
        generate_molecular_network(
            all_queries=all_queries,
            all_references=reference_spectra,
            all_results=all_results,
            config=config,
            output_path=network_out,
        )

    logger.info("Pipeline Finished.")
