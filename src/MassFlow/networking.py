"""
EXPERIMENTAL: Graph construction utilities for MassFlow molecular networking.

This module converts annotation outputs into a GraphML network suitable for
downstream visualization in tools such as Cytoscape or Gephi. The resulting
graph contains query nodes, reference nodes, query-to-reference annotation
edges from prior search results, and query-to-query similarity edges computed
on demand from the processed query spectra.

Note: Molecular networking is currently an experimental feature and is not
part of the stable v1.0 contract.
"""

import logging
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from matchms import Spectrum, calculate_scores

from MassFlow.config import MassFlowConfig
from MassFlow.similarity import SearchResult, get_similarity_engine

logger = logging.getLogger(__name__)


def generate_molecular_network(
    all_queries: List[Spectrum],
    all_references: List[Spectrum],
    all_results: List[SearchResult],
    config: MassFlowConfig,
    output_path: Path,
) -> None:
    """
    Generate a molecular network (GraphML) from query and reference spectra.

    The network is assembled from two edge sources:

    - query-to-reference edges derived from existing annotation results that
      satisfy the configured score and FDR thresholds.
    - query-to-query edges derived from an all-vs-all similarity calculation on
      the processed query spectra.

    Reference nodes are included so annotation hits and de novo query clusters
    appear in the same exported graph.

    Parameters
    ----------
    all_queries : List[matchms.Spectrum]
        The full list of experimental query spectra.
    all_references : List[matchms.Spectrum]
        The full list of reference library spectra.
    all_results : List[SearchResult]
        The computed matches between queries and references.
    config : MassFlowConfig
        The full pipeline configuration, detailing similarity thresholds.
    output_path : Path
        The destination file path for the output GraphML network.

    Returns
    -------
    None

    Notes
    -----
    For consensus similarity, the first underlying engine is reused to compute
    query-to-query edges. If the selected engine does not expose a direct
    ``similarity_function`` suitable for symmetric scoring, the function logs a
    warning and returns without writing a network file.
    """
    logger.info("Generating molecular network...")

    if not all_queries:
        logger.warning("No query spectra available for networking.")
        return

    engine = get_similarity_engine(config.similarity)

    # Calculate query-to-query similarity (All vs All)
    # The factory returns an object with a similarity_function (if it's not a ConsensusEngine)
    # We will use the primary configured algorithm for networking if it's a standard engine
    sim_func = None
    if hasattr(engine, "similarity_function"):
        sim_func = engine.similarity_function
    elif hasattr(engine, "engines") and len(engine.engines) > 0:
        # Fallback to the first engine in a consensus setup for query-to-query network building
        sim_func = engine.engines[0][0].similarity_function
        logger.warning(
            f"Consensus selected. Using {engine.engines[0][0].config.algorithm} for query-to-query networking."
        )

    if sim_func is None:
        logger.warning("Could not determine similarity function for networking.")
        return

    G = nx.Graph()

    # 1. Add Query Nodes
    for i, q in enumerate(all_queries):
        node_id = str(q.get("id", f"query_{i}"))
        mz = q.get("precursor_mz")
        G.add_node(
            node_id,
            node_type="query",
            precursor_mz=float(mz) if mz is not None else 0.0,
        )

    # 2. Add Reference Nodes
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

    # 3. Add Query-to-Reference Edges (From existing results)
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

    # 4. Add Query-to-Query Edges (Chunked calculation to prevent OOM)
    min_score = config.similarity.min_score
    n_queries = len(all_queries)
    chunk_size = 1000

    logger.info(
        f"Computing query-to-query similarity for {n_queries} spectra in chunks..."
    )

    for i in range(0, n_queries, chunk_size):
        end_i = min(i + chunk_size, n_queries)
        q_chunk = all_queries[i:end_i]

        for j in range(i, n_queries, chunk_size):
            end_j = min(j + chunk_size, n_queries)
            r_chunk = all_queries[j:end_j]

            try:
                scores_obj = calculate_scores(
                    references=r_chunk,
                    queries=q_chunk,
                    similarity_function=sim_func,
                    is_symmetric=(i == j),
                )

                scores_data = scores_obj.scores
                if hasattr(scores_data, "to_array"):
                    scores_array = scores_data.to_array()
                else:
                    scores_array = np.asarray(scores_data)

                if (
                    hasattr(scores_array.dtype, "names")
                    and scores_array.dtype.names is not None
                ):
                    score_cols = [
                        c for c in scores_array.dtype.names if "score" in c.lower()
                    ]
                    numeric_scores = scores_array[score_cols[0]]
                else:
                    numeric_scores = scores_array

                # Add edges for scores above threshold, strictly upper triangle
                row_indices, col_indices = np.where(numeric_scores >= min_score)
                for r_idx, c_idx in zip(row_indices, col_indices):
                    global_i = i + r_idx
                    global_j = j + c_idx

                    if global_j > global_i:
                        score_val = float(numeric_scores[r_idx, c_idx])
                        q1_id = str(
                            all_queries[global_i].get("id", f"query_{global_i}")
                        )
                        q2_id = str(
                            all_queries[global_j].get("id", f"query_{global_j}")
                        )
                        G.add_edge(
                            q1_id, q2_id, weight=score_val, edge_type="query_to_query"
                        )

            except Exception as e:
                logger.error(
                    f"Chunked similarity calculation failed at indices {i}, {j}: {e}",
                    exc_info=True,
                )
                continue

    # Export to GraphML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(output_path))
    logger.info(
        f"Molecular network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) saved to {output_path}"
    )
