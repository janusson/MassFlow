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
import scipy.sparse as sp
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

    scores_obj = calculate_scores(
        references=all_queries,
        queries=all_queries,
        similarity_function=sim_func,
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

    # 4. Add Query-to-Query Edges (From new matrix)
    min_score = config.similarity.min_score
    # Only keep upper triangle to avoid duplicate edges (since it's symmetric)
    adj_matrix = np.triu(numeric_scores, k=1)
    adj_matrix[adj_matrix < min_score] = 0.0

    sparse_adj = sp.coo_matrix(adj_matrix)

    for i, j, v in zip(sparse_adj.row, sparse_adj.col, sparse_adj.data):
        q1_id = str(all_queries[i].get("id", f"query_{i}"))
        q2_id = str(all_queries[j].get("id", f"query_{j}"))
        G.add_edge(q1_id, q2_id, weight=float(v), edge_type="query_to_query")

    # Export to GraphML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(output_path))
    logger.info(
        f"Molecular network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) saved to {output_path}"
    )
