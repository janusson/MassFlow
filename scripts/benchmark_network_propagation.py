"""Opt-in graph traversal and one-hop label-voting benchmark.

Build a deterministic mock network (default: 5,000 nodes) and compare available
backends. Neighbor voting is deliberately bounded to direct, fixed seed labels;
predictions are never reused as seeds. NetworkX and python-igraph are optional
benchmark backends; custom CSR uses MassFlow's existing NumPy dependency.

Run with::

    uv run python scripts/benchmark_network_propagation.py
    uv run python scripts/benchmark_network_propagation.py --nodes 20000 --repeats 5
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np

try:
    import networkx as nx
except ImportError:  # optional benchmark backend
    nx = None  # type: ignore[assignment]

try:
    import igraph as ig
except ImportError:  # optional benchmark backend
    ig = None  # type: ignore[assignment]


def _mock_edges(
    node_count: int, *, block_size: int, extra_edges_per_node: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make deterministic disconnected blocks with weighted undirected edges."""
    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    if block_size < 2 or extra_edges_per_node < 0:
        raise ValueError("block_size must be >= 2 and extra_edges_per_node >= 0")

    rng = np.random.default_rng(seed)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []

    for block_start in range(0, node_count, block_size):
        block_end = min(block_start + block_size, node_count)
        block_nodes = np.arange(block_start, block_end, dtype=np.int64)
        if block_nodes.size < 2:
            continue
        left_parts.append(block_nodes)
        right_parts.append(np.roll(block_nodes, -1))

        extra_count = extra_edges_per_node * block_nodes.size
        extra_left = rng.integers(block_start, block_end, size=extra_count)
        extra_right = rng.integers(block_start, block_end, size=extra_count)
        valid = extra_left != extra_right
        left_parts.append(extra_left[valid].astype(np.int64))
        right_parts.append(extra_right[valid].astype(np.int64))

    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    source = np.minimum(left, right)
    target = np.maximum(left, right)
    edge_pairs = np.unique(np.column_stack((source, target)), axis=0)
    weights = rng.uniform(0.2, 1.0, size=edge_pairs.shape[0]).astype(np.float64)
    return edge_pairs[:, 0], edge_pairs[:, 1], weights


def _mock_seed_labels(
    node_count: int, *, label_count: int, seed_fraction: float, seed: int
) -> np.ndarray:
    """Assign fixed seed labels to a deterministic subset of nodes."""
    if label_count < 1 or not 0.0 < seed_fraction <= 1.0:
        raise ValueError("label_count must be positive and seed_fraction in (0, 1]")
    rng = np.random.default_rng(seed)
    labels = np.full(node_count, -1, dtype=np.int64)
    seeded = rng.random(node_count) < seed_fraction
    labels[seeded] = rng.integers(0, label_count, size=int(seeded.sum()))
    labels[: min(node_count, label_count)] = np.arange(min(node_count, label_count))
    return labels


@dataclass(frozen=True)
class CSRAdjacency:
    """A symmetric weighted graph in compressed sparse row form."""

    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    node_count: int


def _networkx_components(graph: Any) -> tuple[int, ...]:
    """Return component sizes in descending order for NetworkX."""
    return tuple(
        sorted((len(part) for part in nx.connected_components(graph)), reverse=True)
    )


def _igraph_components(graph: Any) -> tuple[int, ...]:
    """Return component sizes in descending order for python-igraph."""
    return tuple(sorted(graph.connected_components().sizes(), reverse=True))


def _csr_components(adjacency: CSRAdjacency) -> tuple[int, ...]:
    """Return component sizes in descending order using CSR depth-first traversal."""
    visited = np.zeros(adjacency.node_count, dtype=np.bool_)
    component_sizes: list[int] = []
    for start_node in range(adjacency.node_count):
        if visited[start_node]:
            continue
        visited[start_node] = True
        stack = [start_node]
        component_size = 0
        while stack:
            node = stack.pop()
            component_size += 1
            for neighbor in adjacency.indices[
                adjacency.indptr[node] : adjacency.indptr[node + 1]
            ]:
                neighbor_index = int(neighbor)
                if not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    stack.append(neighbor_index)
        component_sizes.append(component_size)
    return tuple(sorted(component_sizes, reverse=True))


def _networkx_vote(graph: Any, seed_labels: np.ndarray, label_count: int) -> np.ndarray:
    """Perform weighted one-hop voting from direct seed labels with NetworkX."""
    predictions = seed_labels.copy()
    for node in range(seed_labels.size):
        if seed_labels[node] >= 0:
            continue
        votes = np.zeros(label_count, dtype=np.float64)
        for neighbor, attributes in graph[node].items():
            label = int(seed_labels[neighbor])
            if label >= 0:
                votes[label] += float(attributes["weight"])
        if votes.any():
            predictions[node] = int(np.argmax(votes))
    return predictions


def _igraph_vote(graph: Any, seed_labels: np.ndarray, label_count: int) -> np.ndarray:
    """Perform weighted one-hop voting from direct seed labels with igraph."""
    predictions = seed_labels.copy()
    for node in range(seed_labels.size):
        if seed_labels[node] >= 0:
            continue
        votes = np.zeros(label_count, dtype=np.float64)
        for edge_id in graph.incident(node, mode="all"):
            edge = graph.es[edge_id]
            source, target = edge.tuple
            neighbor = target if source == node else source
            label = int(seed_labels[neighbor])
            if label >= 0:
                votes[label] += float(edge["weight"])
        if votes.any():
            predictions[node] = int(np.argmax(votes))
    return predictions


def _csr_vote(
    adjacency: CSRAdjacency, seed_labels: np.ndarray, label_count: int
) -> np.ndarray:
    """Perform weighted one-hop voting with CSR rows and NumPy aggregation."""
    predictions = seed_labels.copy()
    for node in range(seed_labels.size):
        if seed_labels[node] >= 0:
            continue
        row_start = int(adjacency.indptr[node])
        row_end = int(adjacency.indptr[node + 1])
        neighbor_indices = adjacency.indices[row_start:row_end]
        neighbor_labels = seed_labels[neighbor_indices]
        has_seed_neighbor = neighbor_labels >= 0
        if not has_seed_neighbor.any():
            continue
        votes = np.bincount(
            neighbor_labels[has_seed_neighbor],
            weights=adjacency.weights[row_start:row_end][has_seed_neighbor],
            minlength=label_count,
        )
        predictions[node] = int(np.argmax(votes))
    return predictions


def _median_seconds(function: Callable[[], Any], repeats: int) -> float:
    """Warm up a callable and return its median wall-clock duration."""
    function()
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        timings.append(time.perf_counter() - started)
    return median(timings)


def _make_csr_adjacency(
    node_count: int, source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> CSRAdjacency:
    """Build a symmetric weighted CSR adjacency structure with NumPy."""
    row_indices = np.concatenate((source, target))
    column_indices = np.concatenate((target, source))
    values = np.concatenate((weights, weights)).astype(np.float64, copy=False)
    order = np.lexsort((column_indices, row_indices))
    sorted_rows = row_indices[order]
    indices = column_indices[order].astype(np.int64, copy=False)
    sorted_weights = values[order]
    row_counts = np.bincount(sorted_rows, minlength=node_count)
    indptr = np.empty(node_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    return CSRAdjacency(indptr, indices, sorted_weights, node_count)


def main() -> None:
    """Run the graph traversal and label-voting benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes", type=int, default=5000, help="Mock graph node count."
    )
    parser.add_argument(
        "--block-size", type=int, default=100, help="Nodes per component."
    )
    parser.add_argument(
        "--extra-edges-per-node",
        type=int,
        default=4,
        help="Random intra-block edges per node.",
    )
    parser.add_argument("--labels", type=int, default=32, help="Number of seed labels.")
    parser.add_argument("--seed-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=23, help="Synthetic data RNG seed.")
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed repetitions per operation."
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    source, target, weights = _mock_edges(
        args.nodes,
        block_size=args.block_size,
        extra_edges_per_node=args.extra_edges_per_node,
        seed=args.seed,
    )
    seed_labels = _mock_seed_labels(
        args.nodes,
        label_count=args.labels,
        seed_fraction=args.seed_fraction,
        seed=args.seed + 1,
    )
    edge_count = source.size
    seeded_count = int(np.count_nonzero(seed_labels >= 0))
    print(
        f"nodes={args.nodes} edges={edge_count} mean_degree={2 * edge_count / args.nodes:.2f} "
        f"seeded={seeded_count} labels={args.labels} repeats={args.repeats}"
    )

    expected_components: tuple[int, ...] | None = None
    expected_predictions: np.ndarray | None = None

    def report_backend(
        name: str,
        graph: Any,
        component_function: Callable[[Any], tuple[int, ...]],
        vote_function: Callable[[Any, np.ndarray, int], np.ndarray],
    ) -> None:
        nonlocal expected_components, expected_predictions
        components = component_function(graph)
        predictions = vote_function(graph, seed_labels, args.labels)
        if expected_components is None:
            expected_components = components
            expected_predictions = predictions
        elif components != expected_components:
            raise RuntimeError(
                f"{name} component results differ from the reference backend"
            )
        elif expected_predictions is None:
            raise RuntimeError(
                "reference labels were not initialized with the components"
            )
        elif not np.array_equal(predictions, expected_predictions):
            raise RuntimeError(f"{name} vote results differ from the reference backend")

        traversal_seconds = _median_seconds(
            lambda: component_function(graph), args.repeats
        )
        voting_seconds = _median_seconds(
            lambda: vote_function(graph, seed_labels, args.labels), args.repeats
        )
        print(
            f"{name:12s} components={len(components):4d} "
            f"traversal_median_s={traversal_seconds:.6f} "
            f"neighbor_vote_median_s={voting_seconds:.6f}"
        )

    if nx is not None:
        started = time.perf_counter()
        networkx_graph = nx.Graph()
        networkx_graph.add_nodes_from(range(args.nodes))
        networkx_graph.add_weighted_edges_from(
            (int(left), int(right), float(weight))
            for left, right, weight in zip(source, target, weights, strict=True)
        )
        build_seconds = time.perf_counter() - started
        report_backend("networkx", networkx_graph, _networkx_components, _networkx_vote)
        print(f"{'':12s} build_s={build_seconds:.6f}")
    else:
        print("networkx     unavailable (install networkx to include this backend)")

    if ig is not None:
        started = time.perf_counter()
        igraph_graph = ig.Graph(
            n=args.nodes,
            edges=list(zip(source.tolist(), target.tolist(), strict=True)),
            directed=False,
        )
        igraph_graph.es["weight"] = weights.tolist()
        build_seconds = time.perf_counter() - started
        report_backend("igraph", igraph_graph, _igraph_components, _igraph_vote)
        print(f"{'':12s} build_s={build_seconds:.6f}")
    else:
        print(
            "igraph       unavailable (install python-igraph to include this backend)"
        )

    started = time.perf_counter()
    adjacency = _make_csr_adjacency(args.nodes, source, target, weights)
    build_seconds = time.perf_counter() - started
    report_backend("custom-csr", adjacency, _csr_components, _csr_vote)
    print(f"{'':12s} build_s={build_seconds:.6f}")
    print(
        "parity=ok (component sizes and propagated labels match across available backends)"
    )


if __name__ == "__main__":
    main()
