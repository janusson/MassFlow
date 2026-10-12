"""
Opt-in network backend benchmark (experimental, ``scripts/`` is not collected by
pytest).

Two modes are supported:

``--mode backends`` (default)
    Build a deterministic synthetic molecular-similarity network (default 5,000
    nodes, ~8-12 neighbours/node) and compare **graph-representation backends**
    for the operations network-aware annotation needs:

    1. construction / representation conversion,
    2. traversal (connected components),
    3. one-hop weighted neighbour voting, and
    4. a memory proxy (``tracemalloc`` peak of construction).

    Available backends: NetworkX and python-igraph (optional import-time
    extras), SciPy CSR (core dependency), and a dependency-free custom CSR
    baseline. All backends must produce identical traversal and vote results
    before any timing is trusted. A readable table is printed and a
    machine-readable JSON report is written to ``--output-dir`` (default
    ``.benchmarks``).

``--mode spectral``
    Reproduces the original Stage-1 spectral-network construction benchmark
    (``build_spectral_graph`` over synthetic spectra).

Run with::

    uv run python scripts/benchmark_network.py
    uv run python scripts/benchmark_network.py --nodes 20000 --repeats 5
    uv run python scripts/benchmark_network.py --mode spectral --spectra 500
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as _scipy_connected_components

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Allow `python scripts/benchmark_network.py` to import the sibling
    # `scripts.benchmark_network_propagation` package.
    sys.path.insert(0, str(REPO_ROOT))

# Shared, already-tested building blocks (the same helpers back the
# ``tests/benchmarks/test_network_propagation.py`` CSR vote check).
from scripts.benchmark_network_propagation import (  # noqa: E402
    CSRAdjacency,
    _csr_components,
    _csr_vote,
    _igraph_components,
    _igraph_vote,
    _make_csr_adjacency,
    _median_seconds,
    _mock_edges,
    _mock_seed_labels,
    _networkx_components,
    _networkx_vote,
    ig,
    nx,
)

DEFAULT_OUTPUT_DIR = Path(".benchmarks")
SCHEMA_VERSION = "massflow.benchmark.network_backends.v1"


# ---------------------------------------------------------------------------
# SciPy CSR backend
# ---------------------------------------------------------------------------


def _make_scipy_adjacency(
    node_count: int, source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> sp.csr_matrix:
    """Build a symmetric weighted CSR adjacency matrix with SciPy."""
    rows = np.concatenate((source, target)).astype(np.int64, copy=False)
    columns = np.concatenate((target, source)).astype(np.int64, copy=False)
    values = np.concatenate((weights, weights)).astype(np.float64, copy=False)
    matrix = sp.coo_matrix(
        (values, (rows, columns)), shape=(node_count, node_count), dtype=np.float64
    ).tocsr()
    matrix.sort_indices()
    return matrix


def _scipy_components(matrix: sp.csr_matrix) -> tuple[int, ...]:
    """Return component sizes in descending order using SciPy's C routine."""
    _count, labels = _scipy_connected_components(matrix, directed=False)
    sizes = np.bincount(labels)
    return tuple(sorted((int(size) for size in sizes), reverse=True))


def _scipy_vote(
    matrix: sp.csr_matrix, seed_labels: np.ndarray, label_count: int
) -> np.ndarray:
    """One-hop weighted neighbour voting as a sparse matrix product.

    ``A @ S`` (adjacency times a one-hot seed matrix) sums, per node, the edge
    weight to every neighbour carrying each label — the same quantity the
    custom-CSR vote accumulates, so the two must agree exactly.
    """
    node_count = seed_labels.size
    seeded = np.nonzero(seed_labels >= 0)[0]
    predictions = seed_labels.copy()
    if seeded.size == 0:
        return predictions
    seed_matrix = sp.csr_matrix(
        (
            np.ones(seeded.size, dtype=np.float64),
            (seeded, seed_labels[seeded]),
        ),
        shape=(node_count, label_count),
    )
    votes = matrix @ seed_matrix
    row_totals = np.asarray(votes.sum(axis=1)).ravel()
    unseeded = seed_labels < 0
    target = unseeded & (row_totals > 0.0)
    if target.any():
        dense = np.asarray(votes.todense())
        predictions[target] = dense[target].argmax(axis=1)
    return predictions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measure_memory(build: Callable[[], Any]) -> tuple[Any, int]:
    """Return ``(built_object, peak_tracemalloc_bytes)`` for a builder.

    ``tracemalloc`` tracks **Python-level** allocations only; a C-backed
    backend (igraph) will under-report, so the value is reported as an
    explicitly qualified proxy.
    """
    tracemalloc.start()
    try:
        obj = build()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return obj, int(peak)


def _structural_bytes_csr(adjacency: CSRAdjacency) -> int:
    return int(
        adjacency.indptr.nbytes + adjacency.indices.nbytes + adjacency.weights.nbytes
    )


def _structural_bytes_scipy(matrix: sp.csr_matrix) -> int:
    return int(matrix.indptr.nbytes + matrix.indices.nbytes + matrix.data.nbytes)


def _dependency_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "polars": _polars_version(),
        "networkx": getattr(nx, "__version__", None) if nx is not None else None,
        "igraph": getattr(ig, "__version__", None) if ig is not None else None,
    }
    return versions


def _polars_version() -> str | None:
    try:
        import polars as pl

        return pl.__version__
    except ImportError:  # pragma: no cover - polars is a core dependency
        return None


# ---------------------------------------------------------------------------
# Backend comparison
# ---------------------------------------------------------------------------


def _run_backends(args: argparse.Namespace) -> int:
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
    edge_count = int(source.size)
    seeded_count = int(np.count_nonzero(seed_labels >= 0))
    mean_degree = 2.0 * edge_count / args.nodes
    print(
        f"nodes={args.nodes} edges={edge_count} mean_degree={mean_degree:.2f} "
        f"seeded={seeded_count} labels={args.labels} repeats={args.repeats}"
    )

    expected_components: tuple[int, ...] | None = None
    expected_predictions: np.ndarray | None = None
    report_backends: dict[str, Any] = {}

    def check_parity(
        name: str, components: tuple[int, ...], predictions: np.ndarray
    ) -> None:
        nonlocal expected_components, expected_predictions
        if expected_components is None:
            expected_components = components
            expected_predictions = predictions
        elif components != expected_components:
            raise RuntimeError(f"{name} component results differ from the reference")
        elif expected_predictions is None or not np.array_equal(
            predictions, expected_predictions
        ):
            raise RuntimeError(f"{name} vote results differ from the reference")

    def report(
        name: str,
        graph: Any,
        component_function: Callable[[Any], tuple[int, ...]],
        vote_function: Callable[[Any, np.ndarray, int], np.ndarray],
        *,
        construction_seconds: float,
        memory_peak_bytes: int | None,
        memory_structural_bytes: int | None,
    ) -> None:
        components = component_function(graph)
        predictions = vote_function(graph, seed_labels, args.labels)
        check_parity(name, components, predictions)
        traversal_seconds = _median_seconds(
            lambda: component_function(graph), args.repeats
        )
        voting_seconds = _median_seconds(
            lambda: vote_function(graph, seed_labels, args.labels), args.repeats
        )
        report_backends[name] = {
            "available": True,
            "construction_seconds": construction_seconds,
            "traversal_seconds": traversal_seconds,
            "neighbor_vote_seconds": voting_seconds,
            "component_count": len(components),
            "memory_peak_bytes": memory_peak_bytes,
            "memory_structural_bytes": memory_structural_bytes,
        }
        memory_text = (
            f"{memory_peak_bytes / 1024:.0f}KiB"
            if memory_peak_bytes is not None
            else "n/a"
        )
        print(
            f"{name:12s} build_s={construction_seconds:.6f} "
            f"traversal_s={traversal_seconds:.6f} "
            f"vote_s={voting_seconds:.6f} mem_peak={memory_text}"
        )

    # ---- NetworkX (optional) --------------------------------------------------
    if nx is not None:

        def build_networkx() -> Any:
            graph = nx.Graph()
            graph.add_nodes_from(range(args.nodes))
            graph.add_weighted_edges_from(
                (int(left), int(right), float(weight))
                for left, right, weight in zip(source, target, weights, strict=True)
            )
            return graph

        started = time.perf_counter()
        networkx_graph, networkx_peak = _measure_memory(build_networkx)
        build_seconds = time.perf_counter() - started
        report(
            "networkx",
            networkx_graph,
            _networkx_components,
            _networkx_vote,
            construction_seconds=build_seconds,
            memory_peak_bytes=networkx_peak,
            memory_structural_bytes=None,
        )
    else:
        report_backends["networkx"] = {"available": False}
        print("networkx     unavailable (install networkx to include this backend)")

    # ---- python-igraph (optional) --------------------------------------------
    if ig is not None:

        def build_igraph() -> Any:
            graph = ig.Graph(
                n=args.nodes,
                edges=list(zip(source.tolist(), target.tolist(), strict=True)),
                directed=False,
            )
            graph.es["weight"] = weights.tolist()
            return graph

        started = time.perf_counter()
        igraph_graph, igraph_peak = _measure_memory(build_igraph)
        build_seconds = time.perf_counter() - started
        report(
            "igraph",
            igraph_graph,
            _igraph_components,
            _igraph_vote,
            construction_seconds=build_seconds,
            memory_peak_bytes=igraph_peak,
            memory_structural_bytes=None,
        )
    else:
        report_backends["igraph"] = {"available": False}
        print(
            "igraph       unavailable (install python-igraph to include this backend)"
        )

    # ---- SciPy CSR (core dependency) -----------------------------------------
    started = time.perf_counter()
    scipy_matrix, scipy_peak = _measure_memory(
        lambda: _make_scipy_adjacency(args.nodes, source, target, weights)
    )
    build_seconds = time.perf_counter() - started
    report(
        "scipy-csr",
        scipy_matrix,
        _scipy_components,
        _scipy_vote,
        construction_seconds=build_seconds,
        memory_peak_bytes=scipy_peak,
        memory_structural_bytes=_structural_bytes_scipy(scipy_matrix),
    )

    # ---- Custom CSR (dependency-free baseline) -------------------------------
    started = time.perf_counter()
    custom_adjacency, custom_peak = _measure_memory(
        lambda: _make_csr_adjacency(args.nodes, source, target, weights)
    )
    build_seconds = time.perf_counter() - started
    report(
        "custom-csr",
        custom_adjacency,
        _csr_components,
        _csr_vote,
        construction_seconds=build_seconds,
        memory_peak_bytes=custom_peak,
        memory_structural_bytes=_structural_bytes_csr(custom_adjacency),
    )

    print("parity=ok (traversal and one-hop vote agree across every available backend)")

    report_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "mode": "backends",
        "dependencies": _dependency_versions(),
        "graph": {
            "nodes": args.nodes,
            "edges": edge_count,
            "mean_degree": mean_degree,
            "block_size": args.block_size,
            "extra_edges_per_node": args.extra_edges_per_node,
            "seed_fraction": args.seed_fraction,
            "labels": args.labels,
            "seeded_nodes": seeded_count,
            "seed": args.seed,
        },
        "repeats": args.repeats,
        "backends": report_backends,
        "memory_proxy_note": (
            "memory_peak_bytes is a tracemalloc peak (Python-level allocations "
            "only); C-allocated memory (e.g. igraph) is under-reported. "
            "memory_structural_bytes is the CSR array footprint when applicable."
        ),
        "parity": "ok",
    }
    _write_report(report_payload, args.output_dir, prefix="network_backends")
    return 0


# ---------------------------------------------------------------------------
# Spectral construction benchmark (preserved from the original script)
# ---------------------------------------------------------------------------


def _synthetic_spectra(count: int, rng: np.random.Generator) -> list[Any]:
    from matchms import Spectrum

    spectra: list[Any] = []
    for index in range(count):
        peak_count = int(rng.integers(8, 40))
        mz = np.sort(rng.uniform(50.0, 900.0, peak_count)).astype(np.float64)
        intensities = rng.uniform(0.01, 1.0, peak_count).astype(np.float64)
        spectra.append(
            Spectrum(
                mz=mz,
                intensities=intensities,
                metadata={
                    "id": f"bench_{index}",
                    "precursor_mz": float(rng.uniform(200.0, 800.0)),
                    "retention_time": float(rng.uniform(0.0, 1800.0)),
                    "adduct": "[M+H]+",
                    "charge": 1,
                },
            )
        )
    return spectra


def _run_spectral(args: argparse.Namespace) -> int:
    from MassFlow.config import NetworkConfig
    from MassFlow.network.build import build_spectral_graph

    rng = np.random.default_rng(args.seed)
    spectra = _synthetic_spectra(args.spectra, rng)
    cfg = NetworkConfig(
        enabled=True, min_score=0.7, min_matched_peaks=6, top_k_per_node=10
    )

    started = time.perf_counter()
    graph = build_spectral_graph(spectra, cfg)
    elapsed = time.perf_counter() - started

    print(
        f"spectra={len(spectra)} nodes={len(graph.nodes)} "
        f"edges={len(graph.relationships)} seconds={elapsed:.3f}"
    )
    payload: dict[str, Any] = {
        "schema_version": "massflow.benchmark.network_spectral.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "mode": "spectral",
        "spectra": len(spectra),
        "nodes": len(graph.nodes),
        "edges": len(graph.relationships),
        "seconds": elapsed,
    }
    _write_report(payload, args.output_dir, prefix="network_spectral")
    return 0


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------


def _write_report(payload: dict[str, Any], output_dir: Path, *, prefix: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{prefix}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print(f"report={path}")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_network.py",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backends", "spectral"],
        default="backends",
        help="Which network benchmark to run.",
    )
    # Backend-mode options.
    parser.add_argument("--nodes", type=int, default=5000, help="Mock graph nodes.")
    parser.add_argument(
        "--block-size", type=int, default=100, help="Nodes per synthetic component."
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
    # Spectral-mode options.
    parser.add_argument(
        "--spectra", type=int, default=500, help="Synthetic spectra (spectral mode)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the JSON report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.nodes < 2:
        parser.error("--nodes must be at least 2")
    if args.mode == "backends":
        return _run_backends(args)
    return _run_spectral(args)


if __name__ == "__main__":
    raise SystemExit(main())
