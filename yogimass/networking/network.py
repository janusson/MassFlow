"""
Similarity network builder for MS/MS spectra.

Responsibilities:
- Build similarity graphs from spectra or stored libraries (threshold or k-NN).
- Compute pairwise similarities via supported metrics and export edge lists/graphs.
- Provide lightweight node/edge data classes used by reporting.

Depends on:
- ``yogimass.similarity.metrics`` and ``yogimass.similarity.processing`` for scores.
- ``yogimass.similarity.library`` for library-backed nodes.
- Optional ``networkx`` for graph exports.

Avoids:
- CLI/config parsing (handled in ``yogimass.workflow``/``yogimass.cli``).
- Curation or report writing; delegates to ``yogimass.reporting`` when needed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from matchms import Spectrum
from matchms.importing import load_from_mgf

from yogimass.io.mgf import list_mgf_libraries
from yogimass.similarity.library import LocalSpectralLibrary
from yogimass.similarity.metrics import (
    cosine_from_vectors,
    cosine_similarity,
    modified_cosine_similarity,
    spec2vec_vectorize,
)
from yogimass.similarity.processing import SpectrumProcessor
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)

try:  # Optional dependency
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover - optional
    nx = None  # type: ignore


MetricName = str


@dataclass
class SpectrumNode:
    """Node representation for a spectrum in the similarity graph."""

    identifier: str
    precursor_mz: float | None
    metadata: dict[str, Any]
    spectrum: Spectrum | None
    vector: dict[str, float] | None


@dataclass
class SimilarityEdge:
    """Edge representation for a similarity relationship between spectra."""

    source: str
    target: str
    similarity: float
    metric: MetricName


def build_network_from_folder(
    input_dir: str | Path,
    *,
    metric: MetricName = "cosine",
    threshold: float | None = None,
    knn: int | None = None,
    processor: SpectrumProcessor | None = None,
    reference_mz: Sequence[float] | None = None,
    undirected: bool = True,
) -> tuple[list[SpectrumNode], list[SimilarityEdge]]:
    """
    Load spectra from MGF files, process them, and build a similarity network.
    """
    spectra = _load_spectra_from_mgf(input_dir)
    return build_network_from_spectra(
        spectra,
        metric=metric,
        threshold=threshold,
        knn=knn,
        undirected=undirected,
        processor=processor,
        reference_mz=reference_mz,
    )


def build_network_from_library(
    library_path: str | Path,
    *,
    metric: MetricName = "spec2vec",
    threshold: float | None = None,
    knn: int | None = None,
    undirected: bool = True,
) -> tuple[list[SpectrumNode], list[SimilarityEdge]]:
    """
    Build a similarity network from a stored ``LocalSpectralLibrary``.
    """
    library = LocalSpectralLibrary(library_path)
    nodes = [
        SpectrumNode(
            identifier=entry.identifier,
            precursor_mz=entry.precursor_mz,
            metadata=entry.metadata,
            spectrum=None,
            vector=entry.vector,
        )
        for entry in library.iter_entries()
    ]
    return _build_edges(nodes, metric=metric, threshold=threshold, knn=knn, undirected=undirected)


def build_network_from_spectra(
    spectra: Sequence[Spectrum],
    *,
    metric: MetricName = "cosine",
    threshold: float | None = None,
    knn: int | None = None,
    processor: SpectrumProcessor | None = None,
    reference_mz: Sequence[float] | None = None,
    undirected: bool = True,
) -> tuple[list[SpectrumNode], list[SimilarityEdge]]:
    """
    Process spectra and construct a similarity network (threshold or k-NN).
    """
    processor = processor or SpectrumProcessor()
    nodes: list[SpectrumNode] = []
    for idx, spectrum in enumerate(spectra):
        processed = processor.process(spectrum, reference_mz=reference_mz)
        identifier = (
            processed.get("name")
            or processed.get("compound_name")
            or processed.get("spectrum_id")
            or f"spectrum-{idx}"
        )
        nodes.append(_node_from_spectrum(processed, identifier=identifier))
    return _build_edges(nodes, metric=metric, threshold=threshold, knn=knn, undirected=undirected)


def _load_spectra_from_mgf(input_dir: str | Path) -> list[Spectrum]:
    mgf_files = list_mgf_libraries(str(input_dir))
    spectra: list[Spectrum] = []
    for mgf_path in mgf_files:
        for spectrum in load_from_mgf(mgf_path):
            spectra.append(spectrum)
    if not spectra:
        logger.warning("No spectra found in %s", input_dir)
    return spectra


def _node_from_spectrum(spectrum: Spectrum, identifier: str) -> SpectrumNode:
    precursor = spectrum.get("precursor_mz") or spectrum.get("parent_mass")
    metadata = dict(spectrum.metadata or {})
    return SpectrumNode(
        identifier=identifier,
        precursor_mz=float(precursor) if precursor is not None else None,
        metadata=metadata,
        spectrum=spectrum,
        vector=None,
    )


def _build_edges(
    nodes: Sequence[SpectrumNode],
    *,
    metric: MetricName,
    threshold: float | None,
    knn: int | None,
    undirected: bool,
) -> tuple[list[SpectrumNode], list[SimilarityEdge]]:
    if threshold is None and knn is None:
        raise ValueError("Specify either a similarity threshold or k-NN value.")
    if threshold is not None and knn is not None:
        raise ValueError("Choose either threshold or k-NN mode, not both.")

    if threshold is not None:
        edges = _threshold_edges(nodes, metric=metric, threshold=threshold, undirected=undirected)
    else:
        edges = _knn_edges(nodes, metric=metric, k=knn or 0, undirected=undirected)
    return list(nodes), edges


def _threshold_edges(
    nodes: Sequence[SpectrumNode], *, metric: MetricName, threshold: float, undirected: bool
) -> list[SimilarityEdge]:
    edges: list[SimilarityEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    for idx, node in enumerate(nodes):
        for jdx in range(idx + 1, len(nodes)):
            other = nodes[jdx]
            score = _compute_similarity(node, other, metric)
            if score >= threshold:
                pair = _edge_key(node.identifier, other.identifier, undirected)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edges.append(
                    SimilarityEdge(
                        source=pair[0] if undirected else node.identifier,
                        target=pair[1] if undirected else other.identifier,
                        similarity=score,
                        metric=metric,
                    )
                )
    return edges


def _knn_edges(nodes: Sequence[SpectrumNode], *, metric: MetricName, k: int, undirected: bool) -> list[SimilarityEdge]:
    if k <= 0:
        raise ValueError("k-NN mode requires k > 0.")
    edges: list[SimilarityEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    for idx, node in enumerate(nodes):
        scores: list[tuple[str, float]] = []
        for jdx, other in enumerate(nodes):
            if idx == jdx:
                continue
            score = _compute_similarity(node, other, metric)
            scores.append((other.identifier, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        for target_id, score in scores[:k]:
            if score <= 0:
                continue
            pair = _edge_key(node.identifier, target_id, undirected)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append(
                SimilarityEdge(
                    source=pair[0] if undirected else node.identifier,
                    target=pair[1] if undirected else target_id,
                    similarity=score,
                    metric=metric,
                )
            )
    return edges


def _compute_similarity(node_a: SpectrumNode, node_b: SpectrumNode, metric: MetricName) -> float:
    if metric == "cosine":
        return cosine_similarity(_require_spectrum(node_a), _require_spectrum(node_b))
    if metric in {"modified_cosine", "modified-cosine"}:
        return modified_cosine_similarity(_require_spectrum(node_a), _require_spectrum(node_b))
    if metric == "spec2vec":
        vector_a = node_a.vector or spec2vec_vectorize(_require_spectrum(node_a))
        vector_b = node_b.vector or spec2vec_vectorize(_require_spectrum(node_b))
        return cosine_from_vectors(vector_a, vector_b)
    raise ValueError(f"Unsupported metric: {metric}")


def _require_spectrum(node: SpectrumNode) -> Spectrum:
    if node.spectrum is None:
        raise ValueError(f"Spectrum data unavailable for node '{node.identifier}'.")
    return node.spectrum


def _edge_key(source: str, target: str, undirected: bool) -> tuple[str, str]:
    if undirected:
        return tuple(sorted((source, target)))
    return (source, target)


def export_network(
    nodes: Sequence[SpectrumNode],
    edges: Sequence[SimilarityEdge],
    output_path: str | Path,
    *,
    undirected: bool = True,
) -> Path:
    """
    Export a similarity network to CSV, GraphML, or GEXF.
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        _export_edges_csv(edges, output_path)
    elif suffix in {".graphml", ".gexf", ".gml"}:
        _export_with_networkx(nodes, edges, output_path, suffix, undirected=undirected)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")
    return output_path


def _export_edges_csv(edges: Sequence[SimilarityEdge], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "similarity", "metric"])
        for edge in edges:
            writer.writerow([edge.source, edge.target, f"{edge.similarity:.6f}", edge.metric])
    logger.info("Wrote CSV edge list to %s", path)


def _export_with_networkx(
    nodes: Sequence[SpectrumNode],
    edges: Sequence[SimilarityEdge],
    path: Path,
    suffix: str,
    *,
    undirected: bool,
) -> None:
    if nx is None:
        raise ImportError("networkx is required for GraphML/GEXF/GML export.")
    graph = nx.Graph() if undirected else nx.DiGraph()
    for node in nodes:
        attrs = {f"meta_{k}": v for k, v in node.metadata.items() if v is not None}
        if node.precursor_mz is not None:
            attrs["precursor_mz"] = node.precursor_mz
        graph.add_node(node.identifier, **attrs)
    for edge in edges:
        graph.add_edge(edge.source, edge.target, similarity=edge.similarity, metric=edge.metric)

    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".graphml":
        nx.write_graphml(graph, path)
    elif suffix == ".gexf":
        nx.write_gexf(graph, path)
    elif suffix == ".gml":
        nx.write_gml(graph, path)
    else:  # pragma: no cover - guarded earlier
        raise ValueError(f"Unsupported graph suffix: {suffix}")
    logger.info("Wrote graph to %s", path)


__all__ = [
    "MetricName",
    "SimilarityEdge",
    "SpectrumNode",
    "build_network_from_folder",
    "build_network_from_library",
    "build_network_from_spectra",
    "export_network",
]
