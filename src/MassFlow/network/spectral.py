"""
Stage 1 spectral-networking edge construction.

Builds :class:`~MassFlow.network.models.SpectralRelationship` edges between
experimental spectra by reusing the trusted classical similarity engine
(:class:`MassFlow.similarity.SimilarityEngine`) with ``include_decoys=False`` —
decoys are an annotation-only FDR device and never participate in networking.

Determinism
-----------
* Candidate pairs come from :func:`~MassFlow.network.candidates.generate_candidate_pairs`.
* Each unordered candidate pair is scored exactly once as a bounded
  single-query search, so no ``N x N`` score matrix is ever materialized
  (peak memory is bounded by the largest per-node candidate set).
* Edges are emitted in a canonical ``(source, target)`` order, and the
  relationship identifiers are content-derived, so repeated runs on identical
  input produce identical graphs.

Confidence discipline
---------------------
Edges carry similarity only. This module never reads or writes q-values, and
``SpectralRelationship`` has no field that could carry statistical confidence.

The heavy imports (:mod:`MassFlow.config`, :mod:`MassFlow.similarity`) are
performed lazily inside functions so that importing the networking package (or
:mod:`MassFlow.config`) stays lightweight and cycle-free.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from MassFlow.network.candidates import generate_candidate_pairs
from MassFlow.network.models import (
    RelationshipProvenance,
    SpectralRelationship,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MassFlow.config import NetworkConfig

logger = logging.getLogger(__name__)

__all__ = ["build_spectral_relationships"]


def _precursor_mz(spectrum: object) -> float:
    """Return the spectrum precursor m/z as float64, or ``NaN`` when missing."""
    value = spectrum.get("precursor_mz")  # type: ignore[attr-defined]
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _retention_time_seconds(spectrum: object) -> float:
    """Return the retention time in seconds as float64, or ``NaN`` when missing."""
    value = spectrum.get("retention_time")  # type: ignore[attr-defined]
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _network_similarity_config(cfg: NetworkConfig):
    """Map ``NetworkConfig`` onto the classical ``SimilarityConfig`` subset."""
    from MassFlow.config import SimilarityConfig

    return SimilarityConfig(
        algorithm=cfg.algorithm,
        ms1_tolerance=cfg.ms1_tolerance,
        ms2_tolerance=cfg.ms2_tolerance,
        min_score=cfg.min_score,
        min_matched_peaks=cfg.min_matched_peaks,
        rt_tolerance=cfg.rt_tolerance,
    )


def build_spectral_relationships(
    spectra: Sequence[object],
    node_ids: Sequence[str],
    cfg: NetworkConfig,
    *,
    provenance: RelationshipProvenance,
    library_build_id: Optional[int] = None,
) -> list[SpectralRelationship]:
    """
    Score candidate spectrum pairs and return the surviving spectral edges.

    Parameters
    ----------
    spectra : sequence of matchms.Spectrum
        The (already de-duplicated, node-aligned) experimental spectra.
    node_ids : sequence of str
        Stable node identifiers aligned with ``spectra``.
    cfg : NetworkConfig
        Thresholds and similarity algorithm for the network.
    provenance : RelationshipProvenance
        Per-edge provenance recorded on every emitted edge.
    library_build_id : int or None, optional
        Reference-library build row to record on the edges, when applicable.

    Returns
    -------
    list of SpectralRelationship
        Undirected edges above the configured threshold, capped per node by
        ``cfg.top_k_per_node``, in canonical ``(source, target)`` order.

    Raises
    ------
    ValueError
        If ``node_ids`` is not aligned with ``spectra``.
    """
    n = len(spectra)
    if len(node_ids) != n:
        raise ValueError(
            f"node_ids length ({len(node_ids)}) must match spectra length ({n})."
        )
    if n < 2:
        return []

    from MassFlow.similarity import SimilarityEngine

    engine = SimilarityEngine(_network_similarity_config(cfg))

    # Clone spectra and stamp index-derived ids so the engine's results map back
    # to indices. The caller's spectra are never mutated.
    clones = []
    index_by_id: dict[str, int] = {}
    for i, spectrum in enumerate(spectra):
        clone = spectrum.clone()  # type: ignore[attr-defined]
        clone_id = f"__mf_node_{i}"
        clone.set("id", clone_id)
        index_by_id[clone_id] = i
        clones.append(clone)

    precursor_mzs = np.array([_precursor_mz(s) for s in spectra], dtype=np.float64)
    rt_seconds = np.array(
        [_retention_time_seconds(s) for s in spectra], dtype=np.float64
    )
    candidate_pairs = generate_candidate_pairs(
        precursor_mzs,
        tolerance=cfg.ms1_tolerance,
        rt_seconds=rt_seconds,
        rt_tolerance=cfg.rt_tolerance,
    )

    # Group candidate pairs by their (smaller) index so every unordered pair is
    # scored exactly once. Each search is a bounded ``1 x degree`` problem, so
    # the peak score matrix is O(max_degree) rather than O(N^2).
    neighbors: dict[int, list[int]] = {}
    for left, right in candidate_pairs:
        neighbors.setdefault(left, []).append(right)

    best: dict[tuple[int, int], tuple[float, int]] = {}
    for query_index in sorted(neighbors):
        reference_indices = neighbors[query_index]
        results = engine.search(
            query_spectra=[clones[query_index]],
            reference_spectra=[clones[index] for index in reference_indices],
            include_decoys=False,
            top_n=None,
        )
        for row in results:
            reference_index = index_by_id.get(str(row.get("reference_id")))
            if reference_index is None or reference_index == query_index:
                continue
            if query_index < reference_index:
                key = (query_index, reference_index)
            else:
                key = (reference_index, query_index)
            score = float(row["score"])
            matched = int(row["matched_peaks"])
            current = best.get(key)
            if current is None or score > current[0]:
                best[key] = (score, matched)

    logger.debug(
        "Spectral networking: %d node(s), %d candidate pair(s), %d scored edge(s).",
        n,
        len(candidate_pairs),
        len(best),
    )

    ordered = sorted(
        best.items(), key=lambda item: (-item[1][0], item[0][0], item[0][1])
    )

    if cfg.top_k_per_node is not None:
        degree = [0] * n
        kept: list[tuple[tuple[int, int], tuple[float, int]]] = []
        for pair, value in ordered:
            a, b = pair
            if degree[a] < cfg.top_k_per_node and degree[b] < cfg.top_k_per_node:
                kept.append((pair, value))
                degree[a] += 1
                degree[b] += 1
    else:
        kept = list(ordered)

    relationships = [
        SpectralRelationship.create(
            source_node_id=str(node_ids[a]),
            target_node_id=str(node_ids[b]),
            score=score,
            matched_peaks=matched,
            algorithm=engine.config.algorithm,
            provenance=provenance,
            directed=False,
            ms1_tolerance=cfg.ms1_tolerance,
            ms2_tolerance=cfg.ms2_tolerance,
            tolerance_unit="Da",
            library_build_id=library_build_id,
        )
        for (a, b), (score, matched) in kept
    ]
    relationships.sort(key=lambda rel: (rel.source_node_id, rel.target_node_id))
    return relationships
