"""
Unified network annotation export and graph serialization (experimental).

This module turns the DataFrame-level network results produced by
:mod:`MassFlow.network.propagation` into (a) a single, tiered **consensus
annotation table** and (b) a **node/edge graph document** for visualization
(GraphML or JSON).

Scope and governance
--------------------
GraphML/Cytoscape *as the core MassFlow interface* remains a declared non-goal
(``docs/network-aware-annotation-spec.md`` §14). This module provides an
**opt-in, Python-API-only** graph export; it is deliberately not wired into the
CLI or the default pipeline, and it never changes the annotation contract.

* **File I/O** is delegated to :func:`MassFlow.io.save_network_graph`
  (``AGENTS.md`` §6.2) — no other module writes files directly.
* **networkx** is an *optional* backend used only for GraphML. It is imported
  lazily and guarded; it is already provided transitively by ``matchms`` but is
  declared explicitly by the ``graph`` extra. The JSON node-link export is
  dependency-free and always available.
* Tiers are **descriptive**: no tier re-derives a q-value, p-value or score.
  Direct evidence and network hypotheses remain distinct rows.

Tiering
-------
``confidence_tier`` is assigned per row (first matching rule wins):

* **Tier 1 — direct high-confidence**: a non-decoy direct library match with
  ``q_value <= direct_high_confidence_q_value`` (default ``0.01``).
* **Tier 2 — network-supported consensus**: an unambiguous network annotation
  (``network_status == "network_supported"``).
* **Tier 3 — conflicted / low-support network**: a conflicted network annotation
  (``network_conflicted`` / ``direct_match_conflict``) or any other network
  hypothesis.
* Decoys and direct hits that are neither high-confidence nor network-supported
  receive no tier (``null``) — they are not consensus annotations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import polars as pl

from MassFlow import io
from MassFlow.network.propagation import (
    PROPAGATION_OUTPUT_SCHEMA,
    _clean_text,
    _coerce_direct_value,
    _finite_float,
    _row_is_decoy,
    _resolve_candidate_id,
    validate_network_edges,
)

try:  # optional backend — declared by the `graph` extra, shipped by matchms
    import networkx as nx

    _HAS_NETWORKX = True
except ImportError:  # pragma: no cover -- tested via the "no networkx" fallback
    nx = None  # type: ignore[assignment]
    _HAS_NETWORKX = False

logger = logging.getLogger(__name__)

if not _HAS_NETWORKX:  # pragma: no cover -- networkx ships with matchms
    logger.info(
        "networkx is not installed. GraphML export is unavailable; the "
        "dependency-free JSON node-link export remains fully operational. "
        "Install the 'graph' extra for GraphML support."
    )

__all__ = [
    "export_network_annotations",
    "export_network_graph",
    "DEFAULT_DIRECT_HIGH_CONFIDENCE_Q_VALUE",
]

#: Default q-value ceiling for the Tier-1 "direct high-confidence" band.
DEFAULT_DIRECT_HIGH_CONFIDENCE_Q_VALUE = 0.01

TIER_DIRECT_HIGH_CONFIDENCE = 1
TIER_NETWORK_SUPPORTED = 2
TIER_NETWORK_CONFLICTED = 3

_TIER_LABELS: dict[int, str] = {
    TIER_DIRECT_HIGH_CONFIDENCE: "direct_high_confidence",
    TIER_NETWORK_SUPPORTED: "network_supported",
    TIER_NETWORK_CONFLICTED: "network_conflicted_or_low_support",
}

_ANNOTATION_COLUMNS: tuple[str, ...] = tuple(PROPAGATION_OUTPUT_SCHEMA)

GraphFormat = Literal["graphml", "json"]


# ---------------------------------------------------------------------------
# Normalisation / tiering
# ---------------------------------------------------------------------------


def _annotation_schema(*, confidence_tiers: bool) -> dict[str, Any]:
    schema: dict[str, Any] = dict(PROPAGATION_OUTPUT_SCHEMA)
    if confidence_tiers:
        schema["confidence_tier"] = pl.UInt8
        schema["confidence_tier_label"] = pl.Utf8
    return schema


def _empty_annotations(*, confidence_tiers: bool) -> pl.DataFrame:
    return pl.DataFrame(schema=_annotation_schema(confidence_tiers=confidence_tiers))


def _normalize_row(row: dict[str, Any], default_source: str) -> dict[str, Any]:
    """Project an arbitrary annotation row onto the canonical output schema."""
    node_id = _clean_text(row.get("query_id"))
    is_decoy = _row_is_decoy(row)
    evidence_source = _clean_text(row.get("evidence_source")) or default_source
    candidate_id = _clean_text(row.get("candidate_id")) or _resolve_candidate_id(row)

    is_direct = row.get("is_direct_match")
    if is_direct is None:
        is_direct = evidence_source == "direct" and not is_decoy

    network_status = _clean_text(row.get("network_status"))
    if network_status is None:
        if is_decoy:
            network_status = "decoy_match"
        elif evidence_source == "network":
            network_status = "network_supported"
        else:
            network_status = "direct_match"

    neighbor_count = _finite_float(row.get("supporting_neighbor_count"))
    support_score = _finite_float(row.get("network_support_score"))

    record: dict[str, Any] = {name: None for name in _ANNOTATION_COLUMNS}
    record["query_id"] = node_id
    record["evidence_source"] = evidence_source
    record["is_direct_match"] = bool(is_direct)
    record["candidate_id"] = candidate_id
    record["candidate_inchikey"] = _clean_text(
        row.get("candidate_inchikey")
    ) or _clean_text(row.get("inchikey"))
    for name in (
        "reference_id",
        "reference_name",
        "reference_precursor_mz",
        "query_precursor_mz",
        "score",
        "matched_peaks",
        "smiles",
        "inchikey",
        "is_decoy",
        "q_value",
        "p_value",
        "annotation_tier",
        "structural_similarity",
        "mass_error_ppm",
        "score_breakdown",
    ):
        record[name] = _coerce_direct_value(name, row.get(name))
    record["network_support_score"] = (
        support_score if support_score is not None else 0.0
    )
    record["supporting_neighbor_count"] = (
        int(neighbor_count) if neighbor_count is not None and neighbor_count > 0 else 0
    )
    record["supporting_seed_ids"] = list(row.get("supporting_seed_ids") or [])
    record["network_status"] = network_status
    record["network_conflict"] = bool(row.get("network_conflict") or False)
    record["network_conflicting_candidates"] = list(
        row.get("network_conflicting_candidates") or []
    )
    network_rank = _finite_float(row.get("network_rank"))
    record["network_rank"] = int(network_rank) if network_rank is not None else None
    return record


def _dedupe_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["query_id"],
        record["candidate_id"],
        record["evidence_source"],
    )


def _assign_tier(
    record: dict[str, Any], direct_high_confidence_q_value: float
) -> tuple[Optional[int], Optional[str]]:
    """Return the (tier, label) of a normalised annotation row."""
    if record["is_decoy"]:
        return None, None
    q_value = record["q_value"]
    if (
        record["is_direct_match"]
        and q_value is not None
        and q_value <= direct_high_confidence_q_value
    ):
        return TIER_DIRECT_HIGH_CONFIDENCE, _TIER_LABELS[TIER_DIRECT_HIGH_CONFIDENCE]
    if record["network_status"] == "network_supported":
        return TIER_NETWORK_SUPPORTED, _TIER_LABELS[TIER_NETWORK_SUPPORTED]
    if (
        record["network_status"] in ("network_conflicted", "direct_match_conflict")
        or record["evidence_source"] == "network"
    ):
        return TIER_NETWORK_CONFLICTED, _TIER_LABELS[TIER_NETWORK_CONFLICTED]
    return None, None


def _prepare_rows(
    matches: pl.DataFrame,
    propagated: pl.DataFrame,
    *,
    direct_high_confidence_q_value: float,
) -> list[dict[str, Any]]:
    """Normalise, merge (propagated wins), de-duplicate, and tier annotation rows."""
    records = [_normalize_row(row, "direct") for row in matches.to_dicts()] + [
        _normalize_row(row, "network") for row in propagated.to_dicts()
    ]

    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        deduped[_dedupe_key(record)] = record  # later rows win
    merged = list(deduped.values())

    for record in merged:
        tier, label = _assign_tier(record, direct_high_confidence_q_value)
        record["confidence_tier"] = tier
        record["confidence_tier_label"] = label
    return merged


def _sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    tier = record.get("confidence_tier")
    tier_sort = tier if tier is not None else 99
    evidence_rank = 0 if record["evidence_source"] == "direct" else 1
    rank = record.get("network_rank") or 0
    score = (
        record["score"]
        if record["score"] is not None
        else record["network_support_score"]
    )
    q_value = record["q_value"] if record["q_value"] is not None else 1.0
    return (
        tier_sort,
        record["query_id"] or "",
        evidence_rank,
        rank,
        -float(score or 0.0),
        float(q_value),
        record["candidate_id"] or "",
    )


# ---------------------------------------------------------------------------
# Public API — unified tiered annotation table
# ---------------------------------------------------------------------------


def export_network_annotations(
    matches: pl.DataFrame,
    propagated: pl.DataFrame,
    *,
    confidence_tiers: bool = True,
    direct_high_confidence_q_value: float = DEFAULT_DIRECT_HIGH_CONFIDENCE_Q_VALUE,
) -> pl.DataFrame:
    """
    Merge direct spectral matches and network-propagated hypotheses.

    The two tables are normalised onto the canonical annotation schema
    (:data:`MassFlow.network.propagation.PROPAGATION_OUTPUT_SCHEMA`), combined,
    and de-duplicated on ``(query_id, candidate_id, evidence_source)`` with
    ``propagated`` rows taking precedence (so passing the full enriched output of
    :func:`~MassFlow.network.propagation.propagate_annotations` as *propagated*
    does not duplicate its direct rows).

    Original provenance columns — ``is_direct_match``, ``evidence_source``,
    ``network_support_score``, ``supporting_neighbor_count``, ``network_conflict``
    — are preserved verbatim. Direct-match ``score``/``q_value``/``p_value`` are
    never recomputed.

    Parameters
    ----------
    matches : polars.DataFrame
        Direct library search results (``SearchResult`` schema).
    propagated : polars.DataFrame
        Network-propagated hypotheses (typically the output of
        :func:`~MassFlow.network.propagation.propagate_annotations`).
    confidence_tiers : bool
        When ``True`` (default), add ``confidence_tier`` (``UInt8``) and
        ``confidence_tier_label`` (``Utf8``) columns; when ``False`` they are
        omitted.
    direct_high_confidence_q_value : float
        q-value ceiling for the Tier-1 band (default ``0.01``).

    Returns
    -------
    polars.DataFrame
        Unified, deterministically ordered consensus table.

    Raises
    ------
    TypeError
        If either input is not a :class:`polars.DataFrame`.
    """
    if not isinstance(matches, pl.DataFrame):
        raise TypeError(
            f"matches must be a polars.DataFrame, got {type(matches).__name__}."
        )
    if not isinstance(propagated, pl.DataFrame):
        raise TypeError(
            f"propagated must be a polars.DataFrame, got {type(propagated).__name__}."
        )

    records = _prepare_rows(
        matches,
        propagated,
        direct_high_confidence_q_value=direct_high_confidence_q_value,
    )
    schema = _annotation_schema(confidence_tiers=confidence_tiers)
    if not confidence_tiers:
        for record in records:
            record.pop("confidence_tier", None)
            record.pop("confidence_tier_label", None)
    records.sort(key=_sort_key)
    return pl.DataFrame(records, schema=schema)


# ---------------------------------------------------------------------------
# Public API — graph serialization
# ---------------------------------------------------------------------------


def _best_row_per_node(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Choose the most informative annotation row per node (tier, then evidence)."""
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        node_id = record["query_id"]
        if node_id is None:
            continue
        current = best.get(node_id)
        if current is None or _sort_key(record) < _sort_key(current):
            best[node_id] = record
    return best


def _node_attributes(record: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build GraphML-safe node attributes (``None`` values are omitted)."""
    if record is None:
        return {"annotation_status": "unannotated"}

    primary_score = (
        record["score"]
        if record["score"] is not None
        else record["network_support_score"]
    )
    conflicting = record["network_conflicting_candidates"] or []
    attributes: dict[str, Any] = {
        "label": record["query_id"],
        "annotation_status": record["network_status"],
        "evidence_source": record["evidence_source"],
        "reference_id": record["candidate_id"] or record["reference_id"],
        "confidence_tier": record.get("confidence_tier"),
        "primary_score": primary_score,
        "q_value": record["q_value"],
        "network_support_score": record["network_support_score"],
        "supporting_neighbor_count": int(record["supporting_neighbor_count"]),
        "network_conflict": bool(record["network_conflict"]),
        "conflict_count": len(conflicting),
        "conflicting_candidates": ",".join(conflicting) if conflicting else None,
    }
    return {key: value for key, value in attributes.items() if value is not None}


def _build_node_link(
    edges: pl.DataFrame, best_by_node: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Build a deterministic node-link mapping from validated edges + annotations."""
    node_ids = sorted(
        set(edges["source_id"].to_list()) | set(edges["target_id"].to_list())
    )
    nodes = [
        {"id": node_id, **_node_attributes(best_by_node.get(node_id))}
        for node_id in node_ids
    ]
    links = [
        {"source": source, "target": target, "weight": float(weight)}
        for source, target, weight in edges.select(
            ["source_id", "target_id", "weight"]
        ).iter_rows()
    ]
    return {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": links,
    }


def _node_link_to_networkx(node_link: dict[str, Any]) -> Any:
    """Convert a node-link mapping to a NetworkX graph (GraphML backend)."""
    graph = nx.Graph()
    for node in node_link["nodes"]:
        attributes = {key: value for key, value in node.items() if key != "id"}
        graph.add_node(node["id"], **attributes)
    for link in node_link["links"]:
        graph.add_edge(link["source"], link["target"], weight=link["weight"])
    return graph


def export_network_graph(
    edges: pl.DataFrame,
    annotations: pl.DataFrame,
    output_path: Path | str,
    format: GraphFormat = "graphml",
) -> None:
    """
    Export a molecular network with embedded node annotations.

    Node attributes carry the annotation status, reference identity, confidence
    tier and primary score (plus provenance: ``evidence_source``,
    ``network_support_score``, ``network_conflict``, ``conflict_count``). Edge
    weights are preserved. The file write is delegated to
    :func:`MassFlow.io.save_network_graph`.

    Parameters
    ----------
    edges : polars.DataFrame
        Molecular-network edges (validated by
        :func:`~MassFlow.network.propagation.validate_network_edges`).
    annotations : polars.DataFrame
        Annotation rows (e.g. the output of :func:`export_network_annotations`).
        Rows that do not carry ``evidence_source`` default to ``"direct"``.
    output_path : Path or str
        Destination file.
    format : {"graphml", "json"}
        Serialization format. ``"graphml"`` requires the optional ``networkx``
        backend; ``"json"`` is dependency-free.

    Raises
    ------
    TypeError
        If ``edges`` or ``annotations`` is not a :class:`polars.DataFrame`.
    ValueError
        If ``format`` is unsupported.
    ImportError
        If ``format="graphml"`` and ``networkx`` is unavailable.
    """
    if format not in ("graphml", "json"):
        raise ValueError(
            f"Unsupported network graph format: {format!r}. Use 'graphml' or 'json'."
        )
    if not isinstance(edges, pl.DataFrame):
        raise TypeError(
            f"edges must be a polars.DataFrame, got {type(edges).__name__}."
        )
    if not isinstance(annotations, pl.DataFrame):
        raise TypeError(
            f"annotations must be a polars.DataFrame, got {type(annotations).__name__}."
        )

    edges_df = validate_network_edges(edges)
    records = _prepare_rows(
        annotations,
        _empty_annotations(confidence_tiers=False),
        direct_high_confidence_q_value=DEFAULT_DIRECT_HIGH_CONFIDENCE_Q_VALUE,
    )
    best_by_node = _best_row_per_node(records)
    node_link = _build_node_link(edges_df, best_by_node)

    output = Path(output_path)
    if format == "json":
        io.save_network_graph(node_link, output, format="json")
        return

    if not _HAS_NETWORKX:
        raise ImportError(
            "GraphML export requires networkx. Install it with "
            "`pip install massflow[graph]` (or `uv sync --extra graph`)."
        )
    io.save_network_graph(_node_link_to_networkx(node_link), output, format="graphml")
