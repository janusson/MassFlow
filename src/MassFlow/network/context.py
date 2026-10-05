"""
Stage 5 annotation contextualization (non-circular).

High-confidence, directly calibrated library annotations ("seeds") give a family
context to the poorly annotated nodes of the same connected component. This stage
is deliberately **one-directional and non-circular**:

* It reads seeds and the graph and produces only **L5** objects
  (:class:`~MassFlow.network.models.NetworkContext`,
  :class:`~MassFlow.network.models.AnnotationInference`).
* It never reads, writes or re-derives a q-value or p-value: the graph's
  :class:`~MassFlow.network.models.FdrAssessment` set is untouched, and no L5
  object carries statistical confidence.
* Inference uses **only** direct seed annotations (never other inferences), and is
  confined to components that contain at least one direct, calibrated,
  high-confidence seed.
* Every inference records the exact seeds and incident edges that justify it.

The seeds are supplied by the caller (in a future phase they are projected from
the annotation run's ``FileExecutionResult`` rows); this module never runs the
annotation engine.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

from pydantic import ConfigDict, Field

from MassFlow.network.families import connected_components
from MassFlow.network.models import (
    AnnotationInference,
    MolecularGraph,
    NetworkContext,
    NetworkModel,
    NodeIdField,
    RelationshipProvenance,
    spectrum_node_id,
    utc_now_iso,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MassFlow.config import NetworkConfig

logger = logging.getLogger(__name__)

__all__ = [
    "AnnotationSeed",
    "seed_annotation_id",
    "seeds_from_results",
    "Contextualization",
    "contextualize",
    "apply_context",
]


class AnnotationSeed(NetworkModel):
    """
    A direct, calibrated, library annotation of a node.

    This is an **input** record to contextualization (not a graph element). It
    describes a confident annotation the annotation engine already produced; it
    is the *only* thing from which network context and inferences may be derived.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: NodeIdField = Field(..., description="Annotated (query) node identifier.")
    label: str = Field(..., description="Compound/family label from the library hit.")
    q_value: float = Field(..., ge=0.0, le=1.0, description="Query-scoped q-value.")
    calibrated: bool = Field(
        True, description="False when the run's decoy null was empty (not a seed)."
    )
    tier: Optional[str] = Field(
        None, description="Annotation tier label, when assigned."
    )


def seed_annotation_id(node_id: str, label: str) -> str:
    """Return a deterministic identifier for a seed annotation."""
    digest = hashlib.sha256(f"{node_id}\n{label}".encode("utf-8")).hexdigest()[:32]
    return f"ann1:{digest}"


def _optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _row_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row_sort_key(row: Any) -> tuple[float, float, str]:
    q_value = _row_float(row.get("q_value"))
    return (
        1.0 if q_value is None else q_value,
        -(_row_float(row.get("score")) or 0.0),
        str(row.get("reference_id")),
    )


def seeds_from_results(results: Sequence[object]) -> list[AnnotationSeed]:
    """
    Derive seed annotations from annotation-run results.

    For each ``FileExecutionResult`` (duck-typed: it exposes ``query_spectra``,
    ``results`` and ``degraded_mode_flags``) the **best** hit per query spectrum
    (lowest q-value, then highest score) becomes one seed whose ``node_id`` is the
    query spectrum's content-addressed node id. A run flagged ``fdr_uncalibrated``
    yields only ``calibrated=False`` seeds (which :func:`contextualize` ignores), so
    uncalibrated q-values can never seed network context.

    Parameters
    ----------
    results : sequence
        The ``FileExecutionResult`` objects returned by the annotation pipeline.

    Returns
    -------
    list of AnnotationSeed
        Deterministically ordered seeds (by node id, then label).
    """
    seeds: list[AnnotationSeed] = []
    for result in results:
        calibrated = "fdr_uncalibrated" not in getattr(
            result, "degraded_mode_flags", []
        )
        node_by_query_id: dict[str, str] = {}
        for spectrum in getattr(result, "query_spectra", []):
            node_by_query_id[str(spectrum.get("id"))] = str(spectrum_node_id(spectrum))

        rows_by_query: dict[str, list[Any]] = {}
        for row in getattr(result, "results", []):
            query_id = row.get("query_id")
            if query_id is None:
                continue
            rows_by_query.setdefault(str(query_id), []).append(row)

        for query_id, rows in rows_by_query.items():
            node_id = node_by_query_id.get(query_id)
            if node_id is None:
                continue
            rows.sort(key=_row_sort_key)
            best = rows[0]
            label = _optional_text(best.get("reference_name")) or _optional_text(
                best.get("reference_id")
            )
            q_value = _row_float(best.get("q_value"))
            if label is None or q_value is None:
                continue
            seeds.append(
                AnnotationSeed(
                    node_id=node_id,
                    label=label,
                    q_value=min(max(q_value, 0.0), 1.0),
                    calibrated=calibrated,
                    tier=_optional_text(best.get("annotation_tier")),
                )
            )

    seeds.sort(key=lambda seed: (seed.node_id, seed.label))
    unique: list[AnnotationSeed] = []
    seen: set[tuple[str, str]] = set()
    for seed in seeds:
        key = (seed.node_id, seed.label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(seed)
    return unique


@dataclass(frozen=True)
class Contextualization:
    """The L5 output of contextualization: contexts and inferences."""

    contexts: tuple[NetworkContext, ...]
    inferences: tuple[AnnotationInference, ...]


def _is_usable_seed(seed: AnnotationSeed, threshold: float) -> bool:
    return seed.calibrated and seed.q_value <= threshold


def contextualize(
    graph: MolecularGraph,
    seeds: Sequence[AnnotationSeed],
    cfg: "NetworkConfig",
    *,
    config_digest_sha256: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Contextualization:
    """
    Derive family contexts and network-inferred annotations from seed hits.

    Parameters
    ----------
    graph : MolecularGraph
        The network to contextualize.
    seeds : sequence of AnnotationSeed
        Direct library annotations. Only calibrated seeds with
        ``q_value <= cfg.context_seed_q_threshold`` are used.
    cfg : NetworkConfig
        Provides ``build_context`` (the on/off switch) and
        ``context_seed_q_threshold``.
    config_digest_sha256 : str or None, optional
        Digest of the effective configuration, recorded in provenance.
    created_at : str or None, optional
        ISO-8601 timestamp recorded on the context/inference provenance. Defaults
        to the current UTC time; pass a fixed value for a reproducible run.

    Returns
    -------
    Contextualization
        Deterministically ordered contexts and inferences. Empty when
        ``cfg.build_context`` is false or no usable seed exists.
    """
    if not cfg.build_context or not seeds:
        return Contextualization((), ())

    threshold = cfg.context_seed_q_threshold
    usable = [seed for seed in seeds if _is_usable_seed(seed, threshold)]
    if not usable:
        return Contextualization((), ())

    seeds_by_node: dict[str, list[AnnotationSeed]] = {}
    for seed in sorted(usable, key=lambda s: (s.node_id, s.label)):
        seeds_by_node.setdefault(seed.node_id, []).append(seed)

    query_node_ids = {
        node.node_id for node in graph.nodes if node.node_kind == "query_spectrum"
    }
    edges_by_node: dict[str, list[str]] = {}
    for relationship in graph.relationships:
        edges_by_node.setdefault(relationship.source_node_id, []).append(
            relationship.relationship_id
        )
        edges_by_node.setdefault(relationship.target_node_id, []).append(
            relationship.relationship_id
        )

    provenance = RelationshipProvenance(
        algorithm="family_context",
        parameters={"context_seed_q_threshold": threshold},
        config_digest_sha256=config_digest_sha256,
        source_module="MassFlow.network.context",
        created_at=created_at or utc_now_iso(),
    )

    contexts: list[NetworkContext] = []
    inferences: list[AnnotationInference] = []

    for component in connected_components(graph):
        component_seeds = [
            seed
            for node_id in component.member_ids
            for seed in seeds_by_node.get(node_id, [])
        ]
        if not component_seeds:
            # No direct, calibrated seed in this component: no context, no inference.
            continue

        label_counts: dict[str, int] = {}
        for seed in component_seeds:
            label_counts[seed.label] = label_counts.get(seed.label, 0) + 1
        label = min(
            label_counts, key=lambda candidate: (-label_counts[candidate], candidate)
        )

        seed_ids = tuple(
            dict.fromkeys(
                seed_annotation_id(seed.node_id, seed.label) for seed in component_seeds
            )
        )

        contexts.append(
            NetworkContext(
                component_id=component.component_id,
                member_node_ids=component.member_ids,
                member_count=len(component.member_ids),
                label=label,
                seed_annotation_ids=seed_ids,
                provenance=provenance,
            )
        )

        seeded_nodes = {seed.node_id for seed in component_seeds}
        for node_id in component.member_ids:
            if node_id not in query_node_ids or node_id in seeded_nodes:
                continue
            inferences.append(
                AnnotationInference(
                    node_id=node_id,
                    label=label,
                    component_id=component.component_id,
                    supporting_annotation_ids=seed_ids,
                    inferred_from_edge_ids=tuple(
                        sorted(set(edges_by_node.get(node_id, [])))
                    ),
                    provenance=provenance,
                )
            )

    contexts.sort(key=lambda context: context.component_id)
    inferences.sort(key=lambda inference: inference.node_id)
    logger.debug(
        "Contextualization: %d usable seed(s), %d context(s), %d inference(s).",
        len(usable),
        len(contexts),
        len(inferences),
    )
    return Contextualization(tuple(contexts), tuple(inferences))


def apply_context(
    graph: MolecularGraph,
    contexts: Sequence[NetworkContext],
    inferences: Sequence[AnnotationInference],
) -> MolecularGraph:
    """
    Return a copy of *graph* carrying the given L5 contexts and inferences.

    The graph is rebuilt (and therefore re-validated) rather than mutated, so
    dangling references are rejected instead of silently persisted. The original
    graph — including its ``fdr_assessments`` — is returned untouched.

    Parameters
    ----------
    graph : MolecularGraph
        The graph to attach context to.
    contexts : sequence of NetworkContext
        Contexts produced by :func:`contextualize`.
    inferences : sequence of AnnotationInference
        Inferences produced by :func:`contextualize`.

    Returns
    -------
    MolecularGraph
        A new graph with the L5 fields populated.
    """
    return MolecularGraph(
        schema_version=graph.schema_version,
        nodes=graph.nodes,
        features=graph.features,
        relationships=graph.relationships,
        fdr_assessments=graph.fdr_assessments,
        network_contexts=list(contexts),
        inferences=list(inferences),
        provenance=graph.provenance,
    )
