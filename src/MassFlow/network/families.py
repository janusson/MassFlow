"""
Deterministic connected-component (molecular-family) primitive.

The default family notion is the connected component of the graph, where nodes
are linked both by typed relationships and by feature membership (a feature is
adjacent to each of its member spectra). Components are content-addressed from
their sorted member identifiers, so the same component always has the same id
regardless of edge order.

Community detection is **not** implemented here (deferred); this module provides
only the deterministic, classical prerequisite used by
:mod:`MassFlow.network.context`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from MassFlow.network.models import (
    MolecularFamily,
    RelationshipProvenance,
    utc_now_iso,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MassFlow.network.models import MolecularGraph

__all__ = ["Component", "component_id", "connected_components", "detect_families"]


def component_id(member_ids: Sequence[str]) -> str:
    """
    Return the deterministic identifier of a component.

    Parameters
    ----------
    member_ids : sequence of str
        The component's member identifiers (nodes and/or features).

    Returns
    -------
    str
        ``nid1:component:<hex>`` — a SHA-256 digest of the sorted members.
    """
    canonical = "\n".join(sorted(str(member) for member in member_ids))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"nid1:component:{digest}"


@dataclass(frozen=True)
class Component:
    """A connected component: its deterministic id and sorted member ids."""

    component_id: str
    member_ids: tuple[str, ...]


class _UnionFind:
    """Union-find producing the smallest index as the component root."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            high, low = max(root_left, root_right), min(root_left, root_right)
            self._parent[high] = low


def connected_components(graph: "MolecularGraph") -> list[Component]:
    """
    Compute the connected components of a molecular graph.

    Adjacency comes from (a) every relationship's endpoints and (b) each
    feature's membership of its member spectra, so components span both the
    spectrum-node and feature layers.

    Parameters
    ----------
    graph : MolecularGraph
        The graph to analyse.

    Returns
    -------
    list of Component
        Components sorted by ``component_id``. Deterministic.
    """
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for identifier in [node.node_id for node in graph.nodes] + [
        feature.feature_id for feature in graph.features
    ]:
        if identifier not in seen:
            seen.add(identifier)
            ordered_ids.append(identifier)

    index = {identifier: position for position, identifier in enumerate(ordered_ids)}
    union_find = _UnionFind(len(ordered_ids))

    for relationship in graph.relationships:
        source = index.get(relationship.source_node_id)
        target = index.get(relationship.target_node_id)
        if source is not None and target is not None:
            union_find.union(source, target)

    for feature in graph.features:
        feature_index = index.get(feature.feature_id)
        if feature_index is None:
            continue
        for member in feature.spectrum_node_ids:
            member_index = index.get(member)
            if member_index is not None:
                union_find.union(feature_index, member_index)

    grouped: dict[int, list[str]] = {}
    for position, identifier in enumerate(ordered_ids):
        grouped.setdefault(union_find.find(position), []).append(identifier)

    components = [
        Component(
            component_id=component_id(members),
            member_ids=tuple(sorted(members)),
        )
        for members in grouped.values()
    ]
    components.sort(key=lambda component: component.component_id)
    return components


def detect_families(
    graph: "MolecularGraph", *, created_at: Optional[str] = None
) -> list[MolecularFamily]:
    """
    Build molecular-family records from the graph's connected components.

    Each family carries its members, the relationships internal to it, and — when
    the graph has an L5 :class:`~MassFlow.network.models.NetworkContext` for the
    component — the inherited label and its seed annotations. Families are
    deterministic and ordered by ``family_id``.

    Parameters
    ----------
    graph : MolecularGraph
        The graph to analyse.
    created_at : str or None, optional
        ISO-8601 timestamp recorded on the family provenance. Defaults to the
        current UTC time; pass a fixed value for a reproducible run.

    Returns
    -------
    list of MolecularFamily
        One family per connected component.
    """
    provenance = RelationshipProvenance(
        algorithm="connected_components",
        source_module="MassFlow.network.families",
        created_at=created_at or utc_now_iso(),
    )
    contexts_by_component = {
        context.component_id: context for context in graph.network_contexts
    }

    families: list[MolecularFamily] = []
    for component in connected_components(graph):
        members = set(component.member_ids)
        edge_ids = tuple(
            sorted(
                relationship.relationship_id
                for relationship in graph.relationships
                if relationship.source_node_id in members
                and relationship.target_node_id in members
            )
        )
        context = contexts_by_component.get(component.component_id)
        families.append(
            MolecularFamily(
                family_id=component.component_id,
                member_node_ids=component.member_ids,
                member_count=len(component.member_ids),
                edge_ids=edge_ids,
                seed_annotation_ids=(
                    tuple(context.seed_annotation_ids) if context else ()
                ),
                label=context.label if context else None,
                label_source=context.label_source if context else None,
                provenance=provenance,
            )
        )
    families.sort(key=lambda family: family.family_id)
    return families
