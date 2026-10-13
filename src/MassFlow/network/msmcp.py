"""
MSMCP — a local-first, MCP-style machine interface over a molecular graph.

This module exposes a **read-only, transport-agnostic** view of a
:class:`~MassFlow.network.models.MolecularGraph` through a small resource URIs
scheme (``graph``, ``node/{id}``, ``node/{id}/neighbors``, ``node/{id}/families``,
``node/{id}/inferences``, ``family/{id}``, ``edge/{id}``, ``families``,
``provenance``). It is dependency-free and works entirely offline.

Confidence discipline is preserved at the interface: statistical confidence is
returned only from :class:`~MassFlow.network.models.FdrAssessment` (via
:meth:`LocalGraphSource.annotations`), edges expose similarity/Δmass only, and
network-inferred annotations are returned under a distinct collection with
``status="network_inferred"``.

A networked MCP server is **deferred** (it would be an optional extra); the local
source is fully functional on its own, so MassFlow never requires internet-scale
data. A future :class:`GraphSource` implementation may federate to a repository.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from MassFlow.network.models import (
    Feature,
    GraphNode,
    MolecularFamily,
    RelationshipBase,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MassFlow.network.models import MolecularGraph

__all__ = ["GraphSource", "LocalGraphSource"]


class GraphSource(ABC):
    """Read-only, resource-oriented view over a molecular graph."""

    @abstractmethod
    def list_resources(self) -> list[str]:
        """Return the URIs this source can serve."""
        ...

    @abstractmethod
    def read(self, uri: str) -> Any:
        """Return the JSON-serializable resource at *uri*."""
        ...


class LocalGraphSource(GraphSource):
    """
    A local, in-memory, read-only graph source.

    Parameters
    ----------
    graph : MolecularGraph
        The graph document to expose.

    Examples
    --------
    >>> source = LocalGraphSource(graph)  # doctest: +SKIP
    >>> source.read("families")  # doctest: +SKIP
    """

    def __init__(self, graph: "MolecularGraph") -> None:
        self._graph = graph

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "LocalGraphSource":
        """Load a graph JSON document and wrap it (uses the ``io`` boundary)."""
        from MassFlow.io import load_molecular_graph

        return cls(load_molecular_graph(Path(path)))

    # -- resource discovery ------------------------------------------------

    def list_resources(self) -> list[str]:
        graph = self._graph
        uris = ["graph", "provenance", "families"]
        for node in graph.nodes:
            uris.extend(
                [
                    f"node/{node.node_id}",
                    f"node/{node.node_id}/neighbors",
                    f"node/{node.node_id}/families",
                    f"node/{node.node_id}/inferences",
                ]
            )
        for family in graph.families:
            uris.append(f"family/{family.family_id}")
        for relationship in graph.relationships:
            uris.append(f"edge/{relationship.relationship_id}")
        return sorted(uris)

    # -- resource reads ----------------------------------------------------

    def read(self, uri: str) -> Any:
        graph = self._graph
        if uri == "graph":
            return graph.to_dict()
        if uri == "provenance":
            return graph.provenance.model_dump(mode="json")
        if uri == "families":
            return [family.model_dump(mode="json") for family in graph.families]

        parts = uri.split("/")
        if parts[0] == "node" and len(parts) >= 2:
            return self._read_node(parts[1], parts[2:] if len(parts) > 2 else [])
        if parts[0] == "family" and len(parts) == 2:
            return self._family(parts[1]).model_dump(mode="json")
        if parts[0] == "edge" and len(parts) == 2:
            return self._edge(parts[1]).model_dump(mode="json")
        raise KeyError(f"Unknown resource: {uri!r}")

    def _read_node(self, node_id: str, rest: list[str]) -> Any:
        entity: Optional[Union[GraphNode, Feature]] = self._find_node(node_id)
        if entity is None:
            raise KeyError(f"Unknown node: {node_id!r}")
        if not rest:
            return entity.model_dump(mode="json")
        if rest == ["neighbors"]:
            return self.neighbors(node_id)
        if rest == ["families"]:
            return self.families_containing(node_id)
        if rest == ["inferences"]:
            return self.inferences_for(node_id)
        raise KeyError(f"Unknown resource: node/{node_id}/{'/'.join(rest)}")

    # -- convenience queries ----------------------------------------------

    def _find_node(self, node_id: str) -> Optional[Union[GraphNode, Feature]]:
        for node in self._graph.nodes:
            if node.node_id == node_id:
                return node
        for feature in self._graph.features:
            if feature.feature_id == node_id:
                return feature
        return None

    def _family(self, family_id: str) -> MolecularFamily:
        for family in self._graph.families:
            if family.family_id == family_id:
                return family
        raise KeyError(f"Unknown family: {family_id!r}")

    def _edge(self, edge_id: str) -> RelationshipBase:
        for relationship in self._graph.relationships:
            if relationship.relationship_id == edge_id:
                return relationship
        raise KeyError(f"Unknown edge: {edge_id!r}")

    def neighbors(self, node_id: str) -> list[dict[str, Any]]:
        """Return the incident relationships of a node (with the other endpoint)."""
        result: list[dict[str, Any]] = []
        for relationship in self._graph.relationships:
            if relationship.source_node_id == node_id:
                neighbor = relationship.target_node_id
            elif relationship.target_node_id == node_id:
                neighbor = relationship.source_node_id
            else:
                continue
            result.append(
                {
                    "relationship_id": relationship.relationship_id,
                    "relationship_type": relationship.relationship_type,
                    "neighbor_node_id": neighbor,
                    "directed": relationship.directed,
                }
            )
        result.sort(
            key=lambda item: (item["relationship_type"], item["relationship_id"])
        )
        return result

    def edges_between(
        self, source_node_id: str, target_node_id: str
    ) -> list[dict[str, Any]]:
        """Return relationships whose endpoints are the two given nodes."""
        endpoints = {source_node_id, target_node_id}
        return [
            relationship.model_dump(mode="json")
            for relationship in self._graph.relationships
            if {relationship.source_node_id, relationship.target_node_id} == endpoints
        ]

    def families_containing(self, node_id: str) -> list[dict[str, Any]]:
        """Return families that include *node_id* among their members."""
        return [
            family.model_dump(mode="json")
            for family in self._graph.families
            if node_id in family.member_node_ids
        ]

    def inferences_for(self, node_id: str) -> list[dict[str, Any]]:
        """Return network inferences targeting *node_id*."""
        return [
            inference.model_dump(mode="json")
            for inference in self._graph.inferences
            if inference.node_id == node_id
        ]

    def annotations(self, node_id: str) -> Optional[dict[str, Any]]:
        """
        Return the query-scoped FDR assessment for *node_id*, if present.

        This is the interface's **only** statistical-confidence surface; edges and
        inferences never carry confidence.
        """
        assessment = self._graph.query_fdr(node_id)
        return None if assessment is None else assessment.model_dump(mode="json")
