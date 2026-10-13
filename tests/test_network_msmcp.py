"""
MSMCP (local read-only interface) tests.

The local graph source must answer node/edge/family/provenance queries entirely
offline, and must expose statistical confidence only from ``FdrAssessment``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import io
from MassFlow.config import NetworkConfig
from MassFlow.network.build import build_spectral_graph
from MassFlow.network.context import AnnotationSeed, apply_context, contextualize
from MassFlow.network.models import FdrAssessment, MolecularGraph
from MassFlow.network.msmcp import GraphSource, LocalGraphSource

pytestmark = pytest.mark.unit

_MZ = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0])
_INT = np.array([1.0, 0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])


def _spectrum(
    spec_id: str, precursor_mz: float, retention_time: float = 100.0
) -> Spectrum:
    return Spectrum(
        mz=_MZ.copy(),
        intensities=_INT.copy(),
        metadata={
            "id": spec_id,
            "precursor_mz": precursor_mz,
            "retention_time": retention_time,
            "adduct": "[M+H]+",
            "charge": 1,
            "ionmode": "positive",
        },
    )


def _cfg(**overrides: object) -> NetworkConfig:
    payload: dict[str, object] = {
        "enabled": True,
        "min_score": 0.7,
        "min_matched_peaks": 3,
        "top_k_per_node": None,
        "build_features": False,
    }
    payload.update(overrides)
    return NetworkConfig(**payload)  # type: ignore[arg-type]


def _graph() -> MolecularGraph:
    spectra = [
        _spectrum("a", 500.0),
        _spectrum("b", 500.005),
        _spectrum("c", 700.0),
    ]
    return build_spectral_graph(spectra, _cfg())


def test_local_graph_source_is_a_graph_source() -> None:
    assert isinstance(LocalGraphSource(_graph()), GraphSource)


def test_list_resources_covers_graphs_nodes_edges_and_families() -> None:
    graph = _graph()
    uris = LocalGraphSource(graph).list_resources()
    assert "graph" in uris
    assert "provenance" in uris
    assert "families" in uris
    assert all(f"node/{node.node_id}" in uris for node in graph.nodes)
    assert all(f"edge/{rel.relationship_id}" in uris for rel in graph.relationships)
    assert all(f"family/{family.family_id}" in uris for family in graph.families)
    assert uris == sorted(uris)


def test_read_graph_and_provenance() -> None:
    graph = _graph()
    source = LocalGraphSource(graph)
    assert source.read("graph")["nodes"] == graph.to_dict()["nodes"]
    assert source.read("provenance")["builder"] == graph.provenance.builder


def test_read_node_and_its_collections() -> None:
    graph = _graph()
    source = LocalGraphSource(graph)
    node_id = graph.nodes[0].node_id
    assert source.read(f"node/{node_id}")["node_id"] == node_id
    neighbors = source.read(f"node/{node_id}/neighbors")
    assert neighbors and neighbors[0]["neighbor_node_id"] in {
        node.node_id for node in graph.nodes
    }
    assert len(source.read(f"node/{node_id}/families")) == 1
    assert source.read(f"node/{node_id}/inferences") == []


def test_read_family_and_edge() -> None:
    graph = _graph()
    source = LocalGraphSource(graph)
    family = graph.families[0]
    assert source.read(f"family/{family.family_id}")["family_id"] == family.family_id
    relationship = graph.relationships[0]
    assert (
        source.read(f"edge/{relationship.relationship_id}")["relationship_id"]
        == relationship.relationship_id
    )


def test_invalid_resources_raise_key_error() -> None:
    source = LocalGraphSource(_graph())
    with pytest.raises(KeyError):
        source.read("bogus")
    with pytest.raises(KeyError):
        source.read("node/nid1:query:" + "0" * 32)
    with pytest.raises(KeyError):
        source.read("family/nid1:component:" + "0" * 32)


def test_annotations_expose_confidence_only_from_fdr() -> None:
    base = _graph()
    source = LocalGraphSource(base)
    assert source.annotations(base.nodes[0].node_id) is None

    query_id = base.nodes[0].node_id
    graph = MolecularGraph(
        schema_version=base.schema_version,
        nodes=base.nodes,
        relationships=base.relationships,
        features=base.features,
        fdr_assessments={
            query_id: FdrAssessment(q_value=0.004, calibrated=True, library_size=42)
        },
        families=base.families,
        provenance=base.provenance,
    )
    assessment = LocalGraphSource(graph).annotations(query_id)
    assert assessment is not None
    assert assessment["q_value"] == 0.004
    assert assessment["competition_unit"] == "query"


def test_inferences_are_returned_as_a_distinct_collection() -> None:
    base = _graph()
    seeds = [
        AnnotationSeed(node_id=base.nodes[0].node_id, label="Caffeine", q_value=0.001)
    ]
    result = contextualize(base, seeds, _cfg())
    graph = apply_context(base, result.contexts, result.inferences)

    inferred_node = result.inferences[0].node_id
    payload = LocalGraphSource(graph).inferences_for(inferred_node)
    assert len(payload) == 1
    assert payload[0]["status"] == "network_inferred"
    assert "q_value" not in payload[0]


def test_from_file_loads_a_saved_graph(tmp_path: Path) -> None:
    graph = _graph()
    path = tmp_path / "graph.json"
    io.save_molecular_graph(graph, path)

    source = LocalGraphSource.from_file(path)
    assert source.read("graph")["nodes"] == graph.to_dict()["nodes"]
