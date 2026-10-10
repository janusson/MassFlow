"""
Serialization/deserialization tests for the MassFlow graph data layer.

Proves exact float64 round-tripping, deterministic JSON, missing-value handling
(``None`` rather than ``NaN``), schema-version guarding, discriminated-union
rehydration, and JSONL edge streaming.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from matchms import Spectrum
from pydantic import TypeAdapter

from MassFlow.network import (
    AnnotationEvidence,
    ChemicalRelationship,
    FdrAssessment,
    GraphNode,
    GraphProvenance,
    IonIdentityRelationship,
    MolecularGraph,
    NeutralLoss,
    Relationship,
    RelationshipProvenance,
    SpectralRelationship,
)

pytestmark = pytest.mark.unit


def _spectrum(precursor_mz: float = 300.0, spec_id: str = "q1") -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([0.5, 1.0], dtype=np.float64),
        metadata={
            "id": spec_id,
            "precursor_mz": precursor_mz,
            "retention_time": 12.5,
            "adduct": "[M+H]+",
            "charge": 1,
        },
    )


def _provenance(algorithm: str = "cosine") -> RelationshipProvenance:
    return RelationshipProvenance(
        algorithm=algorithm,
        algorithm_version="1.0",
        parameters={"ms2_tolerance": 0.02},
        library_build_id=3,
        config_digest_sha256="deadbeef",
    )


def _graph() -> MolecularGraph:
    query_node = GraphNode.from_query_spectrum(_spectrum())
    library_node = GraphNode.from_library_spectrum(
        _spectrum(precursor_mz=299.9995, spec_id="ref_1"), library_build_id=3
    )
    spectral = SpectralRelationship.create(
        source_node_id=query_node.node_id,
        target_node_id=library_node.node_id,
        score=0.987654321012345,
        matched_peaks=11,
        algorithm="modified_cosine",
        provenance=_provenance("modified_cosine"),
        ms2_tolerance=0.02,
        library_build_id=3,
    )
    chemical = ChemicalRelationship.create(
        source_node_id=query_node.node_id,
        target_node_id=library_node.node_id,
        delta_mass_da=-18.01056468403,
        provenance=_provenance("neutral_loss"),
        neutral_loss=NeutralLoss(label="H2O", formula="H2O", mass_da=18.01056468403),
    )
    ion_identity = IonIdentityRelationship.create(
        source_node_id=query_node.node_id,
        target_node_id=library_node.node_id,
        relationship_kind="adduct",
        delta_mass_da=21.981943,
        provenance=_provenance("adduct_offset"),
        adduct_source="[M+H]+",
        adduct_target="[M+Na]+",
        charge_source=1,
        charge_target=1,
        expected_delta_mass_da=21.981943,
    )
    evidence = AnnotationEvidence.from_search_result(
        {"score": 0.987654321012345, "matched_peaks": 11},
        query_node_id=query_node.node_id,
        reference_node_id=library_node.node_id,
        algorithm="modified_cosine",
    )
    graph = MolecularGraph(
        nodes=[query_node, library_node],
        relationships=[spectral, chemical, ion_identity],
        fdr_assessments={
            query_node.node_id: FdrAssessment(
                q_value=0.004321, calibrated=True, library_size=12345
            )
        },
        provenance=GraphProvenance(library_build_id=3),
    )
    # Evidence is not part of the graph document, but must itself round-trip.
    assert evidence.score == spectral.score
    return graph


def test_dict_round_trip_is_lossless() -> None:
    graph = _graph()
    restored = MolecularGraph.from_dict(graph.to_dict())
    assert restored.to_dict() == graph.to_dict()


def test_json_round_trip_preserves_float64() -> None:
    graph = _graph()
    restored = MolecularGraph.from_json(graph.to_json())
    spectral = next(
        r for r in restored.relationships if r.relationship_type == "spectral"
    )
    assert spectral.score == 0.987654321012345
    chemical = next(
        r for r in restored.relationships if r.relationship_type == "chemical"
    )
    assert chemical.delta_mass_da == -18.01056468403
    fdr_key = next(iter(restored.fdr_assessments))
    assessment = restored.query_fdr(fdr_key)
    assert assessment is not None
    assert assessment.q_value == 0.004321


def test_union_rehydrates_to_concrete_types() -> None:
    graph = _graph()
    restored = MolecularGraph.from_json(graph.to_json())
    kinds = sorted(r.relationship_type for r in restored.relationships)
    assert kinds == ["chemical", "ion_identity", "spectral"]
    assert isinstance(restored.relationships[0], SpectralRelationship)


def test_missing_values_are_null_and_never_nan() -> None:
    query_node = GraphNode.from_query_spectrum(_spectrum())
    graph = MolecularGraph(nodes=[query_node], provenance=GraphProvenance())
    document = graph.to_dict()
    assert document["nodes"][0]["retention_time_seconds"] == 12.5
    assert document["fdr_assessments"] == {}
    text = graph.to_json()
    assert "NaN" not in text
    assert "Infinity" not in text


def test_json_is_deterministic() -> None:
    graph = _graph()
    assert graph.to_json() == graph.to_json()


def test_schema_version_is_guarded() -> None:
    document = _graph().to_dict()
    with pytest.raises(ValueError):
        MolecularGraph.from_dict({**document, "schema_version": "2"})
    with pytest.raises(ValueError):
        MolecularGraph.from_dict({"nodes": [], "provenance": {}})


def test_relationship_jsonl_streams_one_edge_per_line() -> None:
    graph = _graph()
    lines = graph.to_relationship_jsonl().splitlines()
    assert len(lines) == len(graph.relationships)
    adapter: TypeAdapter[Relationship] = TypeAdapter(Relationship)
    rehydrated = [adapter.validate_json(line) for line in lines]
    assert {r.relationship_type for r in rehydrated} == {
        "spectral",
        "chemical",
        "ion_identity",
    }
    # Each line is standalone JSON.
    assert all(json.loads(line)["relationship_id"] for line in lines)


def test_provenance_defaults_are_populated() -> None:
    provenance = RelationshipProvenance(algorithm="cosine")
    assert provenance.massflow_version
    assert provenance.created_at
    assert provenance.source_module == "MassFlow.network"
