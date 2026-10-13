"""
Evidence-separation invariant tests for the MassFlow graph data layer.

These are the highest-value tests of the subsystem: they prove that spectral
similarity, chemical relationships and ion-identity relationships can never
carry primary statistical confidence (q-value/FDR), and that network-facing
containers cannot silently alter it.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchms import Spectrum
from pydantic import ValidationError

from MassFlow.network import (
    AnnotationEvidence,
    ChemicalRelationship,
    FdrAssessment,
    GraphNode,
    GraphProvenance,
    IonIdentityRelationship,
    MolecularGraph,
    RelationshipProvenance,
    SpectralRelationship,
    library_node_id,
    spectrum_node_id,
)

pytestmark = pytest.mark.unit

# Field names that would let an edge masquerade as a statistical-confidence
# claim. None of them may appear on any relationship or evidence model.
_FORBIDDEN_CONFIDENCE_FIELDS = {"q_value", "p_value", "fdr", "confidence"}


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
    return RelationshipProvenance(algorithm=algorithm)


def test_relationships_expose_no_statistical_confidence_field() -> None:
    for model in (
        SpectralRelationship,
        ChemicalRelationship,
        IonIdentityRelationship,
        AnnotationEvidence,
    ):
        leaks = _FORBIDDEN_CONFIDENCE_FIELDS & set(model.model_fields)
        assert not leaks, f"{model.__name__} leaks confidence fields: {leaks}"


def test_fdr_assessment_is_the_only_fdr_carrier() -> None:
    assert set(FdrAssessment.model_fields) >= {"q_value", "p_value", "calibrated"}
    # The relationship base has provenance but no FDR sub-structure.
    assert "provenance" in SpectralRelationship.model_fields
    assert "fdr" not in SpectralRelationship.model_fields
    assert "fdr_assessment" not in SpectralRelationship.model_fields


def _base_graph() -> tuple[MolecularGraph, str, str]:
    query_node = GraphNode.from_query_spectrum(_spectrum())
    library_node = GraphNode.from_library_spectrum(
        _spectrum(precursor_mz=299.999, spec_id="ref_1"), library_build_id=1
    )
    assessment = FdrAssessment(q_value=0.003, calibrated=True, library_size=5000)
    graph = MolecularGraph(
        nodes=[query_node, library_node],
        relationships=[
            SpectralRelationship.create(
                source_node_id=query_node.node_id,
                target_node_id=library_node.node_id,
                score=0.9,
                matched_peaks=8,
                algorithm="cosine",
                provenance=_provenance(),
            )
        ],
        fdr_assessments={query_node.node_id: assessment},
        provenance=GraphProvenance(),
    )
    return graph, query_node.node_id, library_node.node_id


def test_graph_construction_does_not_alter_fdr() -> None:
    graph, query_id, _ = _base_graph()
    before = FdrAssessment(q_value=0.003, calibrated=True, library_size=5000)
    assert graph.query_fdr(query_id) == before
    # Re-serializing the graph cannot change the assessment either.
    restored = MolecularGraph.from_json(graph.to_json())
    assert restored.query_fdr(query_id) == before


def test_query_fdr_returns_none_for_unknown_node() -> None:
    graph, _, _ = _base_graph()
    assert graph.query_fdr("nid1:query:" + "0" * 32) is None


def test_graph_rejects_duplicate_node_ids() -> None:
    node = GraphNode.from_query_spectrum(_spectrum())
    with pytest.raises(ValidationError):
        MolecularGraph(nodes=[node, node], provenance=GraphProvenance())


def test_graph_rejects_dangling_relationship_endpoints() -> None:
    node = GraphNode.from_query_spectrum(_spectrum())
    relationship = SpectralRelationship.create(
        source_node_id=node.node_id,
        target_node_id="nid1:library:" + "a" * 32,
        score=0.9,
        matched_peaks=8,
        algorithm="cosine",
        provenance=_provenance(),
    )
    with pytest.raises(ValidationError):
        MolecularGraph(
            nodes=[node],
            relationships=[relationship],
            provenance=GraphProvenance(),
        )


def test_graph_rejects_fdr_keyed_by_non_query_node() -> None:
    library_node = GraphNode.from_library_spectrum(
        _spectrum(spec_id="ref_1"), library_build_id=1
    )
    with pytest.raises(ValidationError):
        MolecularGraph(
            nodes=[library_node],
            fdr_assessments={
                library_node.node_id: FdrAssessment(
                    q_value=0.01, calibrated=True, library_size=10
                )
            },
            provenance=GraphProvenance(),
        )


def test_relationships_are_frozen() -> None:
    source = spectrum_node_id(_spectrum())
    relationship = ChemicalRelationship.create(
        source_node_id=source,
        target_node_id=library_node_id(1, "r"),
        delta_mass_da=-18.0,
        provenance=_provenance("neutral_loss"),
    )
    with pytest.raises(ValidationError):
        relationship.delta_mass_da = 0.0  # type: ignore[misc]


def test_graph_is_data_only_container() -> None:
    # The container must not expose analysis methods (search/cluster/analyse).
    analysis_like = {
        "search",
        "cluster",
        "cluster_components",
        "analyse",
        "analyze",
        "build",
    }
    assert analysis_like.isdisjoint(dir(MolecularGraph))
