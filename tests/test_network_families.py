"""
Stage 6 molecular-family tests.

Families are deterministic connected components with content-addressed ids,
internal edge references, and (when a context exists) an inherited label.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from matchms import Spectrum
from pydantic import ValidationError

from MassFlow.config import NetworkConfig
from MassFlow.network.build import build_spectral_graph
from MassFlow.network.context import AnnotationSeed, apply_context, contextualize
from MassFlow.network.families import component_id, detect_families
from MassFlow.network.models import (
    MolecularFamily,
    MolecularGraph,
    RelationshipProvenance,
    library_node_id,
)

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


def _graph(**overrides: object) -> MolecularGraph:
    spectra = [
        _spectrum("a", 500.0),
        _spectrum("b", 500.005),
        _spectrum("c", 700.0),
    ]
    return build_spectral_graph(spectra, _cfg(**overrides))


def test_build_attaches_deterministic_families() -> None:
    graph = _graph()
    assert graph.families
    assert [f.family_id for f in graph.families] == sorted(
        f.family_id for f in graph.families
    )
    component_ids = {family.family_id for family in detect_families(graph)}
    assert {f.family_id for f in graph.families} == component_ids


def test_family_members_and_edges_are_consistent() -> None:
    graph = _graph()
    edge_ids = {relationship.relationship_id for relationship in graph.relationships}
    for family in graph.families:
        assert family.member_count == len(family.member_node_ids)
        assert set(family.edge_ids) <= edge_ids


def test_family_id_is_the_component_id_and_matches_members() -> None:
    graph = _graph()
    for family in graph.families:
        assert family.family_id == component_id(family.member_node_ids)


def test_families_disabled() -> None:
    graph = _graph(build_families=False)
    assert graph.families == []


def test_family_inherits_label_from_context() -> None:
    base = _graph()
    seeds = [
        AnnotationSeed(node_id=base.nodes[0].node_id, label="Caffeine", q_value=0.001)
    ]
    result = contextualize(base, seeds, _cfg())
    analysed = apply_context(base, result.contexts, result.inferences)

    families = detect_families(analysed)
    labelled = [family for family in families if family.label is not None]
    assert labelled
    assert labelled[0].label == "Caffeine"
    assert labelled[0].label_source == "seed_annotation"
    assert labelled[0].seed_annotation_ids
    # Unlabelled families carry no label and no seeds.
    unlabelled = [family for family in families if family.label is None]
    assert all(family.seed_annotation_ids == () for family in unlabelled)


def test_detect_families_is_deterministic() -> None:
    graph = _graph()
    first = detect_families(graph)
    second = detect_families(graph)
    assert [f.family_id for f in first] == [f.family_id for f in second]
    assert [f.member_node_ids for f in first] == [f.member_node_ids for f in second]


def test_family_jsonl_export_round_trips() -> None:
    graph = _graph()
    lines = graph.to_family_jsonl().strip().splitlines()
    assert len(lines) == len(graph.families)
    parsed = [json.loads(line) for line in lines]
    assert {record["family_id"] for record in parsed} == {
        family.family_id for family in graph.families
    }


def test_graph_rejects_family_with_unknown_member() -> None:
    base = _graph()
    bad = MolecularFamily(
        family_id=component_id([base.nodes[0].node_id]),
        member_node_ids=(library_node_id(9, "absent"),),
        member_count=1,
        provenance=RelationshipProvenance(algorithm="connected_components"),
    )
    with pytest.raises(ValidationError):
        MolecularGraph(
            nodes=base.nodes,
            relationships=base.relationships,
            families=[bad],
            provenance=base.provenance,
        )


def test_graph_rejects_family_with_unknown_edge() -> None:
    base = _graph()
    bad = MolecularFamily(
        family_id=component_id([base.nodes[0].node_id]),
        member_node_ids=(base.nodes[0].node_id,),
        member_count=1,
        edge_ids=("rid1:spectral:" + "0" * 32,),
        provenance=RelationshipProvenance(algorithm="connected_components"),
    )
    with pytest.raises(ValidationError):
        MolecularGraph(
            nodes=base.nodes,
            relationships=base.relationships,
            families=[bad],
            provenance=base.provenance,
        )
