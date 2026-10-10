"""
Stage 5 contextualization tests (non-circular, confidence-preserving).

These are the most consequential tests of the subsystem: they prove that network
context and network-inferred annotations can never carry or alter primary
statistical confidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matchms import Spectrum
from pydantic import ValidationError

from MassFlow.config import NetworkConfig
from MassFlow.network.build import build_spectral_graph
from MassFlow.network.context import (
    AnnotationSeed,
    apply_context,
    contextualize,
    seed_annotation_id,
    seeds_from_results,
)
from MassFlow.network.families import component_id, connected_components
from MassFlow.network.models import (
    AnnotationInference,
    FdrAssessment,
    MolecularGraph,
    NetworkContext,
    RelationshipProvenance,
    library_node_id,
    spectrum_node_id,
)
from MassFlow.workflow import FileExecutionResult

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
    # a and b are close (one edge); c is isolated.
    spectra = [
        _spectrum("a", 500.0),
        _spectrum("b", 500.005),
        _spectrum("c", 700.0),
    ]
    return build_spectral_graph(spectra, _cfg(**overrides))


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


def test_component_id_is_order_independent() -> None:
    assert component_id(["b", "a"]) == component_id(["a", "b"])
    assert component_id(["a"]).startswith("nid1:component:")


def test_connected_components_groups_edges_and_isolates_singletons() -> None:
    graph = _graph()
    components = connected_components(graph)
    sizes = sorted(len(component.member_ids) for component in components)
    assert sizes == [1, 2]
    assert components == sorted(components, key=lambda c: c.component_id)


def test_feature_membership_joins_components() -> None:
    graph = _graph(build_features=True)
    components = connected_components(graph)
    # a + b + their shared feature form one component.
    assert any(len(component.member_ids) == 3 for component in components)
    feature_ids = {feature.feature_id for feature in graph.features}
    joined = next(c for c in components if len(c.member_ids) == 3)
    assert feature_ids & set(joined.member_ids)


# ---------------------------------------------------------------------------
# Contextualization
# ---------------------------------------------------------------------------


def _seeds_for(
    graph: MolecularGraph, entries: list[tuple[int, str, float]]
) -> list[AnnotationSeed]:
    return [
        AnnotationSeed(node_id=graph.nodes[index].node_id, label=label, q_value=q)
        for index, label, q in entries
    ]


def test_seeded_component_yields_context_and_inference() -> None:
    graph = _graph()
    cfg = _cfg()
    seeds = _seeds_for(graph, [(0, "Caffeine", 0.001)])

    result = contextualize(graph, seeds, cfg)

    assert len(result.contexts) == 1
    assert len(result.inferences) == 1
    context = result.contexts[0]
    assert context.label == "Caffeine"
    assert context.member_count == 2
    assert context.label_source == "seed_annotation"

    inference = result.inferences[0]
    assert inference.node_id == graph.nodes[1].node_id  # the unseeded member
    assert inference.label == "Caffeine"
    assert inference.status == "network_inferred"
    assert inference.component_id == context.component_id
    assert inference.supporting_annotation_ids == (
        seed_annotation_id(graph.nodes[0].node_id, "Caffeine"),
    )


def test_unseeded_component_yields_nothing() -> None:
    graph = _graph()
    # Seed the isolated node c; its component has no unseeded member and the a/b
    # component has no seed, so nothing is produced for either beyond c's context.
    result = contextualize(graph, _seeds_for(graph, [(2, "X", 0.001)]), _cfg())
    assert len(result.contexts) == 1
    assert result.contexts[0].member_count == 1
    assert result.inferences == ()


def test_low_confidence_seed_is_not_used() -> None:
    graph = _graph()
    seeds = _seeds_for(graph, [(0, "Caffeine", 0.5)])
    assert (
        contextualize(graph, seeds, _cfg(context_seed_q_threshold=0.01)).contexts == ()
    )


def test_uncalibrated_seed_is_not_used() -> None:
    graph = _graph()
    seed = AnnotationSeed(
        node_id=graph.nodes[0].node_id,
        label="Caffeine",
        q_value=0.001,
        calibrated=False,
    )
    assert contextualize(graph, [seed], _cfg()).contexts == ()


def test_seeded_nodes_receive_no_inference() -> None:
    graph = _graph()
    seeds = _seeds_for(graph, [(0, "Caffeine", 0.001), (1, "Caffeine", 0.001)])
    result = contextualize(graph, seeds, _cfg())
    assert len(result.contexts) == 1
    assert result.inferences == ()


def test_majority_label_wins_and_ties_break_lexicographically() -> None:
    graph = _graph()
    majority = contextualize(
        graph,
        _seeds_for(graph, [(0, "X", 0.001), (0, "Y", 0.001), (1, "X", 0.001)]),
        _cfg(),
    )
    assert majority.contexts[0].label == "X"

    tie = contextualize(
        graph, _seeds_for(graph, [(0, "Y", 0.001), (1, "X", 0.001)]), _cfg()
    )
    assert tie.contexts[0].label == "X"


def test_seed_for_unknown_node_is_ignored() -> None:
    graph = _graph()
    seed = AnnotationSeed(
        node_id=library_node_id(1, "absent"), label="X", q_value=0.001
    )
    assert contextualize(graph, [seed], _cfg()).contexts == ()


def test_contextualize_is_order_independent_and_deterministic() -> None:
    graph = _graph()
    seeds = _seeds_for(graph, [(0, "X", 0.001), (1, "X", 0.001)])
    forward = contextualize(graph, seeds, _cfg())
    reverse = contextualize(graph, list(reversed(seeds)), _cfg())
    assert [c.component_id for c in forward.contexts] == [
        c.component_id for c in reverse.contexts
    ]
    assert [i.node_id for i in forward.inferences] == [
        i.node_id for i in reverse.inferences
    ]


def test_disabled_context_returns_nothing() -> None:
    graph = _graph()
    seeds = _seeds_for(graph, [(0, "Caffeine", 0.001)])
    assert contextualize(graph, seeds, _cfg(build_context=False)).contexts == ()


def test_inference_records_incident_edge() -> None:
    graph = _graph()
    result = contextualize(graph, _seeds_for(graph, [(0, "Caffeine", 0.001)]), _cfg())
    edge_ids = {relationship.relationship_id for relationship in graph.relationships}
    assert set(result.inferences[0].inferred_from_edge_ids) <= edge_ids
    assert result.inferences[0].inferred_from_edge_ids


# ---------------------------------------------------------------------------
# Confidence-governance invariants
# ---------------------------------------------------------------------------


def test_l5_types_carry_no_statistical_confidence() -> None:
    for model in (NetworkContext, AnnotationInference):
        for field in ("q_value", "p_value", "fdr", "confidence", "score"):
            assert field not in model.model_fields


def test_contextualization_does_not_alter_fdr() -> None:
    base = _graph()
    query_id = base.nodes[0].node_id
    graph = MolecularGraph(
        schema_version=base.schema_version,
        nodes=base.nodes,
        features=base.features,
        relationships=base.relationships,
        fdr_assessments={
            query_id: FdrAssessment(q_value=0.003, calibrated=True, library_size=10)
        },
        provenance=base.provenance,
    )

    result = contextualize(graph, _seeds_for(graph, [(0, "Caffeine", 0.001)]), _cfg())
    updated = apply_context(graph, result.contexts, result.inferences)

    assert updated.fdr_assessments == graph.fdr_assessments
    final_assessment = updated.query_fdr(query_id)
    assert final_assessment is not None
    assert final_assessment.q_value == 0.003
    for inference in updated.inferences:
        assert not hasattr(inference, "q_value")


def test_apply_context_round_trips_and_revalidates() -> None:
    graph = _graph()
    result = contextualize(graph, _seeds_for(graph, [(0, "Caffeine", 0.001)]), _cfg())
    updated = apply_context(graph, result.contexts, result.inferences)

    restored = MolecularGraph.from_json(updated.to_json())
    assert restored.to_dict() == updated.to_dict()
    assert len(restored.network_contexts) == 1
    assert len(restored.inferences) == 1


def test_graph_rejects_context_with_unknown_member() -> None:
    graph = _graph()
    context = NetworkContext(
        component_id=component_id([graph.nodes[0].node_id]),
        member_node_ids=(library_node_id(9, "absent"),),
        member_count=1,
        label="X",
        seed_annotation_ids=("ann1:" + "0" * 32,),
        provenance=RelationshipProvenance(algorithm="family_context"),
    )
    with pytest.raises(ValidationError):
        apply_context(graph, [context], [])


def test_graph_rejects_inference_on_non_query_node() -> None:
    graph = _graph()
    result = contextualize(graph, _seeds_for(graph, [(0, "Caffeine", 0.001)]), _cfg())
    context = result.contexts[0]
    bad = AnnotationInference(
        node_id=library_node_id(1, "not-a-query-node"),
        label="X",
        component_id=context.component_id,
        supporting_annotation_ids=context.seed_annotation_ids,
        provenance=RelationshipProvenance(algorithm="family_context"),
    )
    with pytest.raises(ValidationError):
        apply_context(graph, [context], [bad])


def test_graph_rejects_inference_with_unknown_context() -> None:
    graph = _graph()
    bad = AnnotationInference(
        node_id=graph.nodes[1].node_id,
        label="X",
        component_id=component_id([library_node_id(2, "elsewhere")]),
        supporting_annotation_ids=("ann1:" + "0" * 32,),
        provenance=RelationshipProvenance(algorithm="family_context"),
    )
    with pytest.raises(ValidationError):
        apply_context(graph, [], [bad])


# ---------------------------------------------------------------------------
# Seeds from the annotation run
# ---------------------------------------------------------------------------


def _result(
    spectra: list[Spectrum],
    rows: list[Any],
    *,
    degraded: tuple[str, ...] = (),
) -> FileExecutionResult:
    return FileExecutionResult(
        status="success",
        input_path=Path("query.mgf"),
        query_spectra=list(spectra),
        results=list(rows),
        degraded_mode_flags=list(degraded),
    )


def test_seeds_from_results_picks_best_hit_per_query() -> None:
    spectra = [_spectrum("a", 500.0), _spectrum("b", 500.005)]
    rows = [
        {
            "query_id": "a",
            "reference_id": "r2",
            "reference_name": "Theobromine",
            "q_value": 0.02,
            "score": 0.80,
        },
        {
            "query_id": "a",
            "reference_id": "r1",
            "reference_name": "Caffeine",
            "q_value": 0.001,
            "score": 0.91,
        },
    ]
    seeds = seeds_from_results([_result(spectra, rows)])
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.node_id == str(spectrum_node_id(spectra[0]))
    assert seed.label == "Caffeine"  # lowest q-value wins
    assert seed.q_value == pytest.approx(0.001)
    assert seed.calibrated is True


def test_seeds_from_results_marks_uncalibrated_runs() -> None:
    spectra = [_spectrum("a", 500.0)]
    rows = [
        {
            "query_id": "a",
            "reference_id": "r1",
            "reference_name": "Caffeine",
            "q_value": 0.001,
            "score": 0.9,
        }
    ]
    seeds = seeds_from_results([_result(spectra, rows, degraded=("fdr_uncalibrated",))])
    assert seeds and seeds[0].calibrated is False


def test_seeds_from_results_ignores_unknown_query_ids() -> None:
    spectra = [_spectrum("a", 500.0)]
    rows = [
        {
            "query_id": "missing",
            "reference_id": "r1",
            "reference_name": "X",
            "q_value": 0.001,
            "score": 0.9,
        }
    ]
    assert seeds_from_results([_result(spectra, rows)]) == []


def test_seeds_from_results_is_deterministic() -> None:
    spectra = [_spectrum("a", 500.0), _spectrum("b", 500.005)]
    rows = [
        {
            "query_id": "a",
            "reference_id": "r1",
            "reference_name": "A",
            "q_value": 0.001,
            "score": 0.9,
        },
        {
            "query_id": "b",
            "reference_id": "r2",
            "reference_name": "B",
            "q_value": 0.001,
            "score": 0.8,
        },
    ]
    first = seeds_from_results([_result(spectra, rows)])
    second = seeds_from_results([_result(spectra, rows)])
    assert [s.model_dump() for s in first] == [s.model_dump() for s in second]
