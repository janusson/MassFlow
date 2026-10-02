"""
Stage 1 spectral-network construction tests.

These are integration tests: they exercise the real classical similarity engine
through :func:`MassFlow.network.build.build_spectral_graph`, but only on tiny,
hand-crafted spectra. They verify the P1 exit criteria — deterministic spectral
graph, no q-value/FDR involvement, input spectra unmutated.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import NetworkConfig
from MassFlow.network import build_spectral_graph
from MassFlow.network.models import MolecularGraph, SpectralRelationship

pytestmark = pytest.mark.integration

_PEAKS_MZ = np.array(
    [50.0, 80.0, 110.0, 140.0, 170.0, 200.0, 230.0, 260.0, 290.0, 320.0],
    dtype=np.float64,
)
_PEAKS_INT = np.array(
    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float64
)


def _spectrum(
    precursor_mz: float,
    spec_id: str,
    *,
    mz: np.ndarray = _PEAKS_MZ,
    intensities: np.ndarray = _PEAKS_INT,
    retention_time: float = 100.0,
) -> Spectrum:
    return Spectrum(
        mz=mz.copy(),
        intensities=intensities.copy(),
        metadata={
            "id": spec_id,
            "precursor_mz": precursor_mz,
            "retention_time": retention_time,
            "adduct": "[M+H]+",
            "charge": 1,
        },
    )


def _cfg(**overrides: object) -> NetworkConfig:
    payload: dict[str, object] = {
        "enabled": True,
        "algorithm": "modified_cosine",
        "min_score": 0.7,
        "min_matched_peaks": 3,
        "top_k_per_node": None,
    }
    payload.update(overrides)
    return NetworkConfig(**payload)  # type: ignore[arg-type]


def _degree(graph: MolecularGraph) -> dict[str, int]:
    degree: dict[str, int] = {}
    for rel in graph.relationships:
        degree[rel.source_node_id] = degree.get(rel.source_node_id, 0) + 1
        degree[rel.target_node_id] = degree.get(rel.target_node_id, 0) + 1
    return degree


def _strip_timestamps(document: dict) -> dict:
    """Remove the (inherently time-varying) provenance timestamps."""
    document = copy.deepcopy(document)
    document["provenance"].pop("created_at", None)
    for relationship in document["relationships"]:
        relationship["provenance"].pop("created_at", None)
    for family in document.get("families", []):
        family["provenance"].pop("created_at", None)
    for context in document.get("network_contexts", []):
        context["provenance"].pop("created_at", None)
    for inference in document.get("inferences", []):
        inference["provenance"].pop("created_at", None)
    return document


def test_builds_nodes_and_edges() -> None:
    first = _spectrum(500.0, "a")
    second = _spectrum(500.01, "b")
    third = _spectrum(600.0, "c", mz=_PEAKS_MZ + 20.0)

    graph = build_spectral_graph([first, second, third], _cfg())

    assert len(graph.nodes) == 3
    assert len(graph.relationships) == 1
    edge = graph.relationships[0]
    assert edge.relationship_type == "spectral"
    assert edge.directed is False
    assert edge.score == pytest.approx(1.0)
    assert edge.matched_peaks == 10
    assert {edge.source_node_id, edge.target_node_id} == {
        graph.nodes[0].node_id,
        graph.nodes[1].node_id,
    }


def test_identical_spectra_deduplicate_to_one_node() -> None:
    spectrum = _spectrum(500.0, "a")
    graph = build_spectral_graph([spectrum, spectrum.clone()], _cfg())
    assert len(graph.nodes) == 1
    assert graph.relationships == []


def test_ms1_window_excludes_distant_precursors() -> None:
    # Identical fragmentation, but the precursor gap (2 Da) is far outside the
    # 0.02 Da candidate window, so no edge may form.
    graph = build_spectral_graph([_spectrum(500.0, "a"), _spectrum(502.0, "b")], _cfg())
    assert len(graph.nodes) == 2
    assert graph.relationships == []


def test_matched_peak_threshold_excludes_weak_edges() -> None:
    graph = build_spectral_graph(
        [_spectrum(500.0, "a"), _spectrum(500.01, "b")],
        _cfg(min_matched_peaks=20),
    )
    assert graph.relationships == []


def test_top_k_caps_each_node_degree() -> None:
    spectra = [_spectrum(500.0 + 0.005 * i, f"n{i}") for i in range(4)]
    uncapped = build_spectral_graph(spectra, _cfg())
    assert len(uncapped.relationships) == 6  # complete graph on 4 nodes

    capped = build_spectral_graph(spectra, _cfg(top_k_per_node=1))
    assert all(d <= 1 for d in _degree(capped).values())
    assert len(capped.relationships) == 2  # a perfect matching


def test_build_is_deterministic() -> None:
    spectra = [
        _spectrum(500.0, "a"),
        _spectrum(500.01, "b"),
        _spectrum(600.0, "c", mz=_PEAKS_MZ + 20.0),
    ]
    first = build_spectral_graph(spectra, _cfg())
    second = build_spectral_graph(spectra, _cfg())
    # Identifiers are content-derived and must be identical across runs.
    assert [n.node_id for n in first.nodes] == [n.node_id for n in second.nodes]
    assert [r.relationship_id for r in first.relationships] == [
        r.relationship_id for r in second.relationships
    ]
    # Everything except the inherently time-varying timestamps is identical.
    assert _strip_timestamps(first.to_dict()) == _strip_timestamps(second.to_dict())


def test_input_spectra_are_not_mutated() -> None:
    first = _spectrum(500.0, "a")
    second = _spectrum(500.01, "b")
    before = [s.metadata.copy() for s in (first, second)]
    build_spectral_graph([first, second], _cfg())
    assert [first.metadata, second.metadata] == before
    assert first.get("id") == "a"
    assert second.get("id") == "b"


def test_graph_carries_no_statistical_confidence() -> None:
    graph = build_spectral_graph(
        [_spectrum(500.0, "a"), _spectrum(500.01, "b")], _cfg()
    )
    assert graph.fdr_assessments == {}
    for field in ("q_value", "p_value", "fdr", "confidence"):
        assert field not in SpectralRelationship.model_fields


def test_graph_round_trips_through_json() -> None:
    graph = build_spectral_graph(
        [_spectrum(500.0, "a"), _spectrum(500.01, "b")], _cfg()
    )
    restored = MolecularGraph.from_json(graph.to_json())
    assert restored.to_dict() == graph.to_dict()


def test_scoring_is_bounded_by_candidate_degree(monkeypatch) -> None:
    """Scoring must be candidate-driven, never a single N x N pass (R-1)."""
    import MassFlow.similarity as similarity_module

    # 14 spectra: 12 mutually far apart, plus one close pair. Only that pair is
    # a candidate, so the engine must never be handed the full set at once.
    spectra = [_spectrum(400.0 + 10.0 * i, f"far{i}") for i in range(12)]
    spectra.append(_spectrum(500.0, "near_a"))
    spectra.append(_spectrum(500.005, "near_b"))

    reference_batch_sizes: list[int] = []
    original_search = similarity_module.SimilarityEngine.search

    def spy(self, query_spectra, reference_spectra, **kwargs):
        references = list(reference_spectra)
        reference_batch_sizes.append(len(references))
        return original_search(self, query_spectra, references, **kwargs)

    monkeypatch.setattr(similarity_module.SimilarityEngine, "search", spy)

    graph = build_spectral_graph(spectra, _cfg())

    assert graph.relationships  # the close pair produced an edge
    assert reference_batch_sizes, "expected at least one candidate-driven search"
    assert max(reference_batch_sizes) < len(spectra)
