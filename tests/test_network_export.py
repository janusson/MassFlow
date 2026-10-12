"""
Tests for unified network annotation export and graph serialization.

Covers the tiering rules, merging of direct matches with propagated hypotheses,
conflict handling during graph export, and round-trip graph serialization.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Optional

import polars as pl
import pytest

from MassFlow.io import save_network_graph
from MassFlow.network.export import (
    export_network_annotations,
    export_network_graph,
)
from MassFlow.network.propagation import (
    PROPAGATION_OUTPUT_SCHEMA,
    propagate_annotations,
)

_HAS_NETWORKX = importlib.util.find_spec("networkx") is not None

requires_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX, reason="networkx (optional 'graph' extra) not installed"
)

pytestmark = pytest.mark.unit

_IK_A = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
_IK_B = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _edges(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_id": [r[0] for r in rows],
            "target_id": [r[1] for r in rows],
            "weight": [float(r[2]) for r in rows],
        },
        schema={"source_id": pl.Utf8, "target_id": pl.Utf8, "weight": pl.Float64},
    )


def _match(
    query_id: str,
    reference_id: str,
    *,
    score: float = 0.9,
    q_value: Optional[float] = 0.001,
    is_decoy: bool = False,
    inchikey: Optional[str] = None,
    reference_name: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "reference_id": reference_id,
        "reference_name": reference_name,
        "score": score,
        "q_value": q_value,
        "p_value": 0.05,
        "is_decoy": is_decoy,
        "inchikey": inchikey,
        "matched_peaks": 5,
    }


_MATCH_SCHEMA = {
    "query_id": pl.Utf8,
    "reference_id": pl.Utf8,
    "reference_name": pl.Utf8,
    "score": pl.Float64,
    "q_value": pl.Float64,
    "p_value": pl.Float64,
    "is_decoy": pl.Boolean,
    "inchikey": pl.Utf8,
    "matched_peaks": pl.Int64,
}


def _matches(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_MATCH_SCHEMA)
    return pl.DataFrame(rows, schema=_MATCH_SCHEMA)


def _row(frame: pl.DataFrame, query_id: str, *, source: Optional[str] = None) -> dict:
    hits = frame.filter(pl.col("query_id") == query_id)
    if source is not None:
        hits = hits.filter(pl.col("evidence_source") == source)
    assert hits.height == 1, f"expected one row, got {hits.height}"
    return hits.to_dicts()[0]


def _tier_of(
    frame: pl.DataFrame, query_id: str, *, source: Optional[str] = None
) -> Any:
    return _row(frame, query_id, source=source)["confidence_tier"]


# ---------------------------------------------------------------------------
# Tiering
# ---------------------------------------------------------------------------


def test_tier1_direct_high_confidence() -> None:
    matches = _matches([_match("a", "rA", score=0.95, q_value=0.001, inchikey=_IK_A)])
    table = export_network_annotations(matches, _matches([]))
    row = _row(table, "a")
    assert row["confidence_tier"] == 1
    assert row["confidence_tier_label"] == "direct_high_confidence"


def test_low_confidence_direct_without_network_support_is_untiered() -> None:
    matches = _matches([_match("a", "rA", score=0.9, q_value=0.02, inchikey=_IK_A)])
    table = export_network_annotations(matches, _matches([]))
    row = _row(table, "a")
    assert row["confidence_tier"] is None
    assert row["confidence_tier_label"] is None


def test_tier1_threshold_is_configurable() -> None:
    matches = _matches([_match("a", "rA", score=0.9, q_value=0.02, inchikey=_IK_A)])
    table = export_network_annotations(
        matches, _matches([]), direct_high_confidence_q_value=0.05
    )
    assert _tier_of(table, "a") == 1


def test_tier2_network_supported() -> None:
    edges = _edges([("seed", "target", 0.9)])
    matches = _matches([_match("seed", "rA", score=0.9, q_value=0.001, inchikey=_IK_A)])
    propagated = propagate_annotations(edges, matches)
    table = export_network_annotations(matches, propagated)
    assert _tier_of(table, "target", source="network") == 2


def test_tier3_network_conflicted() -> None:
    edges = _edges([("s1", "centre", 0.5), ("s2", "centre", 0.5)])
    matches = _matches(
        [
            _match("s1", "rA", score=1.0, inchikey=_IK_A),
            _match("s2", "rB", score=1.0, inchikey=_IK_B),
        ]
    )
    propagated = propagate_annotations(edges, matches)
    table = export_network_annotations(matches, propagated)
    conflicted = table.filter(
        (pl.col("query_id") == "centre") & (pl.col("evidence_source") == "network")
    )
    assert conflicted.height == 2
    assert set(conflicted["confidence_tier"].to_list()) == {3}


def test_tier3_direct_match_conflict() -> None:
    edges = _edges([("neighbour", "target", 0.99)])
    matches = _matches(
        [
            _match("target", "rA", score=0.95, q_value=0.001, inchikey=_IK_A),
            _match("neighbour", "rB", score=0.99, q_value=0.001, inchikey=_IK_B),
        ]
    )
    propagated = propagate_annotations(edges, matches)
    table = export_network_annotations(matches, propagated)
    direct = _row(table, "target", source="direct")
    assert direct["network_conflict"] is True
    # A direct high-confidence conflict is still Tier 1 (it is a real direct hit).
    assert direct["confidence_tier"] == 1


def test_decoys_receive_no_tier() -> None:
    matches = _matches(
        [_match("a", "rA", score=0.99, q_value=0.001, is_decoy=True, inchikey=_IK_A)]
    )
    table = export_network_annotations(matches, _matches([]))
    row = _row(table, "a")
    assert row["is_decoy"] is True
    assert row["confidence_tier"] is None


def test_confidence_tiers_can_be_disabled() -> None:
    matches = _matches([_match("a", "rA", inchikey=_IK_A)])
    table = export_network_annotations(matches, _matches([]), confidence_tiers=False)
    assert "confidence_tier" not in table.columns
    assert "confidence_tier_label" not in table.columns


def test_conflicting_node_records_competing_candidates_in_table() -> None:
    edges = _edges([("s1", "centre", 0.5), ("s2", "centre", 0.5)])
    matches = _matches(
        [
            _match("s1", "rA", score=1.0, inchikey=_IK_A),
            _match("s2", "rB", score=1.0, inchikey=_IK_B),
        ]
    )
    propagated = propagate_annotations(edges, matches)
    table = export_network_annotations(matches, propagated)
    conflicted = table.filter(pl.col("network_conflict"))
    assert conflicted.height >= 2
    for record in conflicted.to_dicts():
        assert len(record["network_conflicting_candidates"]) >= 1


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------


def test_propagated_direct_rows_are_not_duplicated() -> None:
    # `propagated` is the full enriched table (direct + network rows).
    edges = _edges([("seed", "target", 0.9)])
    matches = _matches([_match("seed", "rA", score=0.9, inchikey=_IK_A)])
    propagated = propagate_annotations(edges, matches)
    table = export_network_annotations(matches, propagated)
    seed_rows = table.filter(pl.col("query_id") == "seed")
    assert seed_rows.height == 1
    assert seed_rows["evidence_source"].to_list() == ["direct"]


def test_direct_matches_without_network_evidence_are_kept() -> None:
    matches = _matches(
        [
            _match("a", "rA", score=0.9, q_value=0.5, inchikey=_IK_A),
            _match("b", "rB", score=0.8, q_value=0.5, inchikey=_IK_B),
        ]
    )
    table = export_network_annotations(matches, _matches([]))
    assert set(table["query_id"].to_list()) == {"a", "b"}
    # Neither is high-confidence nor network-supported, so neither is tiered.
    assert all(tier is None for tier in table["confidence_tier"].to_list())


def test_provenance_columns_and_direct_values_are_preserved() -> None:
    edges = _edges([("seed", "target", 0.9)])
    matches = _matches(
        [
            _match("seed", "rA", score=0.9123, q_value=0.003, inchikey=_IK_A),
            _match("target", "rB", score=0.4, q_value=0.2, inchikey=_IK_B),
        ]
    )
    propagated = propagate_annotations(edges, matches)
    table = export_network_annotations(matches, propagated)

    for column in (
        "is_direct_match",
        "evidence_source",
        "network_support_score",
        "supporting_neighbor_count",
        "network_conflict",
    ):
        assert column in table.columns

    direct = _row(table, "seed", source="direct")
    assert direct["score"] == pytest.approx(0.9123)
    assert direct["q_value"] == pytest.approx(0.003)


def test_output_schema_is_stable_and_deterministic() -> None:
    matches = _matches([_match("a", "rA", inchikey=_IK_A)])
    first = export_network_annotations(matches, _matches([]))
    second = export_network_annotations(matches, _matches([]))
    expected = list(PROPAGATION_OUTPUT_SCHEMA) + [
        "confidence_tier",
        "confidence_tier_label",
    ]
    assert first.columns == expected
    assert first.equals(second)


def test_empty_inputs_yield_empty_typed_table() -> None:
    table = export_network_annotations(_matches([]), _matches([]))
    assert table.height == 0
    assert table.columns == list(PROPAGATION_OUTPUT_SCHEMA) + [
        "confidence_tier",
        "confidence_tier_label",
    ]


def test_invalid_input_types_raise() -> None:
    with pytest.raises(TypeError):
        export_network_annotations("nope", _matches([]))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        export_network_annotations(_matches([]), "nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Graph export — JSON (dependency-free)
# ---------------------------------------------------------------------------


def _annotation_table() -> pl.DataFrame:
    edges = _edges([("seed", "target", 0.9)])
    matches = _matches(
        [_match("seed", "rA", score=0.95, q_value=0.001, inchikey=_IK_A)]
    )
    propagated = propagate_annotations(edges, matches)
    return export_network_annotations(matches, propagated)


def test_json_export_embeds_node_attributes(tmp_path: Path) -> None:
    edges = _edges([("seed", "target", 0.9)])
    out = tmp_path / "graph.json"
    export_network_graph(edges, _annotation_table(), out, format="json")

    payload = json.loads(out.read_text())
    assert payload["directed"] is False
    nodes = {node["id"]: node for node in payload["nodes"]}
    assert set(nodes) == {"seed", "target"}

    seed = nodes["seed"]
    assert seed["annotation_status"] == "direct_match"
    # ``reference_id`` is the library entry id; the compound identity is exposed
    # separately as ``candidate_id`` (the full InChIKey here).
    assert seed["reference_id"] == "rA"
    assert seed["candidate_id"] == _IK_A
    assert seed["confidence_tier"] == 1
    assert seed["primary_score"] == pytest.approx(0.95)
    assert seed["primary_score_kind"] == "match_score"
    assert seed["evidence_source"] == "direct"

    target = nodes["target"]
    assert target["annotation_status"] == "network_supported"
    assert target["confidence_tier"] == 2

    assert payload["links"] == [{"source": "seed", "target": "target", "weight": 0.9}]


def test_json_export_marks_unannotated_nodes(tmp_path: Path) -> None:
    edges = _edges([("solo", "orphan", 0.5)])
    out = tmp_path / "graph.json"
    export_network_graph(edges, _matches([]), out, format="json")
    nodes = {node["id"]: node for node in json.loads(out.read_text())["nodes"]}
    assert nodes["solo"]["annotation_status"] == "unannotated"
    assert nodes["orphan"]["annotation_status"] == "unannotated"


def test_json_export_flags_conflicted_nodes(tmp_path: Path) -> None:
    edges = _edges([("s1", "centre", 0.5), ("s2", "centre", 0.5)])
    matches = _matches(
        [
            _match("s1", "rA", score=1.0, inchikey=_IK_A),
            _match("s2", "rB", score=1.0, inchikey=_IK_B),
        ]
    )
    propagated = propagate_annotations(edges, matches)
    annotations = export_network_annotations(matches, propagated)

    out = tmp_path / "graph.json"
    export_network_graph(edges, annotations, out, format="json")
    centre = {node["id"]: node for node in json.loads(out.read_text())["nodes"]}[
        "centre"
    ]
    assert centre["annotation_status"] == "network_conflicted"
    assert centre["network_conflict"] is True
    assert centre["conflict_count"] >= 1
    assert centre["confidence_tier"] == 3


def test_json_round_trip_is_lossless(tmp_path: Path) -> None:
    edges = _edges([("seed", "target", 0.9), ("target", "leaf", 0.4)])
    annotations = _annotation_table()
    out = tmp_path / "graph.json"
    export_network_graph(edges, annotations, out, format="json")

    payload = json.loads(out.read_text())
    node_ids = {node["id"] for node in payload["nodes"]}
    assert node_ids == {"seed", "target", "leaf"}
    # Edges are undirected and canonicalised, so compare unordered endpoint pairs.
    link_pairs = {
        frozenset((link["source"], link["target"])) for link in payload["links"]
    }
    assert link_pairs == {
        frozenset(("seed", "target")),
        frozenset(("target", "leaf")),
    }


# ---------------------------------------------------------------------------
# Graph export — GraphML (optional networkx backend)
# ---------------------------------------------------------------------------


@requires_networkx
def test_graphml_round_trip(tmp_path: Path) -> None:
    import networkx as nx

    edges = _edges([("seed", "target", 0.9), ("target", "leaf", 0.4)])
    annotations = _annotation_table()
    out = tmp_path / "graph.graphml"
    export_network_graph(edges, annotations, out, format="graphml")
    assert out.is_file()

    graph = nx.read_graphml(out)
    assert set(graph.nodes()) == {"seed", "target", "leaf"}
    assert graph["seed"]["target"]["weight"] == pytest.approx(0.9)

    seed = graph.nodes["seed"]
    assert seed["annotation_status"] == "direct_match"
    assert seed["reference_id"] == "rA"
    assert seed["candidate_id"] == _IK_A
    assert int(seed["confidence_tier"]) == 1
    assert float(seed["primary_score"]) == pytest.approx(0.95)


@requires_networkx
def test_graphml_flags_conflicted_nodes(tmp_path: Path) -> None:
    import networkx as nx

    edges = _edges([("s1", "centre", 0.5), ("s2", "centre", 0.5)])
    matches = _matches(
        [
            _match("s1", "rA", score=1.0, inchikey=_IK_A),
            _match("s2", "rB", score=1.0, inchikey=_IK_B),
        ]
    )
    propagated = propagate_annotations(edges, matches)
    annotations = export_network_annotations(matches, propagated)

    out = tmp_path / "graph.graphml"
    export_network_graph(edges, annotations, out, format="graphml")
    centre = nx.read_graphml(out).nodes["centre"]
    assert centre["annotation_status"] == "network_conflicted"
    assert str(centre["network_conflict"]).lower() in ("true", "1")


def test_json_export_works_without_networkx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from MassFlow.network import export as export_module

    monkeypatch.setattr(export_module, "_HAS_NETWORKX", False)
    out = tmp_path / "graph.json"
    export_network_graph(
        _edges([("seed", "target", 0.9)]), _annotation_table(), out, format="json"
    )
    assert {node["id"] for node in json.loads(out.read_text())["nodes"]} == {
        "seed",
        "target",
    }


def test_graphml_requires_networkx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from MassFlow.network import export as export_module

    monkeypatch.setattr(export_module, "_HAS_NETWORKX", False)
    with pytest.raises(ImportError, match="networkx"):
        export_network_graph(
            _edges([("seed", "target", 0.9)]),
            _annotation_table(),
            tmp_path / "graph.graphml",
            format="graphml",
        )


def test_graph_export_rejects_bad_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported network graph format"):
        export_network_graph(
            _edges([("a", "b", 0.5)]),
            _matches([]),
            tmp_path / "graph.xyz",
            format="cyjs",  # type: ignore[arg-type]
        )


def test_graph_export_rejects_bad_input_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        export_network_graph("nope", _matches([]), tmp_path / "g.json")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        export_network_graph(_edges([]), "nope", tmp_path / "g.json")  # type: ignore[arg-type]


def test_save_network_graph_rejects_bad_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported network graph format"):
        save_network_graph({}, tmp_path / "graph.xyz", format="graphml2")
