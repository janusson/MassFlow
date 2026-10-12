"""
Network-aware annotation propagation tests (experimental, one hop).

These tests pin the scientifically consequential guarantees of
:mod:`MassFlow.network.propagation`:

* direct-match evidence (score, q-value, p-value, decoy status) is preserved
  byte-for-byte and never rewritten by neighbour votes;
* network support is a *separate* hypothesis, never a library identification;
* propagation is one hop, decoy-free, deterministic, and conflict-aware.

Everything here is dependency-light: the module under test uses only the core
NumPy/SciPy/Polars stack, so no optional graph package is required.
"""

from __future__ import annotations

from typing import Any, Optional

import polars as pl
import pytest
from pydantic import ValidationError

from MassFlow.network.propagation import (
    PROPAGATION_OUTPUT_SCHEMA,
    NetworkPropagationConfig,
    network_seed_id,
    propagate_annotations,
    validate_network_edges,
)

pytestmark = pytest.mark.unit

# Deterministic InChIKeys used as compound identities.
_IK_A = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"  # aspirin
_IK_B = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"  # caffeine
_IK_C = "YAPQBXQYLJRXSA-UHFFFAOYSA-N"  # theobromine


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


_EMPTY_EDGES = _edges([])


def _match(
    query_id: str,
    reference_id: str,
    *,
    score: float = 0.9,
    q_value: Optional[float] = 0.001,
    p_value: Optional[float] = 0.01,
    is_decoy: bool = False,
    inchikey: Optional[str] = None,
    reference_name: Optional[str] = None,
    matched_peaks: Optional[int] = 5,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "reference_id": reference_id,
        "reference_name": reference_name,
        "score": score,
        "q_value": q_value,
        "p_value": p_value,
        "is_decoy": is_decoy,
        "inchikey": inchikey,
        "matched_peaks": matched_peaks,
    }


def _matches(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = {
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
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def _find(
    frame: pl.DataFrame,
    *,
    query_id: str,
    candidate_id: Optional[str] = None,
    evidence_source: Optional[str] = None,
) -> dict[str, Any]:
    hits = frame.filter(pl.col("query_id") == query_id)
    if candidate_id is not None:
        hits = hits.filter(pl.col("candidate_id") == candidate_id)
    if evidence_source is not None:
        hits = hits.filter(pl.col("evidence_source") == evidence_source)
    assert hits.height == 1, f"expected exactly one row, got {hits.height}"
    return hits.to_dicts()[0]


# ---------------------------------------------------------------------------
# validate_network_edges
# ---------------------------------------------------------------------------


def test_validate_requires_dataframe() -> None:
    with pytest.raises(TypeError):
        validate_network_edges("not a frame")  # type: ignore[arg-type]


def test_config_is_frozen_and_forbids_extra_fields() -> None:
    cfg = NetworkPropagationConfig()
    with pytest.raises(ValidationError):
        cfg.seed_min_score = 0.5  # type: ignore[misc]  # frozen=True
    with pytest.raises(ValidationError):
        NetworkPropagationConfig(unknown_field=1)  # type: ignore[call-arg]


def test_validate_requires_columns() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        validate_network_edges(pl.DataFrame({"source_id": ["a"], "target_id": ["b"]}))


def test_validate_empty_returns_empty_schema() -> None:
    result = validate_network_edges(_EMPTY_EDGES)
    assert result.height == 0
    assert result.columns == ["source_id", "target_id", "weight"]


def test_validate_rejects_null_endpoints() -> None:
    frame = pl.DataFrame(
        {
            "source_id": ["a", None],
            "target_id": ["b", "c"],
            "weight": [0.5, 0.5],
        },
        schema={"source_id": pl.Utf8, "target_id": pl.Utf8, "weight": pl.Float64},
    )
    with pytest.raises(ValueError, match="null endpoints"):
        validate_network_edges(frame)


def test_validate_rejects_blank_endpoints() -> None:
    frame = _edges([("a", "  ", 0.5)])
    with pytest.raises(ValueError, match="empty-string endpoints"):
        validate_network_edges(frame)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf"), 1.5])
def test_validate_drops_inappropriate_weights(weight: float) -> None:
    frame = _edges([("a", "b", weight), ("c", "d", 0.5)])
    result = validate_network_edges(frame)
    assert result.height == 1
    assert result.row(0) == ("c", "d", 0.5)


def test_validate_drops_null_weights() -> None:
    frame = pl.DataFrame(
        {"source_id": ["a", "c"], "target_id": ["b", "d"], "weight": [None, 0.5]},
        schema={"source_id": pl.Utf8, "target_id": pl.Utf8, "weight": pl.Float64},
    )
    result = validate_network_edges(frame)
    assert result.height == 1
    assert result.row(0) == ("c", "d", 0.5)


def test_validate_drops_non_numeric_weights() -> None:
    frame = pl.DataFrame(
        {"source_id": ["a", "c"], "target_id": ["b", "d"], "weight": ["n/a", "0.5"]}
    )
    result = validate_network_edges(frame)
    assert result.height == 1
    assert result.row(0) == ("c", "d", 0.5)


def test_validate_keeps_boundary_weights() -> None:
    frame = _edges([("a", "b", 0.0), ("c", "d", 1.0)])
    result = validate_network_edges(frame)
    assert result.height == 2
    assert {result.row(0), result.row(1)} == {("a", "b", 0.0), ("c", "d", 1.0)}


def test_validate_drops_self_loops() -> None:
    frame = _edges([("a", "a", 0.9), ("a", "b", 0.5)])
    result = validate_network_edges(frame)
    assert result.height == 1
    assert result.row(0) == ("a", "b", 0.5)


def test_validate_canonicalises_direction() -> None:
    frame = _edges([("z", "a", 0.5)])
    result = validate_network_edges(frame)
    assert result.row(0) == ("a", "z", 0.5)


def test_validate_dedups_keeping_max_weight_deterministically() -> None:
    frame = _edges([("a", "b", 0.3), ("b", "a", 0.9), ("a", "b", 0.9)])
    result = validate_network_edges(frame)
    assert result.height == 1
    assert result.row(0) == ("a", "b", 0.9)
    # Order-independent: reversed input yields the same canonical table.
    reversed_frame = _edges([("a", "b", 0.9), ("b", "a", 0.9), ("a", "b", 0.3)])
    assert validate_network_edges(reversed_frame).equals(result)


def test_validate_carries_edge_id_and_breaks_ties_lexicographically() -> None:
    frame = pl.DataFrame(
        {
            "source_id": ["a", "b"],
            "target_id": ["b", "a"],
            "weight": [0.9, 0.9],
            "edge_id": ["e2", "e1"],
        }
    )
    result = validate_network_edges(frame, edge_id_col="edge_id")
    assert result.height == 1
    assert result.row(0) == ("a", "b", 0.9, "e1")


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_matches_yields_empty_output() -> None:
    out = propagate_annotations(_edges([("a", "b", 0.9)]), _matches([]))
    assert out.height == 0
    assert out.columns == list(PROPAGATION_OUTPUT_SCHEMA)


def test_empty_edges_passes_direct_rows_through() -> None:
    matches = _matches([_match("a", "r1", inchikey=_IK_A)])
    out = propagate_annotations(_EMPTY_EDGES, matches)
    row = _find(out, query_id="a")
    assert row["evidence_source"] == "direct"
    assert row["candidate_id"] == _IK_A
    assert row["network_status"] == "not_in_network"
    assert row["score"] == pytest.approx(0.9)


def test_isolated_node_receives_no_fabricated_row() -> None:
    # Node c is not an edge endpoint at all: nothing is invented for it.
    edges = _edges([("a", "b", 0.9)])
    matches = _matches([_match("a", "r1", inchikey=_IK_A)])
    out = propagate_annotations(edges, matches)
    assert out.filter(pl.col("query_id") == "c").height == 0


def test_neighbour_without_eligible_seed_gets_no_support() -> None:
    # The only seed has q_value above threshold, so it does not propagate.
    edges = _edges([("a", "b", 0.9)])
    matches = _matches([_match("a", "r1", inchikey=_IK_A, q_value=0.5)])
    out = propagate_annotations(edges, matches)
    assert (
        out.filter(
            (pl.col("query_id") == "b") & (pl.col("evidence_source") == "network")
        ).height
        == 0
    )


# ---------------------------------------------------------------------------
# Single-edge propagation (hand-computed fixture)
# ---------------------------------------------------------------------------


def test_single_seed_propagates_across_one_valid_edge() -> None:
    edges = _edges([("n1", "n2", 0.8)])
    matches = _matches([_match("n1", "r1", score=0.9, inchikey=_IK_A)])
    out = propagate_annotations(edges, matches)

    row = _find(out, query_id="n2", evidence_source="network")
    assert row["candidate_id"] == _IK_A
    # Hand-computed: support = edge_weight * seed_score = 0.8 * 0.9.
    assert row["network_support_score"] == pytest.approx(0.72)
    assert row["supporting_neighbor_count"] == 1
    assert row["supporting_seed_ids"] == [network_seed_id("n1", _IK_A)]
    assert row["network_status"] == "network_supported"
    assert row["network_rank"] == 1
    assert row["network_conflict"] is False


def test_propagated_row_carries_no_direct_evidence() -> None:
    edges = _edges([("n1", "n2", 0.8)])
    matches = _matches([_match("n1", "r1", score=0.9, inchikey=_IK_A)])
    out = propagate_annotations(edges, matches)
    row = _find(out, query_id="n2", evidence_source="network")
    assert row["is_direct_match"] is False
    assert row["score"] is None
    assert row["q_value"] is None
    assert row["p_value"] is None
    assert row["is_decoy"] is None


# ---------------------------------------------------------------------------
# Weighted voting from multiple neighbours
# ---------------------------------------------------------------------------


def test_weighted_voting_from_multiple_neighbours() -> None:
    # centre is a hub; three seeds vote with distinct weights and scores.
    edges = _edges(
        [
            ("s1", "centre", 0.9),
            ("s2", "centre", 0.9),
            ("s3", "centre", 0.4),
        ]
    )
    matches = _matches(
        [
            _match("s1", "rA1", score=0.9, q_value=0.001, inchikey=_IK_A),
            _match("s2", "rA2", score=0.8, q_value=0.001, inchikey=_IK_A),
            _match("s3", "rB", score=0.95, q_value=0.001, inchikey=_IK_B),
        ]
    )
    out = propagate_annotations(edges, matches)
    row = _find(out, query_id="centre", evidence_source="network")
    # A: 0.9*0.9 + 0.9*0.8 = 1.53 ; B: 0.4*0.95 = 0.38 ; A wins by share.
    assert row["candidate_id"] == _IK_A
    assert row["network_support_score"] == pytest.approx(1.53)
    assert row["supporting_neighbor_count"] == 2
    assert row["network_status"] == "network_supported"


def test_weight_seed_by_score_can_be_disabled() -> None:
    edges = _edges([("s1", "centre", 0.5), ("s2", "centre", 0.5)])
    matches = _matches(
        [
            _match("s1", "rA", score=0.9, inchikey=_IK_A),
            _match("s2", "rB", score=0.1, inchikey=_IK_B),
        ]
    )
    cfg = NetworkPropagationConfig(weight_seed_by_score=False)
    out = propagate_annotations(edges, matches, config=cfg)
    # Without score weighting both candidates tie at 0.5 -> conflict.
    statuses = set(
        out.filter(pl.col("evidence_source") == "network")["network_status"].to_list()
    )
    assert statuses == {"network_conflicted"}


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


def test_equal_neighbour_evidence_is_a_conflict() -> None:
    edges = _edges([("s1", "centre", 0.5), ("s2", "centre", 0.5)])
    matches = _matches(
        [
            _match("s1", "rA", score=1.0, inchikey=_IK_A),
            _match("s2", "rB", score=1.0, inchikey=_IK_B),
        ]
    )
    out = propagate_annotations(edges, matches)
    network_rows = out.filter(pl.col("evidence_source") == "network")
    assert network_rows.height == 2
    assert set(network_rows["network_status"].to_list()) == {"network_conflicted"}
    assert network_rows["network_conflict"].all()
    # Both candidate structures are preserved, ranks assigned by support order.
    assert set(network_rows["candidate_id"].to_list()) == {_IK_A, _IK_B}
    assert set(network_rows["network_rank"].to_list()) == {1, 2}
    for row in network_rows.to_dicts():
        assert set(row["network_conflicting_candidates"]) == {_IK_A, _IK_B} - {
            row["candidate_id"]
        }


def test_strong_but_chemically_inconsistent_hits_do_not_collapse() -> None:
    """Two neighbours with strong, disagreeing hits must not be resolved by an
    arbitrary winner."""
    edges = _edges([("s1", "centre", 0.95), ("s2", "centre", 0.9)])
    matches = _matches(
        [
            _match("s1", "rA", score=0.99, inchikey=_IK_A),
            _match("s2", "rB", score=0.99, inchikey=_IK_B),
        ]
    )
    out = propagate_annotations(edges, matches)
    assert out.filter(pl.col("evidence_source") == "network").height == 2


# ---------------------------------------------------------------------------
# Direct evidence is never overwritten
# ---------------------------------------------------------------------------


def test_direct_high_confidence_match_is_not_overwritten_by_neighbours() -> None:
    edges = _edges([("neighbour", "target", 0.99)])
    matches = _matches(
        [
            _match("target", "rA", score=0.95, q_value=0.001, inchikey=_IK_A),
            _match("neighbour", "rB", score=0.99, q_value=0.001, inchikey=_IK_B),
        ]
    )
    out = propagate_annotations(edges, matches)
    target_rows = out.filter(pl.col("query_id") == "target")
    # Only the direct row exists for the target; no competing network promotion.
    assert target_rows.filter(pl.col("evidence_source") == "network").height == 0
    direct = _find(out, query_id="target", evidence_source="direct")
    assert direct["candidate_id"] == _IK_A
    assert direct["score"] == pytest.approx(0.95)
    # The disagreement is surfaced, not hidden.
    assert direct["network_conflict"] is True
    assert direct["network_conflicting_candidates"] == [_IK_B]
    assert direct["network_status"] == "direct_match_conflict"


def test_direct_match_corroborated_by_neighbours_is_not_a_conflict() -> None:
    edges = _edges([("neighbour", "target", 0.9)])
    matches = _matches(
        [
            _match("target", "rA", score=0.95, inchikey=_IK_A),
            _match("neighbour", "rA2", score=0.9, inchikey=_IK_A),
        ]
    )
    out = propagate_annotations(edges, matches)
    direct = _find(out, query_id="target", evidence_source="direct")
    assert direct["network_conflict"] is False
    assert direct["network_status"] == "direct_match"
    assert direct["network_support_score"] == pytest.approx(0.9 * 0.9)
    assert direct["supporting_neighbor_count"] == 1


# ---------------------------------------------------------------------------
# Decoys
# ---------------------------------------------------------------------------


def test_decoys_do_not_propagate() -> None:
    edges = _edges([("s1", "centre", 0.9)])
    matches = _matches([_match("s1", "rA", score=0.99, is_decoy=True, inchikey=_IK_A)])
    out = propagate_annotations(edges, matches)
    assert out.filter(pl.col("evidence_source") == "network").height == 0
    decoy_row = _find(out, query_id="s1")
    assert decoy_row["network_status"] == "decoy_match"
    assert decoy_row["is_decoy"] is True
    assert decoy_row["is_direct_match"] is False


def test_node_with_only_a_decoy_match_may_still_be_propagated_to() -> None:
    edges = _edges([("seed", "candidate", 0.9), ("seed", "centre", 0.9)])
    matches = _matches(
        [
            _match("seed", "rA", score=0.9, inchikey=_IK_A),
            # A decoy hit on the target does not make it "directly annotated".
            _match("centre", "rD", score=0.99, is_decoy=True, inchikey=_IK_C),
        ]
    )
    out = propagate_annotations(edges, matches)
    assert (
        out.filter(
            (pl.col("query_id") == "centre") & (pl.col("evidence_source") == "network")
        ).height
        == 1
    )


# ---------------------------------------------------------------------------
# Multiple direct candidates and unknown identifiers
# ---------------------------------------------------------------------------


def test_multiple_direct_candidates_are_all_preserved() -> None:
    matches = _matches(
        [
            _match("q", "rA", score=0.9, q_value=0.001, inchikey=_IK_A),
            _match("q", "rB", score=0.8, q_value=0.02, inchikey=_IK_B),
        ]
    )
    out = propagate_annotations(_edges([("q", "other", 0.9)]), matches)
    direct = out.filter(
        (pl.col("query_id") == "q") & (pl.col("evidence_source") == "direct")
    )
    assert direct.height == 2
    assert set(direct["candidate_id"].to_list()) == {_IK_A, _IK_B}
    assert set(direct["score"].to_list()) == {0.9, 0.8}


def test_matches_with_unknown_node_are_kept_as_direct_rows() -> None:
    matches = _matches([_match("absent", "rA", inchikey=_IK_A)])
    out = propagate_annotations(_edges([("a", "b", 0.9)]), matches)
    row = _find(out, query_id="absent")
    assert row["network_status"] == "not_in_network"
    assert row["score"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Metadata / confidence-field robustness
# ---------------------------------------------------------------------------


def test_absent_q_value_is_not_a_seed_under_default_gate() -> None:
    edges = _edges([("s", "centre", 0.9)])
    matches = _matches([_match("s", "rA", score=0.9, q_value=None, inchikey=_IK_A)])
    out = propagate_annotations(edges, matches)
    assert out.filter(pl.col("evidence_source") == "network").height == 0


def test_score_only_mode_allows_seeding_without_q_value() -> None:
    edges = _edges([("s", "centre", 0.9)])
    matches = _matches([_match("s", "rA", score=0.9, q_value=None, inchikey=_IK_A)])
    cfg = NetworkPropagationConfig(seed_max_q_value=None)
    out = propagate_annotations(edges, matches, config=cfg)
    assert out.filter(pl.col("evidence_source") == "network").height == 1


def test_reference_id_is_the_identity_fallback_without_inchikey() -> None:
    edges = _edges([("s", "centre", 0.9)])
    matches = _matches([_match("s", "rA", score=0.9, inchikey=None)])
    out = propagate_annotations(edges, matches)
    row = _find(out, query_id="centre", evidence_source="network")
    assert row["candidate_id"] == "rA"
    assert row["candidate_inchikey"] is None


def test_similar_names_do_not_merge_structures() -> None:
    """Identity is the InChIKey, never the name."""
    edges = _edges([("s1", "centre", 0.9), ("s2", "centre", 0.9)])
    matches = _matches(
        [
            _match("s1", "rA", score=1.0, inchikey=_IK_A, reference_name="Caffeine"),
            _match("s2", "rB", score=1.0, inchikey=_IK_B, reference_name="Caffeine"),
        ]
    )
    out = propagate_annotations(edges, matches)
    assert out.filter(pl.col("evidence_source") == "network").height == 2


def test_skeleton_inchikey_is_not_used_as_identity() -> None:
    """A 14-char skeleton key is not stereochemically complete, so two library
    entries sharing it must NOT be collapsed into one candidate."""
    skeleton = _IK_A.split("-")[0]  # 14-char first block (stereochemistry lost)
    edges = _edges([("s1", "centre", 0.9), ("s2", "centre", 0.9)])
    matches = _matches(
        [
            _match("s1", "rA1", score=1.0, inchikey=skeleton),
            _match("s2", "rA2", score=1.0, inchikey=skeleton),
        ]
    )
    out = propagate_annotations(edges, matches)
    network_rows = out.filter(pl.col("evidence_source") == "network")
    assert network_rows.height == 2
    assert set(network_rows["candidate_id"].to_list()) == {"rA1", "rA2"}


def test_network_row_reference_id_is_the_library_id_not_the_inchikey() -> None:
    edges = _edges([("s", "centre", 0.9)])
    matches = _matches([_match("s", "rA", score=0.9, inchikey=_IK_A)])
    out = propagate_annotations(edges, matches)
    row = _find(out, query_id="centre", evidence_source="network")
    assert row["candidate_id"] == _IK_A
    assert row["reference_id"] == "rA"  # library entry id, not the InChIKey


def test_network_row_carries_candidate_precursor_mz_and_smiles() -> None:
    edges = _edges([("s", "centre", 0.9)])
    matches = _matches([_match("s", "rA", score=0.9, inchikey=_IK_A)]).with_columns(
        pl.lit(180.0634).alias("reference_precursor_mz"),
        pl.lit("CC(=O)OC1=CC=CC=C1C(=O)O").alias("smiles"),
    )
    out = propagate_annotations(edges, matches)
    row = _find(out, query_id="centre", evidence_source="network")
    assert row["reference_precursor_mz"] == pytest.approx(180.0634)
    assert row["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"


# ---------------------------------------------------------------------------
# Schema stability and preservation of original values
# ---------------------------------------------------------------------------


def test_output_schema_is_stable() -> None:
    edges = _edges([("n1", "n2", 0.8)])
    matches = _matches([_match("n1", "r1", inchikey=_IK_A)])
    out = propagate_annotations(edges, matches)
    assert out.columns == list(PROPAGATION_OUTPUT_SCHEMA)
    assert out.schema["network_support_score"] == pl.Float64
    assert out.schema["supporting_neighbor_count"] == pl.UInt32
    assert out.schema["supporting_seed_ids"] == pl.List(pl.Utf8)
    assert out.schema["network_rank"] == pl.UInt32


def test_direct_values_are_preserved_verbatim() -> None:
    edges = _edges([("n1", "n2", 0.8)])
    matches = _matches(
        [
            _match(
                "n1",
                "r1",
                score=0.9123,
                q_value=0.004,
                p_value=0.123,
                inchikey=_IK_A,
                reference_name="Aspirin",
                matched_peaks=7,
            )
        ]
    )
    out = propagate_annotations(edges, matches)
    direct = _find(out, query_id="n1", evidence_source="direct")
    original = matches.to_dicts()[0]
    for column in ("score", "q_value", "p_value", "matched_peaks", "reference_name"):
        assert direct[column] == original[column]
    assert direct["q_value"] == pytest.approx(0.004)
    assert direct["p_value"] == pytest.approx(0.123)
    assert direct["is_decoy"] is False


def test_q_and_p_values_are_never_altered_by_network_evidence() -> None:
    edges = _edges([("s", "centre", 0.9), ("s", "other", 0.9)])
    matches = _matches(
        [_match("s", "rA", score=0.9, q_value=0.007, p_value=0.2, inchikey=_IK_A)]
    )
    out = propagate_annotations(edges, matches)
    # Network rows never carry statistical confidence.
    for row in out.filter(pl.col("evidence_source") == "network").to_dicts():
        assert row["q_value"] is None
        assert row["p_value"] is None
        assert row["score"] is None
    # The direct row's statistics are unchanged.
    direct = _find(out, query_id="s", evidence_source="direct")
    assert direct["q_value"] == pytest.approx(0.007)
    assert direct["p_value"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Determinism, parity, and end-to-end scenario
# ---------------------------------------------------------------------------


def test_propagation_is_deterministic_and_order_independent() -> None:
    edges = _edges([("a", "b", 0.9), ("b", "c", 0.8), ("a", "c", 0.4)])
    matches = _matches(
        [
            _match("a", "rA", score=0.9, inchikey=_IK_A),
            _match("c", "rB", score=0.9, inchikey=_IK_B),
        ]
    )
    first = propagate_annotations(edges, matches)
    shuffled_edges = _edges([("a", "c", 0.4), ("b", "c", 0.8), ("b", "a", 0.9)])
    shuffled_matches = _matches(list(reversed(matches.to_dicts())))
    second = propagate_annotations(shuffled_edges, shuffled_matches)
    assert first.equals(second)


def test_reference_implementation_parity() -> None:
    """Compare the SciPy-CSR backend against a naive pure-Python reference."""
    edges = _edges(
        [
            ("a", "b", 0.9),
            ("b", "c", 0.7),
            ("c", "d", 0.6),
            ("d", "a", 0.5),
            ("b", "d", 0.3),
        ]
    )
    matches = _matches(
        [
            _match("a", "rA", score=0.9, inchikey=_IK_A),
            _match("c", "rB", score=0.8, inchikey=_IK_B),
        ]
    )
    out = propagate_annotations(edges, matches)

    # Naive reference: sum edge_weight * seed_score per neighbour, per node.
    seeds: dict[str, tuple[str, float]] = {"a": (_IK_A, 0.9), "c": (_IK_B, 0.8)}
    supports: dict[tuple[str, str], float] = {}
    for left, right, weight in edges.iter_rows():
        for node, neighbour in ((left, right), (right, left)):
            if neighbour in seeds:
                candidate, score = seeds[neighbour]
                key = (node, candidate)
                supports[key] = supports.get(key, 0.0) + weight * score

    for key, value in supports.items():
        row = _find(out, query_id=key[0], candidate_id=key[1])
        assert row["network_support_score"] == pytest.approx(value)


def test_end_to_end_chain_scenario() -> None:
    # a (seed A) -- b -- c (seed B); b is genuinely contested -> conflict.
    edges = _edges([("a", "b", 0.6), ("b", "c", 0.6)])
    matches = _matches(
        [
            _match("a", "rA", score=1.0, inchikey=_IK_A),
            _match("c", "rB", score=1.0, inchikey=_IK_B),
        ]
    )
    out = propagate_annotations(edges, matches)
    b_rows = out.filter(
        (pl.col("query_id") == "b") & (pl.col("evidence_source") == "network")
    )
    assert b_rows.height == 2
    assert b_rows["network_conflict"].all()
