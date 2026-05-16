import pytest

from MassFlow.consensus import ConsensusEngine
from MassFlow.models import (
    AnnotationHit,
    ConsensusConfig,
    ConsensusInput,
)


@pytest.fixture
def base_config():
    return ConsensusConfig(
        engine_weights={"cosine": 0.6, "ms2deepscore": 0.4},
        tie_breaker_strategy="highest_rank",
        flag_rank_discrepancy_threshold=5,
    )


def test_consensus_engine_init(base_config):
    engine = ConsensusEngine(base_config)
    assert engine.config == base_config


def test_consensus_engine_invalid_weight():
    invalid_config = ConsensusConfig(engine_weights={"cosine": 0.0})
    with pytest.raises(
        ValueError, match="sum of engine weights must be greater than 0"
    ):
        ConsensusEngine(invalid_config)


def test_consensus_resolve_empty(base_config):
    engine = ConsensusEngine(base_config)
    empty_input = ConsensusInput(query_id="query_01", hits=[])
    res = engine.resolve(empty_input)
    assert res.query_id == "query_01"
    assert res.best_reference_id is None


def test_consensus_resolve_single_candidate(base_config):
    engine = ConsensusEngine(base_config)
    hit1 = AnnotationHit(engine_id="cosine", reference_id="ref_01", score=0.9, rank=1)
    hit2 = AnnotationHit(
        engine_id="ms2deepscore", reference_id="ref_01", score=0.8, rank=1
    )
    c_in = ConsensusInput(query_id="query_01", hits=[hit1, hit2])

    res = engine.resolve(c_in)
    assert res.best_reference_id == "ref_01"
    # Weighted score: 0.9 * 0.6 + 0.8 * 0.4 = 0.54 + 0.32 = 0.86
    assert pytest.approx(res.best_consensus_score) == 0.86
    assert res.flagged_for_review is False


def test_consensus_tie_breaker_highest_rank(base_config):
    engine = ConsensusEngine(base_config)
    # Give them both identical weighted scores
    hit_a_1 = AnnotationHit(engine_id="cosine", reference_id="ref_A", score=0.5, rank=1)
    hit_a_2 = AnnotationHit(
        engine_id="ms2deepscore", reference_id="ref_A", score=0.5, rank=2
    )

    hit_b_1 = AnnotationHit(engine_id="cosine", reference_id="ref_B", score=0.5, rank=3)
    hit_b_2 = AnnotationHit(
        engine_id="ms2deepscore", reference_id="ref_B", score=0.5, rank=4
    )

    c_in = ConsensusInput(
        query_id="query_tie", hits=[hit_a_1, hit_a_2, hit_b_1, hit_b_2]
    )
    res = engine.resolve(c_in)

    # A has sum of ranks = 3, B has sum of ranks = 7. A should win via highest_rank tie-breaker.
    assert res.best_reference_id == "ref_A"


def test_consensus_tie_breaker_average_score():
    cfg = ConsensusConfig(
        engine_weights={"e1": 0.5, "e2": 0.5}, tie_breaker_strategy="average_score"
    )
    engine = ConsensusEngine(cfg)

    # Tie on weighted score (both are 0.5)
    # A has scores: 0.5, 0.5 -> avg 0.5
    # B has scores: 1.0 (from e1), missing from e2 (weighted score 0.5, but avg is 1.0)
    hit_a_1 = AnnotationHit(engine_id="e1", reference_id="A", score=0.5, rank=1)
    hit_a_2 = AnnotationHit(engine_id="e2", reference_id="A", score=0.5, rank=1)

    hit_b_1 = AnnotationHit(engine_id="e1", reference_id="B", score=1.0, rank=2)

    c_in = ConsensusInput(query_id="q1", hits=[hit_a_1, hit_a_2, hit_b_1])
    res = engine.resolve(c_in)

    assert res.best_reference_id == "B"


def test_consensus_tie_breaker_validator_engine():
    cfg = ConsensusConfig(
        engine_weights={"cosine": 0.5, "ml": 0.5},
        tie_breaker_strategy="validator_engine",
        validator_engine="ml",
    )
    engine = ConsensusEngine(cfg)

    hit_a_1 = AnnotationHit(engine_id="cosine", reference_id="A", score=0.8, rank=1)
    hit_a_2 = AnnotationHit(engine_id="ml", reference_id="A", score=0.2, rank=2)

    hit_b_1 = AnnotationHit(engine_id="cosine", reference_id="B", score=0.2, rank=2)
    hit_b_2 = AnnotationHit(engine_id="ml", reference_id="B", score=0.8, rank=1)

    c_in = ConsensusInput(query_id="q1", hits=[hit_a_1, hit_a_2, hit_b_1, hit_b_2])
    res = engine.resolve(c_in)

    # Tie in weighted score: A=(0.4+0.1)=0.5, B=(0.1+0.4)=0.5
    # Validator is ML. B has ML=0.8, A has ML=0.2. B should win.
    assert res.best_reference_id == "B"


def test_consensus_missing_validator_raises():
    cfg = ConsensusConfig(
        engine_weights={"e1": 1.0}, tie_breaker_strategy="validator_engine"
    )
    engine = ConsensusEngine(cfg)
    hit_a = AnnotationHit(engine_id="e1", reference_id="A", score=0.5, rank=1)
    hit_b = AnnotationHit(engine_id="e1", reference_id="B", score=0.5, rank=2)

    c_in = ConsensusInput(query_id="q1", hits=[hit_a, hit_b])
    with pytest.raises(ValueError, match="validator_engine must be set"):
        engine.resolve(c_in)


def test_check_scientific_credibility_unranked(base_config):
    engine = ConsensusEngine(base_config)
    hit1 = AnnotationHit(engine_id="cosine", reference_id="ref_01", score=0.9, rank=1)
    # ms2deepscore unranked ref_01
    hit2 = AnnotationHit(
        engine_id="ms2deepscore", reference_id="ref_02", score=0.9, rank=1
    )

    c_in = ConsensusInput(query_id="query_01", hits=[hit1, hit2])
    res = engine.resolve(c_in)

    assert res.flagged_for_review is True
    assert "completely unranked" in res.review_reason


def test_check_scientific_credibility_rank_threshold(base_config):
    engine = ConsensusEngine(base_config)
    hit1 = AnnotationHit(engine_id="cosine", reference_id="ref_01", score=0.9, rank=1)
    # ms2deepscore ranked ref_01 as 10th (threshold is 5)
    hit2 = AnnotationHit(
        engine_id="ms2deepscore", reference_id="ref_01", score=0.1, rank=10
    )

    c_in = ConsensusInput(query_id="query_01", hits=[hit1, hit2])
    res = engine.resolve(c_in)

    assert res.flagged_for_review is True
    assert "Threshold" in res.review_reason
