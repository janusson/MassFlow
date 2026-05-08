from MassFlow.consensus import ConsensusEngine
from MassFlow.models import AnnotationHit, ConsensusConfig, ConsensusInput


def test_consensus_flagged_for_review_rank_discrepancy():
    # Setup hits where the top hit of Engine A is ranked poorly in Engine B
    hits = [
        # Engine A (cosine) Top Hit
        AnnotationHit(engine_id="cosine", reference_id="Ref_A", score=0.99, rank=1),
        # Engine A (cosine) Second Hit
        AnnotationHit(engine_id="cosine", reference_id="Ref_B", score=0.80, rank=2),
        # Engine B (neural_net) ranks Ref_B highly, but ranks Ref_A very poorly (discrepancy)
        AnnotationHit(engine_id="neural_net", reference_id="Ref_B", score=0.98, rank=1),
        AnnotationHit(
            engine_id="neural_net", reference_id="Ref_A", score=0.10, rank=10
        ),
    ]

    # Configure consensus engine with a discrepancy threshold of 5
    config = ConsensusConfig(
        engine_weights={"cosine": 0.5, "neural_net": 0.5},
        flag_rank_discrepancy_threshold=5,
        tie_breaker_strategy="highest_rank",
    )
    engine = ConsensusEngine(config)
    c_input = ConsensusInput(query_id="query_1", hits=hits)

    # Resolve
    result = engine.resolve(c_input)

    # Assert it was flagged because cosine's #1 (Ref_A) was ranked 10th by neural_net (10 > 5)
    assert result.flagged_for_review is True
    assert "Discordance:" in result.review_reason
    assert "was ranked #10 by neural_net" in result.review_reason


def test_consensus_not_flagged_for_review():
    # Setup hits where engines agree closely
    hits = [
        AnnotationHit(engine_id="cosine", reference_id="Ref_A", score=0.99, rank=1),
        AnnotationHit(engine_id="cosine", reference_id="Ref_B", score=0.95, rank=2),
        AnnotationHit(engine_id="neural_net", reference_id="Ref_B", score=0.98, rank=1),
        AnnotationHit(engine_id="neural_net", reference_id="Ref_A", score=0.90, rank=2),
    ]

    config = ConsensusConfig(
        engine_weights={"cosine": 0.5, "neural_net": 0.5},
        flag_rank_discrepancy_threshold=5,
        tie_breaker_strategy="highest_rank",
    )
    engine = ConsensusEngine(config)
    c_input = ConsensusInput(query_id="query_2", hits=hits)

    result = engine.resolve(c_input)

    # Ranks are 1 and 2 across engines, well below the threshold of 5
    assert result.flagged_for_review is False
    assert result.review_reason is None


def test_consensus_tie_breaker_highest_rank():
    # Setup an exact tie in consensus score
    hits = [
        # Ref_A: 0.9 * 0.5 + 0.5 * 0.5 = 0.7
        AnnotationHit(engine_id="cosine", reference_id="Ref_A", score=0.9, rank=1),
        AnnotationHit(engine_id="neural_net", reference_id="Ref_A", score=0.5, rank=5),
        # Ref_B: 0.5 * 0.5 + 0.9 * 0.5 = 0.7
        AnnotationHit(engine_id="cosine", reference_id="Ref_B", score=0.5, rank=2),
        AnnotationHit(engine_id="neural_net", reference_id="Ref_B", score=0.9, rank=1),
    ]
    # In highest_rank tie-breaker:
    # Ref_A sum of ranks = 1 + 5 = 6
    # Ref_B sum of ranks = 2 + 1 = 3
    # Ref_B should win the tie because its sum of ranks is lower (better).

    config = ConsensusConfig(
        engine_weights={"cosine": 0.5, "neural_net": 0.5},
        tie_breaker_strategy="highest_rank",
    )
    engine = ConsensusEngine(config)
    c_input = ConsensusInput(query_id="query_3", hits=hits)

    result = engine.resolve(c_input)

    assert result.best_reference_id == "Ref_B"
    assert result.best_consensus_score == 0.7


def test_consensus_flagged_missing_hit():
    # Setup hits where the top hit of Engine A is entirely missing from Engine B
    hits = [
        AnnotationHit(engine_id="cosine", reference_id="Ref_A", score=0.99, rank=1),
        AnnotationHit(engine_id="neural_net", reference_id="Ref_B", score=0.98, rank=1),
        # Ref_A is missing from neural_net entirely
    ]

    config = ConsensusConfig(
        engine_weights={"cosine": 0.5, "neural_net": 0.5},
        flag_rank_discrepancy_threshold=5,
        tie_breaker_strategy="highest_rank",
    )
    engine = ConsensusEngine(config)
    c_input = ConsensusInput(query_id="query_4", hits=hits)

    result = engine.resolve(c_input)

    assert result.flagged_for_review is True
    assert "was completely unranked by" in result.review_reason
