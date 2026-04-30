import pytest
from pydantic import ValidationError

from MassFlow.models import (
    AggregatedCandidate,
    AnnotationHit,
    ConsensusConfig,
    ConsensusInput,
    ConsensusResult,
)


def test_annotation_hit_creation():
    hit = AnnotationHit(
        engine_id="cosine",
        reference_id="ref_01",
        score=0.95,
        rank=1,
        inchikey="AABBCC",
    )
    assert hit.engine_id == "cosine"
    assert hit.score == 0.95
    assert hit.rank == 1


def test_annotation_hit_missing_fields():
    with pytest.raises(ValidationError):
        AnnotationHit(engine_id="cosine")


def test_consensus_input_creation():
    hit = AnnotationHit(engine_id="cosine", reference_id="ref_01", score=0.9, rank=1)
    c_in = ConsensusInput(query_id="query_01", hits=[hit])
    assert c_in.query_id == "query_01"
    assert len(c_in.hits) == 1


def test_aggregated_candidate_creation():
    cand = AggregatedCandidate(reference_id="ref_01", inchikey="AABBCC", smiles=None)
    assert cand.consensus_score == 0.0
    cand.engine_scores["cosine"] = 0.9
    cand.engine_ranks["cosine"] = 1
    assert cand.engine_scores["cosine"] == 0.9


def test_consensus_result_creation():
    cand = AggregatedCandidate(
        reference_id="ref_01", inchikey=None, smiles=None, consensus_score=0.95
    )
    res = ConsensusResult(
        query_id="query_01",
        best_reference_id="ref_01",
        best_consensus_score=0.95,
        candidates=[cand],
    )
    assert res.query_id == "query_01"
    assert res.best_reference_id == "ref_01"
    assert len(res.candidates) == 1


def test_consensus_config_creation():
    cfg = ConsensusConfig(
        engine_weights={"cosine": 0.6, "ms2deepscore": 0.4},
        tie_breaker_strategy="highest_rank",
    )
    assert cfg.engine_weights["cosine"] == 0.6
    assert cfg.tie_breaker_strategy == "highest_rank"
