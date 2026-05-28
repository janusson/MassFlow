"""Tests for SimilarityEngine factory and error behavior.

Covers:
- Factory returns correct engine classes for 'cosine', 'cascade', and 'consensus'
- ML-backed engines raise FileNotFoundError when model_path is missing
- Consensus fallback behavior when allow_consensus_fallback=False
"""

from pathlib import Path

import pytest

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import (
    SimilarityEngine,
    ConsensusEngine,
    CascadeEngine,
    get_similarity_engine,
)


def test_get_similarity_engine_types(monkeypatch):
    cfg1 = SimilarityConfig(algorithm="cosine")
    eng1 = get_similarity_engine(cfg1)
    assert isinstance(eng1, SimilarityEngine)

    # For CascadeEngine initialization, avoid loading ML backends by monkeypatching
    # the SimilarityEngine used inside CascadeEngine to a lightweight fake.
    class FakeSim:
        def __init__(self, config):
            self.config = config

        def search(self, *args, **kwargs):
            return []

    monkeypatch.setattr("MassFlow.similarity.SimilarityEngine", FakeSim)

    cfg2 = SimilarityConfig(algorithm="cascade", cascade_tier2="ms2deepscore")
    eng2 = get_similarity_engine(cfg2)
    assert isinstance(eng2, CascadeEngine)

    # Restore the real SimilarityEngine for consensus test
    monkeypatch.setattr("MassFlow.similarity.SimilarityEngine", SimilarityEngine)

    weights = {"cosine": 0.5, "modified_cosine": 0.5}
    cfg3 = SimilarityConfig(algorithm="consensus", consensus_weights=weights)
    eng3 = get_similarity_engine(cfg3)
    assert isinstance(eng3, ConsensusEngine)


def test_ml_model_path_missing_raises():
    # Spec2Vec
    cfg_spec = SimilarityConfig(
        algorithm="spec2vec", model_path=Path("/no/such/model.bin")
    )
    with pytest.raises(FileNotFoundError):
        SimilarityEngine(cfg_spec)

    # MS2DeepScore
    cfg_ms2 = SimilarityConfig(
        algorithm="ms2deepscore", model_path=Path("/no/such/model.pt")
    )
    with pytest.raises(FileNotFoundError):
        SimilarityEngine(cfg_ms2)


def test_consensus_fallback_flag():
    # When fallback is disabled, require consensus_weights
    cfg_bad = SimilarityConfig(
        algorithm="consensus", consensus_weights=None, allow_consensus_fallback=False
    )
    with pytest.raises(ValueError):
        get_similarity_engine(cfg_bad)

    # When fallback is enabled, return a ConsensusEngine (single cosine fallback)
    cfg_ok = SimilarityConfig(
        algorithm="consensus", consensus_weights=None, allow_consensus_fallback=True
    )
    eng = get_similarity_engine(cfg_ok)
    assert isinstance(eng, ConsensusEngine)
