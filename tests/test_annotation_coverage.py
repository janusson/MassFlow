"""
Comprehensive test suite to achieve 100% coverage on core annotation logic.
"""

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import (
    SimilarityEngine,
    _ms1_prefilter,
    calculate_fdr,
    generate_decoys,
    get_similarity_engine,
)


def test_generate_decoys_uniqueness():
    # Test the branch where shuffling is ineffective
    spec = Spectrum(mz=np.array([100.0, 200.0]), intensities=np.array([1.0, 1.0]))
    decoy = generate_decoys([spec])[0]
    # np.roll is used, so it should be shifted
    assert not np.array_equal(spec.peaks.intensities, decoy.peaks.intensities)


def test_ms1_prefilter_no_precursor():
    # Test cases where query or reference spectra are missing precursor_mz
    query = Spectrum(mz=np.array([100.0]), intensities=np.array([1.0]), metadata={})
    ref = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"precursor_mz": 150.0},
    )

    # Query missing
    indices = _ms1_prefilter([ref], [query], ms1_tolerance=0.1)
    assert len(indices[0]) == 1  # Should match because filter is bypassed

    # Reference missing
    indices = _ms1_prefilter([query], [ref], ms1_tolerance=0.1)
    assert len(indices[0]) == 1  # Should match because filter is bypassed


def test_similarity_engine_ml_model_errors():
    # Test for FileNotFoundError on ML models when the file doesn't exist
    with pytest.raises(FileNotFoundError):
        SimilarityEngine(
            SimilarityConfig(algorithm="spec2vec", model_path="nonexistent.model")
        )

    with pytest.raises(ImportError):
        SimilarityEngine(
            SimilarityConfig(algorithm="ms2deepscore", model_path="nonexistent.model")
        )


def test_get_similarity_engine_fallback_and_cascade():
    # Test consensus fallback
    config = SimilarityConfig(algorithm="consensus", allow_consensus_fallback=True)
    engine = get_similarity_engine(config)
    from MassFlow.similarity import ConsensusEngine

    assert isinstance(
        engine, ConsensusEngine
    )  # Falls back to a single 'cosine' engine wrapped in a ConsensusEngine

    # Test unsupported algorithm
    with pytest.raises(ValueError):
        SimilarityEngine(SimilarityConfig(algorithm="unsupported"))


def test_fdr_edge_cases():
    # Empty inputs
    res = calculate_fdr(np.array([]), np.array([]))
    assert len(res[0]) == 0

    # Only decoys
    scores, qvals, is_target = calculate_fdr(np.array([]), np.array([0.5]))
    assert len(scores) == 1
    assert qvals[0] == 1.0
    assert not is_target[0]
