
import pytest
import numpy as np
import pandas as pd
from matchms import Spectrum
from MassFlow import similarity

@pytest.fixture
def spectrum_a():
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={"id": "A", "smiles": "CCC"}
    )

@pytest.fixture
def spectrum_b():
    # Identical to A
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={"id": "B", "smiles": "CCC"}
    )

@pytest.fixture
def spectrum_c():
    # Different
    return Spectrum(
        mz=np.array([500.0, 600.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "C", "smiles": "CCCl"}
    )

def test_calculate_cosscores_identical(spectrum_a, spectrum_b):
    scores = similarity.calculate_cosscores([spectrum_a], [spectrum_b])
    score_struct = scores.to_array()[0][0]
    
    # Should be perfect match
    assert score_struct['CosineGreedy_score'] > 0.99
    assert score_struct['CosineGreedy_matches'] == 3

def test_calculate_cosscores_different(spectrum_a, spectrum_c):
    scores = similarity.calculate_cosscores([spectrum_a], [spectrum_c])
    score_struct = scores.to_array()[0][0]
    
    # Should be zero match
    assert score_struct['CosineGreedy_score'] == 0.0
    assert score_struct['CosineGreedy_matches'] == 0

def test_top10_cosine_matches(spectrum_a, spectrum_b, spectrum_c):
    # Query A vs Lib [B, C]. A should match B perfectly.
    scores = similarity.top10_cosine_matches([spectrum_b, spectrum_c], [spectrum_a])
    
    best_matches = scores.scores_by_query(spectrum_a, "CosineGreedy_score", sort=True)
    top_match = best_matches[0]


    
    # Check that B is the top match
    assert top_match[0].metadata["id"] == "B"
    assert top_match[1]["CosineGreedy_score"] > 0.99
