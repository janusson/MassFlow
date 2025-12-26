'''
Tests for Yogimass core functions.
Verifies logic ported from original_source.
'''
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from matchms import Spectrum

from yogimass import processing, similarity, io

@pytest.fixture
def mock_spectrum():
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([10.0, 50.0, 100.0], dtype="float"),
        metadata={"name": "test_compound", "inchikey": "ABC", "ionmode": "positive", "precursor_mz": 100.0}
    )

def test_metadata_processing(mock_spectrum):
    """Test that metadata cleaning works."""
    processed = processing.metadata_processing(mock_spectrum)
    assert processed is not None
    assert processed.get("ionmode") == "positive"
    # Adduct derivation logic might add 'charge'
    assert processed.get("charge") == 1

def test_peak_processing(mock_spectrum):
    """Test peak filtering logic."""
    # Mock spectrum has max intensity 100.
    # relative intensity threshold 0.08 => 8.
    # absolute intensity threshold 0.01.
    # All peaks (10, 50, 100) should be kept since 10 > 8.
    
    # Let's create a noisy one
    noisy = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([5.0, 50.0, 100.0], dtype="float"), # 5.0 is 5% of 100, should be dropped (<8%)
        metadata={"name": "test_compound"}
    )
    processed = processing.peak_processing(noisy)
    assert len(processed.peaks.mz) == 2
    assert 100.0 not in processed.peaks.mz
    assert 200.0 in processed.peaks.mz

def test_similarity_scoring(mock_spectrum):
    """Test cosine similarity calculation."""
    # Self-match should be 1.0 (or close)
    # Process the spectrum first to ensure specific filtering/normalization expected by the scorer
    processed = processing.peak_processing(mock_spectrum)
    ref = [processed]
    query = [processed]
    
    scores = similarity.calculate_cosscores(ref, query)
    dense_scores = scores.to_array()
    
    # CosineGreedy returns structured array, access score by name
    score_data = dense_scores[0][0]
    assert score_data['CosineGreedy_score'] > 0.99
    assert score_data['CosineGreedy_matches'] == 3

def test_io_module_mocks():
    """Test that I/O paths are constructed correctly (mocking filesystem)."""
    with patch("glob.glob") as mock_glob:
        mock_glob.return_value = ["/path/to/lib.msp"]
        msps = io.list_msp_libraries("/path/to")
        assert len(msps) == 1
        assert msps[0] == "/path/to/lib.msp"
