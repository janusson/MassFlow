
import pytest
import numpy as np
from matchms import Spectrum
from MassFlow import processing

@pytest.fixture
def mock_spectrum():
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([10.0, 50.0, 100.0], dtype="float"),
        metadata={
            "name": "test_compound",
            "inchikey": "ABC",
            "ionmode": "positive",
            "precursor_mz": 100.0
        }
    )

@pytest.fixture
def noisy_spectrum():
    # Includes low intensity peaks that should be filtered
    return Spectrum(
        mz=np.array([50.0, 100.0, 200.0, 300.0, 1500.0], dtype="float"),
        intensities=np.array([1.0, 10.0, 50.0, 100.0, 50.0], dtype="float"), 
        # 1.0 is 1% of max (100), so barely meets absolute 0.01 threshold but fails relative 0.08 (8%)
        metadata={"name": "noise_compound", "ionmode": "n/a"}
    )

def test_metadata_processing_valid(mock_spectrum):
    """Test that metadata cleaning works for valid input."""
    processed = processing.metadata_processing(mock_spectrum)
    assert processed is not None
    assert processed.get("ionmode") == "positive"
    # Adduct derivation might add 'charge'
    assert processed.get("charge") == 1

def test_metadata_processing_derivations():
    """Test that charge and ionmode are derived correctly."""
    spec = Spectrum(
        mz=np.array([100.0]), intensities=np.array([100.0]),
        metadata={"name": "Compound [M+H]+", "adduct": "[M+H]+"}
    )
    processed = processing.metadata_processing(spec)
    assert processed.get("ionmode") == "positive"
    assert processed.get("charge") == 1

def test_metadata_processing_none():
    """Test that None input returns None."""
    assert processing.metadata_processing(None) is None

def test_peak_processing_filtering(noisy_spectrum):
    """Test peak filtering logic (min intensity, relative intensity, mz range)."""
    processed = processing.peak_processing(noisy_spectrum)
    
    # Check MZ range (1500.0 > 1000 so should be removed if default filters applies, 
    # but select_by_mz is set to mz_to=1000 in code)
    assert 1500.0 not in processed.peaks.mz
    
    # Check relative intensity: max is 100. cutoff is 0.08 * 100 = 8.
    # 1.0 should be removed
    assert 50.0 not in processed.peaks.mz  # matched to intensity 1.0
    
    # 10.0 should be kept (>= 8)? Wait, select_by_relative_intensity uses fraction relative to max.
    # 10/100 = 0.1 > 0.08. So 100.0 m/z should be kept.
    assert 100.0 in processed.peaks.mz
    
    # Check filtering results
    assert len(processed.peaks.mz) == 3 # 100, 200, 300

def test_peak_processing_none():
    assert processing.peak_processing(None) is None
