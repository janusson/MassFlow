"""
Tests for MassFlow processing module.
"""

from unittest.mock import patch

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import processing
from MassFlow.config import ProcessingConfig


@pytest.fixture
def processing_config():
    return ProcessingConfig()


def test_peak_processing_none(processing_config):
    assert processing.peak_processing(None, processing_config) is None


def test_peak_processing_filters_noise(processing_config):
    # Create spectrum with low intensity
    spectrum = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([10.0, 10.0], dtype="float"),
        metadata={"id": "noise_spec"},
    )
    # Set threshold higher than intensity
    processing_config.noise_threshold = 100.0

    result = processing.peak_processing(spectrum, processing_config)
    assert result is None


def test_peak_processing_filters_min_peaks(processing_config):
    # Create spectrum with few peaks
    spectrum = Spectrum(
        mz=np.array([100.0], dtype="float"),
        intensities=np.array([1000.0], dtype="float"),
        metadata={"id": "few_peaks_spec"},
    )
    # Set min peaks higher
    processing_config.min_peaks = 5
    # Ensure noise threshold doesn't kill it
    processing_config.noise_threshold = 0.0

    result = processing.peak_processing(spectrum, processing_config)
    assert result is None


def test_process_spectra_exception_handling(processing_config):
    # Mock metadata_processing to raise exception
    spectra = [
        Spectrum(
            mz=np.array([100.0], dtype="float"),
            intensities=np.array([100.0], dtype="float"),
            metadata={"id": "bad"},
        ),
        Spectrum(
            mz=np.array([200.0], dtype="float"),
            intensities=np.array([200.0], dtype="float"),
            metadata={"id": "good"},
        ),
    ]

    with patch("MassFlow.processing.metadata_processing") as mock_meta:
        # First call raises, second returns spectrum
        def side_effect(spec, conf):
            if spec.get("id") == "bad":
                raise ValueError("Processing Error")
            return spec

        mock_meta.side_effect = side_effect

        # We need to make sure peak_processing also passes the good one
        processing_config.min_peaks = 1
        processing_config.noise_threshold = 0.0

        results = list(processing.process_spectra(spectra, processing_config))

        assert len(results) == 1
        assert results[0].get("id") == "good"


def test_process_spectra_drops_none(processing_config):
    spectra = [
        None,
        Spectrum(
            mz=np.array([100.0], dtype="float"),
            intensities=np.array([100.0], dtype="float"),
            metadata={"id": "good"},
        ),
    ]
    processing_config.min_peaks = 1
    processing_config.noise_threshold = 0.0

    results = list(processing.process_spectra(spectra, processing_config))
    assert len(results) == 1


@pytest.fixture
def mock_spectrum():
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([10.0, 50.0, 100.0], dtype="float"),
        metadata={
            "name": "test_compound",
            "inchikey": "ABC",
            "precursor_mz": 100.0,
            "adduct": "[M+H]+",
        },
    )


@pytest.fixture
def noisy_spectrum():
    # Includes low intensity peaks that should be filtered
    return Spectrum(
        mz=np.array([50.0, 100.0, 200.0, 300.0, 1500.0], dtype="float"),
        intensities=np.array([1.0, 10.0, 50.0, 100.0, 50.0], dtype="float"),
        metadata={"name": "noise_compound"},
    )


def test_metadata_processing_valid(mock_spectrum):
    """Test that metadata cleaning works for valid input."""
    processed = processing.metadata_processing(mock_spectrum)
    assert processed is not None
    # Adduct derivation might add 'charge'
    assert processed.get("charge") == 1


def test_metadata_processing_with_config(mock_spectrum):
    """Test metadata injection from config."""
    config = ProcessingConfig(instrument="Orbitrap", mode="negative")
    processed = processing.metadata_processing(mock_spectrum, config)
    assert processed.get("instrument") == "Orbitrap"
    assert processed.get("ionmode") == "negative"


def test_metadata_processing_derivations():
    """Test that charge and ionmode are derived correctly."""
    spec = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([100.0]),
        metadata={"name": "Compound [M+H]+", "adduct": "[M+H]+"},
    )
    processed = processing.metadata_processing(spec)
    assert processed.get("ionmode") == "positive"
    assert processed.get("charge") == 1


def test_metadata_processing_none():
    """Test that None input returns None."""
    assert processing.metadata_processing(None) is None


def test_peak_processing_noise_threshold(noisy_spectrum):
    """Test peak filtering with explicit noise_threshold."""
    # intensities: 1.0, 10.0, 50.0, 100.0, 50.0
    # Set threshold to 15.0
    config = ProcessingConfig(noise_threshold=15.0, min_peaks=1, mz_max=2000.0)
    processed = processing.peak_processing(noisy_spectrum, config)

    # Should keep 50.0, 100.0, 50.0 (indices 2, 3, 4) -> m/z 200, 300, 1500
    expected_mzs = [200.0, 300.0, 1500.0]
    assert np.allclose(processed.peaks.mz, expected_mzs)


def test_peak_processing_fallback_min_intensity(noisy_spectrum):
    """Test peak filtering falling back to min_intensity if noise_threshold is 0."""
    # intensities: 1.0, 10.0, 50.0, 100.0, 50.0
    # Set noise_threshold to 0, min_intensity to 40.0
    config = ProcessingConfig(
        noise_threshold=0.0, min_intensity=40.0, min_peaks=1, mz_max=2000.0
    )
    processed = processing.peak_processing(noisy_spectrum, config)

    # Should keep 50.0, 100.0, 50.0
    expected_mzs = [200.0, 300.0, 1500.0]
    assert np.allclose(processed.peaks.mz, expected_mzs)


def test_process_spectra(mock_spectrum):
    """Test process_spectra generator."""
    spectra_in = [mock_spectrum, None, mock_spectrum]
    config = ProcessingConfig(noise_threshold=0.0, min_peaks=1)

    results = list(processing.process_spectra(spectra_in, config))
    assert len(results) == 2


def test_peak_processing_mz_range():
    """Test peak filtering by mz_min and mz_max."""
    spectrum = Spectrum(
        mz=np.array([50.0, 100.0, 500.0, 1200.0], dtype="float"),
        intensities=np.array([100.0, 100.0, 100.0, 100.0], dtype="float"),
        metadata={"id": "range_spec"},
    )
    config = ProcessingConfig(
        mz_min=75.0, mz_max=1000.0, noise_threshold=0.0, min_peaks=1
    )
    processed = processing.peak_processing(spectrum, config)
    assert processed is not None
    # Should keep 100.0 and 500.0
    expected_mzs = [100.0, 500.0]
    assert np.allclose(processed.peaks.mz, expected_mzs)


def test_peak_processing_normalize_intensity():
    """Test peak intensity normalization."""
    spectrum = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([500.0, 1000.0], dtype="float"),
        metadata={"id": "norm_spec"},
    )
    # With normalization
    config_norm = ProcessingConfig(
        normalize_intensity=True, noise_threshold=0.0, min_peaks=1
    )
    processed_norm = processing.peak_processing(spectrum.clone(), config_norm)
    assert processed_norm is not None
    assert np.max(processed_norm.peaks.intensities) == 1.0
    assert np.allclose(processed_norm.peaks.intensities, [0.5, 1.0])

    # Without normalization
    config_no_norm = ProcessingConfig(
        normalize_intensity=False, noise_threshold=0.0, min_peaks=1
    )
    processed_no_norm = processing.peak_processing(spectrum.clone(), config_no_norm)
    assert processed_no_norm is not None
    assert np.max(processed_no_norm.peaks.intensities) == 1000.0


def test_metadata_processing_toggles_disabled():
    """Test that when metadata toggles are disabled, the filters are bypassed."""
    # A spectrum with uncleaned names and list-based RT
    spec = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([100.0]),
        metadata={
            "name": "   MESSY NAME [M+H]+  ",
            "retention_time": [0.017],  # mzML style list
            "adduct": "[M+H]+",
        },
    )

    config = ProcessingConfig(
        clean_metadata=False,
        add_retention_time=False,
        clean_compound_name=False,
        derive_adduct_from_name=False,
        derive_formula_from_name=False,
        derive_ionmode=False,
        make_charge_int=False,
        repair_inchi_inchikey_smiles=False,
    )

    processed = processing.metadata_processing(spec, config)

    # Assertions to ensure it was NOT cleaned
    assert processed.get("compound_name") == "   MESSY NAME [M+H]+  "
    assert processed.get("charge") is None  # Never derived


def test_peak_processing_toggles_disabled():
    """Test that when peak toggles are disabled, the spectrum remains untouched."""
    spec = Spectrum(
        mz=np.array([10.0, 100.0, 2000.0]),
        intensities=np.array([1.0, 10.0, 100.0]),
        metadata={"id": "test"},
    )

    config = ProcessingConfig(
        filter_by_intensity=False,
        noise_threshold=50.0,  # Should be ignored
        filter_min_peaks=False,
        min_peaks=10,  # Should be ignored
        filter_by_mz=False,
        mz_max=1000.0,  # Should be ignored
        reduce_to_top_n_peaks=False,
        n_max=1,  # Should be ignored
        normalize_intensity=False,
    )

    processed = processing.peak_processing(spec, config)

    # Assertions to ensure peaks were untouched
    assert len(processed.peaks.mz) == 3
    assert np.allclose(processed.peaks.mz, [10.0, 100.0, 2000.0])
    assert np.allclose(processed.peaks.intensities, [1.0, 10.0, 100.0])
