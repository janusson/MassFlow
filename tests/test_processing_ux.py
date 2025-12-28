"""
Tests for UX improvements in processing module.
"""
import pytest
from unittest.mock import MagicMock, patch
from yogimass import processing

@pytest.fixture
def mock_spectra_list():
    return [MagicMock(), MagicMock(), MagicMock()]

@patch("yogimass.processing.metadata_processing")
@patch("yogimass.processing.peak_processing")
@patch("yogimass.processing.load_from_mgf")
@patch("yogimass.processing.tqdm")
def test_clean_mgf_uses_tqdm(mock_tqdm, mock_load, mock_peak, mock_meta, mock_spectra_list):
    """Test that tqdm is used when cleaning MGF libraries."""
    mock_load.return_value = mock_spectra_list
    # The iterator returned by tqdm should be the input list
    mock_tqdm.return_value = mock_spectra_list

    # Configure mock processors to just return the input
    mock_meta.side_effect = lambda x: x
    mock_peak.side_effect = lambda x: x

    processing.clean_mgf_library("dummy.mgf")

    # Verify tqdm was initialized with the list
    mock_tqdm.assert_called_once()
    args, kwargs = mock_tqdm.call_args
    assert args[0] == mock_spectra_list
    assert kwargs.get("desc") == "Cleaning MGF spectra"
    assert kwargs.get("unit") == "spectra"

@patch("yogimass.processing.metadata_processing")
@patch("yogimass.processing.peak_processing")
@patch("yogimass.processing.load_from_msp")
@patch("yogimass.processing.tqdm")
def test_clean_msp_uses_tqdm(mock_tqdm, mock_load, mock_peak, mock_meta, mock_spectra_list):
    """Test that tqdm is used when cleaning MSP libraries."""
    mock_load.return_value = mock_spectra_list
    mock_tqdm.return_value = mock_spectra_list

    mock_meta.side_effect = lambda x: x
    mock_peak.side_effect = lambda x: x

    processing.clean_msp_library("dummy.msp")

    mock_tqdm.assert_called_once()
    args, kwargs = mock_tqdm.call_args
    assert args[0] == mock_spectra_list
    assert kwargs.get("desc") == "Cleaning MSP spectra"
    assert kwargs.get("unit") == "spectra"
