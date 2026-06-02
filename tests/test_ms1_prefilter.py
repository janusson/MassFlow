import numpy as np
import pytest
from matchms import Spectrum
from src.MassFlow.similarity import _ms1_prefilter

def create_mock_spectrum(precursor_mz: float | None = None) -> Spectrum:
    """
    Create a minimal mock spectrum for MS1 pre-filtering testing.

    Parameters
    ----------
    precursor_mz : float | None, optional
        The precursor m/z value for the spectrum. If None, the metadata
        dictionary will be empty (missing precursor). If np.nan, the
        precursor will be set to np.nan explicitly.

    Returns
    -------
    matchms.Spectrum
        A matchms Spectrum object with dummy peaks and the specified
        precursor m/z in its metadata.

    Examples
    --------
    >>> spec = create_mock_spectrum(100.5)
    >>> print(spec.get("precursor_mz"))
    100.5
    """
    spectrum_peaks_mz = np.array([10.0, 20.0], dtype=np.float64)
    spectrum_peaks_intensities = np.array([100.0, 50.0], dtype=np.float64)

    metadata = {}
    if precursor_mz is not None:
        metadata["precursor_mz"] = precursor_mz

    return Spectrum(
        mz=spectrum_peaks_mz,
        intensities=spectrum_peaks_intensities,
        metadata=metadata
    )


def test_ms1_prefilter_da_tolerance():
    """
    Test the Da tolerance path of the MS1 pre-filter.
    Verifies that absolute Dalton windows are calculated and filtered correctly,
    including exact boundary conditions.
    """
    query_spectra = [
        create_mock_spectrum(100.0),
        create_mock_spectrum(200.0),
    ]

    reference_spectra = [
        create_mock_spectrum(100.0),      # Exact match for query 0
        create_mock_spectrum(100.02),     # Boundary match for query 0
        create_mock_spectrum(100.020001), # Just outside boundary for query 0
        create_mock_spectrum(99.98),      # Boundary match for query 0
        create_mock_spectrum(99.979999),  # Just outside boundary for query 0
        create_mock_spectrum(200.01),     # Within boundary for query 1
    ]

    ms1_tolerance = 0.02

    idx_row, idx_col = _ms1_prefilter(
        reference_spectra,
        query_spectra,
        ms1_tolerance=ms1_tolerance,
        resolution_ppm=None
    )

    # query 0 should match ref 0, 1, 3
    # query 1 should match ref 5
    expected_matches = {
        (0, 0),
        (1, 0),
        (3, 0),
        (5, 1)
    }

    actual_matches = set(zip(idx_row.tolist(), idx_col.tolist()))

    assert actual_matches == expected_matches, f"Expected {expected_matches}, got {actual_matches}"

def test_ms1_prefilter_ppm_tolerance():
    """
    Test the PPM tolerance path of the MS1 pre-filter.
    Verifies that relative mass errors are calculated and filtered correctly,
    including exact boundary conditions.
    """
    # 5.0 ppm of 100.0 is 0.0005
    query_spectra = [
        create_mock_spectrum(100.0),
    ]

    reference_spectra = [
        create_mock_spectrum(100.0),          # Exact match
        create_mock_spectrum(100.0005),       # Upper boundary match
        create_mock_spectrum(100.0005001),    # Just outside upper boundary
        create_mock_spectrum(99.9995),        # Lower boundary match
        create_mock_spectrum(99.9994999),     # Just outside lower boundary
    ]

    ms1_tolerance = 0.02
    resolution_ppm = 5.0

    idx_row, idx_col = _ms1_prefilter(
        reference_spectra,
        query_spectra,
        ms1_tolerance=ms1_tolerance,
        resolution_ppm=resolution_ppm
    )

    # query 0 should match ref 0, 1, 3
    expected_matches = {
        (0, 0),
        (1, 0),
        (3, 0)
    }

    actual_matches = set(zip(idx_row.tolist(), idx_col.tolist()))

    assert actual_matches == expected_matches, f"Expected {expected_matches}, got {actual_matches}"

def test_ms1_prefilter_missing_precursor():
    """
    Test that spectra with missing or np.nan precursors correctly bypass the MS1 pre-filter.
    When a reference or query spectrum is missing a precursor, it should be compared
    against all possible counterparts.
    """
    query_spectra = [
        create_mock_spectrum(100.0),   # query 0
        create_mock_spectrum(None),    # query 1 (missing precursor entirely)
        create_mock_spectrum(np.nan),  # query 2 (nan precursor)
    ]

    reference_spectra = [
        create_mock_spectrum(100.0),   # ref 0
        create_mock_spectrum(200.0),   # ref 1
        create_mock_spectrum(None),    # ref 2 (missing)
        create_mock_spectrum(np.nan),  # ref 3 (nan)
    ]

    ms1_tolerance = 0.02

    idx_row, idx_col = _ms1_prefilter(
        reference_spectra,
        query_spectra,
        ms1_tolerance=ms1_tolerance,
        resolution_ppm=None
    )

    # Expected behavior:
    # Query 0 (100.0) matches Ref 0 (100.0), Ref 2 (missing), Ref 3 (nan)
    # Query 1 (missing) matches Ref 0, Ref 1, Ref 2, Ref 3
    # Query 2 (nan) matches Ref 0, Ref 1, Ref 2, Ref 3
    # Ref 2 (missing) matches Query 0, Query 1, Query 2
    # Ref 3 (nan) matches Query 0, Query 1, Query 2

    expected_matches = {
        # Query 0 matches
        (0, 0),
        (2, 0),
        (3, 0),

        # Query 1 matches (missing precursor compares to all refs)
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),

        # Query 2 matches (nan precursor compares to all refs)
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2)

        # Ref 2 matches (missing precursor compares to all queries)
        # Note: (2, 0), (2, 1), (2, 2) covered above

        # Ref 3 matches (nan precursor compares to all queries)
    }

    actual_matches = set(zip(idx_row.tolist(), idx_col.tolist()))

    assert actual_matches == expected_matches, f"Expected {expected_matches}, got {actual_matches}"

def test_ms1_prefilter_ppm_missing_precursor():
    """
    Test that spectra with missing or np.nan precursors correctly bypass the MS1 pre-filter
    when using ppm tolerance.
    """
    query_spectra = [
        create_mock_spectrum(100.0),   # query 0
        create_mock_spectrum(None),    # query 1
    ]

    reference_spectra = [
        create_mock_spectrum(100.0),   # ref 0
        create_mock_spectrum(200.0),   # ref 1
        create_mock_spectrum(np.nan),  # ref 2
    ]

    ms1_tolerance = 0.02

    idx_row, idx_col = _ms1_prefilter(
        reference_spectra,
        query_spectra,
        ms1_tolerance=ms1_tolerance,
        resolution_ppm=5.0
    )

    # Expected behavior:
    # Query 0 matches Ref 0, Ref 2
    # Query 1 matches Ref 0, Ref 1, Ref 2

    expected_matches = {
        (0, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1)
    }

    actual_matches = set(zip(idx_row.tolist(), idx_col.tolist()))

    assert actual_matches == expected_matches, f"Expected {expected_matches}, got {actual_matches}"
