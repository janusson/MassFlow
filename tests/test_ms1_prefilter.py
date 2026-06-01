import math
import numpy as np
import pytest
from matchms import Spectrum
from MassFlow.similarity import _ms1_prefilter


def _create_mock_spectrum(precursor_mz: float | None = None, use_nan: bool = False, missing: bool = False) -> Spectrum:
    """
    Creates a mock matchms.Spectrum object for testing.

    Parameters
    ----------
    precursor_mz : float, optional
        The precursor m/z value to assign to the spectrum.
    use_nan : bool
        If True, forces the precursor_mz metadata field to be np.nan.
    missing : bool
        If True, omits precursor_mz entirely.

    Returns
    -------
    Spectrum
        A valid matchms Spectrum object with explicit float64 precision.

    Examples
    --------
    >>> spec = _create_mock_spectrum(precursor_mz=150.0)
    >>> print(spec.get("precursor_mz"))
    150.0
    """
    mz = np.array([100.0, 200.0], dtype=np.float64)
    intensities = np.array([10.0, 20.0], dtype=np.float64)
    metadata = {}
    if use_nan:
        metadata["precursor_mz"] = np.nan
    elif missing:
        pass
    elif precursor_mz is not None:
        metadata["precursor_mz"] = np.float64(precursor_mz)
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)


def test_ms1_prefilter_da_tolerance():
    """Test MS1 Da Tolerance (Absolute Mass Error)."""
    ref = _create_mock_spectrum(precursor_mz=200.0)

    q_match_exact = _create_mock_spectrum(precursor_mz=200.0)
    q_match_high = _create_mock_spectrum(precursor_mz=200.019)
    q_match_low = _create_mock_spectrum(precursor_mz=199.981)

    q_nomatch_high = _create_mock_spectrum(precursor_mz=200.021)
    q_nomatch_low = _create_mock_spectrum(precursor_mz=199.979)

    queries = [q_match_exact, q_match_high, q_match_low, q_nomatch_high, q_nomatch_low]

    idx_row, idx_col = _ms1_prefilter([ref], queries, ms1_tolerance=0.02, resolution_ppm=None)

    assert len(idx_row) == 3
    assert set(idx_col) == {0, 1, 2}
    assert all(row == 0 for row in idx_row)


def test_ms1_prefilter_ppm_tolerance():
    """Test MS1 PPM Tolerance (Relative Mass Error)."""
    # Mass error formula: (|m_exp - m_theo| / m_theo) * 10^6 <= 5.0
    # Where m_theo is the reference mass.

    ref = _create_mock_spectrum(precursor_mz=1000.0)

    q_match_exact = _create_mock_spectrum(precursor_mz=1000.0)

    # 5 ppm of 1000.0 is 0.005 Da
    # Query at 1000.0049
    q_match_high = _create_mock_spectrum(precursor_mz=1000.0049)
    # Query at 999.9951
    q_match_low = _create_mock_spectrum(precursor_mz=999.9951)

    # Query at 1000.0051 (Error > 5 ppm)
    q_nomatch_high = _create_mock_spectrum(precursor_mz=1000.0051)
    # Query at 999.9949 (Error > 5 ppm)
    q_nomatch_low = _create_mock_spectrum(precursor_mz=999.9949)

    queries = [q_match_exact, q_match_high, q_match_low, q_nomatch_high, q_nomatch_low]

    idx_row, idx_col = _ms1_prefilter([ref], queries, ms1_tolerance=0.02, resolution_ppm=5.0)

    assert len(idx_row) == 3
    assert set(idx_col) == {0, 1, 2}
    assert all(row == 0 for row in idx_row)


def test_ms1_prefilter_boundary_conditions():
    """Test boundary conditions exactly at the threshold cutoff limits."""
    # Da tolerance bound tests: exactly at cutoff
    ref_da_bound = _create_mock_spectrum(precursor_mz=200.0)

    q_da_boundary_high = _create_mock_spectrum(precursor_mz=200.02)
    q_da_boundary_low = _create_mock_spectrum(precursor_mz=199.98)

    idx_row_da, idx_col_da = _ms1_prefilter([ref_da_bound], [q_da_boundary_high, q_da_boundary_low], ms1_tolerance=0.02, resolution_ppm=None)
    assert set(idx_col_da) == {0, 1}

    # PPM bound tests: exactly at cutoff based on theoretical mass (1000.0).
    # Cutoffs should be exactly 999.995 and 1000.005.
    ref_ppm_bound = _create_mock_spectrum(precursor_mz=1000.0)

    q_ppm_boundary_high = _create_mock_spectrum(precursor_mz=1000.005)
    q_ppm_boundary_low = _create_mock_spectrum(precursor_mz=999.995)

    idx_row_ppm, idx_col_ppm = _ms1_prefilter([ref_ppm_bound], [q_ppm_boundary_high, q_ppm_boundary_low], ms1_tolerance=0.02, resolution_ppm=5.0)

    # Both boundaries exactly at 5ppm error should match
    assert len(idx_row_ppm) == 2
    assert set(idx_col_ppm) == {0, 1}


def test_ms1_prefilter_missing_precursors():
    """Test behavior with missing or NaN precursors."""
    # The prefilter must handle np.nan values safely according to constraints
    ref_valid = _create_mock_spectrum(precursor_mz=200.0)
    ref_nan = _create_mock_spectrum(use_nan=True)
    ref_missing = _create_mock_spectrum(missing=True)

    q_valid = _create_mock_spectrum(precursor_mz=200.0)
    q_nan = _create_mock_spectrum(use_nan=True)
    q_missing = _create_mock_spectrum(missing=True)

    refs = [ref_valid, ref_nan, ref_missing]
    queries = [q_valid, q_nan, q_missing]

    idx_row, idx_col = _ms1_prefilter(refs, queries, ms1_tolerance=0.02, resolution_ppm=None)

    matches = set(zip(idx_row, idx_col))

    assert (0, 0) in matches

    # ref_missing (None) bypasses and matches all queries
    assert (2, 0) in matches
    assert (2, 1) in matches
    assert (2, 2) in matches

    # q_missing (None) bypasses and matches all refs
    assert (0, 2) in matches
    assert (1, 2) in matches
    assert (2, 2) in matches

    # NaN values do not cause errors (safely handled), but they do not match valid m/z or each other
    assert (0, 1) not in matches
    assert (1, 0) not in matches
    assert (1, 1) not in matches
