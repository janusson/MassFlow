"""Unit tests for _ms1_prefilter in MassFlow.similarity

Covers both fixed Da tolerance and ppm resolution paths and verifies that
missing precursor_mz values bypass the filter (i.e., entries are included).
"""

import math
import numpy as np
from matchms import Spectrum

from MassFlow.similarity import _ms1_prefilter


def make_mock_spectrum(spectrum_id: str, precursor_mz: float) -> Spectrum:
    """
    Creates a mock matchms.Spectrum object for MS1 pre-filter testing.

    Parameters
    ----------
    spectrum_id : str
        The unique identifier for the spectrum.
    precursor_mz : float
        The precursor m/z value. Missing values should be represented with np.nan.

    Returns
    -------
    Spectrum
        A matchms.Spectrum object with np.float64 precision arrays for mz and intensities.

    Examples
    --------
    >>> spec = make_mock_spectrum("q1", 100.0)
    >>> print(spec.get("precursor_mz"))
    100.0
    """
    return Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata={"id": spectrum_id, "precursor_mz": precursor_mz},
    )


def test_ms1_prefilter_da_tolerance():
    refs = [
        make_mock_spectrum("r1", 100.0),
        make_mock_spectrum("r2", 100.04),
        make_mock_spectrum("r3", 100.06),
    ]
    queries = [
        make_mock_spectrum("q1", 100.0),
    ]

    # Da Tolerance path: 0.05 Da
    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.05, resolution_ppm=None)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    # Matches:
    # q1 (100.0) matches r1 (100.0) - diff 0.0
    # q1 (100.0) matches r2 (100.04) - diff 0.04
    # q1 (100.0) does NOT match r3 (100.06) - diff 0.06

    matched_ref_indices = {r for r, c in pairs if c == 0}
    assert matched_ref_indices == {0, 1}


def test_ms1_prefilter_ppm_tolerance():
    refs = [
        make_mock_spectrum("r1", 100.0),
        make_mock_spectrum("r2", 100.0004),
        make_mock_spectrum("r3", 100.0006),
    ]
    queries = [
        make_mock_spectrum("q1", 100.0),
    ]

    # PPM Tolerance path: 5.0 ppm
    # ppm error = (|100.0 - ref| / 100.0) * 1e6 <= 5.0
    # which implies |100.0 - ref| <= 0.0005 Da

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.0, resolution_ppm=5.0)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    matched_ref_indices = {r for r, c in pairs if c == 0}
    assert matched_ref_indices == {0, 1}


def test_ms1_prefilter_missing_precursor_bypass():
    # Use np.nan for missing precursors based on MassFlow coding standards
    refs = [
        make_mock_spectrum("r1", 100.0),
        make_mock_spectrum("r2", np.nan),
    ]
    queries = [
        make_mock_spectrum("q1", 100.0),
        make_mock_spectrum("q2", np.nan),
    ]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.1, resolution_ppm=None)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    # q1 (100.0) should match r1 (100.0) and r2 (np.nan, which bypasses the filter)
    q1_rows = {r for r, c in pairs if c == 0}
    assert q1_rows == {0, 1}

    # q2 (np.nan) should bypass the filter and match both r1 and r2
    q2_rows = {r for r, c in pairs if c == 1}
    assert q2_rows == {0, 1}


def test_ms1_prefilter_boundary_conditions():
    # Evaluating matches exactly at the threshold cutoff limits.
    refs = [
        make_mock_spectrum("r1", 100.05),
        make_mock_spectrum("r2", 99.95),
        make_mock_spectrum("r3", 100.05001),
        make_mock_spectrum("r4", 99.94999),
    ]
    queries = [
        make_mock_spectrum("q1", 100.0),
    ]

    # Da Tolerance exactly 0.05
    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.05, resolution_ppm=None)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    matched_ref_indices = {r for r, c in pairs if c == 0}
    # Matches r1 and r2, but not r3 and r4
    assert matched_ref_indices == {0, 1}
