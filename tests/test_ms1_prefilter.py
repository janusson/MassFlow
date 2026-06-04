"""Unit tests for _ms1_prefilter in MassFlow.similarity

Covers both fixed Da tolerance and ppm resolution paths and verifies that
missing precursor_mz values bypass the filter (i.e., entries are included).
"""

from typing import Optional
import numpy as np
from matchms import Spectrum

from MassFlow.similarity import _ms1_prefilter


def make_spec(spectrum_id: str, precursor_mz: float) -> Spectrum:
    """
    Create a mock matchms Spectrum object for testing.

    Parameters
    ----------
    spectrum_id : str
        The unique identifier for the spectrum.
    precursor_mz : float
        The precursor m/z value. Should be np.nan if undefined or missing.

    Returns
    -------
    Spectrum
        A matchms.Spectrum object initialized with explicit float64
        m/z and intensity arrays, and corresponding metadata.

    Examples
    --------
    >>> spec = make_spec("q1", 100.0)
    >>> print(spec.get("precursor_mz"))
    100.0
    """
    metadata = {
        "id": spectrum_id,
        "precursor_mz": precursor_mz
    }

    return Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata=metadata,
    )


def test_ms1_prefilter_da_tolerance_basic():
    refs = [make_spec("r1", 100.0), make_spec("r2", 101.0), make_spec("r3", 110.0)]
    queries = [make_spec("q1", 100.02), make_spec("q2", np.nan)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.05, resolution_ppm=None)

    pairs = list(zip(rows.tolist(), cols.tolist()))

    q1_rows = {r for r, c in pairs if c == 0}
    assert q1_rows == {0}

    q2_rows = {r for r, c in pairs if c == 1}
    assert q2_rows == {0, 1, 2}

def test_ms1_prefilter_ppm_resolution():
    refs = [make_spec("r1", 100.0), make_spec("r2", 100.2), make_spec("r3", 200.0)]
    queries = [make_spec("q1", 100.0), make_spec("q2", 100.001)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.02, resolution_ppm=10)

    matched_pairs = list(zip(rows.tolist(), cols.tolist()))

    q1_rows = {r for r, c in matched_pairs if c == 0}
    assert q1_rows == {0}

    q2_rows = {r for r, c in matched_pairs if c == 1}
    assert q2_rows == {0}

def test_ms1_prefilter_missing_precursor_bypass():
    refs = [make_spec("r1", 100.0), make_spec("r2", np.nan)]
    queries = [make_spec("q1", 100.0), make_spec("q2", np.nan)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.1, resolution_ppm=None)

    pairs = list(zip(rows.tolist(), cols.tolist()))

    q1_rows = {r for r, c in pairs if c == 0}
    assert q1_rows == {0, 1}

    q2_rows = {r for r, c in pairs if c == 1}
    assert q2_rows == {0, 1}

def test_ms1_prefilter_da_boundary():
    """Test boundary conditions for Da tolerance paths."""
    refs = [make_spec("r1", 100.0)]

    # 0.0625 Da tolerance boundaries (binary exact for floating point)
    queries = [
        make_spec("q1", 99.9375),    # exactly on lower bound (should match)
        make_spec("q2", 100.0625),   # exactly on upper bound (should match)
        make_spec("q3", 99.93749),   # just outside lower bound (should NOT match)
        make_spec("q4", 100.06251)   # just outside upper bound (should NOT match)
    ]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.0625, resolution_ppm=None)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    # Check that q1 (col 0) matches r1 (row 0)
    assert (0, 0) in pairs
    # Check that q2 (col 1) matches r1 (row 0)
    assert (0, 1) in pairs
    # Check that q3 and q4 do not match
    assert not any(c == 2 for r, c in pairs)
    assert not any(c == 3 for r, c in pairs)

def test_ms1_prefilter_ppm_boundary():
    """Test boundary conditions for PPM resolution paths."""
    refs = [make_spec("r1", 100.0)]

    # Formula used in _ms1_prefilter: ppm_tol_da = resolution_ppm * query_mz / 1e6
    # So |ref_mz - query_mz| <= ppm_tol_da  --> |100.0 - query_mz| <= 5.0 * query_mz / 1e6
    # For upper bound: query_mz - 5e-6 * query_mz = 100.0 => query_mz * (1 - 5e-6) = 100.0 => query_mz = 100.0 / 0.999995 = 100.0005000025
    # For lower bound: query_mz + 5e-6 * query_mz = 100.0 => query_mz * (1 + 5e-6) = 100.0 => query_mz = 100.0 / 1.000005 = 99.9995000025

    queries = [
        make_spec("q1", 100.0 / 1.000005),   # exactly on lower bound
        make_spec("q2", 100.0 / 0.999995),   # exactly on upper bound
        make_spec("q3", (100.0 / 1.000005) - 1e-6),  # just outside lower bound
        make_spec("q4", (100.0 / 0.999995) + 1e-6)   # just outside upper bound
    ]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.0, resolution_ppm=5.0)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    # Check that q1 (col 0) matches r1 (row 0)
    assert (0, 0) in pairs
    # Check that q2 (col 1) matches r1 (row 0)
    assert (0, 1) in pairs
    # Check that q3 and q4 do not match
    assert not any(c == 2 for r, c in pairs)
    assert not any(c == 3 for r, c in pairs)
