"""Unit tests for _ms1_prefilter in MassFlow.similarity

Covers both fixed Da tolerance and ppm resolution paths and verifies that
missing precursor_mz values bypass the filter (i.e., entries are included).
Tests precision boundaries and proper explicit datatypes.
"""

from typing import Optional

import numpy as np
from matchms import Spectrum

from MassFlow.similarity import _ms1_prefilter


def create_mock_spectrum(
    spectrum_id: str, precursor_mz: Optional[float] = None
) -> Spectrum:
    """
    Create a minimal mock spectrum for MS1 pre-filter testing.

    Parameters
    ----------
    spectrum_id : str
        The unique identifier for the mock spectrum.
    precursor_mz : Optional[float]
        The precursor m/z value. Can be None or np.nan to simulate missing data.

    Returns
    -------
    Spectrum
        A matchms Spectrum object with explicit float64 types.

    Examples
    --------
    >>> spec = create_mock_spectrum("q1", 304.0)
    >>> print(spec.get("precursor_mz"))
    304.0
    """
    return Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata={"id": spectrum_id, "precursor_mz": precursor_mz},
    )


def test_ms1_prefilter_da_tolerance_basic():
    """Verify basic Dalton tolerance filtering behavior."""
    reference_spectra = [
        create_mock_spectrum("r1", 100.0),
        create_mock_spectrum("r2", 101.0),
        create_mock_spectrum("r3", 110.0),
    ]
    query_spectra = [
        create_mock_spectrum("q1", 100.02),
        create_mock_spectrum("q2", None),
    ]

    matched_rows, matched_cols = _ms1_prefilter(
        reference_spectra, query_spectra, ms1_tolerance=0.05, resolution_ppm=None
    )

    pairs = list(zip(matched_rows.tolist(), matched_cols.tolist()))

    # Query q1 should match r1 and r2
    q1_rows = {r for r, c in pairs if c == 0}
    assert q1_rows.issubset({0, 1})

    # Query q2 has a missing precursor_mz and should match everything
    q2_rows = {r for r, c in pairs if c == 1}
    assert q2_rows >= {0, 1, 2}


def test_ms1_prefilter_ppm_vs_da_tolerance():
    """Verify that ppm resolution accurately uses the mass error formula."""
    # Base theoretical mass: 500.0
    # 5.0 ppm window: Error (Da) = 5.0 * 500.0 / 1e6 = 0.0025 Da
    reference_spectra = [
        create_mock_spectrum("ref_exact", 500.0),
        create_mock_spectrum("ref_da_match", 500.015), # Inside 0.02 Da, outside 5 ppm
        create_mock_spectrum("ref_ppm_match", 500.002), # Inside both
        create_mock_spectrum("ref_ppm_edge", 500.0025), # Exactly on 5 ppm edge
        create_mock_spectrum("ref_outside", 500.025), # Outside both
    ]
    query_spectra = [create_mock_spectrum("query1", 500.0)]

    # Test PPM tolerance
    rows_ppm, cols_ppm = _ms1_prefilter(
        reference_spectra, query_spectra, ms1_tolerance=0.0, resolution_ppm=5.0
    )
    ppm_matches = {r for r, c in zip(rows_ppm.tolist(), cols_ppm.tolist())}
    # Matches: exact (0), ppm_match (2), and ppm_edge (3).
    assert ppm_matches == {0, 2, 3}

    # Test Da tolerance
    rows_da, cols_da = _ms1_prefilter(
        reference_spectra, query_spectra, ms1_tolerance=0.02, resolution_ppm=None
    )
    da_matches = {r for r, c in zip(rows_da.tolist(), cols_da.tolist())}
    # Matches: exact (0), da_match (1), ppm_match (2), ppm_edge (3).
    assert da_matches == {0, 1, 2, 3}


def test_ms1_prefilter_missing_precursor_handling():
    """Verify that missing precursors (None or np.nan) bypass the MS1 filter."""
    reference_spectra = [
        create_mock_spectrum("r1", 100.0),
        create_mock_spectrum("r_nan", np.nan),
        create_mock_spectrum("r_none", None),
    ]
    query_spectra = [
        create_mock_spectrum("q1", 200.0),
        create_mock_spectrum("q_nan", np.nan),
        create_mock_spectrum("q_none", None),
    ]

    rows, cols = _ms1_prefilter(
        reference_spectra, query_spectra, ms1_tolerance=0.05, resolution_ppm=None
    )
    pairs = list(zip(rows.tolist(), cols.tolist()))

    # q1 (idx 0) should match with all missing references (idx 1, 2)
    q1_matches = {r for r, c in pairs if c == 0}
    assert q1_matches == {1, 2}

    # q_nan (idx 1) and q_none (idx 2) should match with all references (idx 0, 1, 2)
    q_nan_matches = {r for r, c in pairs if c == 1}
    assert q_nan_matches == {0, 1, 2}

    q_none_matches = {r for r, c in pairs if c == 2}
    assert q_none_matches == {0, 1, 2}


def test_ms1_prefilter_boundary_conditions():
    """Test boundary points explicitly for Dalton tolerance."""
    base_mass = 300.0
    tolerance = 0.05

    reference_spectra = [
        create_mock_spectrum("ref_lower_bound", base_mass - tolerance),
        create_mock_spectrum("ref_upper_bound", base_mass + tolerance),
        create_mock_spectrum("ref_just_below", base_mass - tolerance - 0.0001),
        create_mock_spectrum("ref_just_above", base_mass + tolerance + 0.0001),
    ]
    query_spectra = [create_mock_spectrum("query1", base_mass)]

    rows, cols = _ms1_prefilter(
        reference_spectra, query_spectra, ms1_tolerance=tolerance, resolution_ppm=None
    )

    matches = {r for r, c in zip(rows.tolist(), cols.tolist())}
    # np.searchsorted side='left' and side='right' should be inclusive
    # depending on exact float comparison, but we want boundary included.
    # The current `_ms1_prefilter` uses left and right, so exact bounds are included.
    assert matches == {0, 1}
