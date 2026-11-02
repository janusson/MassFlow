"""Unit tests for _ms1_prefilter in MassFlow.similarity

Covers both fixed Da tolerance and ppm resolution paths and verifies that
missing precursor_mz values bypass the filter (i.e., entries are included).
"""

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.similarity import _ms1_prefilter, _ms1_prefilter_arrays


def make_mock_spectrum(spectrum_id: str, precursor_mz: float | None) -> Spectrum:
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


def test_ms1_prefilter_arrays_missing_precursor_bypass():
    # Similar to test_ms1_prefilter_missing_precursor_bypass but for the array version
    ref_mzs = np.array([100.0, np.nan], dtype=np.float64)
    queries = [
        make_mock_spectrum("q1", 100.0),
        make_mock_spectrum("q2", np.nan),
    ]

    rows, cols = _ms1_prefilter_arrays(
        ref_mzs, queries, ms1_tolerance=0.1, resolution_ppm=None
    )
    pairs = list(zip(rows.tolist(), cols.tolist()))

    # q1 (100.0) should match r1 (100.0) and r2 (np.nan, which bypasses the filter)
    q1_rows = {r for r, c in pairs if c == 0}
    assert q1_rows == {0, 1}

    # q2 (np.nan) should bypass the filter and match both r1 and r2
    q2_rows = {r for r, c in pairs if c == 1}
    assert q2_rows == {0, 1}


@pytest.mark.parametrize(
    ("ref_values", "query_values", "ms1_tolerance", "resolution_ppm", "expected"),
    [
        pytest.param([100.0], [100.0], 0.05, None, [(0, 0)], id="finite-da"),
        pytest.param([np.nan], [100.0], 0.1, None, [(0, 0)], id="missing-ref-nan"),
        pytest.param([None], [100.0], 0.1, None, [(0, 0)], id="missing-ref-none"),
        pytest.param([100.0], [np.nan], 0.1, None, [(0, 0)], id="missing-query-nan"),
        pytest.param([100.0], [None], 0.1, None, [(0, 0)], id="missing-query-none"),
        pytest.param([np.nan], [np.nan], 0.1, None, [(0, 0)], id="both-missing-once"),
        pytest.param([0.0], [100.0], 0.1, None, [(0, 0)], id="zero-ref"),
        pytest.param([-1.0], [100.0], 0.1, None, [(0, 0)], id="negative-ref"),
        pytest.param([np.inf], [100.0], 0.1, None, [], id="positive-infinity-ref"),
        pytest.param([-np.inf], [100.0], 0.1, None, [], id="negative-infinity-ref"),
        pytest.param([100.0], [0.0], 0.1, None, [], id="zero-query"),
        pytest.param([100.0], [-1.0], 0.1, None, [], id="negative-query"),
        pytest.param([100.0], [np.inf], 0.1, None, [], id="positive-infinity-query"),
        pytest.param([100.0], [-np.inf], 0.1, None, [], id="negative-infinity-query"),
        pytest.param([np.inf], [np.inf], 0.1, None, [], id="both-positive-infinity"),
        pytest.param([-np.inf], [-np.inf], 0.1, None, [], id="both-negative-infinity"),
        pytest.param(
            [100.05, 100.05001], [100.0], 0.05, None, [(0, 0)], id="da-boundary"
        ),
        pytest.param(
            [100.0005, 100.0006], [100.0], 0.0, 5.0, [(0, 0)], id="ppm-boundary"
        ),
        pytest.param(
            [100.0, np.nan, 0.0, 200.0],
            [100.0, np.nan, 0.0],
            0.1,
            None,
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
                (3, 1),
            ],
            id="mixed-with-deduplication",
        ),
    ],
)
def test_prefilter_paths_agree_on_missing_and_invalid_masses(
    ref_values: list[float | None],
    query_values: list[float | None],
    ms1_tolerance: float,
    resolution_ppm: float | None,
    expected: list[tuple[int, int]],
) -> None:
    references = [
        make_mock_spectrum(f"r{index}", value) for index, value in enumerate(ref_values)
    ]
    queries = [
        make_mock_spectrum(f"q{index}", value)
        for index, value in enumerate(query_values)
    ]

    object_rows, object_cols = _ms1_prefilter(
        references, queries, ms1_tolerance, resolution_ppm
    )
    array_rows, array_cols = _ms1_prefilter_arrays(
        np.asarray(ref_values, dtype=np.float64),
        queries,
        ms1_tolerance,
        resolution_ppm,
    )

    object_pairs = sorted(zip(object_rows.tolist(), object_cols.tolist()))
    array_pairs = sorted(zip(array_rows.tolist(), array_cols.tolist()))
    assert object_pairs == sorted(expected)
    assert array_pairs == sorted(expected)


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
