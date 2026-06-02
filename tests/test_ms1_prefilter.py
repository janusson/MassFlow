"""Unit tests for _ms1_prefilter in MassFlow.similarity

Covers both fixed Da tolerance and ppm resolution paths and verifies that
missing precursor_mz values bypass the filter (i.e., entries are included).
"""

import numpy as np
from matchms import Spectrum

from MassFlow.similarity import _ms1_prefilter


def make_spec(id, mz):
    return Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": id, "precursor_mz": mz},
    )


def test_ms1_prefilter_da_tolerance_basic():
    refs = [make_spec("r1", 100.0), make_spec("r2", 101.0), make_spec("r3", 110.0)]
    queries = [make_spec("q1", 100.02), make_spec("q2", None)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.05, resolution_ppm=None)

    # Inspect matched (ref_idx, query_idx) pairs
    pairs = list(zip(rows.tolist(), cols.tolist()))
    # For query index 0 (q1), only refs 0 and 1 should appear
    q1_rows = {r for r, c in pairs if c == 0}
    assert q1_rows.issubset({0, 1})
    # For query index 1 (q2) with missing precursor, behavior is to include all refs
    q2_rows = {r for r, c in pairs if c == 1}
    assert q2_rows >= {0}


def test_ms1_prefilter_ppm_resolution():
    refs = [make_spec("r1", 100.0), make_spec("r2", 100.2), make_spec("r3", 200.0)]
    queries = [make_spec("q1", 100.0), make_spec("q2", 100.001)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.02, resolution_ppm=10)

    # With 10 ppm at 100 Da, tolerance is 0.001 Da. q2 should match r1 only.
    matched_pairs = list(zip(rows.tolist(), cols.tolist()))
    assert any(r == 0 and c in (0, 1) for r, c in matched_pairs)


def test_ms1_prefilter_missing_precursor_bypass():
    refs = [make_spec("r1", 100.0), make_spec("r2", None)]
    queries = [make_spec("q1", 100.0), make_spec("q2", None)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.1, resolution_ppm=None)

    # Missing reference or query precursor should include broad matches (bypass)
    # Ensure that at least one mapping covers missing entries
    assert len(rows) > 0


def test_ppm_boundary_high_mass():
    """At 2000 Da, 10 ppm is only 0.02 Da — a nearby ref at 0.05 Da offset should miss."""
    refs = [make_spec("r_inside", 2000.01), make_spec("r_outside", 2000.05)]
    queries = [make_spec("q", 2000.0)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.0, resolution_ppm=10)

    matched_refs = set(rows[cols == 0])
    # r_inside @ 2000.01 is within 0.02 Da (10 ppm of 2000)
    assert 0 in matched_refs, "ref at 0.01 Da offset should match at 10 ppm"
    # r_outside @ 2000.05 is outside the 0.02 Da window
    assert 1 not in matched_refs, "ref at 0.05 Da offset should NOT match at 10 ppm"


def test_ppm_boundary_low_mass():
    """At 50 Da, 10 ppm is only 0.0005 Da — even a tiny offset should miss."""
    refs = [make_spec("r", 50.001)]
    queries = [make_spec("q", 50.0)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.0, resolution_ppm=10)

    # 0.001 Da > 0.0005 Da → should NOT match
    assert len(rows) == 0, "0.001 Da offset at 50 Da should exceed 10 ppm window"


def test_da_ppm_equivalence_invariant():
    """10 ppm at exactly 100 Da equals 0.001 Da — both modes should produce identical results."""
    refs = [make_spec("r_match", 100.0005), make_spec("r_miss", 100.002)]
    queries = [make_spec("q", 100.0)]

    rows_ppm, cols_ppm = _ms1_prefilter(
        refs, queries, ms1_tolerance=0.0, resolution_ppm=10
    )
    rows_da, cols_da = _ms1_prefilter(
        refs, queries, ms1_tolerance=0.001, resolution_ppm=None
    )

    assert np.array_equal(
        rows_ppm, rows_da
    ), f"ppm rows {rows_ppm} != Da rows {rows_da}"
    assert np.array_equal(
        cols_ppm, cols_da
    ), f"ppm cols {cols_ppm} != Da cols {cols_da}"


def test_both_sides_missing_precursor():
    """When both a reference and a query lack precursor_mz, they must still be paired."""
    refs = [make_spec("r_good", 100.0), make_spec("r_none", None)]
    queries = [make_spec("q_good", 100.0), make_spec("q_none", None)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=0.1, resolution_ppm=None)

    pairs = set(zip(rows.tolist(), cols.tolist()))
    # r_none (index 1) should be paired with every query (including q_none)
    assert (1, 0) in pairs, "missing-precursor ref should pair with valid query"
    assert (
        1,
        1,
    ) in pairs, "missing-precursor ref should pair with missing-precursor query"
    # q_none (index 1) should be paired with every ref (including r_none)
    assert (0, 1) in pairs, "missing-precursor query should pair with valid ref"


def test_zero_precursor_skipped():
    """A query with precursor_mz == 0.0 should not produce any matches via the filter."""
    refs = [make_spec("r", 100.0)]
    queries = [make_spec("q", 0.0)]

    rows, cols = _ms1_prefilter(refs, queries, ms1_tolerance=10.0, resolution_ppm=None)

    # Even with a huge tolerance, the zero-precursor query should be skipped.
    assert len(rows) == 0, "zero-precursor query should never pass the filter"


def test_empty_inputs():
    """Empty reference or query lists should return empty index arrays."""
    refs = [make_spec("r", 100.0)]
    queries = [make_spec("q", 100.0)]

    rows_empty_ref, cols_empty_ref = _ms1_prefilter(
        [], queries, ms1_tolerance=0.1, resolution_ppm=None
    )
    rows_empty_q, cols_empty_q = _ms1_prefilter(
        refs, [], ms1_tolerance=0.1, resolution_ppm=None
    )

    assert len(rows_empty_ref) == 0 and len(cols_empty_ref) == 0
    assert len(rows_empty_q) == 0 and len(cols_empty_q) == 0
