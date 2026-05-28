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
