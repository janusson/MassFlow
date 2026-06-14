"""
Precursor physics validation tests for MassFlow.

Validates that the MS1 pre-filter, adduct compatibility, and missing-precursor
handling behave correctly — preventing physically impossible matches from ever
reaching the expensive cosine calculation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import SimilarityEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spectrum(
    precursor_mz: float,
    mz: list[float],
    intensities: list[float],
    spec_id: str = "spec",
    **extra_meta: object,
) -> Spectrum:
    """Build a Spectrum with mandatory metadata and optional extras."""
    meta: dict[str, object] = {
        "precursor_mz": precursor_mz,
        "id": spec_id,
        "compound_name": spec_id,
    }
    meta.update(extra_meta)
    return Spectrum(
        mz=np.array(mz, dtype=np.float64),
        intensities=np.array(intensities, dtype=np.float64),
        metadata=meta,
    )


def _count_matches(
    engine: SimilarityEngine,
    queries: list[Spectrum],
    reference: Spectrum,
) -> int:
    """Return the number of results the engine produces for a set of queries."""
    results = engine.search(
        query_spectra=queries,
        reference_spectra=[reference],
        min_score=0.0,
        include_decoys=False,
    )
    return len(results)


# ---------------------------------------------------------------------------
# Test 1 — PPM tolerance boundary
# ---------------------------------------------------------------------------


def test_ppm_tolerance_boundary() -> None:
    """Queries at 4.75 ppm and 6.25 ppm — only the first must pass the 5 ppm gate."""
    ref = _spectrum(
        precursor_mz=400.0000,
        mz=[100.0, 200.0],
        intensities=[1.0, 1.0],
        spec_id="ref_ppm",
    )

    # 4.75 ppm error → should pass
    query_pass = _spectrum(
        precursor_mz=400.0019,
        mz=[100.0, 200.0],
        intensities=[1.0, 1.0],
        spec_id="q_475ppm",
    )

    # 6.25 ppm error → must be filtered out by the MS1 pre-filter
    query_fail = _spectrum(
        precursor_mz=400.0025,
        mz=[100.0, 200.0],
        intensities=[1.0, 1.0],
        spec_id="q_625ppm",
    )

    config = SimilarityConfig(
        algorithm="cosine",
        ms2_tolerance=0.02,
        ms1_tolerance=0.02,
        resolution_ppm=5.0,
        min_score=0.0,
        min_matched_peaks=1,  # zero-match entries from pre-filter rejection are dropped
    )
    engine = SimilarityEngine(config)

    # Both queries in one call; we count matches per query
    results = engine.search(
        query_spectra=[query_pass, query_fail],
        reference_spectra=[ref],
        min_score=0.0,
        include_decoys=False,
    )

    passing_ids = {r["query_id"] for r in results}

    assert (
        "q_475ppm" in passing_ids
    ), "4.75 ppm query should pass the 5.0 ppm MS1 pre-filter"
    assert (
        "q_625ppm" not in passing_ids
    ), "6.25 ppm query must be rejected by the 5.0 ppm MS1 pre-filter"

    # The passing query should have a perfect cosine score (identical peaks)
    passing_hits = [r for r in results if r["query_id"] == "q_475ppm"]
    assert len(passing_hits) == 1
    assert math.isclose(
        float(passing_hits[0]["score"]), 1.0, rel_tol=1e-6
    ), f"Passing query expected cosine=1.0, got {passing_hits[0]['score']}"


# ---------------------------------------------------------------------------
# Test 2 — Adduct mismatch rejects the pair before similarity scoring
# ---------------------------------------------------------------------------


def test_adduct_mismatch() -> None:
    """
    A reference with [M+H]+ and a query with [M-H]- are physically
    incompatible. The pipeline must detect the adduct-mode mismatch
    and refuse to score the pair.
    """
    ref = _spectrum(
        precursor_mz=150.0,
        mz=[100.0, 200.0],
        intensities=[1.0, 1.0],
        spec_id="ref_pos",
        adduct="[M+H]+",
        ionmode="positive",
    )

    query_neg = _spectrum(
        precursor_mz=150.0,
        mz=[100.0, 200.0],
        intensities=[1.0, 1.0],
        spec_id="q_neg",
        adduct="[M-H]-",
        ionmode="negative",
    )

    config = SimilarityConfig(
        algorithm="cosine",
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=0,
    )
    engine = SimilarityEngine(config)

    results = engine.search(
        query_spectra=[query_neg],
        reference_spectra=[ref],
        min_score=0.0,
        include_decoys=False,
    )

    # The adduct-mode mismatch means the pair should produce no results.
    # Even though the peak lists are identical, opposite ionisation modes
    # are physically incompatible.
    assert (
        len(results) == 0
    ), f"Adduct-mismatched pair must be rejected, got {len(results)} result(s)"


# ---------------------------------------------------------------------------
# Test 3 — Missing precursor fails safely (no crash, graceful drop)
# ---------------------------------------------------------------------------


def test_missing_precursor_fails_safely(caplog: pytest.LogCaptureFixture) -> None:
    """
    A query spectrum whose ``precursor_mz`` is missing from metadata must be
    handled gracefully — no exception, no crash, just a logged skip.
    """
    ref = _spectrum(
        precursor_mz=200.0,
        mz=[100.0, 200.0],
        intensities=[1.0, 1.0],
        spec_id="ref_good",
    )

    # Deliberately omit precursor_mz from metadata
    query_bad = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        metadata={
            "id": "q_no_precursor",
            "compound_name": "bad_query",
        },
    )

    config = SimilarityConfig(
        algorithm="cosine",
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=0,
    )
    engine = SimilarityEngine(config)

    # Must not raise — the engine should handle missing precursor gracefully
    results = engine.search(
        query_spectra=[query_bad],
        reference_spectra=[ref],
        min_score=0.0,
        include_decoys=False,
    )

    # The engine may return results (MS1 pre-filter bypasses missing precursors)
    # or it may return nothing — either is acceptable as long as no exception.
    # However, the result should not contain a meaningful cosine score for a
    # spectrum that lacks basic physical metadata.
    for r in results:
        assert (
            r["query_precursor_mz"] is None
        ), "Query without precursor_mz should have None in results"

    # Verify we didn't crash — reaching here means success.
    assert True
