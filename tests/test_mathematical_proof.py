"""
Mathematical proof tests for the SimilarityEngine.

These tests validate the exact numerical accuracy of cosine and modified cosine
spectral similarity scoring using hand-crafted matchms.Spectrum objects. No file
I/O or external datasets are involved — every input is a pristine numpy array.
"""

from __future__ import annotations

import math

import numpy as np
from matchms import Spectrum

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import SimilarityEngine


def _make_spectrum(
    precursor_mz: float,
    mz: list[float],
    intensities: list[float],
    spec_id: str = "spec",
) -> Spectrum:
    """Build a matchms.Spectrum with the mandatory metadata fields."""
    return Spectrum(
        mz=np.array(mz, dtype=np.float64),
        intensities=np.array(intensities, dtype=np.float64),
        metadata={
            "precursor_mz": precursor_mz,
            "id": spec_id,
            "compound_name": spec_id,
        },
    )


def _single_hit_score(
    engine: SimilarityEngine,
    query: Spectrum,
    reference: Spectrum,
) -> float:
    """Run a 1-vs-1 search and return the top hit's score."""
    results = engine.search(
        query_spectra=[query],
        reference_spectra=[reference],
        min_score=0.0,
        include_decoys=False,
    )
    assert len(results) > 0, f"No results returned by {engine.config.algorithm}"
    return float(results[0]["score"])


# ---------------------------------------------------------------------------
# Test 1: Perfectly proportional spectra → cosine score must be exactly 1.0
# ---------------------------------------------------------------------------
def test_perfect_match() -> None:
    """Reference [100,200]@[1,1] vs Query [100,200]@[0.5,0.5] → cosine=1.0000."""
    ref = _make_spectrum(
        precursor_mz=150.0,
        mz=[100.0, 200.0],
        intensities=[1.0, 1.0],
        spec_id="ref_perfect",
    )
    query = _make_spectrum(
        precursor_mz=150.0,
        mz=[100.0, 200.0],
        intensities=[0.5, 0.5],
        spec_id="query_perfect",
    )

    config = SimilarityConfig(
        algorithm="cosine",
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=0,
    )
    engine = SimilarityEngine(config)

    score = _single_hit_score(engine, query, ref)
    assert math.isclose(score, 1.0, rel_tol=1e-6), (
        f"Perfect match expected cosine=1.0000, got {score}"
    )


# ---------------------------------------------------------------------------
# Test 2: Disjoint peak sets → cosine score must be exactly 0.0
# ---------------------------------------------------------------------------
def test_orthogonal_miss() -> None:
    """Reference [100,200] vs Query [300,400] → cosine=0.0000."""
    ref = _make_spectrum(
        precursor_mz=250.0,
        mz=[100.0, 200.0],
        intensities=[1.0, 1.0],
        spec_id="ref_ortho",
    )
    query = _make_spectrum(
        precursor_mz=350.0,
        mz=[300.0, 400.0],
        intensities=[1.0, 1.0],
        spec_id="query_ortho",
    )

    config = SimilarityConfig(
        algorithm="cosine",
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=0,
    )
    engine = SimilarityEngine(config)

    score = _single_hit_score(engine, query, ref)
    assert math.isclose(score, 0.0, abs_tol=1e-9), (
        f"Orthogonal miss expected cosine=0.0000, got {score}"
    )


# ---------------------------------------------------------------------------
# Test 3: Predictable partial overlap → cosine = 0.5000
# ---------------------------------------------------------------------------
def test_predictable_partial() -> None:
    """Ref [100,200,300]@[1,1,0] vs Qry [100,200,300]@[1,0,1] → cosine=0.5000."""
    ref = _make_spectrum(
        precursor_mz=200.0,
        mz=[100.0, 200.0, 300.0],
        intensities=[1.0, 1.0, 0.0],
        spec_id="ref_partial",
    )
    query = _make_spectrum(
        precursor_mz=200.0,
        mz=[100.0, 200.0, 300.0],
        intensities=[1.0, 0.0, 1.0],
        spec_id="query_partial",
    )

    config = SimilarityConfig(
        algorithm="cosine",
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=0,
    )
    engine = SimilarityEngine(config)

    score = _single_hit_score(engine, query, ref)
    # With CosineGreedy, only peaks that *both* have non-zero intensity in the
    # reference contribute to the dot product. The greedy matching selects the
    # best-scoring peak pair at 100.0 (both have intensity 1.0), then at 200.0
    # the query has 0.0 intensity → skipped, at 300.0 the reference has 0.0
    # intensity → skipped. So only one peak pair contributes: score = 1*1 / (√2 * √2) = 0.5.
    assert math.isclose(score, 0.5, rel_tol=1e-6), (
        f"Partial overlap expected cosine=0.5000, got {score}"
    )


# ---------------------------------------------------------------------------
# Test 4: Exact precursor-mass shift → modified cosine = 1.0, cosine = 0.0
# ---------------------------------------------------------------------------
def test_exact_shift() -> None:
    """Ref (prec 150, peaks [100]) vs Qry (prec 166, peaks [116])."""
    ref = _make_spectrum(
        precursor_mz=150.0,
        mz=[100.0],
        intensities=[1.0],
        spec_id="ref_shift",
    )
    query = _make_spectrum(
        precursor_mz=166.0,
        mz=[116.0],
        intensities=[1.0],
        spec_id="query_shift",
    )

    # --- Standard cosine ---
    # Precursors differ by 16 Da (≫ 0.02 tolerance), so the MS1 pre-filter
    # produces zero (row, col) pairs. The entire scores array remains zero →
    # cosine score is 0.0.
    cosine_config = SimilarityConfig(
        algorithm="cosine",
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=0,
    )
    cosine_engine = SimilarityEngine(cosine_config)

    cosine_score = _single_hit_score(cosine_engine, query, ref)
    assert math.isclose(cosine_score, 0.0, abs_tol=1e-9), (
        f"Exact-shift cosine expected 0.0000, got {cosine_score}"
    )

    # --- Modified cosine ---
    # Modified cosine accounts for the precursor mass difference (166 − 150 = 16).
    # The fragment at m/z 100 in the reference is matched to m/z 116 in the
    # query after applying the precursor shift → perfect match.
    mod_cosine_config = SimilarityConfig(
        algorithm="modified_cosine",
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=0,
    )
    mod_cosine_engine = SimilarityEngine(mod_cosine_config)

    mod_score = _single_hit_score(mod_cosine_engine, query, ref)
    assert math.isclose(mod_score, 1.0, rel_tol=1e-6), (
        f"Exact-shift modified cosine expected 1.0000, got {mod_score}"
    )
