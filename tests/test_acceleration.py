"""
Tests for the Phase 2 algorithmic acceleration components.

Covers:
- Numba-accelerated peak/neutral-loss prefilter: hand-computed matching
  semantics, conservative candidate generation, equivalence with the pure
  NumPy fallback, and identical end-to-end search results with the prefilter
  enabled vs disabled.
- HNSW index wrapper: spectral binning correctness, build/query/save/load,
  validation errors, and recall on non-metric spectral data.
- CascadeEngine HNSW integration: exhaustive-candidate equivalence with the
  full cascade, graceful fallback when hnswlib is missing or misconfigured,
  and index caching.

HNSW tests require the optional ``hnswlib`` dependency (the ``[hnsw]``
extra) and skip when it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import hnsw as hnsw_module
from MassFlow.acceleration import (
    _HAS_NUMBA,
    _count_tolerance_matches_numba,
    _count_tolerance_matches_python,
    build_flat_peak_arrays,
    prefilter_candidate_pairs,
)
from MassFlow.config import SimilarityConfig
from MassFlow.hnsw import HNSWSpectralIndex, bin_spectra, spectrum_to_binned_vector
from MassFlow.similarity import (
    CascadeEngine,
    SearchResult,
    SimilarityEngine,
)

_HAS_HNSWLIB = hnsw_module._HAS_HNSWLIB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spectrum(
    spectrum_id: str,
    mz: list[float],
    precursor_mz: float,
    intensities: list[float] | None = None,
) -> Spectrum:
    """Build a Spectrum with sorted m/z values and a precursor."""
    mz_array = np.asarray(sorted(mz), dtype=np.float64)
    if intensities is None:
        intensities_array: np.ndarray = np.ones(mz_array.size, dtype=np.float64)
    else:
        intensities_array = np.asarray(intensities, dtype=np.float64)
    return Spectrum(
        mz=mz_array,
        intensities=intensities_array,
        metadata={"id": spectrum_id, "precursor_mz": precursor_mz},
    )


def make_random_library(count: int, seed: int = 7) -> list[Spectrum]:
    """Generate a deterministic random library of *count* spectra."""
    rng = np.random.default_rng(seed)
    spectra: list[Spectrum] = []
    for index in range(count):
        n_peaks = max(3, int(rng.poisson(20)))
        mz = np.sort(rng.uniform(50.0, 900.0, size=n_peaks)).astype(np.float64)
        intensities = rng.uniform(0.01, 1.0, size=n_peaks).astype(np.float64)
        spectra.append(
            Spectrum(
                mz=mz,
                intensities=intensities,
                metadata={
                    "id": f"ref_{index:04d}",
                    "precursor_mz": float(rng.uniform(100.0, 1000.0)),
                },
            )
        )
    return spectra


def summarize_results(results: list[SearchResult]) -> list[tuple]:
    """Reduce SearchResult dicts to a comparable, order-independent form."""
    return sorted(
        (
            result["query_id"],
            result["reference_id"],
            round(float(result["score"]), 6),
            int(result["matched_peaks"]),
        )
        for result in results
    )


# ---------------------------------------------------------------------------
# Numba prefilter — match counting semantics
# ---------------------------------------------------------------------------


class TestToleranceMatchCounting:
    """Two-pointer tolerance match counting (both implementations)."""

    @pytest.mark.parametrize(
        "count_fn",
        [_count_tolerance_matches_numba, _count_tolerance_matches_python],
        ids=["numba", "python"],
    )
    def test_identical_arrays(self, count_fn) -> None:
        """Identical arrays match on every peak."""
        mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        assert count_fn(mz, mz, tolerance=0.02) == 3

    @pytest.mark.parametrize(
        "count_fn",
        [_count_tolerance_matches_numba, _count_tolerance_matches_python],
        ids=["numba", "python"],
    )
    def test_disjoint_arrays(self, count_fn) -> None:
        """Disjoint arrays produce zero matches."""
        a = np.array([100.0, 200.0], dtype=np.float64)
        b = np.array([500.0, 600.0], dtype=np.float64)
        assert count_fn(a, b, tolerance=0.02) == 0

    @pytest.mark.parametrize(
        "count_fn",
        [_count_tolerance_matches_numba, _count_tolerance_matches_python],
        ids=["numba", "python"],
    )
    def test_tolerance_boundary_is_inclusive(self, count_fn) -> None:
        """Peaks exactly at the tolerance boundary match (<= semantics)."""
        a = np.array([100.0], dtype=np.float64)
        b = np.array([100.02], dtype=np.float64)
        assert count_fn(a, b, tolerance=0.02) == 1
        c = np.array([100.02001], dtype=np.float64)
        assert count_fn(a, c, tolerance=0.02) == 0

    @pytest.mark.parametrize(
        "count_fn",
        [_count_tolerance_matches_numba, _count_tolerance_matches_python],
        ids=["numba", "python"],
    )
    def test_empty_arrays(self, count_fn) -> None:
        """Empty arrays produce zero matches."""
        empty = np.empty(0, dtype=np.float64)
        assert count_fn(empty, empty, tolerance=0.02) == 0
        assert count_fn(np.array([100.0]), empty, tolerance=0.02) == 0

    @pytest.mark.parametrize(
        "count_fn",
        [_count_tolerance_matches_numba, _count_tolerance_matches_python],
        ids=["numba", "python"],
    )
    def test_merge_counts_a_maximal_interval_matching(self, count_fn) -> None:
        """The two-pointer merge counts a maximal interval matching.

        This property makes the prefilter conservative: the merge count is a
        maximum-cardinality matching of tolerance-compatible peaks, so it is
        always >= the intensity-greedy matched-peak count produced by matchms
        scoring. Skipping pairs below the threshold can therefore never lose
        a pair the exact scoring would have kept.
        """
        # Each ref peak pairs off with the nearest query peak: 2 matches.
        ref = np.array([99.99, 100.01], dtype=np.float64)
        query = np.array([99.99, 100.01], dtype=np.float64)
        assert count_fn(ref, query, tolerance=0.02) == 2

        # Only one query peak is available: at most one match, even though
        # two ref peaks fall within tolerance of it.
        ref_single = np.array([99.99, 100.01], dtype=np.float64)
        query_single = np.array([100.0], dtype=np.float64)
        assert count_fn(ref_single, query_single, tolerance=0.02) == 1


# ---------------------------------------------------------------------------
# Numba prefilter — candidate generation
# ---------------------------------------------------------------------------


class TestPrefilterCandidates:
    """Candidate-pair generation semantics."""

    def test_cosine_gates_on_exact_mass(self) -> None:
        """Cosine keeps only pairs with enough raw m/z matches."""
        ref = make_spectrum("r1", [100.0, 200.0, 300.0], precursor_mz=500.0)
        query_match = make_spectrum("q1", [100.0, 200.0, 300.0], precursor_mz=900.0)
        query_few = make_spectrum("q2", [100.0, 400.0, 500.0], precursor_mz=500.0)

        rows, cols = prefilter_candidate_pairs(
            [ref],
            [query_match, query_few],
            tolerance=0.02,
            min_matches=3,
            algorithm="cosine",
        )
        pairs = set(zip(rows.tolist(), cols.tolist()))
        assert (0, 0) in pairs  # 3 raw m/z matches
        assert (0, 1) not in pairs  # only 1 raw m/z match

    def test_modified_cosine_gates_on_neutral_loss(self) -> None:
        """Modified cosine keeps pairs matching in neutral-loss space."""
        # Same fragments at different precursor masses: identical neutral
        # losses (500 - 100 = 400, etc.; 900 - 500 = 400, etc.).
        ref = make_spectrum("r1", [100.0, 200.0, 300.0], precursor_mz=500.0)
        query_nl_match = make_spectrum("q1", [500.0, 600.0, 700.0], precursor_mz=900.0)
        query_no_match = make_spectrum("q2", [110.0, 220.0, 330.0], precursor_mz=500.0)

        rows, cols = prefilter_candidate_pairs(
            [ref],
            [query_nl_match, query_no_match],
            tolerance=0.02,
            min_matches=3,
            algorithm="modified_cosine",
        )
        pairs = set(zip(rows.tolist(), cols.tolist()))
        assert (0, 0) in pairs  # 3 neutral-loss matches
        assert (0, 1) not in pairs  # neither raw m/z nor NL matches

    def test_missing_precursor_bypasses_gate(self) -> None:
        """Pairs with a missing precursor always stay candidates."""
        ref = make_spectrum("r1", [100.0, 200.0, 300.0], precursor_mz=500.0)
        query_without_precursor = Spectrum(
            mz=np.array([900.0, 1000.0, 1100.0], dtype=np.float64),
            intensities=np.ones(3, dtype=np.float64),
            metadata={"id": "q_no_precursor"},
        )

        rows, cols = prefilter_candidate_pairs(
            [ref],
            [query_without_precursor],
            tolerance=0.02,
            min_matches=10,
            algorithm="modified_cosine",
        )
        assert (0, 0) in set(zip(rows.tolist(), cols.tolist()))

    def test_zero_min_matches_returns_full_grid(self) -> None:
        """min_matches <= 0 disables prefiltering entirely."""
        refs = make_random_library(4)
        queries = make_random_library(3, seed=8)
        rows, cols = prefilter_candidate_pairs(
            refs,
            queries,
            tolerance=0.02,
            min_matches=0,
            algorithm="modified_cosine",
        )
        pairs = set(zip(rows.tolist(), cols.tolist()))
        assert pairs == {(r, q) for r in range(4) for q in range(3)}

    def test_empty_inputs(self) -> None:
        """Empty references or queries yield empty candidate arrays."""
        ref = make_spectrum("r1", [100.0, 200.0], precursor_mz=300.0)
        rows, cols = prefilter_candidate_pairs(
            [], [ref], tolerance=0.02, min_matches=1, algorithm="cosine"
        )
        assert rows.size == 0 and cols.size == 0

    @pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
    def test_numba_and_python_implementations_agree(self) -> None:
        """The JIT and fallback kernels produce identical candidate pairs."""
        refs = make_random_library(25)
        queries = make_random_library(8, seed=99)
        tolerance = 0.02
        min_matches = 3

        ref_mz_flat, ref_offsets, ref_nl_flat, ref_valid = build_flat_peak_arrays(refs)
        query_mz_flat, query_offsets, query_nl_flat, query_valid = (
            build_flat_peak_arrays(queries)
        )

        numba_rows, numba_cols = _count_free_numba_call(
            ref_mz_flat,
            ref_offsets,
            ref_nl_flat,
            query_mz_flat,
            query_offsets,
            query_nl_flat,
            tolerance,
            min_matches,
            ref_valid,
            query_valid,
        )
        python_rows, python_cols = _count_free_python_call(
            ref_mz_flat,
            ref_offsets,
            ref_nl_flat,
            query_mz_flat,
            query_offsets,
            query_nl_flat,
            tolerance,
            min_matches,
            ref_valid,
            query_valid,
        )
        assert set(zip(numba_rows.tolist(), numba_cols.tolist())) == set(
            zip(python_rows.tolist(), python_cols.tolist())
        )

    def test_candidate_set_contains_all_threshold_pairs(self) -> None:
        """The prefilter never drops a pair that exact scoring would keep.

        For every pair with matched_peaks >= min_matched_peaks in a full
        modified-cosine scoring run, the pair must appear in the prefilter
        candidate set (the conservativeness guarantee).
        """
        refs = make_random_library(30)
        queries = make_random_library(10, seed=123)
        min_matched = 3

        config = SimilarityConfig(
            algorithm="modified_cosine",
            ms2_tolerance=0.02,
            min_score=0.0,
            min_matched_peaks=min_matched,
            enable_numba_prefilter=False,
        )
        full_results = SimilarityEngine(config).search(
            query_spectra=queries,
            reference_spectra=refs,
            include_decoys=False,
        )
        kept_pairs = {
            (r["reference_id"], r["query_id"])
            for r in full_results
            if r["matched_peaks"] >= min_matched
        }

        rows, cols = prefilter_candidate_pairs(
            refs,
            queries,
            tolerance=0.02,
            min_matches=min_matched,
            algorithm="modified_cosine",
        )
        candidate_pairs = {
            (str(refs[r].get("id")), str(queries[c].get("id")))
            for r, c in zip(rows.tolist(), cols.tolist())
        }
        assert kept_pairs.issubset(candidate_pairs)


def _count_free_numba_call(
    ref_mz_flat,
    ref_offsets,
    ref_nl_flat,
    query_mz_flat,
    query_offsets,
    query_nl_flat,
    tolerance,
    min_matches,
    ref_valid,
    query_valid,
):
    """Call the JIT prefilter kernel directly."""
    from MassFlow.acceleration import _prefilter_pairs_numba

    return _prefilter_pairs_numba(
        ref_mz_flat,
        ref_offsets,
        ref_nl_flat,
        query_mz_flat,
        query_offsets,
        query_nl_flat,
        tolerance,
        min_matches,
        True,
        True,
        ref_valid,
        query_valid,
    )


def _count_free_python_call(
    ref_mz_flat,
    ref_offsets,
    ref_nl_flat,
    query_mz_flat,
    query_offsets,
    query_nl_flat,
    tolerance,
    min_matches,
    ref_valid,
    query_valid,
):
    """Call the pure-Python prefilter kernel directly."""
    from MassFlow.acceleration import _prefilter_pairs_python

    return _prefilter_pairs_python(
        ref_mz_flat,
        ref_offsets,
        ref_nl_flat,
        query_mz_flat,
        query_offsets,
        query_nl_flat,
        tolerance,
        min_matches,
        True,
        True,
        ref_valid,
        query_valid,
    )


class TestFlatArrayConstruction:
    """Flat-array representation used by the prefilter."""

    def test_offsets_and_neutral_losses(self) -> None:
        """Offsets bound each spectrum; neutral losses ascend."""
        spectra = [
            make_spectrum("a", [100.0, 200.0, 300.0], precursor_mz=500.0),
            make_spectrum("b", [150.0], precursor_mz=400.0),
        ]
        mz_flat, offsets, nl_flat, precursor_valid = build_flat_peak_arrays(spectra)
        assert mz_flat.tolist() == [100.0, 200.0, 300.0, 150.0]
        assert offsets.tolist() == [0, 3, 4]
        # Ascending neutral losses: 500-300, 500-200, 500-100 then 400-150.
        assert nl_flat.tolist() == [200.0, 300.0, 400.0, 250.0]
        assert precursor_valid.tolist() == [True, True]

    def test_missing_precursor_flags_invalid(self) -> None:
        """Spectra without precursors are flagged and zero-filled."""
        with_precursor = make_spectrum("a", [100.0], precursor_mz=200.0)
        without_precursor = Spectrum(
            mz=np.array([100.0], dtype=np.float64),
            intensities=np.array([1.0], dtype=np.float64),
            metadata={"id": "b"},
        )
        _, _, _, precursor_valid = build_flat_peak_arrays(
            [with_precursor, without_precursor]
        )
        assert precursor_valid.tolist() == [True, False]


# ---------------------------------------------------------------------------
# Engine-level prefilter equivalence
# ---------------------------------------------------------------------------


class TestPrefilterEngineEquivalence:
    """End-to-end equivalence of the prefiltered modified-cosine path."""

    def test_prefiltered_search_matches_full_scoring(self) -> None:
        """Prefilter on/off produce identical results (with decoys)."""
        refs = make_random_library(40)
        rng = np.random.default_rng(5)
        queries: list[Spectrum] = []
        for index in range(10):
            base = refs[index]
            mz = base.peaks.mz + rng.uniform(-0.005, 0.005, base.peaks.mz.size)
            queries.append(
                Spectrum(
                    mz=np.sort(mz),
                    intensities=base.peaks.intensities.copy(),
                    metadata={
                        "id": f"query_{index:04d}",
                        "precursor_mz": base.get("precursor_mz"),
                    },
                )
            )

        config_on = SimilarityConfig(
            algorithm="modified_cosine",
            ms2_tolerance=0.02,
            min_score=0.0,
            min_matched_peaks=3,
            enable_numba_prefilter=True,
        )
        config_off = SimilarityConfig(
            algorithm="modified_cosine",
            ms2_tolerance=0.02,
            min_score=0.0,
            min_matched_peaks=3,
            enable_numba_prefilter=False,
        )

        results_on = SimilarityEngine(config_on).search(
            query_spectra=queries,
            reference_spectra=refs,
            include_decoys=True,
        )
        results_off = SimilarityEngine(config_off).search(
            query_spectra=queries,
            reference_spectra=refs,
            include_decoys=True,
        )
        assert summarize_results(results_on) == summarize_results(results_off)
        assert len(results_on) > 0, "Sanity check: the search should find hits."

    def test_prefilter_disabled_via_config_matches_full(self) -> None:
        """enable_numba_prefilter=False uses the untouched full path."""
        refs = make_random_library(15)
        rng = np.random.default_rng(21)
        queries: list[Spectrum] = []
        for index in range(5):
            base = refs[index]
            mz = base.peaks.mz + rng.uniform(-0.005, 0.005, base.peaks.mz.size)
            queries.append(
                Spectrum(
                    mz=np.sort(mz),
                    intensities=base.peaks.intensities.copy(),
                    metadata={
                        "id": f"query_{index}",
                        "precursor_mz": base.get("precursor_mz"),
                    },
                )
            )
        config = SimilarityConfig(
            algorithm="modified_cosine",
            ms2_tolerance=0.02,
            min_score=0.0,
            min_matched_peaks=2,
            enable_numba_prefilter=False,
        )
        results = SimilarityEngine(config).search(
            query_spectra=queries,
            reference_spectra=refs,
            include_decoys=False,
        )
        assert results  # near-duplicate queries must find their sources

    def test_cosine_path_untouched_by_prefilter_flag(self) -> None:
        """Cosine scoring ignores the prefilter flag and keeps MS1 gating."""
        refs = make_random_library(12)
        queries = make_random_library(4, seed=77)
        config_on = SimilarityConfig(
            algorithm="cosine",
            ms2_tolerance=0.02,
            min_score=0.0,
            min_matched_peaks=2,
            enable_numba_prefilter=True,
        )
        config_off = SimilarityConfig(
            algorithm="cosine",
            ms2_tolerance=0.02,
            min_score=0.0,
            min_matched_peaks=2,
            enable_numba_prefilter=False,
        )
        results_on = SimilarityEngine(config_on).search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        results_off = SimilarityEngine(config_off).search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        assert summarize_results(results_on) == summarize_results(results_off)


# ---------------------------------------------------------------------------
# Spectral binning
# ---------------------------------------------------------------------------


class TestSpectralBinning:
    """Binned vectorization used by the HNSW index."""

    def test_hand_computed_two_channel_vector(self) -> None:
        """Channels bin exact m/z and neutral losses; vector is unit-norm."""
        # No precursor: the neutral-loss channel must be all zeros.
        spectrum = Spectrum(
            mz=np.array([0.5, 1.5, 1.75, 9.5], dtype=np.float64),
            intensities=np.array([2.0, 3.0, 1.0, 7.0], dtype=np.float64),
            metadata={"id": "binned"},
        )
        vector = spectrum_to_binned_vector(
            spectrum, bin_width=1.0, mz_min=0.0, mz_max=4.0
        )
        assert vector.shape == (8,)  # 2 channels × 4 bins
        assert vector.dtype == np.float32
        # Channel 0 (exact m/z): bin 0 → 2.0; bin 1 → 3.0 + 1.0 = 4.0.
        expected_mz_channel = np.array([2.0, 4.0, 0.0, 0.0], dtype=np.float32)
        normalized = expected_mz_channel / np.linalg.norm(expected_mz_channel)
        assert np.allclose(vector[:4], normalized)
        # Channel 1 (neutral losses): zero — no precursor information.
        assert np.count_nonzero(vector[4:]) == 0
        # Full vector is unit-normalized.
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)

    def test_neutral_loss_channel_hand_computed(self) -> None:
        """The second half of the vector bins precursor_mz - fragment_mz."""
        spectrum = Spectrum(
            mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
            intensities=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            metadata={"id": "nl", "precursor_mz": 500.0},
        )
        vector = spectrum_to_binned_vector(
            spectrum, bin_width=100.0, mz_min=0.0, mz_max=500.0
        )
        assert vector.shape == (10,)  # 2 channels × 5 bins

        # Channel 0: m/z 100 → bin 1, 200 → bin 2, 300 → bin 3.
        exact_channel = np.zeros(5, dtype=np.float32)
        exact_channel[1] = 1.0
        exact_channel[2] = 2.0
        exact_channel[3] = 3.0
        exact_channel /= np.linalg.norm(exact_channel)
        assert np.allclose(vector[:5], exact_channel / np.sqrt(2.0), atol=1e-6)

        # Channel 1: neutral losses 400 → bin 4, 300 → bin 3, 200 → bin 2.
        nl_channel = np.zeros(5, dtype=np.float32)
        nl_channel[2] = 3.0
        nl_channel[3] = 2.0
        nl_channel[4] = 1.0
        nl_channel /= np.linalg.norm(nl_channel)
        assert np.allclose(vector[5:], nl_channel / np.sqrt(2.0), atol=1e-6)

    def test_out_of_range_peaks_ignored(self) -> None:
        """Peaks outside [mz_min, mz_max) do not contribute."""
        spectrum = Spectrum(
            mz=np.array([-5.0, 0.0, 1.0, 2001.0], dtype=np.float64),
            intensities=np.array([9.0, 1.0, 1.0, 9.0], dtype=np.float64),
            metadata={"id": "range"},
        )
        vector = spectrum_to_binned_vector(
            spectrum, bin_width=1.0, mz_min=0.0, mz_max=2.0
        )
        # Channel 0: bins [1, 1] normalized; channel 1: zeros (no precursor).
        expected = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(vector, expected / np.linalg.norm(expected))

    def test_empty_spectrum_yields_zero_vector(self) -> None:
        """Spectra with no in-range peaks map to the zero vector."""
        spectrum = Spectrum(
            mz=np.array([3000.0], dtype=np.float64),
            intensities=np.array([1.0], dtype=np.float64),
            metadata={"id": "empty"},
        )
        vector = spectrum_to_binned_vector(
            spectrum, bin_width=1.0, mz_min=0.0, mz_max=2000.0
        )
        assert vector.shape == (4000,)  # 2 channels × 2000 bins
        assert np.count_nonzero(vector) == 0

    def test_bin_spectra_matrix(self) -> None:
        """bin_spectra returns a normalized (n, 2*dim) float32 matrix."""
        spectra = make_random_library(6)
        matrix = bin_spectra(spectra, bin_width=10.0, mz_min=0.0, mz_max=1000.0)
        assert matrix.shape == (6, 200)  # 2 channels × 100 bins
        assert matrix.dtype == np.float32
        norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms[norms > 0], 1.0)


# ---------------------------------------------------------------------------
# HNSW index
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_HNSWLIB, reason="hnswlib not installed")
@pytest.mark.optional
class TestHNSWSpectralIndex:
    """hnswlib-backed index wrapper."""

    def test_build_and_query_nearest_self(self) -> None:
        """Each indexed spectrum's nearest neighbour is itself."""
        spectra = make_random_library(30)
        index = HNSWSpectralIndex.from_spectra(
            spectra,
            bin_width=2.0,
            mz_min=0.0,
            mz_max=1000.0,
            m=16,
            ef_construction=200,
            max_elements=100,
        )
        assert len(index) == 30
        query_vectors = bin_spectra(spectra, bin_width=2.0, mz_min=0.0, mz_max=1000.0)
        candidate_ids, distances = index.query(query_vectors[:5], k=1, ef_search=64)
        for position, per_query in enumerate(candidate_ids):
            assert per_query[0] == str(spectra[position].get("id"))
            assert distances[position][0] <= 1e-6  # cosine distance ~ 0

    def test_query_matches_brute_force(self) -> None:
        """With generous ef, HNSW returns the brute-force binned top-k."""
        spectra = make_random_library(60)
        index = HNSWSpectralIndex.from_spectra(
            spectra,
            bin_width=2.0,
            mz_min=0.0,
            mz_max=1000.0,
            m=16,
            ef_construction=200,
            max_elements=100,
        )
        vectors = bin_spectra(spectra, bin_width=2.0, mz_min=0.0, mz_max=1000.0)
        k = 10
        candidate_ids, _ = index.query(vectors[:5], k=k, ef_search=256)

        for query_position in range(5):
            query_vector = vectors[query_position]
            # einsum avoids macOS Accelerate's spurious matmul warnings.
            cosines = np.einsum(
                "ij,j->i",
                vectors.astype(np.float64),
                query_vector.astype(np.float64),
            )
            expected_ids = {
                str(spectra[pos].get("id"))
                for pos in np.argsort(cosines)[::-1][:k].tolist()
            }
            assert set(candidate_ids[query_position]) == expected_ids

    def test_shifted_analogue_retrieval_via_neutral_loss_channel(self) -> None:
        """The 2-channel index retrieves precursors-shifted analogues.

        The analogue shares the target's neutral-loss profile but has
        completely disjoint exact m/z values (a modified-cosine match that
        classical cosine cannot see). An exact-m/z-only index would be
        blind to it; the neutral-loss channel must rank it first.
        """
        target = Spectrum(
            mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
            intensities=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            metadata={"id": "target", "precursor_mz": 500.0},
        )
        # Same neutral losses (400, 300, 200), precursor shifted +50, all
        # fragment m/z values disjoint from the target's. Fragment
        # intensities stay correlated with their neutral losses (real
        # analogues share fragment chemistry, not exact intensities).
        analogue = Spectrum(
            mz=np.array([150.0, 250.0, 350.0], dtype=np.float64),
            intensities=np.array([1.2, 1.9, 3.1], dtype=np.float64),
            metadata={"id": "analogue", "precursor_mz": 550.0},
        )
        distractors = make_random_library(40, seed=13)

        library = [analogue] + distractors
        index = HNSWSpectralIndex.from_spectra(
            library,
            bin_width=10.0,
            mz_min=0.0,
            mz_max=600.0,
            m=16,
            ef_construction=200,
            max_elements=100,
        )

        target_vector = spectrum_to_binned_vector(
            target, bin_width=10.0, mz_min=0.0, mz_max=600.0
        )

        # Sanity: the exact-m/z channels are orthogonal (disjoint bins),
        # while the neutral-loss channels are near-identical — only the
        # neutral-loss channel can retrieve the analogue.
        analogue_vector = spectrum_to_binned_vector(
            analogue, bin_width=10.0, mz_min=0.0, mz_max=600.0
        )
        mz_overlap = float(
            np.dot(target_vector[:60], analogue_vector[:60])
            / max(
                np.linalg.norm(target_vector[:60])
                * np.linalg.norm(analogue_vector[:60]),
                1e-12,
            )
        )
        nl_overlap = float(
            np.dot(target_vector[60:], analogue_vector[60:])
            / max(
                np.linalg.norm(target_vector[60:])
                * np.linalg.norm(analogue_vector[60:]),
                1e-12,
            )
        )
        assert mz_overlap < 0.01, "Analogue exact-m/z channel must be disjoint."
        assert nl_overlap > 0.99, "Analogue neutral-loss channel must match."

        # HNSW query for the target must return the shifted analogue first.
        candidate_ids, _ = index.query(target_vector.reshape(1, -1), k=5, ef_search=128)
        assert candidate_ids[0][0] == "analogue"
        # And the analogue must also be the brute-force 2-channel top-1.
        library_vectors = bin_spectra(library, bin_width=10.0, mz_min=0.0, mz_max=600.0)
        cosines = np.einsum(
            "ij,j->i",
            library_vectors.astype(np.float64),
            target_vector.astype(np.float64),
        )
        assert str(library[int(np.argmax(cosines))].get("id")) == "analogue"

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """A saved index restores ids, graph, and query behaviour."""
        spectra = make_random_library(20)
        index = HNSWSpectralIndex.from_spectra(
            spectra,
            bin_width=2.0,
            mz_min=0.0,
            mz_max=1000.0,
            m=8,
            ef_construction=100,
            max_elements=50,
        )
        index_path = tmp_path / "index.bin"
        index.save(index_path)

        restored = HNSWSpectralIndex.load(index_path)
        assert len(restored) == 20
        assert restored.dim == index.dim
        assert restored.m == index.m

        query_vectors = bin_spectra(
            spectra[:3], bin_width=2.0, mz_min=0.0, mz_max=1000.0
        )
        original_candidates, _ = index.query(query_vectors, k=3, ef_search=32)
        restored_candidates, _ = restored.query(query_vectors, k=3, ef_search=32)
        assert original_candidates == restored_candidates

    def test_non_metric_recall_with_generous_parameters(self) -> None:
        """Generous M/ef_construction/ef_search preserve recall.

        Modified cosine is non-metric, so graph quality is validated against
        the brute-force ranking in the binned cosine space: with the default
        tuning (M=32, ef_construction=400) and ef_search >= k, retrieval
        recall of the exact top-5 must be perfect on this random library.
        """
        spectra = make_random_library(200, seed=11)
        index = HNSWSpectralIndex.from_spectra(
            spectra,
            bin_width=2.0,
            mz_min=0.0,
            mz_max=1000.0,
            m=32,
            ef_construction=400,
            max_elements=400,
        )
        vectors = bin_spectra(spectra, bin_width=2.0, mz_min=0.0, mz_max=1000.0)
        k = 5
        candidate_ids, _ = index.query(vectors, k=20, ef_search=200)

        recall = 0.0
        for query_position in range(len(spectra)):
            # einsum avoids macOS Accelerate's spurious matmul warnings.
            cosines = np.einsum(
                "ij,j->i",
                vectors.astype(np.float64),
                vectors[query_position].astype(np.float64),
            )
            exact_top_k = {
                str(spectra[pos].get("id"))
                for pos in np.argsort(cosines)[::-1][:k].tolist()
            }
            retrieved = set(candidate_ids[query_position])
            recall += len(exact_top_k & retrieved) / k
        assert recall / len(spectra) == 1.0

    def test_validation_errors(self) -> None:
        """Invalid construction/query parameters raise ValueError."""
        with pytest.raises(ValueError):
            HNSWSpectralIndex(dim=100, m=0)
        with pytest.raises(ValueError):
            HNSWSpectralIndex(dim=100, m=16, ef_construction=8)

        index = HNSWSpectralIndex(dim=10, m=8, ef_construction=64)
        with pytest.raises(ValueError):
            index.query(np.zeros((1, 10), dtype=np.float32), k=5)
        with pytest.raises(ValueError):
            index.add_items(np.zeros((1, 3), dtype=np.float32), ["x"])

    def test_constructor_requires_hnswlib(self, monkeypatch) -> None:
        """A missing hnswlib raises RuntimeError with install guidance."""
        monkeypatch.setattr(hnsw_module, "_HAS_HNSWLIB", False)
        with pytest.raises(RuntimeError, match="hnswlib"):
            HNSWSpectralIndex(dim=10)


# ---------------------------------------------------------------------------
# CascadeEngine HNSW integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_HNSWLIB, reason="hnswlib not installed")
@pytest.mark.optional
class TestCascadeHNSWIntegration:
    """Seamless HNSW candidate retrieval inside the cascade engine."""

    def _cascade_config(self, **hnsw_kwargs) -> SimilarityConfig:
        """Build a cascade config with classical stages and HNSW settings."""
        return SimilarityConfig(
            algorithm="cascade",
            ms2_tolerance=0.02,
            min_score=0.0,
            min_matched_peaks=1,
            cascade_lower_bound=0.1,
            cascade_upper_bound=0.0,
            cascade_stages=["cosine", "modified_cosine"],
            **hnsw_kwargs,
        )

    def test_hnsw_exhaustive_candidates_match_full_cascade(self) -> None:
        """HNSW with k == library size reproduces the exact cascade."""
        refs = make_random_library(60, seed=31)
        rng = np.random.default_rng(6)
        queries: list[Spectrum] = []
        for index in range(8):
            base = refs[index]
            mz = base.peaks.mz + rng.uniform(-0.005, 0.005, base.peaks.mz.size)
            queries.append(
                Spectrum(
                    mz=np.sort(mz),
                    intensities=base.peaks.intensities.copy(),
                    metadata={
                        "id": f"cascade_q_{index}",
                        "precursor_mz": base.get("precursor_mz"),
                    },
                )
            )

        hnsw_config = self._cascade_config(
            hnsw_enabled=True,
            hnsw_m=8,
            hnsw_ef_construction=64,
            hnsw_ef_search=60,
            hnsw_candidates_per_query=60,
            hnsw_bin_width=2.0,
            hnsw_mz_max=1000.0,
        )
        full_config = self._cascade_config(hnsw_enabled=False)

        hnsw_results = CascadeEngine(hnsw_config).search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        full_results = CascadeEngine(full_config).search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        assert summarize_results(hnsw_results) == summarize_results(full_results)

    def test_hnsw_candidates_are_subset_of_full_results(self) -> None:
        """Sub-linear candidate retrieval only narrows the exact cascade."""
        refs = make_random_library(100, seed=41)
        rng = np.random.default_rng(42)
        queries: list[Spectrum] = []
        for index in range(5):
            base = refs[index]
            mz = base.peaks.mz + rng.uniform(-0.005, 0.005, base.peaks.mz.size)
            queries.append(
                Spectrum(
                    mz=np.sort(mz),
                    intensities=base.peaks.intensities.copy(),
                    metadata={
                        "id": f"hnsw_query_{index}",
                        "precursor_mz": base.get("precursor_mz"),
                    },
                )
            )

        hnsw_config = self._cascade_config(
            hnsw_enabled=True,
            hnsw_m=16,
            hnsw_ef_construction=200,
            hnsw_ef_search=128,
            hnsw_candidates_per_query=30,
            hnsw_bin_width=2.0,
            hnsw_mz_max=1000.0,
        )
        full_config = self._cascade_config(hnsw_enabled=False)

        engine = CascadeEngine(hnsw_config)
        hnsw_results = engine.search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        full_results = CascadeEngine(full_config).search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        hnsw_summary = set(summarize_results(hnsw_results))
        full_summary = set(summarize_results(full_results))
        assert hnsw_summary.issubset(full_summary)
        assert len(hnsw_summary) > 0

    def test_hnsw_index_cached_across_searches(self) -> None:
        """Repeated searches against the same library reuse one index."""
        refs = make_random_library(40, seed=51)
        queries_a = make_random_library(3, seed=52)
        queries_b = make_random_library(3, seed=53)

        config = self._cascade_config(
            hnsw_enabled=True,
            hnsw_m=8,
            hnsw_ef_construction=64,
            hnsw_ef_search=40,
            hnsw_candidates_per_query=40,
            hnsw_bin_width=2.0,
            hnsw_mz_max=1000.0,
        )
        engine = CascadeEngine(config)
        engine.search(
            query_spectra=queries_a, reference_spectra=refs, include_decoys=False
        )
        first_index = engine._hnsw_index
        engine.search(
            query_spectra=queries_b, reference_spectra=refs, include_decoys=False
        )
        assert engine._hnsw_index is first_index

    def test_hnsw_invalid_ef_falls_back_to_full_cascade(self) -> None:
        """ef_search < k triggers the exact-scoring fallback gracefully."""
        refs = make_random_library(50, seed=61)
        queries = make_random_library(4, seed=62)

        bad_config = self._cascade_config(
            hnsw_enabled=True,
            hnsw_m=8,
            hnsw_ef_construction=64,
            hnsw_ef_search=10,
            hnsw_candidates_per_query=50,
            hnsw_bin_width=2.0,
            hnsw_mz_max=1000.0,
        )
        full_config = self._cascade_config(hnsw_enabled=False)

        bad_results = CascadeEngine(bad_config).search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        full_results = CascadeEngine(full_config).search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        assert summarize_results(bad_results) == summarize_results(full_results)

    def test_hnsw_missing_library_falls_back(self, monkeypatch) -> None:
        """Cascade degrades to exact scoring when hnswlib is unavailable."""
        refs = make_random_library(30, seed=71)
        queries = make_random_library(3, seed=72)

        config = self._cascade_config(
            hnsw_enabled=True,
            hnsw_bin_width=2.0,
            hnsw_mz_max=1000.0,
        )
        engine = CascadeEngine(config)
        monkeypatch.setattr(hnsw_module, "_HAS_HNSWLIB", False)

        results = engine.search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        full_engine = CascadeEngine(self._cascade_config(hnsw_enabled=False))
        full_results = full_engine.search(
            query_spectra=queries, reference_spectra=refs, include_decoys=False
        )
        assert summarize_results(results) == summarize_results(full_results)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestHNSWConfig:
    """HNSW settings exposed on SimilarityConfig."""

    def test_defaults_are_recall_friendly(self) -> None:
        """Defaults favor recall: M=32, ef_construction=400, ef_search=200."""
        config = SimilarityConfig()
        assert config.hnsw_enabled is False
        assert config.hnsw_m == 32
        assert config.hnsw_ef_construction == 400
        assert config.hnsw_ef_search == 200
        assert config.hnsw_candidates_per_query == 200
        assert config.enable_numba_prefilter is True

    def test_ef_construction_below_m_rejected(self) -> None:
        """ef_construction < M is rejected (unstable graph construction)."""
        with pytest.raises(ValueError, match="hnsw_ef_construction"):
            SimilarityConfig(
                algorithm="cascade",
                hnsw_enabled=True,
                hnsw_m=64,
                hnsw_ef_construction=32,
            )

    def test_invalid_binning_range_rejected(self) -> None:
        """Non-positive bin width or inverted range are rejected."""
        with pytest.raises(ValueError):
            SimilarityConfig(hnsw_enabled=True, hnsw_bin_width=0.0)
        with pytest.raises(ValueError, match="hnsw_mz_min"):
            SimilarityConfig(hnsw_enabled=True, hnsw_mz_min=2000.0, hnsw_mz_max=1000.0)

    @pytest.mark.optional
    def test_hnsw_parameters_flow_to_index(self) -> None:
        """Config values reach the constructed HNSW index (needs hnswlib)."""
        if not _HAS_HNSWLIB:
            pytest.skip("hnswlib not installed")
        spectra = make_random_library(10, seed=81)
        config = self._cascade_config(
            hnsw_enabled=True,
            hnsw_m=24,
            hnsw_ef_construction=300,
            hnsw_bin_width=5.0,
            hnsw_mz_min=0.0,
            hnsw_mz_max=1000.0,
        )
        engine = CascadeEngine(config)
        index = engine._get_hnsw_index(spectra)
        assert index is not None
        assert index.m == 24
        assert index.ef_construction == 300
        # Two-channel layout: exact m/z + neutral loss, 2 × (1000 / 5).
        assert index.dim == 400

    def _cascade_config(self, **kwargs) -> SimilarityConfig:
        return SimilarityConfig(
            algorithm="cascade",
            cascade_stages=["cosine", "modified_cosine"],
            **kwargs,
        )
