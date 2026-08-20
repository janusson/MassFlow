"""
Comprehensive coverage tests for MassFlow similarity.py:
- _ms1_prefilter_arrays
- yield_fixed_chunks
- _handle_lazy_reference_spectra (top_n grouping, chunking)
- calculate_mass_error_ppm
- _is_missing
- _adduct_modes_compatible
- generate_decoys (edge cases)
- calculate_empirical_p_values (edge cases)
- SimilarityEngine (error handling, edge cases, ref_precursor_mzs, ref_is_decoy)
- get_similarity_engine
"""

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import similarity
from MassFlow.config import SimilarityConfig


# ==============================================================================
# Helper fixtures
# ==============================================================================


def _make_spectrum(precursor_mz, spec_id="test", n_peaks=5, **meta):
    rng = np.random.default_rng(42)
    mz = np.sort(rng.uniform(50.0, 500.0, n_peaks)).astype(np.float64)
    intensities = rng.uniform(0.1, 1.0, n_peaks).astype(np.float64)
    metadata = {"id": spec_id, "precursor_mz": precursor_mz, **meta}
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)


# ==============================================================================
# calculate_mass_error_ppm
# ==============================================================================


def test_calculate_mass_error_ppm():
    err = similarity.calculate_mass_error_ppm(200.0, 200.0)
    assert err == 0.0

    err = similarity.calculate_mass_error_ppm(200.02, 200.0)
    assert abs(err - 100.0) < 1.0  # ~100 ppm


# ==============================================================================
# yield_fixed_chunks
# ==============================================================================


def test_yield_fixed_chunks_empty():
    chunks = list(similarity.yield_fixed_chunks([], chunk_size=100))
    assert chunks == []


def test_yield_fixed_chunks_smaller_than_chunk():
    s1 = _make_spectrum(100.0, "s1")
    s2 = _make_spectrum(200.0, "s2")
    chunks = list(similarity.yield_fixed_chunks([s1, s2], chunk_size=100))
    assert len(chunks) == 1
    assert len(chunks[0]) == 2


def test_yield_fixed_chunks_exact_chunk():
    s = [_make_spectrum(100.0, f"s{i}") for i in range(3)]
    chunks = list(similarity.yield_fixed_chunks(s, chunk_size=3))
    assert len(chunks) == 1
    assert len(chunks[0]) == 3


def test_yield_fixed_chunks_multi_chunk():
    s = [_make_spectrum(100.0, f"s{i}") for i in range(5)]
    chunks = list(similarity.yield_fixed_chunks(s, chunk_size=2))
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [2, 2, 1]


# ==============================================================================
# _is_missing
# ==============================================================================


def test_is_missing_none():
    assert similarity._is_missing(None) is True


def test_is_missing_nan():
    result = similarity._is_missing(float("nan"))
    assert bool(result) is True


def test_is_missing_valid():
    result = similarity._is_missing(100.0)
    assert bool(result) is False


def test_is_missing_invalid_string():
    assert similarity._is_missing("not_a_number") is True


# ==============================================================================
# _adduct_modes_compatible
# ==============================================================================


def test_adduct_modes_compatible_both_none():
    assert similarity._adduct_modes_compatible(None, None) is True


def test_adduct_modes_compatible_one_none():
    assert similarity._adduct_modes_compatible("[M+H]+", None) is True
    assert similarity._adduct_modes_compatible(None, "[M-H]-") is True


def test_adduct_modes_compatible_both_positive():
    assert similarity._adduct_modes_compatible("[M+H]+", "[M+Na]+") is True


def test_adduct_modes_compatible_positive_vs_negative():
    assert similarity._adduct_modes_compatible("[M+H]+", "[M-H]-") is False


def test_adduct_modes_compatible_negative_vs_positive():
    assert similarity._adduct_modes_compatible("[M-H]-", "[M+H]+") is False


# ==============================================================================
# _ms1_prefilter_arrays
# ==============================================================================


def test_ms1_prefilter_arrays_basic():
    ref_mzs = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    q1 = _make_spectrum(200.0, "q1")
    q2 = _make_spectrum(400.0, "q2")
    queries = [q1, q2]

    rows, cols = similarity._ms1_prefilter_arrays(ref_mzs, queries, ms1_tolerance=0.5)
    # q1 (precursor=200) should match ref index 1
    assert len(rows) > 0


def test_ms1_prefilter_arrays_with_ppm():
    ref_mzs = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    q1 = _make_spectrum(200.0, "q1")
    queries = [q1]

    rows, cols = similarity._ms1_prefilter_arrays(
        ref_mzs, queries, ms1_tolerance=0.5, resolution_ppm=10.0
    )
    # ppm-based window
    assert len(rows) > 0


def test_ms1_prefilter_arrays_missing_ref():
    """Reference precursors <= 0 are treated as missing."""
    ref_mzs = np.array([0.0, -1.0, 200.0], dtype=np.float64)
    q1 = _make_spectrum(200.0, "q1")
    queries = [q1]

    rows, cols = similarity._ms1_prefilter_arrays(ref_mzs, queries, ms1_tolerance=0.5)
    # The two missing refs should match, plus the exact match
    # Actually missing refs (<= 0) bypass the filter
    assert len(rows) >= 3


def test_ms1_prefilter_arrays_missing_query():
    """Queries with missing precursors bypass filter."""
    ref_mzs = np.array([100.0, 200.0], dtype=np.float64)
    q1 = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "q_missing"},
    )
    queries = [q1]

    rows, cols = similarity._ms1_prefilter_arrays(ref_mzs, queries, ms1_tolerance=0.5)
    # Missing query should match all refs
    assert len(rows) >= 2


def test_ms1_prefilter_arrays_no_matches():
    ref_mzs = np.array([100.0], dtype=np.float64)
    q1 = _make_spectrum(9000.0, "far_away")
    queries = [q1]

    rows, cols = similarity._ms1_prefilter_arrays(ref_mzs, queries, ms1_tolerance=0.5)
    assert len(rows) == 0


# ==============================================================================
# generate_decoys
# ==============================================================================


def test_generate_decoys_basic():
    s = _make_spectrum(200.0, "target", n_peaks=5)
    decoys = similarity.generate_decoys([s])
    assert len(decoys) == 1
    d = decoys[0]
    assert d.get("is_decoy") is True
    assert "decoy" in str(d.get("id"))


def test_generate_decoys_randomizes_fragment_positions():
    """Entropy-based decoys jitter fragment positions away from the source.

    Phase 3 replaces naive intensity shuffling with entropy-preserving
    decoy generation: decoys preserve the precursor m/z and the spectral
    entropy of the source, but fragment positions are jittered so decoys
    share no fragments with their source at scoring tolerance. The old
    contract (decoys keep the exact m/z array) is intentionally retired.
    """
    s = _make_spectrum(200.0, "target", n_peaks=10)
    decoys = similarity.generate_decoys([s])
    decoy_mz = decoys[0].peaks.mz
    assert not np.array_equal(decoy_mz, s.peaks.mz)
    # Precursor m/z (metadata) is still preserved exactly.
    assert float(decoys[0].get("precursor_mz")) == pytest.approx(
        float(s.get("precursor_mz")), rel=1e-12
    )


def test_generate_decoys_all_same_intensity():
    """When all intensities are identical, use taper."""
    s = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensities=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        metadata={"id": "uniform", "precursor_mz": 200.0},
    )
    decoys = similarity.generate_decoys([s])
    assert len(decoys) == 1
    # Intensities should be different from original
    assert not np.array_equal(decoys[0].peaks.intensities, s.peaks.intensities)


def test_generate_decoys_few_peaks():
    """Low peak count spectra get jitter."""
    s = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([1.0, 2.0], dtype=np.float64),
        metadata={"id": "few_peaks", "precursor_mz": 200.0},
    )
    decoys = similarity.generate_decoys([s])
    assert len(decoys) == 1


def test_generate_decoys_handles_compound_name():
    s = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([0.5, 1.0], dtype=np.float64),
        metadata={"id": "named", "precursor_mz": 200.0, "compound_name": "Caffeine"},
    )
    decoys = similarity.generate_decoys([s])
    assert "Caffeine_decoy" in str(decoys[0].get("compound_name"))


def test_generate_decoys_custom_seed():
    s = _make_spectrum(200.0, "target", n_peaks=10)
    decoys1 = similarity.generate_decoys([s], random_seed=123)
    decoys2 = similarity.generate_decoys([s], random_seed=456)
    # Different seeds should produce different shuffled intensities
    assert not np.array_equal(
        decoys1[0].peaks.intensities, decoys2[0].peaks.intensities
    )


# ==============================================================================
# calculate_empirical_p_values
# ==============================================================================


def test_calculate_empirical_p_values_empty_decoys():
    target = np.array([0.9, 0.8, 0.7])
    decoy = np.array([])
    p_vals = similarity.calculate_empirical_p_values(target, decoy)
    np.testing.assert_array_equal(p_vals, np.ones_like(target))


def test_calculate_empirical_p_values_basic():
    target = np.array([0.9])
    decoy = np.array([0.5, 0.3, 0.1])
    p_vals = similarity.calculate_empirical_p_values(target, decoy)
    assert p_vals[0] < 1.0


# ==============================================================================
# SimilarityEngine
# ==============================================================================


class TestSimilarityEngine:
    """Cover SimilarityEngine edge cases."""

    def test_unsupported_algorithm_raises(self):
        """Test that SimilarityEngine raises for unsupported algorithm (bypassed Pydantic validation)."""
        cfg = SimilarityConfig(algorithm="cosine")
        cfg.algorithm = "invalid"  # type: ignore
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            similarity.SimilarityEngine(cfg)

    def test_get_similarity_engine_factory(self):
        cfg = SimilarityConfig(algorithm="cosine")
        engine = similarity.get_similarity_engine(cfg)
        assert isinstance(engine, similarity.SimilarityEngine)

    def test_search_empty_queries(self):
        cfg = SimilarityConfig(algorithm="cosine", min_score=0.0)
        engine = similarity.SimilarityEngine(cfg)
        ref = _make_spectrum(200.0, "ref")
        results = engine.search([], [ref])
        assert results == []

    def test_search_empty_references(self):
        cfg = SimilarityConfig(algorithm="cosine", min_score=0.0)
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query")
        results = engine.search([q], [])
        assert results == []

    def test_search_without_decoys(self):
        cfg = SimilarityConfig(algorithm="cosine", min_score=0.0, ms1_tolerance=10.0)
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=20)
        ref = _make_spectrum(200.0, "ref", n_peaks=20)
        results = engine.search([q], [ref], include_decoys=False)
        # Should have results without decoys
        assert isinstance(results, list)

    def test_search_direct(self):
        """Test search with standard args."""
        cfg = SimilarityConfig(
            algorithm="cosine",
            min_score=0.0,
            ms1_tolerance=1000.0,
            ms2_tolerance=1000.0,
        )
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=10, adduct="[M+H]+")
        ref = _make_spectrum(200.0, "ref", n_peaks=10, adduct="[M+H]+")
        results = engine.search(
            [q],
            [ref],
            include_decoys=False,
        )
        assert isinstance(results, list)

    def test_search_with_min_score_override(self):
        cfg = SimilarityConfig(algorithm="cosine", min_score=0.0, ms1_tolerance=10.0)
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=20)
        ref = _make_spectrum(200.0, "ref", n_peaks=20)
        results = engine.search([q], [ref], min_score=2.0, include_decoys=False)
        # Min score > 1.0 should filter everything
        assert len([r for r in results if not r.get("is_decoy")]) == 0

    def test_search_with_top_n(self):
        cfg = SimilarityConfig(algorithm="cosine", min_score=0.0, ms1_tolerance=10.0)
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=20)
        ref1 = _make_spectrum(200.0, "ref1", n_peaks=20)
        ref2 = _make_spectrum(200.0, "ref2", n_peaks=20)
        ref3 = _make_spectrum(200.0, "ref3", n_peaks=20)
        results = engine.search([q], [ref1, ref2, ref3], top_n=2, include_decoys=False)
        non_decoy = [r for r in results if not r.get("is_decoy")]
        assert len(non_decoy) <= 2

    def test_search_rt_filtering(self):
        cfg = SimilarityConfig(
            algorithm="cosine", min_score=0.0, ms1_tolerance=10.0, rt_tolerance=1.0
        )
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(
            200.0, "query", n_peaks=20, adduct="[M+H]+", retention_time=10.0
        )
        ref = _make_spectrum(
            200.0, "ref", n_peaks=20, adduct="[M+H]+", retention_time=100.0
        )
        results = engine.search([q], [ref], include_decoys=False)
        # RT diff = 90, tolerance = 1 → should be filtered
        non_decoy = [r for r in results if not r.get("is_decoy")]
        assert len(non_decoy) == 0

    def test_search_adduct_mode_incompatible(self):
        cfg = SimilarityConfig(algorithm="cosine", min_score=0.0, ms1_tolerance=10.0)
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=20, adduct="[M+H]+")
        ref = _make_spectrum(200.0, "ref", n_peaks=20, adduct="[M-H]-")
        results = engine.search([q], [ref], include_decoys=False)
        non_decoy = [r for r in results if not r.get("is_decoy")]
        assert len(non_decoy) == 0

    def test_search_modified_cosine(self):
        cfg = SimilarityConfig(
            algorithm="modified_cosine", min_score=0.0, ms2_tolerance=0.1
        )
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=20)
        ref = _make_spectrum(200.0, "ref", n_peaks=20)
        results = engine.search([q], [ref], include_decoys=False)
        assert isinstance(results, list)

    def test_search_modified_cosine_with_ppm_warning(self):
        """When resolution_ppm is set with modified_cosine, it logs but does not crash."""
        cfg = SimilarityConfig(
            algorithm="modified_cosine",
            min_score=0.0,
            ms2_tolerance=0.1,
            resolution_ppm=10.0,
        )
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=20)
        ref = _make_spectrum(200.0, "ref", n_peaks=20)
        results = engine.search([q], [ref], include_decoys=False)
        assert isinstance(results, list)

    def test_get_similarity_engine_unsupported(self):
        cfg = SimilarityConfig(algorithm="cosine")
        cfg.algorithm = "unsupported"  # type: ignore
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            similarity.get_similarity_engine(cfg)


class TestLazyReferenceHandling:
    """Cover _handle_lazy_reference_spectra decorator (top_n grouping, chunking)."""

    def test_lazy_reference_with_iterator_input(self):
        """When reference_spectra is an iterator, the decorator chunks it."""
        cfg = SimilarityConfig(algorithm="cosine", min_score=0.0, ms1_tolerance=10.0)
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=20)

        def ref_iter():
            for i in range(3):
                yield _make_spectrum(200.0, f"ref{i}", n_peaks=20)

        results = engine.search([q], ref_iter(), include_decoys=False)
        non_decoy = [r for r in results if not r.get("is_decoy")]
        assert len(non_decoy) == 3

    def test_lazy_reference_with_top_n_and_iterator(self):
        """When top_n is specified with lazy references, grouping works."""
        cfg = SimilarityConfig(algorithm="cosine", min_score=0.0, ms1_tolerance=10.0)
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=20)

        def ref_iter():
            for i in range(5):
                yield _make_spectrum(200.0, f"ref{i}", n_peaks=20)

        results = engine.search([q], ref_iter(), top_n=2, include_decoys=False)
        non_decoy = [r for r in results if not r.get("is_decoy")]
        assert len(non_decoy) <= 2

    def test_lazy_reference_with_empty_chunk(self):
        """Empty chunks are skipped by the decorator."""
        cfg = SimilarityConfig(algorithm="cosine", min_score=0.0, ms1_tolerance=10.0)
        engine = similarity.SimilarityEngine(cfg)
        q = _make_spectrum(200.0, "query", n_peaks=20)

        # Create an iterable that is not a list/tuple but empty
        results = engine.search([q], iter([]), include_decoys=False)
        assert results == []


class TestCalculateFDREdgeCases:
    """Cover calculate_fdr with edge cases already partially covered; fill gaps."""

    def test_both_empty(self):
        sorted_scores, q_values, is_target = similarity.calculate_fdr(
            np.array([]), np.array([])
        )
        assert len(sorted_scores) == 0
        assert len(q_values) == 0
        assert len(is_target) == 0

    def test_only_targets(self):
        sorted_scores, q_values, is_target = similarity.calculate_fdr(
            np.array([0.9, 0.8]), np.array([])
        )
        assert len(sorted_scores) == 2
        assert np.all(is_target)

    def test_only_decoys(self):
        sorted_scores, q_values, is_target = similarity.calculate_fdr(
            np.array([]), np.array([0.5, 0.3])
        )
        assert len(sorted_scores) == 2
        assert not np.any(is_target)
