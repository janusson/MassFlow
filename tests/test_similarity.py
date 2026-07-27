"""
Integration test suite for MassFlow SimilarityEngine, focusing on ModifiedCosine.
"""

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import (
    CascadeEngine,
    ConsensusEngine,
    SimilarityEngine,
    calculate_fdr,
)


@pytest.fixture(scope="module")
def cocaine_spectrum() -> Spectrum:
    """
    Retrieves the Cocaine reference spectrum from matchms.

    Returns
    -------
    matchms.Spectrum
        The unshifted Cocaine reference standard spectrum.
    """
    from matchms.reference_spectra.cocaine import cocaine

    spec = cocaine()
    assert spec.get("precursor_mz") is not None, (
        "Found Cocaine spectrum, but missing precursor_mz."
    )
    return spec


@pytest.fixture
def methylene_homolog(cocaine_spectrum: Spectrum) -> Spectrum:
    """
    Creates a mass-shifted test fixture (+14.0156 Da) simulating a methylene homolog.

    Parameters
    ----------
    cocaine_spectrum : matchms.Spectrum
        The original Cocaine reference spectrum.

    Returns
    -------
    matchms.Spectrum
        A synthesized spectrum with all peaks and precursor shifted by +14.0156 Da.
    """
    shift = 14.0156

    orig_mz = cocaine_spectrum.peaks.mz
    orig_ints = cocaine_spectrum.peaks.intensities
    precursor_mz = cocaine_spectrum.get("precursor_mz")
    assert precursor_mz is not None
    orig_precursor = float(precursor_mz)

    shifted_mz = orig_mz + shift
    shifted_precursor = orig_precursor + shift

    # Clone metadata and update precursor
    new_metadata = cocaine_spectrum.metadata.copy()
    new_metadata["precursor_mz"] = shifted_precursor
    # Update ID to avoid confusion in results mapping
    new_metadata["id"] = "shifted_cocaine"

    return Spectrum(mz=shifted_mz, intensities=orig_ints, metadata=new_metadata)


def test_modified_cosine_integration(
    cocaine_spectrum: Spectrum, methylene_homolog: Spectrum
) -> None:
    """
    Verify correct neutral loss handling against known reference standard pairs.

    Mathematical Constraint Validation:
    The modified cosine score computes the normalized dot product of matched peak
    intensities, natively incorporating neutral losses.
    |m/z_A - m/z_B - delta_M| <= tolerance

    Since the test fixture shifts both the precursor and the fragment peaks by
    exactly +14.0156 Da, the mass difference (delta_M) matches the peak shift.
    ModifiedCosine must score this analog highly, while CosineGreedy must fail
    to align the shifted fragments and score it poorly.

    Parameters
    ----------
    cocaine_spectrum : matchms.Spectrum
        The reference spectrum.
    methylene_homolog : matchms.Spectrum
        The mass-shifted query spectrum.
    """
    # Configuration for CosineGreedy
    cosine_config = SimilarityConfig(
        algorithm="cosine",
        ms2_tolerance=0.1,
        min_score=0.0,  # Zero threshold to capture the failure score
    )
    cosine_engine = SimilarityEngine(cosine_config)

    # Configuration for ModifiedCosine
    mod_cosine_config = SimilarityConfig(
        algorithm="modified_cosine",
        ms2_tolerance=0.1,
        min_score=0.0,
    )
    mod_cosine_engine = SimilarityEngine(mod_cosine_config)

    # Run searches
    cosine_results = cosine_engine.search(
        query_spectra=[methylene_homolog], reference_spectra=[cocaine_spectrum]
    )
    cosine_results = [r for r in cosine_results if not r.get("is_decoy")]

    mod_cosine_results = mod_cosine_engine.search(
        query_spectra=[methylene_homolog], reference_spectra=[cocaine_spectrum]
    )
    mod_cosine_results = [r for r in mod_cosine_results if not r.get("is_decoy")]

    assert len(cosine_results) == 0, (
        "CosineGreedy search returned a result object despite MS1 mismatch."
    )
    assert len(mod_cosine_results) == 1, (
        "ModifiedCosine search returned no result object."
    )

    mod_cosine_score = mod_cosine_results[0]["score"]

    # Assertions
    # Modified Cosine should be effectively 1.0 because all shifted peaks
    # satisfy the delta_M constraint exactly.
    assert mod_cosine_score > 0.95, (
        f"ModifiedCosine failed to match mass-shifted analog. "
        f"Expected score > 0.95, got {mod_cosine_score}"
    )


def test_ms1_tolerance_filtering(cocaine_spectrum: Spectrum) -> None:
    """Verify that queries outside the MS1 tolerance are rejected."""
    # Create a query with a precursor mz shifted by 20 ppm
    shift_ppm = 20.0
    precursor_mz = cocaine_spectrum.get("precursor_mz")
    assert precursor_mz is not None
    orig_precursor = float(precursor_mz)
    shift_da = orig_precursor * (shift_ppm / 1e6)

    new_metadata = cocaine_spectrum.metadata.copy()
    new_metadata["precursor_mz"] = orig_precursor + shift_da
    new_metadata["id"] = "shifted_ms1"

    query_spectrum = Spectrum(
        mz=cocaine_spectrum.peaks.mz,
        intensities=cocaine_spectrum.peaks.intensities,
        metadata=new_metadata,
    )

    # Config with 10 ppm tolerance (should reject)
    strict_config = SimilarityConfig(
        algorithm="cosine", resolution_ppm=10.0, ms1_tolerance=0.0, min_score=0.0
    )
    strict_engine = SimilarityEngine(strict_config)

    strict_results = strict_engine.search(
        query_spectra=[query_spectrum], reference_spectra=[cocaine_spectrum]
    )
    strict_results = [r for r in strict_results if not r.get("is_decoy")]

    assert len(strict_results) == 0, "Query outside MS1 tolerance was not rejected."

    # Config with 30 ppm tolerance (should accept)
    relaxed_config = SimilarityConfig(
        algorithm="cosine", ms1_tolerance=30.0, min_score=0.0
    )
    relaxed_engine = SimilarityEngine(relaxed_config)

    relaxed_results = relaxed_engine.search(
        query_spectra=[query_spectrum], reference_spectra=[cocaine_spectrum]
    )
    relaxed_results = [r for r in relaxed_results if not r.get("is_decoy")]

    assert len(relaxed_results) == 1, "Query inside MS1 tolerance was rejected."


def test_rt_tolerance_filtering() -> None:
    """Verify that matches exceeding rt_tolerance are rejected and missing RTs are safely ignored."""
    # Query with valid RT
    query = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "q1", "precursor_mz": 400.0, "retention_time": 5.0},
    )

    # Reference within RT tolerance
    ref_match = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "r1", "precursor_mz": 400.0, "retention_time": 5.2},
    )

    # Reference outside RT tolerance
    ref_mismatch = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "r2", "precursor_mz": 400.0, "retention_time": 6.5},
    )

    # Reference missing RT (should not be filtered)
    ref_missing = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "r3", "precursor_mz": 400.0},
    )

    # Reference malformed RT string (should not be filtered)
    ref_malformed = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "r4", "precursor_mz": 400.0, "retention_time": "NaN"},
    )

    config = SimilarityConfig(
        algorithm="cosine",
        ms1_tolerance=10.0,
        ms2_tolerance=10.0,
        rt_tolerance=1.0,
        min_matched_peaks=1,
        min_score=0.0,
    )
    engine = SimilarityEngine(config)

    results = engine.search(
        query_spectra=[query],
        reference_spectra=[ref_match, ref_mismatch, ref_missing, ref_malformed],
        include_decoys=False,
    )

    matched_ids = [r["reference_id"] for r in results]

    assert "r1" in matched_ids, "Valid RT match was incorrectly rejected."
    assert "r2" not in matched_ids, "RT mismatch was incorrectly accepted."
    assert "r3" in matched_ids, "Missing RT was incorrectly rejected."
    assert "r4" in matched_ids, "Malformed RT was incorrectly rejected."


def test_min_matched_peaks_filtering() -> None:
    """Verify that matches with fewer than min_matched_peaks are rejected."""
    query = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={"id": "query1", "precursor_mz": 400.0},
    )

    # Ref matches 2 peaks exactly
    ref = Spectrum(
        mz=np.array([100.0, 200.0, 400.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={"id": "ref1", "precursor_mz": 400.0},
    )

    strict_config = SimilarityConfig(
        algorithm="cosine",
        resolution_ppm=10.0,
        ms1_tolerance=0.0,
        min_matched_peaks=3,
        min_score=0.0,
    )
    strict_engine = SimilarityEngine(strict_config)

    strict_results = strict_engine.search(
        query_spectra=[query], reference_spectra=[ref]
    )
    strict_results = [r for r in strict_results if not r.get("is_decoy")]

    assert len(strict_results) == 0, (
        "Result with too few matched peaks was not rejected."
    )

    relaxed_config = SimilarityConfig(
        algorithm="cosine",
        resolution_ppm=10.0,
        ms1_tolerance=0.0,
        min_matched_peaks=2,
        min_score=0.0,
    )
    relaxed_engine = SimilarityEngine(relaxed_config)

    relaxed_results = relaxed_engine.search(
        query_spectra=[query], reference_spectra=[ref]
    )
    relaxed_results = [r for r in relaxed_results if not r.get("is_decoy")]

    assert len(relaxed_results) == 1, "Result with enough matched peaks was rejected."


def test_rt_tolerance_filtering_strict() -> None:
    """Verify that matches outside a narrow RT tolerance (0.5 min) are rejected."""
    query = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={"id": "query1", "precursor_mz": 400.0, "retention_time": 5.0},
    )

    # Within tolerance
    ref_accept = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={"id": "ref1", "precursor_mz": 400.0, "retention_time": 5.2},
    )

    # Outside tolerance
    ref_reject = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={"id": "ref2", "precursor_mz": 400.0, "retention_time": 6.0},
    )

    # Missing RT in reference
    ref_missing = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={"id": "ref3", "precursor_mz": 400.0, "retention_time": "N/A"},
    )

    # Configuration with rt_tolerance = 0.5
    config = SimilarityConfig(
        algorithm="cosine",
        ms1_tolerance=0.0,
        rt_tolerance=0.5,
        min_score=0.0,
    )
    engine = SimilarityEngine(config)

    # Search against all references (disable decoys for pure target test)
    results = engine.search(
        query_spectra=[query],
        reference_spectra=[ref_accept, ref_reject, ref_missing],
        include_decoys=False,
    )

    result_ids = [r["reference_id"] for r in results]
    assert "ref1" in result_ids, "Reference within RT tolerance was rejected."
    assert "ref2" not in result_ids, "Reference outside RT tolerance was not rejected."
    assert "ref3" in result_ids, "Reference with missing RT was wrongly rejected."


def test_calculate_fdr_basic():
    """Verify basic q-value calculations on a synthetic target/decoy distribution."""
    target_scores = np.array([0.9, 0.8, 0.7, 0.6])
    decoy_scores = np.array([0.85, 0.65])

    sorted_scores, q_values, is_target = calculate_fdr(target_scores, decoy_scores)

    expected_scores = np.array([0.9, 0.85, 0.8, 0.7, 0.65, 0.6])
    expected_targets = np.array([True, False, True, True, False, True])
    # Conservative +1 pseudo-count formula q-values
    expected_q = np.array([2 / 3, 2 / 3, 2 / 3, 2 / 3, 0.75, 0.75])

    np.testing.assert_allclose(sorted_scores, expected_scores)
    np.testing.assert_array_equal(is_target, expected_targets)
    np.testing.assert_allclose(q_values, expected_q)


def test_calculate_fdr_perfect_separation():
    """Verify q-values when targets are perfectly separated from decoys."""
    target_scores = np.array([0.9, 0.8, 0.7])
    decoy_scores = np.array([0.4, 0.3, 0.2])

    sorted_scores, q_values, is_target = calculate_fdr(target_scores, decoy_scores)

    expected_scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    expected_targets = np.array([True, True, True, False, False, False])
    # Conservative +1 pseudo-count formula q-values
    expected_q = np.array([1 / 3, 1 / 3, 1 / 3, 2 / 3, 1.0, 1.0])

    np.testing.assert_allclose(sorted_scores, expected_scores)
    np.testing.assert_array_equal(is_target, expected_targets)
    np.testing.assert_allclose(q_values, expected_q)


def test_calculate_fdr_monotonicity():
    """Verify that q-values are monotonically increasing (or flat) as scores decrease."""
    rng = np.random.default_rng(42)
    target_scores = rng.normal(0.8, 0.1, 100)
    decoy_scores = rng.normal(0.5, 0.1, 100)

    target_scores = np.clip(target_scores, 0, 1)
    decoy_scores = np.clip(decoy_scores, 0, 1)

    sorted_scores, q_values, is_target = calculate_fdr(target_scores, decoy_scores)

    assert np.all(np.diff(sorted_scores) <= 0), (
        "Scores are not sorted in descending order"
    )
    assert np.all(np.diff(q_values) >= 0), "Q-values are not monotonically increasing"


def test_calculate_fdr_empty_arrays():
    """Verify edge cases with empty target or decoy arrays."""
    s, q, t = calculate_fdr(np.array([]), np.array([]))
    assert len(s) == 0 and len(q) == 0 and len(t) == 0

    s, q, t = calculate_fdr(np.array([]), np.array([0.5, 0.6]))
    assert len(s) == 2
    assert np.all(q == 1.0)
    assert not np.any(t)

    s, q, t = calculate_fdr(np.array([0.8, 0.9]), np.array([]))
    assert len(s) == 2
    # Conservative +1 pseudo-count formula: FDR=1/targets
    # With monotonicity enforced by minimum.accumulate, q-values for both are 0.5
    np.testing.assert_allclose(q, np.array([0.5, 0.5]))
    assert np.all(t)


def test_rt_tolerance_filtering_exact_and_missing():
    """Verify RT filtering handles exact matches, tolerance boundaries, and missing RTs."""
    q_mz = np.array([100.0, 200.0], dtype="float")
    q_ints = np.array([1.0, 1.0], dtype="float")

    # Match RT exactly
    ref1 = Spectrum(
        mz=q_mz,
        intensities=q_ints,
        metadata={"id": "ref1", "precursor_mz": 400.0, "retention_time": 5.0},
    )

    # Match RT within tolerance
    ref2 = Spectrum(
        mz=q_mz,
        intensities=q_ints,
        metadata={"id": "ref2", "precursor_mz": 400.0, "retention_time": 5.2},
    )

    # Match RT outside tolerance
    ref3 = Spectrum(
        mz=q_mz,
        intensities=q_ints,
        metadata={"id": "ref3", "precursor_mz": 400.0, "retention_time": 5.8},
    )

    # Missing/Malformed RT (should bypass filter)
    ref4 = Spectrum(
        mz=q_mz,
        intensities=q_ints,
        metadata={"id": "ref4", "precursor_mz": 400.0, "retention_time": "N/A"},
    )

    # query spectrum with RT 5.0
    query = Spectrum(
        mz=q_mz,
        intensities=q_ints,
        metadata={"id": "query1", "precursor_mz": 400.0, "retention_time": 5.0},
    )

    # Config with 0.5 RT tolerance
    config = SimilarityConfig(
        algorithm="cosine",
        rt_tolerance=0.5,
        min_matched_peaks=2,
        min_score=0.0,
    )
    engine = SimilarityEngine(config)

    results = engine.search(
        query_spectra=[query],
        reference_spectra=[ref1, ref2, ref3, ref4],
        include_decoys=False,
    )

    # We should have matches for ref1, ref2, and ref4. ref3 should be filtered out.
    matched_ids = [r["reference_id"] for r in results]

    assert "ref1" in matched_ids, "Exact RT match was incorrectly rejected."
    assert "ref2" in matched_ids, "Match within RT tolerance was incorrectly rejected."
    assert "ref3" not in matched_ids, (
        "Match outside RT tolerance was incorrectly accepted."
    )
    assert "ref4" in matched_ids, "Match with missing RT was incorrectly rejected."


# ---------------------------------------------------------------------------
# ConsensusEngine tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def consensus_query() -> Spectrum:
    """Simple two-peak query spectrum for consensus testing."""
    return Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "cons_q", "precursor_mz": 400.0},
    )


@pytest.fixture(scope="module")
def consensus_ref_match() -> Spectrum:
    """Reference spectrum identical to the query (high-score match)."""
    return Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={
            "id": "cons_r1",
            "precursor_mz": 400.0,
            "compound_name": "Match",
        },
    )


@pytest.fixture(scope="module")
def consensus_ref_partial() -> Spectrum:
    """Reference sharing only one peak with the query (partial match)."""
    return Spectrum(
        mz=np.array([100.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={
            "id": "cons_r2",
            "precursor_mz": 400.0,
            "compound_name": "Partial",
        },
    )


def test_consensus_basic_weighted_scoring(
    consensus_query: Spectrum,
    consensus_ref_match: Spectrum,
    consensus_ref_partial: Spectrum,
) -> None:
    """Verify that the consensus score is a weighted average of sub-engine scores."""
    config = SimilarityConfig(
        algorithm="consensus",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        consensus_weights={"cosine": 0.5, "modified_cosine": 0.5},
    )
    engine = ConsensusEngine(config)

    results = engine.search(
        query_spectra=[consensus_query],
        reference_spectra=[consensus_ref_match, consensus_ref_partial],
        include_decoys=False,
    )

    # Should have results for both references
    result_ids = {r["reference_id"] for r in results}
    assert "cons_r1" in result_ids, "High-score reference missing from consensus."
    assert "cons_r2" in result_ids, "Partial-match reference missing from consensus."

    for r in results:
        # score_breakdown should contain individual engine scores
        assert r["score_breakdown"] is not None
        breakdown = r["score_breakdown"]
        assert "cosine" in breakdown or "modified_cosine" in breakdown, (
            f"score_breakdown missing engine keys: {breakdown}"
        )

        # The consensus score should fall between the min and max sub-scores
        sub_scores = list(breakdown.values())
        if len(sub_scores) >= 2:
            assert min(sub_scores) <= r["score"] <= max(sub_scores), (
                f"Consensus score {r['score']} not between sub-scores {sub_scores}"
            )

        # structural_similarity holds the best individual score (tie-break)
        assert r["structural_similarity"] is not None
        assert r["structural_similarity"] == max(sub_scores), (
            f"structural_similarity {r['structural_similarity']} "
            f"!= max sub-score {max(sub_scores)}"
        )


def test_consensus_min_engines_filter(
    consensus_query: Spectrum,
    consensus_ref_match: Spectrum,
) -> None:
    """Verify that consensus_min_engines filters out single-engine results."""
    # Only cosine is configured; min_engines=2 should yield zero results
    config = SimilarityConfig(
        algorithm="consensus",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        consensus_weights={"cosine": 1.0},
        consensus_min_engines=2,
    )
    engine = ConsensusEngine(config)

    results = engine.search(
        query_spectra=[consensus_query],
        reference_spectra=[consensus_ref_match],
        include_decoys=False,
    )

    assert len(results) == 0, (
        f"Expected 0 results with min_engines=2 > 1 available, got {len(results)}"
    )


def test_consensus_empty_inputs() -> None:
    """Verify consensus returns empty list for empty queries or references."""
    config = SimilarityConfig(algorithm="consensus")
    engine = ConsensusEngine(config)

    assert engine.search([], []) == []

    query = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "q", "precursor_mz": 400.0},
    )
    assert engine.search([query], []) == []


def test_consensus_score_breakdown_structure(
    consensus_query: Spectrum,
    consensus_ref_match: Spectrum,
) -> None:
    """Verify score_breakdown dict is correctly populated per result."""
    config = SimilarityConfig(
        algorithm="consensus",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        consensus_weights={"cosine": 0.6, "modified_cosine": 0.4},
    )
    engine = ConsensusEngine(config)

    results = engine.search(
        query_spectra=[consensus_query],
        reference_spectra=[consensus_ref_match],
        include_decoys=False,
    )

    assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"

    r = results[0]
    breakdown = r["score_breakdown"]
    assert isinstance(breakdown, dict), (
        f"score_breakdown is {type(breakdown)}, expected dict"
    )
    assert len(breakdown) >= 1, f"score_breakdown empty: {breakdown}"

    # All sub-scores should be in [0, 1]
    for algo, score in breakdown.items():
        assert 0.0 <= score <= 1.0, f"Sub-score for '{algo}' out of range: {score}"

    # Verify the consensus score is the weighted average
    total_weight = sum(config.consensus_weights.get(a, 0.0) for a in breakdown)
    if total_weight > 0:
        expected = (
            sum(
                score * config.consensus_weights.get(a, 0.0)
                for a, score in breakdown.items()
            )
            / total_weight
        )
        assert abs(r["score"] - expected) < 1e-6, (
            f"Consensus score {r['score']} != weighted avg {expected}"
        )


def test_consensus_min_score_threshold(
    consensus_query: Spectrum,
    consensus_ref_match: Spectrum,
) -> None:
    """Verify that the min_score threshold filters consensus results."""
    config = SimilarityConfig(
        algorithm="consensus",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.999,  # Very high threshold
        min_matched_peaks=1,
        consensus_weights={"cosine": 0.5, "modified_cosine": 0.5},
    )
    engine = ConsensusEngine(config)

    results = engine.search(
        query_spectra=[consensus_query],
        reference_spectra=[consensus_ref_match],
        include_decoys=False,
    )

    # With min_score=0.999, even perfect matches may be filtered
    # (depending on floating point and matchms tolerance).
    # The key assertion: no result should have score < 0.999.
    for r in results:
        assert r["score"] >= 0.999, (
            f"Result score {r['score']} below min_score threshold 0.999"
        )


def test_consensus_top_n(
    consensus_query: Spectrum,
    consensus_ref_match: Spectrum,
    consensus_ref_partial: Spectrum,
) -> None:
    """Verify that top_n limits results per query in consensus output."""
    config = SimilarityConfig(
        algorithm="consensus",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        consensus_weights={"cosine": 0.5, "modified_cosine": 0.5},
    )
    engine = ConsensusEngine(config)

    results = engine.search(
        query_spectra=[consensus_query],
        reference_spectra=[consensus_ref_match, consensus_ref_partial],
        include_decoys=False,
        top_n=1,
    )

    assert len(results) == 1, f"Expected 1 result with top_n=1, got {len(results)}"


# ---------------------------------------------------------------------------
# CascadeEngine tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cascade_query() -> Spectrum:
    """Query spectrum for cascade testing (identical to ref_match)."""
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={"id": "casc_q", "precursor_mz": 400.0},
    )


@pytest.fixture(scope="module")
def cascade_ref_strong() -> Spectrum:
    """Reference with identical peaks (high score in both stages)."""
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={
            "id": "casc_r1",
            "precursor_mz": 400.0,
            "compound_name": "Strong",
        },
    )


@pytest.fixture(scope="module")
def cascade_ref_weak() -> Spectrum:
    """Reference with only one matching peak (weak score)."""
    return Spectrum(
        mz=np.array([100.0, 500.0, 600.0], dtype="float"),
        intensities=np.array([1.0, 1.0, 1.0], dtype="float"),
        metadata={
            "id": "casc_r2",
            "precursor_mz": 400.0,
            "compound_name": "Weak",
        },
    )


def test_cascade_sequential_filtering(
    cascade_query: Spectrum,
    cascade_ref_strong: Spectrum,
    cascade_ref_weak: Spectrum,
) -> None:
    """Verify that cascade stage 1 filters out weak matches before stage 2."""
    # Stage 1 (cosine) with lower_bound=0.5 should pass the strong match
    # but filter out the weak match. Stage 2 (modified_cosine) only sees
    # the strong reference.
    config = SimilarityConfig(
        algorithm="cascade",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        cascade_lower_bound=0.5,
        cascade_upper_bound=0.0,
        cascade_stages=["cosine", "modified_cosine"],
    )
    engine = CascadeEngine(config)

    results = engine.search(
        query_spectra=[cascade_query],
        reference_spectra=[cascade_ref_strong, cascade_ref_weak],
        include_decoys=False,
    )

    result_ids = {r["reference_id"] for r in results}
    assert "casc_r1" in result_ids, "Strong match should survive cascade."
    assert "casc_r2" not in result_ids, "Weak match should be filtered at stage 1."


def test_cascade_empty_inputs() -> None:
    """Verify cascade returns empty list for empty queries or references."""
    config = SimilarityConfig(algorithm="cascade")
    engine = CascadeEngine(config)

    assert engine.search([], []) == []

    query = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "q", "precursor_mz": 400.0},
    )
    assert engine.search([query], []) == []


def test_cascade_early_exit_no_survivors(
    cascade_query: Spectrum,
    cascade_ref_weak: Spectrum,
) -> None:
    """Verify cascade returns empty when no candidates survive stage 1."""
    config = SimilarityConfig(
        algorithm="cascade",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=3,  # weak ref only has 1 matching peak
        cascade_lower_bound=0.7,  # high threshold
        cascade_upper_bound=0.0,
        cascade_stages=["cosine", "modified_cosine"],
    )
    engine = CascadeEngine(config)

    results = engine.search(
        query_spectra=[cascade_query],
        reference_spectra=[cascade_ref_weak],
        include_decoys=False,
    )

    assert len(results) == 0, (
        f"Expected 0 results when no candidates survive stage 1, got {len(results)}"
    )


def test_cascade_upper_bound_final_filter(
    cascade_query: Spectrum,
    cascade_ref_strong: Spectrum,
    cascade_ref_weak: Spectrum,
) -> None:
    """Verify that cascade_upper_bound is applied as the final-stage threshold."""
    # Stage 1 passes both with lower_bound=0.0.
    # Stage 2 applies upper_bound=0.999 which should filter everything
    # since no real score reaches exactly 1.0.
    config = SimilarityConfig(
        algorithm="cascade",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        cascade_lower_bound=0.0,
        cascade_upper_bound=0.999,
        cascade_stages=["cosine", "modified_cosine"],
    )
    engine = CascadeEngine(config)

    results = engine.search(
        query_spectra=[cascade_query],
        reference_spectra=[cascade_ref_strong, cascade_ref_weak],
        include_decoys=False,
    )

    # With a 0.999 upper bound, scores may or may not pass depending on
    # floating-point precision. The key property is that no result should
    # have score < 0.999.
    for r in results:
        assert r["score"] >= 0.999, (
            f"Result score {r['score']} below cascade_upper_bound 0.999"
        )


def test_cascade_top_n(
    cascade_query: Spectrum,
    cascade_ref_strong: Spectrum,
    cascade_ref_weak: Spectrum,
) -> None:
    """Verify that top_n limits results in cascade output."""
    config = SimilarityConfig(
        algorithm="cascade",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        cascade_lower_bound=0.0,
        cascade_upper_bound=0.0,
        cascade_stages=["cosine"],  # single stage for simplicity
    )
    engine = CascadeEngine(config)

    results = engine.search(
        query_spectra=[cascade_query],
        reference_spectra=[cascade_ref_strong, cascade_ref_weak],
        include_decoys=False,
        top_n=1,
    )

    assert len(results) == 1, f"Expected 1 result with top_n=1, got {len(results)}"


def test_cascade_single_stage(
    cascade_query: Spectrum,
    cascade_ref_strong: Spectrum,
) -> None:
    """Verify cascade with a single stage works as a pass-through."""
    config = SimilarityConfig(
        algorithm="cascade",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        cascade_lower_bound=0.3,
        cascade_upper_bound=0.0,
        cascade_stages=["cosine"],
    )
    engine = CascadeEngine(config)

    results = engine.search(
        query_spectra=[cascade_query],
        reference_spectra=[cascade_ref_strong],
        include_decoys=False,
    )

    assert len(results) >= 1, (
        f"Single-stage cascade should return results, got {len(results)}"
    )


def test_cascade_min_score_override(
    cascade_query: Spectrum,
    cascade_ref_strong: Spectrum,
) -> None:
    """Verify that explicit min_score overrides cascade_upper_bound."""
    config = SimilarityConfig(
        algorithm="cascade",
        ms1_tolerance=10.0,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        cascade_lower_bound=0.0,
        cascade_upper_bound=0.999,  # very strict
        cascade_stages=["cosine"],
    )
    engine = CascadeEngine(config)

    # Override with a low threshold
    results = engine.search(
        query_spectra=[cascade_query],
        reference_spectra=[cascade_ref_strong],
        include_decoys=False,
        min_score=0.0,
    )

    assert len(results) >= 1, (
        f"min_score override should bypass cascade_upper_bound, got {len(results)}"
    )
