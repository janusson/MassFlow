"""
Integration test suite for MassFlow SimilarityEngine, focusing on ModifiedCosine.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import SimilarityEngine, calculate_fdr


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
    if spec.get("precursor_mz") is None:
        pytest.fail("Found Cocaine spectrum, but missing precursor_mz.")
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
    orig_precursor = float(cocaine_spectrum.get("precursor_mz"))

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
        tolerance=0.1,
        min_score=0.0,  # Zero threshold to capture the failure score
    )
    cosine_engine = SimilarityEngine(cosine_config)

    # Configuration for ModifiedCosine
    mod_cosine_config = SimilarityConfig(
        algorithm="modified_cosine",
        tolerance=0.1,
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

    assert (
        len(cosine_results) == 0
    ), "CosineGreedy search returned a result object despite MS1 mismatch."
    assert (
        len(mod_cosine_results) == 1
    ), "ModifiedCosine search returned no result object."

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
    orig_precursor = float(cocaine_spectrum.get("precursor_mz"))
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

    assert (
        len(strict_results) == 0
    ), "Result with too few matched peaks was not rejected."

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


@pytest.mark.experimental
def test_spec2vec_initialization():
    """Verify spec2vec engine initialization handles mock models correctly."""
    import sys
    from unittest.mock import MagicMock

    mock_gensim = MagicMock()
    mock_spec2vec = MagicMock()

    config = SimilarityConfig(algorithm="spec2vec", model_path=Path("dummy.model"))
    with (
        patch.dict(
            sys.modules,
            {
                "gensim": mock_gensim,
                "gensim.models": mock_gensim.models,
                "spec2vec": mock_spec2vec,
            },
        ),
        patch.object(Path, "exists", return_value=True),
    ):
        engine = SimilarityEngine(config)
        mock_gensim.models.Word2Vec.load.assert_called_once_with("dummy.model")
        mock_spec2vec.Spec2Vec.assert_called_once()
        assert engine.similarity_function == mock_spec2vec.Spec2Vec.return_value


@pytest.mark.experimental
def test_ms2deepscore_initialization():
    """Verify ms2deepscore engine initialization handles mock models correctly."""
    import sys
    from unittest.mock import MagicMock

    mock_ms2deepscore = MagicMock()

    config = SimilarityConfig(algorithm="ms2deepscore", model_path=Path("dummy.model"))
    with (
        patch.dict(
            sys.modules,
            {
                "ms2deepscore": mock_ms2deepscore,
                "ms2deepscore.models": mock_ms2deepscore.models,
            },
        ),
        patch.object(Path, "exists", return_value=True),
    ):
        engine = SimilarityEngine(config)
        mock_ms2deepscore.models.load_model.assert_called_once_with("dummy.model")
        mock_ms2deepscore.MS2DeepScore.assert_called_once()
        assert engine.similarity_function == mock_ms2deepscore.MS2DeepScore.return_value


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

    assert np.all(
        np.diff(sorted_scores) <= 0
    ), "Scores are not sorted in descending order"
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
