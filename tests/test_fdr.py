"""Unit tests for calculate_fdr in MassFlow.similarity

These tests check behavior on edge-cases: no decoys, no targets, ties, and general monotonicity.

Phase 3 additions cover entropy-based decoy generation:
- Shannon entropy computation (nats) with strict baseline noise filtering.
- Exact entropy preservation between targets and entropy-based decoys.
- Statistical validation that target-decoy entropy distributions do not
  systematically diverge (KS test + per-pair deltas).
"""

import numpy as np
import pytest
from matchms import Spectrum
from pydantic import ValidationError

from MassFlow.config import ProcessingConfig
from MassFlow.similarity import (
    calculate_fdr,
    compare_target_decoy_entropy,
    generate_decoys,
    spectral_entropy,
)


def _is_non_decreasing(arr: np.ndarray) -> bool:
    return bool(np.all(np.diff(arr) >= -1e-8))


def test_calculate_fdr_no_decoys_properties():
    targets = np.array([0.9, 0.8, 0.7], dtype=float)
    decoys = np.array([], dtype=float)

    sorted_scores, q_values, is_target = calculate_fdr(targets, decoys)

    assert sorted_scores.shape == q_values.shape
    assert sorted_scores.size == 3
    assert q_values.size == 3
    assert np.all(q_values >= 0.0) and np.all(q_values <= 1.0)
    # q-values should be non-decreasing as score decreases
    assert _is_non_decreasing(q_values)
    # All entries are targets when no decoys present
    assert np.all(is_target)


def test_calculate_fdr_no_targets_returns_all_ones_and_is_target_false():
    targets = np.array([], dtype=float)
    decoys = np.array([0.5, 0.4], dtype=float)

    sorted_scores, q_values, is_target = calculate_fdr(targets, decoys)

    assert sorted_scores.size == 2
    assert q_values.size == 2
    assert np.allclose(q_values, 1.0)
    assert not np.any(is_target)


def test_calculate_fdr_ties_put_targets_before_decoys():
    targets = np.array([0.8], dtype=float)
    decoys = np.array([0.8], dtype=float)

    sorted_scores, q_values, is_target = calculate_fdr(targets, decoys)

    # When scores tie, targets must be ordered before decoys (is_target True first)
    assert is_target[0]
    assert not is_target[1]


def test_calculate_fdr_mixed_case_monotonicity_and_bounds():
    targets = np.array([0.9, 0.88, 0.5], dtype=float)
    decoys = np.array([0.85, 0.4], dtype=float)

    sorted_scores, q_values, is_target = calculate_fdr(targets, decoys)

    assert sorted_scores.size == 5
    assert q_values.size == 5
    assert np.all(q_values >= 0.0) and np.all(q_values <= 1.0)
    assert _is_non_decreasing(q_values)


# ---------------------------------------------------------------------------
# Entropy-based decoy generation (Phase 3)
# ---------------------------------------------------------------------------


def _realistic_spectral_library(
    count: int, seed: int = 42, include_noise: bool = True
) -> list[Spectrum]:
    """
    Generate a deterministic library of realistic random spectra.

    Fragment intensities follow a log-normal decay profile (a few intense
    peaks and a long tail); when ``include_noise`` is set, low-abundance
    baseline noise peaks (0.01-0.5% of total ion current) are appended to
    exercise the strict noise-thresholding requirement.

    Parameters
    ----------
    count : int
        Number of spectra.
    seed : int, optional
        RNG seed.
    include_noise : bool, optional
        Append baseline noise peaks below the default relative threshold.

    Returns
    -------
    list[Spectrum]
    """
    rng = np.random.default_rng(seed)
    spectra: list[Spectrum] = []
    for index in range(count):
        n_peaks = int(rng.integers(5, 30))
        mz = np.sort(rng.uniform(50.0, 900.0, size=n_peaks)).astype(np.float64)
        intensities = np.exp(rng.normal(-1.0, 1.5, size=n_peaks)).astype(np.float64)
        if include_noise and n_peaks >= 5:
            n_noise = int(rng.integers(1, 4))
            noise_intensities = (
                intensities.sum() * rng.uniform(0.0001, 0.005, size=n_noise)
            ).astype(np.float64)
            noise_mz = np.sort(rng.uniform(50.0, 900.0, size=n_noise)).astype(
                np.float64
            )
            mz = np.sort(np.concatenate([mz, noise_mz])).astype(np.float64)
            intensities = np.concatenate([intensities, noise_intensities]).astype(
                np.float64
            )
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


class TestSpectralEntropy:
    """Shannon entropy computation over normalized fragment intensities."""

    def test_uniform_intensities_reach_maximum_entropy(self) -> None:
        """n equal-intensity peaks have entropy ln(n)."""
        assert spectral_entropy(np.array([1.0, 1.0])) == pytest.approx(
            np.log(2.0), rel=1e-12
        )
        assert spectral_entropy(np.array([1.0, 1.0, 1.0])) == pytest.approx(
            np.log(3.0), rel=1e-12
        )

    def test_single_peak_has_zero_entropy(self) -> None:
        """A deterministic single-peak spectrum carries no information."""
        assert spectral_entropy(np.array([1.0])) == 0.0
        assert spectral_entropy(np.array([1.0, 0.0])) == 0.0
        assert spectral_entropy(np.array([])) == 0.0
        assert spectral_entropy(np.zeros(3)) == 0.0

    def test_hand_computed_entropy(self) -> None:
        """Sqrt-weighted spectral entropy matches the definition.

        w = [sqrt(2), 1, 1] -> p = w / sum(w); H = -Σ p ln p. The sqrt
        weighting differs from plain Shannon over linear intensities
        (p = [0.5, 0.25, 0.25] -> 1.5 ln 2), which over-weights the base
        peak and under-weights low-abundance fragments.
        """
        weights = np.sqrt(np.array([2.0, 1.0, 1.0]))
        probabilities = weights / weights.sum()
        expected = float(-np.sum(probabilities * np.log(probabilities)))
        assert spectral_entropy(np.array([2.0, 1.0, 1.0])) == pytest.approx(
            expected, rel=1e-12
        )

        plain_shannon = 1.5 * np.log(2.0)
        assert expected != pytest.approx(plain_shannon, rel=1e-6)

    def test_scale_invariance(self) -> None:
        """Entropy is invariant to global intensity scaling."""
        intensities = np.array([10.0, 4.0, 2.0, 1.0])
        assert spectral_entropy(intensities) == pytest.approx(
            spectral_entropy(intensities * 137.5), rel=1e-12
        )

    def test_noise_peaks_skew_unfiltered_entropy(self) -> None:
        """Sub-1%-base-peak noise alters entropy unless baseline-filtered.

        With the default 1%-of-base-peak floor the noise peak is excluded;
        with a permissive floor it is retained and changes the
        sqrt-weighted entropy estimate.
        """
        intensities = np.array([100.0, 2.0, 0.1])

        def expected_entropy(values: np.ndarray) -> float:
            weights = np.sqrt(values)
            probabilities = weights / weights.sum()
            return float(-np.sum(probabilities * np.log(probabilities)))

        filtered = spectral_entropy(intensities, min_relative_intensity=0.01)
        noise_included = spectral_entropy(intensities, min_relative_intensity=0.0001)
        # Base peak = 100: the 0.1 peak is below 1.0 (1%) but above 0.01.
        assert filtered == pytest.approx(
            expected_entropy(np.array([100.0, 2.0])), rel=1e-12
        )
        assert noise_included == pytest.approx(expected_entropy(intensities), rel=1e-12)
        assert filtered != pytest.approx(noise_included, rel=1e-9)


class TestEntropyDecoyGeneration:
    """Entropy-preserving decoy generation."""

    def test_precursor_and_identity_preserved(self) -> None:
        """Decoys keep precursor m/z and suffixed identifiers."""
        targets = _realistic_spectral_library(5, seed=1)
        decoys = generate_decoys(targets, random_seed=42)
        assert len(decoys) == len(targets)
        for target, decoy in zip(targets, decoys):
            assert float(decoy.get("precursor_mz")) == pytest.approx(
                float(target.get("precursor_mz")), rel=1e-12
            )
            assert str(decoy.get("id")).endswith("_decoy")
            assert decoy.get("is_decoy") is True

    def test_entropy_preserved_exactly_per_spectrum(self) -> None:
        """Each decoy's entropy equals its source spectrum's entropy.

        The decoy intensity profile is a permutation of the noise-filtered
        target profile, so the normalized distribution — and therefore the
        Shannon entropy — is preserved up to floating-point rounding.
        """
        targets = _realistic_spectral_library(100, seed=7)
        decoys = generate_decoys(targets, random_seed=123)
        for target, decoy in zip(targets, decoys):
            target_entropy = spectral_entropy(
                np.asarray(target.peaks.intensities, dtype=np.float64)
            )
            decoy_entropy = spectral_entropy(
                np.asarray(decoy.peaks.intensities, dtype=np.float64)
            )
            assert decoy_entropy == pytest.approx(target_entropy, abs=1e-12)
            assert float(decoy.get("spectral_entropy")) == pytest.approx(
                target_entropy, abs=1e-12
            )

    def test_target_decoy_entropy_distributions_do_not_diverge(self) -> None:
        """Statistical validation of target-decoy entropy calibration.

        Across a realistic random library with baseline noise peaks, the
        target and decoy entropy distributions must not systematically
        diverge: per-pair deltas are at floating-point precision, the
        two-sample Kolmogorov-Smirnov test finds no distributional
        difference, and the mean delta is unbiased (zero).
        """
        from scipy import stats

        targets = _realistic_spectral_library(300, seed=11)
        decoys = generate_decoys(targets, random_seed=99)

        target_entropies = np.array(
            [
                spectral_entropy(np.asarray(s.peaks.intensities, dtype=np.float64))
                for s in targets
            ]
        )
        decoy_entropies = np.array(
            [
                spectral_entropy(np.asarray(s.peaks.intensities, dtype=np.float64))
                for s in decoys
            ]
        )

        deltas = np.abs(target_entropies - decoy_entropies)
        # Exact preservation: per-pair delta bounded by fp rounding.
        assert deltas.max() <= 1e-9
        # No systematic bias in either direction.
        assert abs(float(np.mean(target_entropies - decoy_entropies))) <= 1e-12
        # Distributional equality: KS test must not reject. The statistic's
        # smallest nonzero value is 1/n (one discrete step); identical
        # distributions can only reach that step.
        ks_statistic, ks_p_value = stats.ks_2samp(
            target_entropies, decoy_entropies, method="asymp"
        )
        assert ks_statistic <= 1.0 / len(target_entropies) + 1e-9
        assert ks_p_value > 0.05

    def test_compare_target_decoy_entropy_diagnostic(self) -> None:
        """The workflow diagnostic reports near-zero divergence."""
        targets = _realistic_spectral_library(50, seed=21)
        decoys = generate_decoys(targets, random_seed=22)
        comparison = compare_target_decoy_entropy(targets, decoys)
        assert comparison["compared_pairs"] == 50
        assert comparison["mean_abs_entropy_delta"] <= 1e-9
        assert comparison["max_abs_entropy_delta"] <= 1e-9
        assert comparison["mean_target_entropy"] == pytest.approx(
            comparison["mean_decoy_entropy"], rel=1e-12
        )

    def test_compare_target_decoy_entropy_detects_divergence(self) -> None:
        """Mismatched distributions produce a large reported divergence."""
        targets = _realistic_spectral_library(40, seed=31)
        # Uniform intensities maximize entropy — systematically different
        # from the log-normal target profiles.
        uniform_spectra = [
            Spectrum(
                mz=np.linspace(50.0, 900.0, 12),
                intensities=np.ones(12, dtype=np.float64),
                metadata={"id": f"uni_{i}", "precursor_mz": 500.0},
            )
            for i in range(40)
        ]
        comparison = compare_target_decoy_entropy(targets, uniform_spectra)
        assert comparison["mean_abs_entropy_delta"] > 0.1

    def test_compare_target_decoy_entropy_empty(self) -> None:
        """Empty inputs yield NaN statistics."""
        comparison = compare_target_decoy_entropy([], [])
        assert np.isnan(comparison["mean_abs_entropy_delta"])
        assert comparison["compared_pairs"] == 0

    def test_randomizes_fragmentation_pathways(self) -> None:
        """Decoy fragment positions differ from their source positions.

        With the default ±1.0 Da jitter, essentially all decoy peaks move
        outside the 0.02 Da scoring tolerance of their source position,
        breaking the fragment-position correlation that naive intensity
        shuffling retains.
        """
        targets = _realistic_spectral_library(100, seed=41)
        decoys = generate_decoys(targets, random_seed=42, mz_shift_da=1.0)
        n_close = 0
        n_total = 0
        for target, decoy in zip(targets, decoys):
            target_mz = np.asarray(target.peaks.mz, dtype=np.float64)
            decoy_mz = np.asarray(decoy.peaks.mz, dtype=np.float64)
            n_total += min(target_mz.size, decoy_mz.size)
            # Sorted order is preserved approximately (jitter << peak gaps),
            # so compare position-by-position.
            for target_value, decoy_value in zip(target_mz, decoy_mz):
                if abs(target_value - decoy_value) <= 0.02:
                    n_close += 1
        # Expected fraction within tolerance is ~2% (uniform ±1 Da jitter).
        assert n_close / n_total < 0.05

    def test_deterministic_generation(self) -> None:
        """The same seed reproduces identical decoys."""
        targets = _realistic_spectral_library(20, seed=51)
        first = generate_decoys(targets, random_seed=7)
        second = generate_decoys(targets, random_seed=7)
        for decoy_a, decoy_b in zip(first, second):
            assert np.array_equal(decoy_a.peaks.mz, decoy_b.peaks.mz)
            assert np.array_equal(decoy_a.peaks.intensities, decoy_b.peaks.intensities)

    def test_degenerate_spectra_handled(self) -> None:
        """Single-peak and zero-intensity spectra produce safe decoys."""
        single_peak = Spectrum(
            mz=np.array([100.0]),
            intensities=np.array([1.0]),
            metadata={"id": "single", "precursor_mz": 150.0},
        )
        zero_intensity = Spectrum(
            mz=np.array([100.0, 200.0]),
            intensities=np.zeros(2),
            metadata={"id": "zero", "precursor_mz": 250.0},
        )
        decoys = generate_decoys([single_peak, zero_intensity], random_seed=3)
        assert len(decoys) == 2
        assert decoys[0].peaks.mz.size == 1
        assert decoys[0].get("spectral_entropy") == 0.0
        assert decoys[1].peaks.mz.size == 2
        assert decoys[1].get("spectral_entropy") == 0.0


class TestDecoyProcessingConfig:
    """ProcessingConfig enforces the decoy noise-thresholding contract."""

    def test_defaults_enforce_strict_baseline_filtering(self) -> None:
        """Defaults: 1% relative intensity floor, ±1.0 Da jitter."""
        config = ProcessingConfig()
        assert config.decoy_min_relative_intensity == 0.01
        assert config.decoy_mz_shift_da == 1.0

    def test_non_positive_thresholds_rejected(self) -> None:
        """A zero or out-of-range relative threshold is rejected."""
        with pytest.raises(ValidationError):
            ProcessingConfig(decoy_min_relative_intensity=0.0)
        with pytest.raises(ValidationError):
            ProcessingConfig(decoy_min_relative_intensity=1.5)
        with pytest.raises(ValidationError):
            ProcessingConfig(decoy_mz_shift_da=0.0)
