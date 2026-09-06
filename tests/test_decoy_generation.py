"""Tests for decoy generation in MassFlow.similarity

Checks that generate_decoys preserves precursor_mz and ids, and that intensities
are altered for typical spectra. Edge case: identical intensities should be
handled by tapering rather than leaving identical arrays.
"""

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.similarity import generate_decoys

pytestmark = pytest.mark.scientific


def make_spectrum(
    spec_id: str, precursor_mz: float = 100.0, intensities=None
) -> Spectrum:
    if intensities is None:
        intensities = np.array([10.0, 5.0, 1.0], dtype=float)
    return Spectrum(
        mz=np.array([100.0, 150.0, 200.0]),
        intensities=np.array(intensities),
        metadata={"id": spec_id, "precursor_mz": precursor_mz},
    )


def test_generate_decoys_preserve_precursor_and_id():
    s = make_spectrum("ref1", 123.45)
    decoys = generate_decoys([s], random_seed=0)
    assert len(decoys) == 1
    d = decoys[0]
    # precursor_mz must be preserved
    assert float(d.get("precursor_mz")) == float(s.get("precursor_mz"))
    # id should be suffixed with _decoy
    assert str(d.get("id")).endswith("_decoy")


def test_generate_decoys_intensity_shuffled_or_tapered():
    # Normal case: varied intensities should be shuffled
    s = make_spectrum("ref2", intensities=[10.0, 5.0, 1.0])
    d = generate_decoys([s], random_seed=1)[0]
    orig = s.peaks.intensities
    new = d.peaks.intensities
    # For varied intensities, expect different ordering or values
    assert not np.array_equal(orig, new)


def test_generate_decoys_handles_identical_intensities():
    # Edge case: identical intensities should be tapered rather than identical
    s = make_spectrum("ref3", intensities=[1.0, 1.0, 1.0])
    d = generate_decoys([s], random_seed=2)[0]
    new = d.peaks.intensities
    # Expect not identical to original and not all equal
    assert not np.array_equal(s.peaks.intensities, new)
    assert not np.allclose(new, new[0])


# ---------------------------------------------------------------------------
# Entropy decoy physics (audit fixes)
# ---------------------------------------------------------------------------


def _sqrt_weighted_entropy(intensities: np.ndarray) -> float:
    """Reference implementation: H = -Σ p ln p with p ∝ I**0.5."""
    weights = np.sqrt(intensities)
    probabilities = weights / weights.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def test_spectral_entropy_applies_sqrt_weighting():
    """The entropy estimate uses I**0.5 weighting, not raw intensities.

    For intensities [4, 1] the sqrt weights are [2, 1], giving
    p = [2/3, 1/3]. Raw linear normalization would give p = [0.8, 0.2] —
    a different (base-peak-dominated) distribution.
    """
    from MassFlow.similarity import spectral_entropy

    intensities = np.array([4.0, 1.0])
    expected = _sqrt_weighted_entropy(intensities)
    assert spectral_entropy(intensities) == pytest.approx(expected, rel=1e-12)

    linear_probabilities = intensities / intensities.sum()
    plain_shannon = float(-np.sum(linear_probabilities * np.log(linear_probabilities)))
    assert expected != pytest.approx(plain_shannon, rel=1e-6)


def test_spectral_entropy_filters_sub_one_percent_base_peak_noise():
    """Peaks below 1% of the base peak are excluded before weighting."""
    from MassFlow.similarity import spectral_entropy

    intensities = np.array([100.0, 2.0, 0.5])
    filtered = spectral_entropy(intensities, min_relative_intensity=0.01)
    unfiltered = spectral_entropy(intensities, min_relative_intensity=0.001)

    # 1% of the base peak (100) = 1.0: the 0.5 peak is noise and removed.
    assert filtered == pytest.approx(
        _sqrt_weighted_entropy(np.array([100.0, 2.0])), rel=1e-12
    )
    # A permissive 0.1% floor keeps the noise peak.
    assert unfiltered == pytest.approx(_sqrt_weighted_entropy(intensities), rel=1e-12)
    assert filtered != pytest.approx(unfiltered, rel=1e-9)


def test_decoy_entropy_matches_filtered_target_entropy():
    """Decoy entropy strictly equals the filtered target entropy.

    The decoy intensity profile is a permutation of the baseline-filtered
    target profile, so the sqrt-weighted spectral entropy matches exactly
    (up to floating-point rounding) and no noise peak leaks into the decoy.
    """
    from MassFlow.similarity import spectral_entropy

    s = Spectrum(
        mz=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64),
        # 400/500 peaks are sub-1%-of-base-peak chemical noise.
        intensities=np.array([1000.0, 500.0, 200.0, 5.0, 2.0], dtype=np.float64),
        metadata={"id": "noisy_target", "precursor_mz": 600.0},
    )
    decoy = generate_decoys([s], random_seed=11)[0]

    target_entropy = spectral_entropy(np.asarray(s.peaks.intensities, dtype=np.float64))
    decoy_entropy = spectral_entropy(
        np.asarray(decoy.peaks.intensities, dtype=np.float64)
    )
    assert decoy_entropy == pytest.approx(target_entropy, abs=1e-12)
    assert float(decoy.get("spectral_entropy")) == pytest.approx(
        target_entropy, abs=1e-12
    )

    # Strict baseline filtering before decoy construction: the two noise
    # peaks (5.0 and 2.0 < 1% of base peak 1000) never reach the decoy.
    decoy_intensities = np.asarray(decoy.peaks.intensities, dtype=np.float64)
    assert decoy_intensities.size == 3
    assert decoy_intensities.min() >= 0.01 * decoy_intensities.max()
    assert set(np.round(decoy_intensities, 6).tolist()) == {
        1000.0,
        500.0,
        200.0,
    }
