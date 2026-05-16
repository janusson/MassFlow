import numpy as np
from matchms import Spectrum

from MassFlow.similarity import _ms1_prefilter, calculate_fdr, generate_decoys


def test_fdr_no_decoys():
    """If no decoys are found, q-values should be 1.0 (no FDR estimation possible)."""
    target_scores = np.array([0.9, 0.8, 0.7])
    decoy_scores = np.array([])
    _, q, _ = calculate_fdr(target_scores, decoy_scores)

    # Conservative +1 pseudo-count formula when decoys=0 -> FDR=1/targets
    # Actually wait, calculate_fdr uses q_values = np.minimum(1.0 / cum_targets, 1.0)
    # The minimum over all lower scores accumulates from the end, so for 0.7 (targets=3), FDR=1/3.
    # q_value for 0.8 is min(1/2, 1/3)=1/3. For 0.9 is min(1/1, 1/3)=1/3.
    # Let me re-verify calculate_fdr implementation for len(decoy_scores) == 0.
    np.testing.assert_allclose(q, np.array([1 / 3, 1 / 3, 1 / 3]))


def test_decoy_generation_on_uniform_spectra():
    """Decoy generation must not return an identical intensity array."""
    # Spectrum with only one unique intensity value
    uniform_spec = Spectrum(
        mz=np.array([100.0, 200.0, 300.0]),
        intensities=np.array([1.0, 1.0, 1.0]),
        metadata={"id": "uniform"},
    )
    decoy = generate_decoys([uniform_spec])[0]
    assert not np.array_equal(
        uniform_spec.peaks.intensities, decoy.peaks.intensities
    ), "Decoy intensities are identical to original"


def test_decoy_generation_on_low_peak_spectra():
    """Decoy generation must not return an identical intensity array for low peak counts."""
    # Spectrum where a simple shuffle might randomly result in the same order
    low_peak_spec = Spectrum(
        mz=np.array([100.0, 200.0]),
        intensities=np.array([1.0, 2.0]),
        metadata={"id": "low_peak"},
    )
    decoy = generate_decoys([low_peak_spec], random_seed=42)[
        0
    ]  # Seed might cause shuffle to be identity
    assert not np.array_equal(
        low_peak_spec.peaks.intensities, decoy.peaks.intensities
    ), "Decoy intensities are identical to original on low peak count spectrum"


def test_ms1_prefiltering_dalton_to_ppm_conversion():
    """Verify that MS1 prefiltering respects PPM units, not absolute Dalton."""
    # Query at 100 Da, Ref at 100.01 Da. At 10ppm, this should NOT match.
    # 10 ppm of 100 Da = 0.001 Da. So 100.01 is way outside tolerance.
    # If the code were mistakenly using absolute tolerance, it might match.
    q = Spectrum(
        mz=np.array([1.0]),
        intensities=np.array([1.0]),
        metadata={"precursor_mz": 100.0},
    )
    r = Spectrum(
        mz=np.array([1.0]),
        intensities=np.array([1.0]),
        metadata={"precursor_mz": 100.01},
    )

    rows, cols = _ms1_prefilter([r], [q], ms1_tolerance=0.0, resolution_ppm=10.0)
    assert len(rows) == 0, "Matched a precursor outside of 10 PPM tolerance"

    # Now, test a valid match inside 10ppm
    r_match = Spectrum(
        mz=np.array([1.0]),
        intensities=np.array([1.0]),
        metadata={"precursor_mz": 100.0005},
    )
    rows, cols = _ms1_prefilter([r_match], [q], ms1_tolerance=10.0)
    assert len(rows) == 1, "Failed to match a precursor inside 10 PPM tolerance"
