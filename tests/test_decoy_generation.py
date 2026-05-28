"""Tests for decoy generation in MassFlow.similarity

Checks that generate_decoys preserves precursor_mz and ids, and that intensities
are altered for typical spectra. Edge case: identical intensities should be
handled by tapering rather than leaving identical arrays.
"""

import numpy as np
from matchms import Spectrum

from MassFlow.similarity import generate_decoys


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
