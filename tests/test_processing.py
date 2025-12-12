import numpy as np
import pytest

pytest.importorskip("matchms", reason="Processing depends on matchms.")

from matchms import Spectrum

from yogimass.similarity.processing import SpectrumProcessor


def _spectrum(peaks, intensities, dtype="float32", **metadata):
    return Spectrum(
        mz=np.asarray(peaks, dtype=dtype),
        intensities=np.asarray(intensities, dtype=dtype),
        metadata=metadata,
    )


def test_normalize_tic_basepeak():
    s = _spectrum([10.0, 20.0], [1.0, 3.0])
    p = SpectrumProcessor(normalization="tic")
    out = p.normalize(s)
    assert out.peaks.intensities[0] == pytest.approx(1.0 / 4.0)
    p2 = SpectrumProcessor(normalization="basepeak")
    out2 = p2.normalize(s)
    assert out2.peaks.intensities[1] == pytest.approx(1.0)


def test_filter_noise_absolute_relative_thresholds():
    s = _spectrum([10.0, 20.0, 30.0], [0.001, 1.0, 0.1])
    p = SpectrumProcessor(min_relative_intensity=0.2, min_absolute_intensity=0.05)
    out = p.filter_noise(s)
    # Expect only the 1.0 peak to remain given a relative threshold of 0.2
    assert out.peaks.mz.tolist() == [20.0]


def test_top_n_peak_selection():
    peaks = list(range(100))
    intensities = list(reversed(range(100)))  # highest intensity at 0
    s = _spectrum(peaks, intensities)
    p = SpectrumProcessor(max_peaks=5)
    out = p.filter_noise(s)
    assert len(out.peaks.mz) <= 5


def test_mz_window_deduplication_keeps_max():
    s = _spectrum([10.0, 10.005, 20.0], [1.0, 2.0, 0.5])
    p = SpectrumProcessor(mz_dedup_tolerance=0.01)
    out = p.filter_noise(s)
    # Peaks 10.0 and 10.005 deduped to a single peak, keeping the apex (2.0),
    # which is then normalized to 1.0 in the output
    assert len(out.peaks.mz) == 2
    intensities = out.peaks.intensities.tolist()
    mz = out.peaks.mz.tolist()
    assert pytest.approx(1.0) in intensities
    assert pytest.approx(10.005) in mz


def test_empty_input_returns_empty():
    s = _spectrum([], [])
    p = SpectrumProcessor()
    out = p.process(s)
    assert out.peaks.mz.size == 0
    assert out.peaks.intensities.size == 0


def test_negative_intensities_raise():
    s = _spectrum([10.0], [-1.0])
    p = SpectrumProcessor()
    with pytest.raises(ValueError):
        p.filter_noise(s)
    with pytest.raises(ValueError):
        p.normalize(s)


def test_reproducibility():
    s = _spectrum([10.0, 20.0, 30.0], [100.0, 50.0, 1.0])
    p = SpectrumProcessor(min_relative_intensity=0.02, max_peaks=2)
    out1 = p.process(s)
    out2 = p.process(s)
    assert np.allclose(out1.peaks.mz, out2.peaks.mz)
    assert np.allclose(out1.peaks.intensities, out2.peaks.intensities)


def test_extremely_dense_spectrum_top_n():
    # create 10000 peaks with descending intensities
    mz = np.linspace(50.0, 550.0, 10000)
    intensities = np.linspace(10000, 1, 10000)
    s = _spectrum(mz, intensities)
    p = SpectrumProcessor(max_peaks=50)
    out = p.filter_noise(s)
    assert len(out.peaks.mz) <= 50


def test_all_zero_intensities_with_normalization():
    s = _spectrum([100.0, 200.0], [0.0, 0.0])
    p = SpectrumProcessor(normalization="tic")
    out = p.normalize(s)
    assert np.all(out.peaks.intensities == 0.0)
    assert out.peaks.intensities.dtype == np.float64


def test_equal_intensity_peaks_top_n_is_deterministic():
    mz = [10.0, 20.0, 30.0, 40.0, 50.0]
    intensities = [1.0, 1.0, 1.0, 1.0, 1.0]
    s = _spectrum(mz, intensities)
    p = SpectrumProcessor(max_peaks=3)
    out = p.filter_noise(s)
    assert out.peaks.mz.tolist() == [30.0, 40.0, 50.0]


def test_extremely_small_mz_dedup_tolerance_no_dedup():
    s = _spectrum([10.0, 10.0001], [1.0, 0.9])
    p = SpectrumProcessor(mz_dedup_tolerance=1e-9)
    out = p.filter_noise(s)
    assert out.peaks.mz.tolist() == pytest.approx([10.0, 10.0001])
    assert len(out.peaks.mz) == 2


def test_mixed_nans_and_negative_intensity_raises():
    s = _spectrum([10.0, 20.0, 30.0], [np.nan, -1.0, 0.5])
    p = SpectrumProcessor()
    with pytest.raises(ValueError):
        p.filter_noise(s)


def test_float_dtype_enforced():
    s = _spectrum([10.0, 20.0], [1.0, 2.0])
    p = SpectrumProcessor(float_dtype=np.float32, normalization="basepeak")
    out = p.process(s)
    assert out.peaks.intensities.dtype == np.float32
    assert out.peaks.mz.dtype == np.float32
