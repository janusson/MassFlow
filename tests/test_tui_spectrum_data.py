"""
Tests for MassFlow.tui.spectrum_data — pure NumPy spectrum helpers.
"""

import math

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.tui.spectrum_data import (
    annotation_status,
    display_entropy,
    downsample_peaks,
    format_mz,
    format_retention_time,
    mirror_align,
    peak_bounds,
    summarize_spectrum,
)
from MassFlow.tui.state import SpectrumSummary


def make_spectrum(**metadata) -> Spectrum:
    """Build a small matchms Spectrum with float64 peaks."""
    mz = np.array([50.0, 100.0, 150.0, 200.0], dtype=np.float64)
    intensities = np.array([10.0, 100.0, 40.0, 5.0], dtype=np.float64)
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)


class TestDownsamplePeaks:
    def test_small_arrays_untouched(self):
        mz = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        intensities = np.array([0.5, 0.6, 0.7], dtype=np.float64)
        out_mz, out_intensities = downsample_peaks(mz, intensities, max_peaks=10)
        np.testing.assert_array_equal(out_mz, mz)
        np.testing.assert_array_equal(out_intensities, intensities)
        assert out_mz.dtype == np.float64
        assert out_intensities.dtype == np.float64

    def test_keeps_tallest_peak_per_group(self):
        mz = np.arange(100, dtype=np.float64)
        intensities = np.ones(100, dtype=np.float64)
        intensities[42] = 99.0  # tallest in its group
        out_mz, out_intensities = downsample_peaks(mz, intensities, max_peaks=10)
        assert out_mz.size == 10
        assert out_mz[4] == 42.0
        assert out_intensities[4] == 99.0

    def test_exact_boundary(self):
        mz = np.arange(10, dtype=np.float64)
        intensities = np.arange(10, dtype=np.float64)
        out_mz, out_intensities = downsample_peaks(mz, intensities, max_peaks=10)
        assert out_mz.size == 10

    def test_invalid_max_peaks(self):
        mz = np.arange(5, dtype=np.float64)
        intensities = np.ones(5, dtype=np.float64)
        with pytest.raises(ValueError):
            downsample_peaks(mz, intensities, max_peaks=0)

    def test_empty_arrays(self):
        mz = np.zeros(0, dtype=np.float64)
        intensities = np.zeros(0, dtype=np.float64)
        out_mz, out_intensities = downsample_peaks(mz, intensities)
        assert out_mz.size == 0
        assert out_intensities.size == 0

    def test_result_is_a_copy(self):
        mz = np.arange(5, dtype=np.float64)
        intensities = np.ones(5, dtype=np.float64)
        out_mz, out_intensities = downsample_peaks(mz, intensities)
        out_mz[0] = -1.0
        assert mz[0] == 0.0


class TestDisplayEntropy:
    def test_uniform(self):
        entropy = display_entropy(np.ones(4, dtype=np.float64))
        assert math.isclose(entropy, math.log(4), rel_tol=1e-12)

    def test_single_peak_is_zero(self):
        assert display_entropy(np.array([5.0])) == 0.0

    def test_empty_is_nan(self):
        assert math.isnan(display_entropy(np.zeros(0, dtype=np.float64)))

    def test_zero_total_is_nan(self):
        assert math.isnan(display_entropy(np.zeros(3, dtype=np.float64)))

    def test_scale_invariance(self):
        base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        assert math.isclose(
            display_entropy(base),
            display_entropy(base * 1000.0),
            rel_tol=1e-12,
        )


class TestSummarizeSpectrum:
    def test_basic_fields(self):
        spectrum = make_spectrum(
            id="scan=42",
            precursor_mz=150.1234,
            retention_time=90.0,
            charge=1,
            ionmode="positive",
            adduct="[M+H]+",
            compound_name="Caffeine",
        )
        summary = summarize_spectrum(spectrum)
        assert isinstance(summary, SpectrumSummary)
        assert summary.spectrum_id == "scan=42"
        assert summary.precursor_mz == pytest.approx(150.1234)
        assert summary.retention_time_seconds == pytest.approx(90.0)
        assert summary.charge == 1
        assert summary.ionmode == "positive"
        assert summary.adduct == "[M+H]+"
        assert summary.compound_name == "Caffeine"
        assert summary.num_peaks == 4
        assert summary.base_peak_mz == pytest.approx(100.0)
        assert summary.base_peak_intensity == pytest.approx(100.0)
        assert summary.total_ion_current == pytest.approx(155.0)

    def test_missing_metadata_becomes_none(self):
        spectrum = make_spectrum()
        summary = summarize_spectrum(spectrum)
        assert summary.spectrum_id == "unknown"
        assert summary.precursor_mz is None
        assert summary.charge is None
        assert summary.compound_name is None

    def test_empty_peaks(self):
        spectrum = Spectrum(
            mz=np.zeros(0, dtype=np.float64),
            intensities=np.zeros(0, dtype=np.float64),
            metadata={"id": "empty"},
        )
        summary = summarize_spectrum(spectrum)
        assert summary.num_peaks == 0
        assert summary.base_peak_mz is None
        assert summary.total_ion_current is None

    def test_precursor_nan_becomes_none(self):
        spectrum = make_spectrum(id="s", precursor_mz=float("nan"))
        summary = summarize_spectrum(spectrum)
        assert summary.precursor_mz is None

    def test_charge_list_takes_first(self):
        spectrum = make_spectrum(id="s", charge=[2, 3])
        assert summarize_spectrum(spectrum).charge == 2

    def test_float64_preserved(self):
        spectrum = make_spectrum(id="s")
        summary = summarize_spectrum(spectrum)
        assert summary.mz_array.dtype == np.float64
        assert summary.intensity_array.dtype == np.float64

    def test_downsamples_large_spectra(self):
        mz = np.linspace(50.0, 500.0, 100_000, dtype=np.float64)
        intensities = np.ones(100_000, dtype=np.float64)
        spectrum = Spectrum(mz=mz, intensities=intensities, metadata={"id": "big"})
        summary = summarize_spectrum(spectrum, max_peaks=500)
        assert summary.num_peaks <= 500


class TestPeakBounds:
    def test_normal(self):
        mz = np.array([50.0, 400.0], dtype=np.float64)
        intensities = np.array([1.0, 7.0], dtype=np.float64)
        assert peak_bounds(mz, intensities) == (50.0, 400.0, 7.0)

    def test_empty(self):
        assert peak_bounds(np.zeros(0), np.zeros(0)) == (0.0, 1.0, 0.0)

    def test_flat_range_gets_padded(self):
        mz = np.array([50.0, 50.0], dtype=np.float64)
        intensities = np.array([1.0, 1.0], dtype=np.float64)
        x_min, x_max, _ = peak_bounds(mz, intensities)
        assert x_max > x_min


class TestMirrorAlign:
    def test_perfect_overlap(self):
        query_mz = np.array([100.0, 200.0, 300.0])
        query_intensities = np.array([1.0, 2.0, 3.0])
        peaks = mirror_align(
            query_mz, query_intensities, query_mz, query_intensities, tolerance=0.02
        )
        assert len(peaks) == 3
        assert all(peak.matched for peak in peaks)

    def test_unmatched_on_both_sides(self):
        query_mz = np.array([100.0, 200.0])
        query_intensities = np.array([1.0, 2.0])
        ref_mz = np.array([150.0, 250.0])
        ref_intensities = np.array([3.0, 4.0])
        peaks = mirror_align(
            query_mz, query_intensities, ref_mz, ref_intensities, tolerance=0.02
        )
        assert len(peaks) == 4
        assert not any(peak.matched for peak in peaks)
        # Query peaks carry zero reference intensity and vice versa.
        by_mz = {peak.mz: peak for peak in peaks}
        assert by_mz[100.0].reference_intensity == 0.0
        assert by_mz[150.0].query_intensity == 0.0

    def test_partial_overlap_within_tolerance(self):
        query_mz = np.array([100.0, 200.0])
        query_intensities = np.array([1.0, 2.0])
        ref_mz = np.array([100.01, 300.0])  # 100.01 within 0.02 of 100.0
        ref_intensities = np.array([5.0, 6.0])
        peaks = mirror_align(
            query_mz, query_intensities, ref_mz, ref_intensities, tolerance=0.02
        )
        matched = [peak for peak in peaks if peak.matched]
        assert len(matched) == 1
        assert matched[0].query_intensity == 1.0
        assert matched[0].reference_intensity == 5.0

    def test_result_sorted_by_mz(self):
        query_mz = np.array([300.0, 100.0])
        query_intensities = np.array([1.0, 2.0])
        ref_mz = np.array([200.0])
        ref_intensities = np.array([3.0])
        peaks = mirror_align(
            query_mz, query_intensities, ref_mz, ref_intensities, tolerance=0.02
        )
        mz_values = [peak.mz for peak in peaks]
        assert mz_values == sorted(mz_values)


class TestFormatters:
    def test_format_mz(self):
        assert format_mz(123.456789) == "123.4568"
        assert format_mz(None) == "n/a"
        assert format_mz(float("nan")) == "n/a"

    def test_format_retention_time(self):
        assert format_retention_time(180.0) == "3.00 min"
        assert format_retention_time(None) == "n/a"

    def test_annotation_status(self):
        assert annotation_status(0.95) == "matched"
        assert annotation_status(0.7) == "putative"
        assert annotation_status(None) == "unknown"
        assert annotation_status(float("nan")) == "unknown"
        assert annotation_status(0.9) == "matched"
