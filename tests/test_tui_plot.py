"""
Tests for MassFlow.tui.plot — pure text renderers for the visualizer.
"""

import math

import numpy as np

from MassFlow.tui.plot import (
    render_axis_labels,
    render_mirror_plot,
    render_score_gauge,
    render_stick_plot,
)


def _peaks():
    mz = np.array([50.0, 100.0, 150.0, 200.0], dtype=np.float64)
    intensities = np.array([0.25, 1.0, 0.5, 0.75], dtype=np.float64)
    return mz, intensities


class TestRenderStickPlot:
    def test_shape_and_layout(self):
        lines = render_stick_plot(*_peaks(), width=60, height=10)
        # 10 plot rows + axis rule + tick labels.
        assert len(lines) == 12
        assert all(len(line) == 60 for line in lines)
        assert set(lines[-2]) == {"─"}
        # Ticks mention the m/z bounds.
        assert "50.00" in lines[-1]
        assert "200.00" in lines[-1]

    def test_title_prepended(self):
        lines = render_stick_plot(*_peaks(), width=60, height=10, title="hello")
        assert lines[0] == "hello"
        assert len(lines) == 13

    def test_peak_heights_monotonic_in_intensity(self):
        mz = np.array([100.0, 200.0], dtype=np.float64)
        intensities = np.array([1.0, 0.2], dtype=np.float64)
        lines = render_stick_plot(mz, intensities, width=40, height=6)
        plot_rows = lines[:6]

        def stem_count(rows, column):
            return sum(1 for row in rows if row[column] in "│█")

        # Collect the filled height of every occupied column, left to right.
        counts = [stem_count(plot_rows, column) for column in range(40)]
        counts = [count for count in counts if count > 0]
        assert len(counts) == 2
        assert counts[0] > counts[1]

    def test_marker_drawn_on_top_row(self):
        mz = np.array([100.0], dtype=np.float64)
        intensities = np.array([1.0], dtype=np.float64)
        lines = render_stick_plot(mz, intensities, width=20, height=6, marker_mz=100.0)
        assert "▼" in lines[0]

    def test_empty_spectrum_renders_axis(self):
        lines = render_stick_plot([], [], width=40, height=6)
        assert len(lines) == 8
        assert all(set(line) <= {" "} for line in lines[:6])

    def test_custom_window(self):
        mz = np.array([100.0, 200.0], dtype=np.float64)
        intensities = np.array([1.0, 1.0], dtype=np.float64)
        lines = render_stick_plot(
            mz, intensities, width=40, height=6, x_min=0.0, x_max=400.0
        )
        assert "0.00" in lines[-1]
        assert "400.00" in lines[-1]

    def test_width_and_height_clamped(self):
        lines = render_stick_plot(*_peaks(), width=1, height=1)
        assert len(lines) == 5  # height clamped to 3 + axis + labels
        assert all(len(line) >= 20 for line in lines)


class TestRenderMirrorPlot:
    def test_shape(self):
        lines = render_mirror_plot(
            np.array([100.0]),
            np.array([1.0]),
            np.array([100.0]),
            np.array([1.0]),
            tolerance=0.02,
            width=50,
            half_height=4,
        )
        # 2 * half_height plot rows + axis + labels.
        assert len(lines) == 10
        assert all(len(line) == 50 for line in lines)

    def test_matched_peaks_span_the_axis(self):
        query_mz = np.array([100.0])
        query_intensities = np.array([1.0])
        lines = render_mirror_plot(
            query_mz,
            query_intensities,
            query_mz,
            query_intensities,
            tolerance=0.02,
            width=50,
            half_height=4,
        )
        axis = lines[8]
        assert "┬" in axis
        # Query stem above the axis (rows 0..3), reference stem below (rows 4..7).
        above = any("│" in row for row in lines[0:4])
        below = any("│" in row for row in lines[4:8])
        assert above
        assert below

    def test_unmatched_peaks_no_joint(self):
        lines = render_mirror_plot(
            np.array([100.0]),
            np.array([1.0]),
            np.array([200.0]),
            np.array([1.0]),
            tolerance=0.02,
            width=50,
            half_height=4,
        )
        assert "┬" not in lines[8]

    def test_title(self):
        lines = render_mirror_plot(
            np.array([100.0]),
            np.array([1.0]),
            np.array([100.0]),
            np.array([1.0]),
            tolerance=0.02,
            width=50,
            half_height=4,
            title="mirror",
        )
        assert lines[0] == "mirror"

    def test_empty_inputs(self):
        lines = render_mirror_plot(
            [], [], [], [], tolerance=0.02, width=50, half_height=4
        )
        assert len(lines) == 10


class TestScoreGauge:
    def test_full_and_empty(self):
        assert render_score_gauge(1.0, width=5) == "█████"
        assert render_score_gauge(0.0, width=5) == "░░░░░"

    def test_half(self):
        assert render_score_gauge(0.5, width=10) == "█████░░░░░"

    def test_clamped(self):
        assert render_score_gauge(1.5, width=4) == "████"
        assert render_score_gauge(-0.5, width=4) == "░░░░"

    def test_nan_is_empty(self):
        assert render_score_gauge(math.nan, width=4) == "░░░░"

    def test_width_floor(self):
        assert render_score_gauge(1.0, width=0) == "█"


class TestAxisLabels:
    def test_contains_both_ends(self):
        line = render_axis_labels(10.5, 20.5, width=40)
        assert line.startswith("10.50")
        assert line.endswith("20.50")
        assert len(line) == 40

    def test_width_clamped(self):
        assert len(render_axis_labels(0.0, 1.0, width=1)) >= 20
