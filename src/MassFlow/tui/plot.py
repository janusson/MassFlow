"""
Pure text renderers for the MassFlow terminal visualizer.

Every function turns numeric data into a list of plain strings — no Textual,
no Rich markup, no colour codes — so the glyph math is fully unit-testable
and the widgets stay thin. The renderers intentionally use single-width
Unicode glyphs (``│``, ``█``, ``─``, ``┬``, ``▼``) that render correctly in
Ghostty, iTerm2, Kitty, and stock terminal fonts.

Float64 throughout; missing values never reach these functions (callers
substitute the ``"n/a"`` strings).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from MassFlow.tui.spectrum_data import MirrorPeak, mirror_align

_STEM = "│"
_CAP = "█"
_AXIS = "─"
_MARKER = "▼"
_MATCHED_JOINT = "┬"
_FILL = "░"
_EMPTY = " "


def _column_positions(
    mz: np.ndarray,
    x_min: float,
    x_max: float,
    width: int,
) -> np.ndarray:
    """Map peak m/z values to integer columns of a ``width``-wide plot."""
    span = max(x_max - x_min, 1e-12)
    scaled = (mz - x_min) / span * (width - 1)
    return np.clip(np.rint(scaled).astype(np.int64), 0, width - 1)


def _peak_heights(
    intensities: np.ndarray, height: int, intensity_max: float
) -> np.ndarray:
    """Normalize intensities to bar heights in ``1..height`` (0 when absent)."""
    if height < 1:
        return np.zeros(0, dtype=np.int64)
    if intensity_max <= 0.0:
        return np.zeros(intensities.size, dtype=np.int64)
    normalized = intensities / intensity_max
    return np.maximum(np.ceil(normalized * height).astype(np.int64), 0)


def render_stick_plot(
    mz: np.ndarray | Sequence[float],
    intensities: np.ndarray | Sequence[float],
    *,
    width: int = 78,
    height: int = 12,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    marker_mz: Optional[float] = None,
    title: Optional[str] = None,
) -> list[str]:
    """Render a centroid stick plot (m/z vs intensity) as text lines.

    Parameters
    ----------
    mz, intensities : sequence of float
        Peak arrays. Intensity is normalized to the tallest peak.
    width : int
        Plot width in characters (clamped to ``>= 20``).
    height : int
        Plot height in rows (clamped to ``>= 3``).
    x_min, x_max : float, optional
        m/z window; defaults to the data extent.
    marker_mz : float, optional
        Precursor m/z to annotate with ``▼`` on the top row.
    title : str, optional
        Left-aligned plain-text title line prepended to the plot.

    Returns
    -------
    list[str]
        One string per plot row, title first when given. The last two rows
        are the axis rule and the m/z tick labels.
    """
    width = max(width, 20)
    height = max(height, 3)
    mz_array = np.asarray(mz, dtype=np.float64)
    intensity_array = np.asarray(intensities, dtype=np.float64)

    x_low, x_high, intensity_max = _bounds(mz_array, intensity_array, x_min, x_max)

    lines: list[str] = []
    if title:
        lines.append(title[:width])

    if mz_array.size == 0:
        for _ in range(height):
            lines.append(_EMPTY * width)
        lines.append(_AXIS * width)
        lines.append(_axis_label_line(x_low, x_high, width))
        return lines

    columns = _column_positions(mz_array, x_low, x_high, width)
    bar_heights = _peak_heights(intensity_array, height, intensity_max)

    # Build an occupancy grid: cap row gets ``█``, occupied rows below get
    # ``│``. Multiple peaks may share a column; the tallest wins.
    grid = np.zeros((height, width), dtype=np.int8)  # 0 empty, 1 stem, 2 cap
    for column, bar_height in zip(columns, bar_heights):
        if bar_height <= 0:
            continue
        column = int(column)
        cap_row = height - bar_height
        for row in range(cap_row, height):
            if grid[row, column] == 0:
                grid[row, column] = 1
        grid[cap_row, column] = 2

    marker_column: Optional[int] = None
    if marker_mz is not None and math.isfinite(marker_mz):
        marker_position = _column_positions(
            np.asarray([marker_mz], dtype=np.float64), x_low, x_high, width
        )[0]
        marker_column = int(marker_position)

    for row in range(height):
        chars = [_EMPTY] * width
        for column in range(width):
            cell = int(grid[row, column])
            if cell == 2:
                chars[column] = _CAP
            elif cell == 1:
                chars[column] = _STEM
        if marker_column is not None and row == 0:
            chars[marker_column] = _MARKER
        lines.append("".join(chars))

    lines.append(_AXIS * width)
    lines.append(_axis_label_line(x_low, x_high, width))
    return lines


def render_mirror_plot(
    query_mz: np.ndarray | Sequence[float],
    query_intensities: np.ndarray | Sequence[float],
    reference_mz: np.ndarray | Sequence[float],
    reference_intensities: np.ndarray | Sequence[float],
    *,
    tolerance: float,
    width: int = 78,
    half_height: int = 6,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    title: Optional[str] = None,
) -> list[str]:
    """Render a mirror plot: query peaks up, reference peaks down.

    Peaks are aligned within ``tolerance`` (see
    :func:`MassFlow.tui.spectrum_data.mirror_align`); matched pairs render as
    a continuous stem across the central axis, unmatched peaks as one-sided
    stems. The central axis row uses ``┬`` at matched positions.

    Returns
    -------
    list[str]
        ``2 * half_height + 2`` rows (title line prepended when given).
    """
    width = max(width, 20)
    half_height = max(half_height, 2)

    aligned: list[MirrorPeak] = mirror_align(
        query_mz, query_intensities, reference_mz, reference_intensities, tolerance
    )

    query_max = max((peak.query_intensity for peak in aligned), default=0.0)
    reference_max = max((peak.reference_intensity for peak in aligned), default=0.0)

    all_mz = [peak.mz for peak in aligned]
    if all_mz:
        x_low = float(min(all_mz)) if x_min is None else float(x_min)
        x_high = float(max(all_mz)) if x_max is None else float(x_max)
    else:
        x_low = x_min if x_min is not None else 0.0
        x_high = x_max if x_max is not None else 1.0
    if x_high <= x_low:
        x_high = x_low + 1.0

    lines: list[str] = []
    if title:
        lines.append(title[:width])

    grid = np.zeros((2 * half_height, width), dtype=np.int8)
    matched_columns: set[int] = set()

    for peak in aligned:
        column = int(
            _column_positions(
                np.asarray([peak.mz], dtype=np.float64), x_low, x_high, width
            )[0]
        )
        query_height = _scaled_height(peak.query_intensity, query_max, half_height)
        reference_height = _scaled_height(
            peak.reference_intensity, reference_max, half_height
        )
        if peak.matched:
            matched_columns.add(column)
        # Query bars occupy rows just above the center line: rows
        # half_height - query_height .. half_height - 1.
        for offset in range(query_height):
            row = half_height - 1 - offset
            if 0 <= row < 2 * half_height:
                grid[row, column] = 1
        # Reference bars occupy rows just below the center line.
        for offset in range(reference_height):
            row = half_height + offset
            if 0 <= row < 2 * half_height:
                grid[row, column] = 1

    for row in range(2 * half_height):
        chars = [_EMPTY] * width
        for column in range(width):
            if int(grid[row, column]):
                chars[column] = _STEM
        lines.append("".join(chars))

    axis_chars = [_AXIS] * width
    for column in matched_columns:
        axis_chars[column] = _MATCHED_JOINT
    lines.append("".join(axis_chars))
    lines.append(_axis_label_line(x_low, x_high, width))
    return lines


def _scaled_height(intensity: float, maximum: float, height: int) -> int:
    if maximum <= 0.0 or intensity <= 0.0:
        return 0
    return max(1, int(math.ceil(intensity / maximum * height)))


def _bounds(
    mz: np.ndarray,
    intensities: np.ndarray,
    x_min: Optional[float],
    x_max: Optional[float],
) -> tuple[float, float, float]:
    if mz.size:
        low = float(np.min(mz))
        high = float(np.max(mz))
        intensity_max = float(np.max(intensities)) if intensities.size else 0.0
    else:
        low, high, intensity_max = 0.0, 1.0, 0.0
    low = low if x_min is None or not math.isfinite(x_min) else float(x_min)
    high = high if x_max is None or not math.isfinite(x_max) else float(x_max)
    if not math.isfinite(low):
        low = 0.0
    if not math.isfinite(high) or high <= low:
        high = low + 1.0
    if not math.isfinite(intensity_max) or intensity_max < 0.0:
        intensity_max = 0.0
    return low, high, intensity_max


def _axis_label_line(x_min: float, x_max: float, width: int) -> str:
    left = f"{x_min:.2f}"
    right = f"{x_max:.2f}"
    gap = max(width - len(left) - len(right), 1)
    return left + _EMPTY * gap + right


def render_axis_labels(x_min: float, x_max: float, width: int = 78) -> str:
    """Render a standalone m/z tick label line."""
    return _axis_label_line(float(x_min), float(x_max), max(width, 20))


def render_score_gauge(score: float, width: int = 12) -> str:
    """Render an intensity gauge (``█``/``░``) for a score in ``[0, 1]``."""
    width = max(width, 1)
    if not math.isfinite(score):
        return _FILL * width
    clamped = min(max(score, 0.0), 1.0)
    filled = int(round(clamped * width))
    return _CAP * filled + _FILL * (width - filled)
