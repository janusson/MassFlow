"""
Pure NumPy helpers for describing and shaping MS/MS spectra for display.

No Textual, no Rich, no filesystem access. Every function accepts and returns
plain values (``numpy.float64`` arrays, floats, strings, dataclasses) so the
science-facing behaviour of the console is testable headlessly.

Float64 is preserved end-to-end: no downcast to float32 anywhere in this
module. Missing scientific values are ``NaN``/``None``, never zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from MassFlow.tui.state import SpectrumSummary

# Preview hard-limit for stored peak arrays. Real MS/MS spectra can carry tens
# of thousands of peaks; the console keeps at most this many (max-pooled) per
# spectrum so the UI stays responsive without changing the science of the
# plotted envelope.
DEFAULT_MAX_PEAKS = 2000


def downsample_peaks(
    mz: np.ndarray,
    intensities: np.ndarray,
    max_peaks: int = DEFAULT_MAX_PEAKS,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample peak arrays to at most ``max_peaks`` pairs via max-pooling.

    The m/z axis is split into contiguous groups; within each group the most
    intense peak is retained with its original (unmodified) float64 values.
    This preserves the tallest fragment peaks — the ones that dominate visual
    interpretation and cosine scoring — while discarding low-intensity
    crowding.

    Parameters
    ----------
    mz : np.ndarray
        Peak m/z values (float64, monotonically non-decreasing).
    intensities : np.ndarray
        Peak intensities aligned with ``mz``.
    max_peaks : int
        Maximum number of peaks to retain. Must be positive.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Downsampled ``(mz, intensities)`` arrays, both float64.
    """
    mz = np.asarray(mz, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if max_peaks < 1:
        raise ValueError("max_peaks must be a positive integer.")
    n = mz.size
    if n <= max_peaks:
        return mz.copy(), intensities.copy()

    group_size = math.ceil(n / max_peaks)
    num_groups = math.ceil(n / group_size)
    out_mz = np.empty(num_groups, dtype=np.float64)
    out_intensity = np.empty(num_groups, dtype=np.float64)

    for group in range(num_groups):
        start = group * group_size
        stop = min(start + group_size, n)
        segment_mz = mz[start:stop]
        segment_intensity = intensities[start:stop]
        winner = int(np.argmax(segment_intensity))
        out_mz[group] = segment_mz[winner]
        out_intensity[group] = segment_intensity[winner]

    return out_mz, out_intensity


def display_entropy(intensities: np.ndarray) -> float:
    """Shannon entropy (nats) of an intensity profile, for display only.

    Intensities are normalized to sum to 1 before the entropy is computed.
    Returns ``NaN`` when the profile is empty or has zero total intensity so
    the UI can render "n/a" instead of a misleading 0.0.
    """
    intensities = np.asarray(intensities, dtype=np.float64)
    if intensities.size == 0:
        return math.nan
    total = float(np.sum(intensities))
    if not math.isfinite(total) or total <= 0.0:
        return math.nan
    probabilities = intensities / total
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def _optional_float(value: Any) -> Optional[float]:
    """Coerce a metadata value to float, mapping missing/NaN to ``None``."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _optional_int(value: Any) -> Optional[int]:
    """Coerce a metadata value to int, mapping missing/non-numeric to ``None``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def summarize_spectrum(
    spectrum: Any,
    *,
    max_peaks: int = DEFAULT_MAX_PEAKS,
) -> SpectrumSummary:
    """Reduce a matchms ``Spectrum`` to a :class:`SpectrumSummary`.

    Parameters
    ----------
    spectrum : matchms.Spectrum
        The spectrum to summarize. Duck-typed: any object with
        ``.get(key)`` and ``.peaks`` attribute works, but matchms spectra are
        the intended input.
    max_peaks : int
        Peak retention limit for the stored arrays (see
        :func:`downsample_peaks`).

    Returns
    -------
    SpectrumSummary
        Display summary including downsampled float64 peak arrays.
    """
    peaks = getattr(spectrum, "peaks", None)
    if peaks is not None:
        mz = np.asarray(peaks.mz, dtype=np.float64)
        intensities = np.asarray(peaks.intensities, dtype=np.float64)
    else:
        mz = np.zeros(0, dtype=np.float64)
        intensities = np.zeros(0, dtype=np.float64)

    mz, intensities = downsample_peaks(mz, intensities, max_peaks)

    base_peak_index = int(np.argmax(intensities)) if intensities.size else -1
    if base_peak_index >= 0:
        base_peak_mz: Optional[float] = float(mz[base_peak_index])
        base_peak_intensity: Optional[float] = float(intensities[base_peak_index])
    else:
        base_peak_mz = None
        base_peak_intensity = None

    total_ion_current = float(np.sum(intensities)) if intensities.size else None

    spectrum_id = _optional_str(spectrum.get("id")) or _optional_str(
        spectrum.get("spectrum_id")
    )
    if spectrum_id is None:
        scans = spectrum.get("scans")
        if scans is not None:
            spectrum_id = f"scan={scans}"

    return SpectrumSummary(
        spectrum_id=spectrum_id or "unknown",
        precursor_mz=_optional_float(spectrum.get("precursor_mz")),
        retention_time_seconds=_optional_float(spectrum.get("retention_time")),
        num_peaks=int(mz.size),
        charge=_optional_int(spectrum.get("charge")),
        ionmode=_optional_str(spectrum.get("ionmode")),
        adduct=_optional_str(spectrum.get("adduct")),
        compound_name=_optional_str(spectrum.get("compound_name"))
        or _optional_str(spectrum.get("name")),
        base_peak_mz=base_peak_mz,
        base_peak_intensity=base_peak_intensity,
        total_ion_current=total_ion_current,
        spectral_entropy=display_entropy(intensities),
        mz_array=mz,
        intensity_array=intensities,
    )


def peak_bounds(mz: np.ndarray, intensities: np.ndarray) -> tuple[float, float, float]:
    """Return ``(mz_min, mz_max, intensity_max)`` for a peak list.

    Degenerate inputs degrade to a safe plotting window of ``(0.0, 1.0, 0.0)``
    rather than raising, so an empty spectrum still renders an empty axis.
    """
    mz = np.asarray(mz, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if mz.size == 0 or intensities.size == 0:
        return 0.0, 1.0, 0.0

    finite_mz = mz[np.isfinite(mz)]
    if finite_mz.size:
        mz_min = float(finite_mz.min())
        mz_max = float(finite_mz.max())
    else:
        mz_min, mz_max = 0.0, 1.0
    if mz_max <= mz_min:
        mz_max = mz_min + 1.0

    finite_intensities = intensities[np.isfinite(intensities)]
    intensity_max = float(finite_intensities.max()) if finite_intensities.size else 0.0
    if intensity_max <= 0.0:
        intensity_max = 0.0
    return mz_min, mz_max, intensity_max


@dataclass(frozen=True)
class MirrorPeak:
    """One aligned peak in a mirror plot."""

    mz: float
    query_intensity: float
    reference_intensity: float
    matched: bool


def mirror_align(
    query_mz: np.ndarray | Sequence[float],
    query_intensities: np.ndarray | Sequence[float],
    reference_mz: np.ndarray | Sequence[float],
    reference_intensities: np.ndarray | Sequence[float],
    tolerance: float,
) -> list[MirrorPeak]:
    """Align query and reference peaks within ``tolerance`` for a mirror plot.

    A greedy two-pointer pass (both arrays assumed sorted by m/z) pairs each
    query peak with the nearest reference peak within ``tolerance``. Unmatched
    peaks are retained with a zero intensity on the opposing side, so the
    mirror plot is an honest picture of the overlap rather than a filtered
    one.

    Parameters
    ----------
    query_mz, query_intensities : np.ndarray
        Query peak arrays (float64, sorted by m/z).
    reference_mz, reference_intensities : np.ndarray
        Reference peak arrays (float64, sorted by m/z).
    tolerance : float
        m/z tolerance (Da) for declaring a peak "matched".

    Returns
    -------
    list[MirrorPeak]
        Peaks sorted by m/z, each carrying query/reference intensities and a
        matched flag.
    """
    query_mz = np.asarray(query_mz, dtype=np.float64)
    query_intensities = np.asarray(query_intensities, dtype=np.float64)
    reference_mz = np.asarray(reference_mz, dtype=np.float64)
    reference_intensities = np.asarray(reference_intensities, dtype=np.float64)

    # The matching pass assumes ascending m/z; sort defensively (real spectra
    # are always sorted, but the console must not crash on hand-built data).
    if query_mz.size > 1 and bool(np.any(np.diff(query_mz) < 0)):
        order = np.argsort(query_mz)
        query_mz = query_mz[order]
        query_intensities = query_intensities[order]
    if reference_mz.size > 1 and bool(np.any(np.diff(reference_mz) < 0)):
        order = np.argsort(reference_mz)
        reference_mz = reference_mz[order]
        reference_intensities = reference_intensities[order]

    peaks: list[MirrorPeak] = []
    query_index = 0
    reference_index = 0

    while query_index < query_mz.size and reference_index < reference_mz.size:
        query_mass = float(query_mz[query_index])
        reference_mass = float(reference_mz[reference_index])
        if abs(query_mass - reference_mass) <= tolerance:
            peaks.append(
                MirrorPeak(
                    mz=(query_mass + reference_mass) / 2.0,
                    query_intensity=float(query_intensities[query_index]),
                    reference_intensity=float(reference_intensities[reference_index]),
                    matched=True,
                )
            )
            query_index += 1
            reference_index += 1
        elif query_mass < reference_mass:
            peaks.append(
                MirrorPeak(
                    mz=query_mass,
                    query_intensity=float(query_intensities[query_index]),
                    reference_intensity=0.0,
                    matched=False,
                )
            )
            query_index += 1
        else:
            peaks.append(
                MirrorPeak(
                    mz=reference_mass,
                    query_intensity=0.0,
                    reference_intensity=float(reference_intensities[reference_index]),
                    matched=False,
                )
            )
            reference_index += 1

    while query_index < query_mz.size:
        peaks.append(
            MirrorPeak(
                mz=float(query_mz[query_index]),
                query_intensity=float(query_intensities[query_index]),
                reference_intensity=0.0,
                matched=False,
            )
        )
        query_index += 1

    while reference_index < reference_mz.size:
        peaks.append(
            MirrorPeak(
                mz=float(reference_mz[reference_index]),
                query_intensity=0.0,
                reference_intensity=float(reference_intensities[reference_index]),
                matched=False,
            )
        )
        reference_index += 1

    return peaks


def format_mz(value: Optional[float], digits: int = 4) -> str:
    """Format a precursor m/z for display; ``n/a`` for missing values."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def format_retention_time(seconds: Optional[float]) -> str:
    """Format retention time as minutes; ``n/a`` for missing values."""
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    return f"{seconds / 60.0:.2f} min"


def annotation_status(score: Optional[float]) -> str:
    """Classify a hit score as ``matched``, ``putative``, or ``unknown``.

    Mirrors the ``Annotation_Status`` convention of the core exporter:
    scores >= 0.9 are "Matched", any other non-null score is "Putative",
    missing scores are "Unknown".
    """
    if score is None or not math.isfinite(score):
        return "unknown"
    if score >= 0.9:
        return "matched"
    return "putative"
