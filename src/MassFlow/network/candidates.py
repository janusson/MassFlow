"""
Deterministic candidate-pair generation for spectral networking (Stage 1).

Networking must not score every pair of spectra blindly. This module produces
the candidate pairs whose precursors fall within an MS1 window (and, optionally,
whose retention times fall within an RT window), in a deterministic order, before
any expensive spectral scoring happens.

The implementation is a classical sliding-window scan over m/z-sorted indices:
``O(N log N + P)`` for ``N`` spectra and ``P`` candidate pairs. It performs no
spectral scoring and imports no similarity machinery.

A later phase may replace this with the experimental HNSW index
(:mod:`MassFlow.hnsw`) for sub-linear candidacy; the function signature is the
stable seam for that change.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["generate_candidate_pairs"]


def generate_candidate_pairs(
    precursor_mzs: ArrayLike,
    *,
    tolerance: float,
    tolerance_ppm: Optional[float] = None,
    rt_seconds: Optional[ArrayLike] = None,
    rt_tolerance: Optional[float] = None,
    treat_missing_rt_as_compatible: bool = False,
) -> list[tuple[int, int]]:
    """
    Generate candidate spectrum pairs within a precursor-mass window.

    Parameters
    ----------
    precursor_mzs : sequence of float
        Precursor m/z value per spectrum (float64). Missing values (``NaN``)
        are excluded from candidacy — a spectrum with no precursor cannot be
        windowed.
    tolerance : float
        Absolute component of the precursor m/z window, in **Da**: the floor of
        the admitted precursor difference for two spectra to be a candidate
        pair. Must be positive unless ``tolerance_ppm`` supplies a window. A
        caller that must not be more selective than another stage's own
        prefilter passes that prefilter's tolerance here, which keeps candidacy
        a strict superset of it.
    tolerance_ppm : float or None, optional
        Relative component of the window, in **ppm**. The effective window for a
        pair is ``max(tolerance, tolerance_ppm * mz / 1e6)``, evaluated on the
        **lower** precursor m/z of the pair, so the relation is symmetric and
        the window grows with mass — a fixed Da window is a drifting ppm window,
        and on a high-resolution instrument that drift is a selectivity defect.
        Must be positive when given.
    rt_seconds : sequence of float or None, optional
        Retention time in **seconds** per spectrum, aligned with
        ``precursor_mzs``. Required when ``rt_tolerance`` is set.
    rt_tolerance : float or None, optional
        Optional maximum absolute retention-time difference (seconds). When
        set, pairs whose retention times are missing (``NaN``) are normally
        excluded (fail closed; see ``treat_missing_rt_as_compatible``).
    treat_missing_rt_as_compatible : bool, optional
        When ``True``, a pair whose retention time is missing on either side is
        treated as RT-compatible instead of being excluded. This is intended for
        *feature grouping* (where an LC peak may legitimately lack an RT and the
        m/z channel is still informative), not for spectral candidacy, whose
        default remains fail-closed.

    Returns
    -------
    list of tuple[int, int]
        Sorted ``(i, j)`` index pairs with ``i < j``, where both precursors are
        present and within the effective window ``max(tolerance,
        tolerance_ppm * mz / 1e6)`` (and within ``rt_tolerance`` when given).
        Deterministic for identical inputs.

    Raises
    ------
    ValueError
        If ``tolerance`` is negative, if neither ``tolerance`` nor
        ``tolerance_ppm`` supplies a positive window, if ``tolerance_ppm`` is
        not positive when given, if ``rt_seconds`` is missing while
        ``rt_tolerance`` is set, or if the input lengths are inconsistent.

    Examples
    --------
    >>> generate_candidate_pairs([100.0, 100.01, 200.0], tolerance=0.02)
    [(0, 1)]
    """
    if tolerance < 0.0:
        raise ValueError(f"tolerance must not be negative; got {tolerance!r}.")
    if tolerance_ppm is not None and tolerance_ppm <= 0.0:
        raise ValueError(
            f"tolerance_ppm must be positive when given; got {tolerance_ppm!r}."
        )
    if tolerance == 0.0 and tolerance_ppm is None:
        raise ValueError(
            "tolerance must be positive when no ppm window is given; got 0.0."
        )

    mz_array = np.asarray(precursor_mzs, dtype=np.float64)
    n = mz_array.size

    if rt_tolerance is not None:
        if rt_seconds is None:
            raise ValueError("rt_tolerance requires rt_seconds.")
        rt_array = np.asarray(rt_seconds, dtype=np.float64)
        if rt_array.size != n:
            raise ValueError(
                f"rt_seconds length ({rt_array.size}) must match precursor_mzs "
                f"length ({n})."
            )
    else:
        rt_array = None

    # Sort by precursor m/z so the m/z window becomes a contiguous scan. Ties
    # are broken by index for determinism.
    order = np.argsort(mz_array, kind="stable")
    order = order[np.isfinite(mz_array[order])]
    sorted_mz = mz_array[order]

    # Effective window per sorted position: the Da floor raised by the ppm
    # component. It is evaluated on the pair's *lower* m/z and is monotone
    # non-decreasing in m/z, so the sweep below stays a valid two-pointer scan.
    if tolerance_ppm is None:
        windows = np.full(sorted_mz.size, tolerance, dtype=np.float64)
    else:
        windows = np.maximum(tolerance, tolerance_ppm * sorted_mz / 1e6)

    pairs: list[tuple[int, int]] = []
    for position, i in enumerate(order):
        upper = sorted_mz[position] + windows[position]
        for following in range(position + 1, order.size):
            j = int(order[following])
            if sorted_mz[following] > upper:
                break
            if rt_array is not None:
                rt_i = rt_array[i]
                rt_j = rt_array[j]
                if not (np.isfinite(rt_i) and np.isfinite(rt_j)):
                    if not treat_missing_rt_as_compatible:
                        continue
                elif abs(rt_i - rt_j) > rt_tolerance:  # type: ignore[operator]
                    continue
            pairs.append((int(i), int(j)) if i < j else (int(j), int(i)))

    # Canonical ordering: by first then second index.
    pairs.sort()
    return pairs
