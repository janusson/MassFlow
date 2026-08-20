"""
Numba-accelerated pre-filtering for pairwise spectral comparisons.

This module finalizes the experimental work in
``scripts/experiments/test_numba_prefilter.py``. The prefilter counts
tolerance-matched peaks between a reference and a query spectrum in two
complementary frames:

- **exact mass frame** — raw fragment m/z values (the gate for cosine
  scoring), and
- **neutral-loss frame** — ``precursor_mz - fragment_mz`` (the gate for
  modified-cosine scoring, whose precursor-aligned matching is equivalent to
  matching in neutral-loss space).

A query--reference pair is a *candidate* only when the match count in either
frame reaches ``min_matches``. Because the two-pointer merge count is an upper
bound on the greedy matched-peak count that matchms computes (every greedily
matched peak pair is one of the counted tolerance matches), skipping pairs
below the threshold is **conservative**: it can never remove a pair that the
subsequent exact scoring would have kept. Pairs where either spectrum lacks a
``precursor_mz`` bypass the gate entirely, preserving recall for spectra
without precursor information.

Two implementations are provided and are numerically identical:

- a ``numba``-JIT implementation used when numba is importable, and
- a pure NumPy/Python fallback for environments without numba.

The flat-array representation (concatenated sorted m/z vectors plus an int64
offset array) keeps the JIT kernels allocation-light and cache-friendly.

Example
-------
>>> rows, cols = prefilter_candidate_pairs(
...     references, queries, tolerance=0.02, min_matches=3,
...     algorithm="modified_cosine",
... )
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from matchms import Spectrum

logger = logging.getLogger(__name__)

_NUMBA_INSTALL_MSG = (
    "numba is not installed; the peak/neutral-loss prefilter falls back to a "
    "pure NumPy implementation with identical results (slower). Install numba "
    "for the JIT-accelerated path: pip install numba"
)

try:
    from numba import njit  # type: ignore[import-not-found]

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover -- exercised via the no-numba fallback
    _HAS_NUMBA = False
    logger.info("%s", _NUMBA_INSTALL_MSG)


# ---------------------------------------------------------------------------
# Missing-precursor helper
# ---------------------------------------------------------------------------


def _precursor_is_missing(precursor_mz: Optional[float]) -> bool:
    """Return ``True`` when a precursor m/z is absent or NaN."""
    if precursor_mz is None:
        return True
    try:
        return bool(np.isnan(float(precursor_mz)))
    except (ValueError, TypeError):
        return True


# ---------------------------------------------------------------------------
# Flat-array construction
# ---------------------------------------------------------------------------


def build_flat_peak_arrays(
    spectra: Sequence[Spectrum],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a list of spectra into contiguous arrays.

    Returns a 4-tuple ``(mz_flat, offsets, nl_flat, precursor_valid)``:

    - ``mz_flat`` — concatenation of each spectrum's ascending fragment m/z
      values (``float64``).
    - ``offsets`` — int64 array of length ``n + 1`` where ``offsets[i]``
      bounds spectrum *i*'s segment in ``mz_flat`` and ``nl_flat``.
    - ``nl_flat`` — concatenation of each spectrum's **ascending**
      neutral-loss values (``precursor_mz - mz``); zero-filled for spectra
      without a precursor.
    - ``precursor_valid`` — bool array flagging spectra with a usable
      precursor.

    Parameters
    ----------
    spectra : sequence of Spectrum
        Spectra to flatten. Fragment m/z values are assumed ascending
        (MassFlow's processing pipeline guarantees this invariant).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    mz_segments: list[np.ndarray] = []
    nl_segments: list[np.ndarray] = []
    n_spectra = len(spectra)
    offsets = np.zeros(n_spectra + 1, dtype=np.int64)
    precursor_valid = np.zeros(n_spectra, dtype=bool)

    for index, spectrum in enumerate(spectra):
        mz_array = np.ascontiguousarray(np.asarray(spectrum.peaks.mz, dtype=np.float64))
        mz_segments.append(mz_array)
        offsets[index + 1] = offsets[index] + mz_array.size

        precursor_mz = spectrum.get("precursor_mz")
        if not _precursor_is_missing(precursor_mz):
            precursor_valid[index] = True
            # m/z reversed is ascending in neutral-loss space.
            neutral_losses = float(precursor_mz) - mz_array[::-1]
            nl_segments.append(np.ascontiguousarray(neutral_losses, dtype=np.float64))
        else:
            nl_segments.append(np.zeros(mz_array.size, dtype=np.float64))

    if mz_segments:
        mz_flat = np.concatenate(mz_segments)
        nl_flat = np.concatenate(nl_segments)
    else:
        mz_flat = np.empty(0, dtype=np.float64)
        nl_flat = np.empty(0, dtype=np.float64)

    return mz_flat, offsets, nl_flat, precursor_valid


# ---------------------------------------------------------------------------
# Numba-JIT kernels
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True)
    def _count_tolerance_matches_numba(
        sorted_a: np.ndarray,
        sorted_b: np.ndarray,
        tolerance: float,
    ) -> int:
        """Count tolerance matches between two ascending float64 arrays.

        Two-pointer merge: with both arrays ascending, every tolerance match
        pair is encountered exactly once. The count is an upper bound on the
        greedy matched-peak count used by matchms scoring.
        """
        count = 0
        index_a = 0
        index_b = 0
        size_a = sorted_a.shape[0]
        size_b = sorted_b.shape[0]
        while index_a < size_a and index_b < size_b:
            difference = sorted_a[index_a] - sorted_b[index_b]
            if abs(difference) <= tolerance:
                count += 1
                index_a += 1
                index_b += 1
            elif difference < -tolerance:
                index_a += 1
            else:
                index_b += 1
        return count

    @njit(cache=True)
    def _prefilter_pairs_numba(
        ref_mz_flat: np.ndarray,
        ref_offsets: np.ndarray,
        ref_nl_flat: np.ndarray,
        query_mz_flat: np.ndarray,
        query_offsets: np.ndarray,
        query_nl_flat: np.ndarray,
        tolerance: float,
        min_matches: int,
        check_mz: bool,
        check_nl: bool,
        ref_precursor_valid: np.ndarray,
        query_precursor_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return candidate ``(rows, cols)`` index arrays (int64).

        Two passes: the first counts candidates per reference so the second
        can allocate the exact output size, bounding peak memory to the true
        candidate count instead of the worst-case ``n_refs * n_queries``.
        """
        n_refs = ref_offsets.shape[0] - 1
        n_queries = query_offsets.shape[0] - 1

        counts = np.zeros(n_refs, dtype=np.int64)
        for ref_index in range(n_refs):
            ref_start = ref_offsets[ref_index]
            ref_end = ref_offsets[ref_index + 1]
            ref_mz_view = ref_mz_flat[ref_start:ref_end]
            ref_nl_view = ref_nl_flat[ref_start:ref_end]
            for query_index in range(n_queries):
                query_start = query_offsets[query_index]
                query_end = query_offsets[query_index + 1]

                if (
                    not ref_precursor_valid[ref_index]
                    or not query_precursor_valid[query_index]
                ):
                    counts[ref_index] += 1
                    continue

                query_mz_view = query_mz_flat[query_start:query_end]
                query_nl_view = query_nl_flat[query_start:query_end]

                passes = False
                if check_mz:
                    if (
                        _count_tolerance_matches_numba(
                            ref_mz_view, query_mz_view, tolerance
                        )
                        >= min_matches
                    ):
                        passes = True
                if not passes and check_nl:
                    if (
                        _count_tolerance_matches_numba(
                            ref_nl_view, query_nl_view, tolerance
                        )
                        >= min_matches
                    ):
                        passes = True
                if passes:
                    counts[ref_index] += 1

        total = counts.sum()
        rows = np.empty(total, dtype=np.int64)
        cols = np.empty(total, dtype=np.int64)
        position = 0

        for ref_index in range(n_refs):
            ref_start = ref_offsets[ref_index]
            ref_end = ref_offsets[ref_index + 1]
            ref_mz_view = ref_mz_flat[ref_start:ref_end]
            ref_nl_view = ref_nl_flat[ref_start:ref_end]
            for query_index in range(n_queries):
                query_start = query_offsets[query_index]
                query_end = query_offsets[query_index + 1]

                if (
                    not ref_precursor_valid[ref_index]
                    or not query_precursor_valid[query_index]
                ):
                    rows[position] = ref_index
                    cols[position] = query_index
                    position += 1
                    continue

                query_mz_view = query_mz_flat[query_start:query_end]
                query_nl_view = query_nl_flat[query_start:query_end]

                passes = False
                if check_mz:
                    if (
                        _count_tolerance_matches_numba(
                            ref_mz_view, query_mz_view, tolerance
                        )
                        >= min_matches
                    ):
                        passes = True
                if not passes and check_nl:
                    if (
                        _count_tolerance_matches_numba(
                            ref_nl_view, query_nl_view, tolerance
                        )
                        >= min_matches
                    ):
                        passes = True
                if passes:
                    rows[position] = ref_index
                    cols[position] = query_index
                    position += 1

        return rows[:position], cols[:position]

else:  # pragma: no cover -- numba is installed in the supported environment
    _count_tolerance_matches_numba = None
    _prefilter_pairs_numba = None


# ---------------------------------------------------------------------------
# Pure NumPy/Python fallback (identical semantics)
# ---------------------------------------------------------------------------


def _count_tolerance_matches_python(
    sorted_a: np.ndarray,
    sorted_b: np.ndarray,
    tolerance: float,
) -> int:
    """Pure-Python mirror of the JIT two-pointer match counter."""
    count = 0
    index_a = 0
    index_b = 0
    size_a = sorted_a.shape[0]
    size_b = sorted_b.shape[0]
    while index_a < size_a and index_b < size_b:
        difference = sorted_a[index_a] - sorted_b[index_b]
        if abs(difference) <= tolerance:
            count += 1
            index_a += 1
            index_b += 1
        elif difference < -tolerance:
            index_a += 1
        else:
            index_b += 1
    return count


def _prefilter_pairs_python(
    ref_mz_flat: np.ndarray,
    ref_offsets: np.ndarray,
    ref_nl_flat: np.ndarray,
    query_mz_flat: np.ndarray,
    query_offsets: np.ndarray,
    query_nl_flat: np.ndarray,
    tolerance: float,
    min_matches: int,
    check_mz: bool,
    check_nl: bool,
    ref_precursor_valid: np.ndarray,
    query_precursor_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Python mirror of :func:`_prefilter_pairs_numba`."""
    n_refs = ref_offsets.shape[0] - 1
    n_queries = query_offsets.shape[0] - 1

    counts = np.zeros(n_refs, dtype=np.int64)
    for ref_index in range(n_refs):
        ref_start = ref_offsets[ref_index]
        ref_end = ref_offsets[ref_index + 1]
        ref_mz_view = ref_mz_flat[ref_start:ref_end]
        ref_nl_view = ref_nl_flat[ref_start:ref_end]
        for query_index in range(n_queries):
            query_start = query_offsets[query_index]
            query_end = query_offsets[query_index + 1]

            if (
                not ref_precursor_valid[ref_index]
                or not query_precursor_valid[query_index]
            ):
                counts[ref_index] += 1
                continue

            query_mz_view = query_mz_flat[query_start:query_end]
            query_nl_view = query_nl_flat[query_start:query_end]

            passes = False
            if check_mz:
                if (
                    _count_tolerance_matches_python(
                        ref_mz_view, query_mz_view, tolerance
                    )
                    >= min_matches
                ):
                    passes = True
            if not passes and check_nl:
                if (
                    _count_tolerance_matches_python(
                        ref_nl_view, query_nl_view, tolerance
                    )
                    >= min_matches
                ):
                    passes = True
            if passes:
                counts[ref_index] += 1

    total = int(counts.sum())
    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    position = 0

    for ref_index in range(n_refs):
        ref_start = ref_offsets[ref_index]
        ref_end = ref_offsets[ref_index + 1]
        ref_mz_view = ref_mz_flat[ref_start:ref_end]
        ref_nl_view = ref_nl_flat[ref_start:ref_end]
        for query_index in range(n_queries):
            query_start = query_offsets[query_index]
            query_end = query_offsets[query_index + 1]

            if (
                not ref_precursor_valid[ref_index]
                or not query_precursor_valid[query_index]
            ):
                rows[position] = ref_index
                cols[position] = query_index
                position += 1
                continue

            query_mz_view = query_mz_flat[query_start:query_end]
            query_nl_view = query_nl_flat[query_start:query_end]

            passes = False
            if check_mz:
                if (
                    _count_tolerance_matches_python(
                        ref_mz_view, query_mz_view, tolerance
                    )
                    >= min_matches
                ):
                    passes = True
            if not passes and check_nl:
                if (
                    _count_tolerance_matches_python(
                        ref_nl_view, query_nl_view, tolerance
                    )
                    >= min_matches
                ):
                    passes = True
            if passes:
                rows[position] = ref_index
                cols[position] = query_index
                position += 1

    return rows[:position], cols[:position]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prefilter_candidate_pairs(
    references: Sequence[Spectrum],
    queries: Sequence[Spectrum],
    tolerance: float,
    min_matches: int,
    algorithm: str = "modified_cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate ``(ref_index, query_index)`` pairs for exact scoring.

    Parameters
    ----------
    references : sequence of Spectrum
        Reference library spectra.
    queries : sequence of Spectrum
        Query spectra.
    tolerance : float
        Fragment mass tolerance (Da) applied in both the m/z and neutral-loss
        frames. Use the same value passed to the matchms scoring function.
    min_matches : int
        Minimum number of matched peaks required to keep a pair. When
        ``min_matches <= 0`` every pair is returned (no prefiltering).
    algorithm : str, optional
        ``"cosine"`` gates on exact m/z matches only; ``"modified_cosine"``
        (default) gates on m/z **or** neutral-loss matches, matching the
        shifted-alignment semantics of modified cosine.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(rows, cols)`` int64 index arrays identifying candidate pairs.

    Examples
    --------
    >>> rows, cols = prefilter_candidate_pairs(
    ...     references, queries, tolerance=0.02, min_matches=3
    ... )
    """
    n_refs = len(references)
    n_queries = len(queries)

    if n_refs == 0 or n_queries == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    if min_matches <= 0:
        rows = np.repeat(np.arange(n_refs, dtype=np.int64), n_queries)
        cols = np.tile(np.arange(n_queries, dtype=np.int64), n_refs)
        return rows, cols

    check_mz = algorithm in ("cosine", "modified_cosine")
    check_nl = algorithm == "modified_cosine"

    ref_mz_flat, ref_offsets, ref_nl_flat, ref_precursor_valid = build_flat_peak_arrays(
        references
    )
    query_mz_flat, query_offsets, query_nl_flat, query_precursor_valid = (
        build_flat_peak_arrays(queries)
    )

    kernel = _prefilter_pairs_numba if _HAS_NUMBA else _prefilter_pairs_python
    return kernel(  # type: ignore[operator]
        ref_mz_flat,
        ref_offsets,
        ref_nl_flat,
        query_mz_flat,
        query_offsets,
        query_nl_flat,
        float(tolerance),
        int(min_matches),
        check_mz,
        check_nl,
        ref_precursor_valid,
        query_precursor_valid,
    )


__all__ = [
    "_HAS_NUMBA",
    "build_flat_peak_arrays",
    "prefilter_candidate_pairs",
]
