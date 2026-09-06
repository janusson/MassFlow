"""
Spectral similarity search engine for MassFlow.

This module encapsulates the logic for comparing experimental mass spectra against
reference libraries. It provides a unified interface (`SimilarityEngine`) to classical
similarity algorithms (Cosine, Modified Cosine) backed by matchms. It handles score
calculation, result filtering/formatting, decoy generation, and FDR estimation.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.metadata
import logging
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
)

import numpy as np
from matchms import Spectrum, calculate_scores

if TYPE_CHECKING:
    from MassFlow.hnsw import HNSWSpectralIndex

try:
    from matchms.similarity import (
        CosineGreedy,  # type: ignore[attr-defined]
        ModifiedCosine,  # type: ignore[attr-defined]
    )
except ImportError:
    from matchms.similarity import CosineGreedy
    from matchms.similarity import ModifiedCosineGreedy as ModifiedCosine

from MassFlow.config import SimilarityConfig
from MassFlow.models import TriageProfile
from MassFlow.protocols import MLEngineProtocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional machine-learning imports
# ---------------------------------------------------------------------------
# Deep learning and vector-based similarity engines (spec2vec, ms2deepscore,
# consensus, cascade) depend on PyTorch, Gensim, Spec2Vec, and MS2DeepScore.
# These are provided via the ``[ml]`` extra and are NOT required for the core
# classical scoring pipeline (cosine / modified_cosine).

_ML_INSTALL_MSG = (
    "This scoring engine requires the machine-learning extras. "
    "Install them with: pip install massflow[ml]"
)

_HAS_TORCH = False
_HAS_GENSIM = False
_HAS_SPEC2VEC = False
_HAS_MS2DEEPSCORE = False
_HAS_ML = False

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    pass

try:
    import gensim  # noqa: F401

    _HAS_GENSIM = True
except ImportError:
    pass

try:
    import spec2vec  # noqa: F401

    _HAS_SPEC2VEC = True
except ImportError:
    pass

try:
    import ms2deepscore  # noqa: F401

    _HAS_MS2DEEPSCORE = True
except ImportError:
    pass

_HAS_ML = _HAS_TORCH and _HAS_GENSIM and _HAS_SPEC2VEC and _HAS_MS2DEEPSCORE

if not _HAS_ML:
    missing = []
    if not _HAS_TORCH:
        missing.append("torch")
    if not _HAS_GENSIM:
        missing.append("gensim")
    if not _HAS_SPEC2VEC:
        missing.append("spec2vec")
    if not _HAS_MS2DEEPSCORE:
        missing.append("ms2deepscore")
    logger.info(
        "Machine-learning extras not fully available (missing: %s). "
        "Classical cosine / modified_cosine scoring remains fully functional. "
        "Install ML engines with: pip install massflow[ml]",
        ", ".join(missing),
    )


def calculate_mass_error_ppm(query_mz: float, ref_mz: float) -> float:
    """Calculate the mass error in parts-per-million (ppm).

    Mass error (ppm) = |query_mz - ref_mz| / ref_mz * 1e6

    Parameters
    ----------
    query_mz : float
        The precursor m/z of the query spectrum.
    ref_mz : float
        The precursor m/z of the reference spectrum.

    Returns
    -------
    float
        The absolute mass error in ppm.
    """
    return abs(query_mz - ref_mz) / ref_mz * 1e6


def yield_fixed_chunks(
    spectra: Iterable[Spectrum], chunk_size: int = 10000
) -> Iterator[List[Spectrum]]:
    """Yield spectra grouped into fixed-size chunks to optimize vectorized similarity operations."""
    current_chunk: List[Spectrum] = []

    for s in spectra:
        current_chunk.append(s)
        if len(current_chunk) >= chunk_size:
            yield current_chunk
            current_chunk = []

    if current_chunk:
        yield current_chunk


def _handle_lazy_reference_spectra(func):
    """Decorator to allow search engines to lazily process generator inputs in chunks."""

    @wraps(func)
    def wrapper(
        self,
        query_spectra,
        reference_spectra,
        min_score=None,
        top_n=None,
        include_decoys=True,
        **kwargs,
    ):
        if not isinstance(reference_spectra, (list, tuple)):
            all_results = []
            processed_count = 0

            # When chunking, pre-computed ref_precursor_mzs / ref_is_decoy
            # arrays are aligned with the *full* library, not individual chunks.
            # Strip them to avoid incorrect indexing in sub-calls.
            chunk_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("ref_precursor_mzs", "ref_is_decoy")
            }

            for chunk in yield_fixed_chunks(reference_spectra, chunk_size=10000):
                if not chunk:
                    continue

                processed_count += len(chunk)
                logger.debug(
                    f"Streaming library... scored against {processed_count} reference spectra so far."
                )

                all_results.extend(
                    func(
                        self,
                        query_spectra,
                        chunk,
                        min_score=min_score,
                        top_n=top_n,
                        include_decoys=include_decoys,
                        **chunk_kwargs,
                    )
                )

            if top_n is not None:
                grouped = defaultdict(list)
                for r in all_results:
                    grouped[r["query_id"]].append(r)

                final_res = []
                for q_id, hits in grouped.items():
                    hits.sort(key=lambda x: x["score"], reverse=True)
                    final_res.extend(hits[:top_n])
                return final_res
            return all_results
        return func(
            self,
            query_spectra,
            reference_spectra,
            min_score,
            top_n,
            include_decoys,
            **kwargs,
        )

    return wrapper


class SearchResult(TypedDict):
    """Structured dictionary for search results."""

    query_id: str
    query_precursor_mz: float | None
    reference_id: str
    reference_name: str | None
    reference_precursor_mz: float | None
    score: float
    matched_peaks: int
    smiles: str | None
    inchikey: str | None
    is_decoy: bool
    q_value: float
    p_value: float | None
    annotation_tier: str | None
    structural_similarity: float | None
    mass_error_ppm: float | None
    score_breakdown: dict[str, float] | None


def _is_missing(val):
    if val is None:
        return True
    try:
        return np.isnan(float(val))
    except (ValueError, TypeError):
        return True


def _adduct_modes_compatible(ref_adduct: str | None, query_adduct: str | None) -> bool:
    """Return False if adducts belong to opposite ionisation modes.

    Positive-mode adducts (e.g. ``[M+H]+``) and negative-mode adducts
    (e.g. ``[M-H]-``) are physically incompatible. If either adduct is
    ``None`` the pair is allowed through (no opinion).
    """
    if ref_adduct is None or query_adduct is None:
        return True
    ref_pos = "+" in ref_adduct
    ref_neg = "-" in ref_adduct
    qry_pos = "+" in query_adduct
    qry_neg = "-" in query_adduct
    if (ref_pos and qry_neg) or (ref_neg and qry_pos):
        return False
    return True


def _ms1_prefilter(
    all_references: List[Spectrum],
    query_spectra: List[Spectrum],
    ms1_tolerance: float,
    resolution_ppm: Optional[float] = None,
) -> tuple[np.ndarray, ...]:
    """Perform MS1 precursor m/z pre-filtering using Da tolerance or optionally PPM resolution.

    For PPM mode, the half-window is computed as ``resolution_ppm * query_mz / 1e6``
    (i.e., using the query m/z as the denominator). This matches the pre-filter gate
    documented by ``calculate_mass_error_ppm`` to within second-order error for
    typical sub-100-ppm tolerances.
    """
    ref_mzs_raw: list[Optional[float]] = [s.get("precursor_mz") for s in all_references]  # type: ignore[assignment]
    query_mzs_raw: list[Optional[float]] = [
        q.get("precursor_mz") for q in query_spectra
    ]  # type: ignore[assignment]

    # Track missing precursors to allow them to bypass the MS1 filter
    ref_missing = np.array([_is_missing(r) for r in ref_mzs_raw], dtype=bool)
    query_missing = np.array([_is_missing(q) for q in query_mzs_raw], dtype=bool)

    ref_mzs = np.array(
        [float(r) if r is not None and not _is_missing(r) else 0.0 for r in ref_mzs_raw]
    )
    query_mzs = np.array(
        [
            float(q) if q is not None and not _is_missing(q) else 0.0
            for q in query_mzs_raw
        ]
    )

    # Use binary search for O(Q * log(R)) complexity, avoiding O(R * Q) dense arrays.
    # Both Da and PPM tolerance paths share the same logic, differing only in the
    # half-window size computed per query.
    query_mzs_indexed = list(enumerate(query_mzs))
    ref_mzs_sorted_indices = np.argsort(ref_mzs)
    ref_mzs_sorted = ref_mzs[ref_mzs_sorted_indices]

    rows: List[int] = []
    cols: List[int] = []
    for query_idx, query_mz in query_mzs_indexed:
        if query_mz > 0:
            if resolution_ppm is not None:
                # Use the query m/z as the reference for the ppm → Da conversion
                # (see docstring note on denominator convention).
                half_window = resolution_ppm * query_mz / 1e6
            else:
                half_window = ms1_tolerance

            min_mz = query_mz - half_window
            max_mz = query_mz + half_window

            start_idx = np.searchsorted(ref_mzs_sorted, min_mz, side="left")
            end_idx = np.searchsorted(ref_mzs_sorted, max_mz, side="right")

            original_indices = ref_mzs_sorted_indices[start_idx:end_idx]
            rows.extend(original_indices)
            cols.extend([query_idx] * len(original_indices))

    # Also include spectra with missing precursors, as they bypass the filter
    for i in np.where(ref_missing)[0]:
        rows.extend([i] * len(query_mzs))
        cols.extend(range(len(query_mzs)))
    for i in np.where(query_missing)[0]:
        rows.extend(range(len(ref_mzs)))
        cols.extend([i] * len(ref_mzs))

    return np.array(rows), np.array(cols)


def _ms1_prefilter_arrays(
    ref_mzs: np.ndarray,
    query_spectra: List[Spectrum],
    ms1_tolerance: float,
    resolution_ppm: Optional[float] = None,
) -> tuple[np.ndarray, ...]:
    """Perform MS1 precursor m/z pre-filtering using pre-computed reference arrays.

    This is the L2-cached variant of ``_ms1_prefilter``. Instead of extracting
    precursor_mz from Spectrum objects, it consumes a flat ``float64`` numpy
    array produced during ``_init_worker``. Query precursor values are still
    extracted from the (typically small) query set; they are not pre-cached
    because the query list is per-file and relatively small.

    Returns the same (rows, cols) sparse index arrays as ``_ms1_prefilter``.
    """
    query_mzs_raw: list[Optional[float]] = [
        q.get("precursor_mz")
        for q in query_spectra  # type: ignore[assignment]
    ]
    query_missing = np.array([_is_missing(q) for q in query_mzs_raw], dtype=bool)
    query_mzs = np.array(
        [
            float(q) if q is not None and not _is_missing(q) else 0.0
            for q in query_mzs_raw
        ]
    )

    # References with mz <= 0 are treated as missing.
    ref_missing = ref_mzs <= 0.0
    ref_mzs_safe = np.where(ref_missing, 0.0, ref_mzs)

    query_mzs_indexed = list(enumerate(query_mzs))
    ref_mzs_sorted_indices = np.argsort(ref_mzs_safe)
    ref_mzs_sorted = ref_mzs_safe[ref_mzs_sorted_indices]

    rows: List[int] = []
    cols: List[int] = []
    for query_idx, query_mz in query_mzs_indexed:
        if query_mz > 0:
            if resolution_ppm is not None:
                half_window = resolution_ppm * query_mz / 1e6
            else:
                half_window = ms1_tolerance

            min_mz = query_mz - half_window
            max_mz = query_mz + half_window

            start_idx = np.searchsorted(ref_mzs_sorted, min_mz, side="left")
            end_idx = np.searchsorted(ref_mzs_sorted, max_mz, side="right")

            original_indices = ref_mzs_sorted_indices[start_idx:end_idx]
            rows.extend(original_indices)
            cols.extend([query_idx] * len(original_indices))

    # Include spectra with missing precursors (bypass filter)
    for i in np.where(ref_missing)[0]:
        rows.extend([i] * len(query_mzs))
        cols.extend(range(len(query_mzs)))
    for i in np.where(query_missing)[0]:
        rows.extend(range(len(ref_mzs)))
        cols.extend([i] * len(ref_mzs))

    return np.array(rows), np.array(cols)


# ---------------------------------------------------------------------------
# Entropy-based decoy generation (FDR calibration)
# ---------------------------------------------------------------------------
# Defaults for the entropy-preserving decoy generator. Spectral cosine /
# modified cosine are not metrics and naive fragment shuffling biases the
# target-decoy null distribution, so decoys are generated to preserve the
# precursor m/z and the spectral entropy — the ion information content —
# of their source spectra while randomizing the fragmentation pathways
# (intensity-to-position pairing and fragment positions).
#
# Spectral entropy (Li et al., Nat. Methods 2021) applies a square-root
# intensity weighting (w = I^0.5) before normalization so that a few intense
# base peaks do not dominate the information estimate while low-abundance
# peaks retain meaningful weight.  A hard baseline filter (peaks below
# ``min_relative_intensity`` × the base peak, default 1%) is applied FIRST so
# chemical noise cannot artificially inflate the entropy.

_DEFAULT_DECOY_MIN_RELATIVE_INTENSITY: float = 0.01
_DEFAULT_DECOY_MZ_SHIFT_DA: float = 1.0


def spectral_entropy(
    intensities: np.ndarray,
    min_relative_intensity: float = _DEFAULT_DECOY_MIN_RELATIVE_INTENSITY,
) -> float:
    """Compute the spectral entropy (nats) of fragment intensities.

    Follows the spectral-entropy definition of Li et al. (Nat. Methods 2021):

    1. **Hard baseline filter** — peaks below ``min_relative_intensity`` ×
       the base peak (default 1% of the base peak) are removed, so
       low-abundance chemical noise cannot artificially inflate the
       information content.
    2. **Square-root intensity weighting** — the remaining intensities are
       weighted as ``w = I**0.5`` and normalized to a probability vector;
       the entropy is ``H = -Σ p ln p``.  The sqrt weighting prevents a few
       intense base peaks from dominating the estimate (raw linear
       intensities would over-weight them).

    Parameters
    ----------
    intensities : np.ndarray
        One-dimensional fragment intensities. The entropy is invariant to
        global intensity scaling, so raw or normalized intensities give the
        same result.
    min_relative_intensity : float, optional
        Baseline noise floor as a fraction of the **base peak** (the maximum
        intensity). Peaks below ``min_relative_intensity × base_peak`` are
        excluded before weighting. Defaults to 0.01 (1% of the base peak).

    Returns
    -------
    float
        Spectral entropy in nats (natural logarithm). Returns 0.0 when fewer
        than two peaks survive filtering (a single-peak spectrum carries no
        information).

    Examples
    --------
    >>> spectral_entropy(np.array([1.0, 1.0]))
    0.6931...  # ln(2)
    """
    intensity_array = np.asarray(intensities, dtype=np.float64)
    if intensity_array.size == 0:
        return 0.0

    base_peak = float(np.max(intensity_array))
    if not np.isfinite(base_peak) or base_peak <= 0.0:
        return 0.0

    # Strict baseline filtering BEFORE the information estimate: noise
    # peaks below the relative floor of the base peak are discarded.
    keep_mask = intensity_array >= min_relative_intensity * base_peak
    kept = intensity_array[keep_mask]
    if kept.size < 2:
        return 0.0

    # Spectral entropy weighting: I^0.5 (sqrt) before normalization.
    weights = np.sqrt(kept)
    probabilities = weights / np.sum(weights)
    return float(-np.sum(probabilities * np.log(probabilities)))


def compare_target_decoy_entropy(
    target_spectra: Sequence[Spectrum],
    decoy_spectra: Sequence[Spectrum],
    min_relative_intensity: float = _DEFAULT_DECOY_MIN_RELATIVE_INTENSITY,
) -> dict[str, float]:
    """Compare Shannon entropy distributions of targets and decoys.

    Diagnostic for FDR calibration: entropy-preserving decoys should produce
    a target and decoy entropy distribution that do not systematically
    diverge. A large ``mean_abs_entropy_delta`` indicates biased decoy
    generation (e.g. unfiltered noise peaks or degenerate intensity
    profiles) and should be investigated before trusting q-values.

    Parameters
    ----------
    target_spectra : sequence of Spectrum
        Reference (target) spectra.
    decoy_spectra : sequence of Spectrum
        Generated decoy spectra, aligned one-to-one with the targets.
    min_relative_intensity : float, optional
        Relative intensity floor passed to :func:`spectral_entropy`.

    Returns
    -------
    dict[str, float]
        Summary with ``mean_target_entropy``, ``mean_decoy_entropy``,
        ``mean_abs_entropy_delta``, ``max_abs_entropy_delta`` and
        ``compared_pairs``. Empty inputs yield NaN statistics.

    Examples
    --------
    >>> compare_target_decoy_entropy(targets, decoys)
    {'mean_target_entropy': ..., 'mean_decoy_entropy': ..., ...}
    """
    target_entropies = np.array(
        [
            spectral_entropy(
                np.asarray(spectrum.peaks.intensities, dtype=np.float64),
                min_relative_intensity,
            )
            for spectrum in target_spectra
        ],
        dtype=np.float64,
    )
    decoy_entropies = np.array(
        [
            spectral_entropy(
                np.asarray(spectrum.peaks.intensities, dtype=np.float64),
                min_relative_intensity,
            )
            for spectrum in decoy_spectra
        ],
        dtype=np.float64,
    )

    compared_pairs = min(target_entropies.size, decoy_entropies.size)
    if compared_pairs == 0:
        return {
            "mean_target_entropy": np.nan,
            "mean_decoy_entropy": np.nan,
            "mean_abs_entropy_delta": np.nan,
            "max_abs_entropy_delta": np.nan,
            "compared_pairs": 0.0,
        }

    deltas = np.abs(
        target_entropies[:compared_pairs] - decoy_entropies[:compared_pairs]
    )
    return {
        "mean_target_entropy": float(np.mean(target_entropies)),
        "mean_decoy_entropy": float(np.mean(decoy_entropies)),
        "mean_abs_entropy_delta": float(np.mean(deltas)),
        "max_abs_entropy_delta": float(np.max(deltas)),
        "compared_pairs": float(compared_pairs),
    }


def generate_decoys(
    spectra: List[Spectrum],
    random_seed: int = 42,
    min_relative_intensity: float = _DEFAULT_DECOY_MIN_RELATIVE_INTENSITY,
    mz_shift_da: float = _DEFAULT_DECOY_MZ_SHIFT_DA,
) -> List[Spectrum]:
    """Generate entropy-preserving decoy spectra for target-decoy FDR.

    Naive fragment shuffling is replaced with entropy-based decoy generation.
    Each decoy:

    1. **Preserves the precursor m/z** of its source spectrum.
    2. **Preserves the spectral entropy** — the information content computed
       with the spectral-entropy weighting (``I**0.5``) after a strict
       baseline filter (peaks below ``min_relative_intensity`` × the base
       peak are removed) — exactly, up to floating-point rounding.
    3. **Randomizes the fragmentation pathways**: intensities are reassigned
       across peaks (random permutation of the filtered intensity profile)
       and fragment positions are jittered by ``±mz_shift_da``, so decoys
       share no fragment positions with their source at scoring tolerance.

    Because the filtered intensity profile is permuted (not resampled), the
    weighted intensity distribution — and therefore the spectral entropy —
    is preserved exactly. This prevents the systematic entropy divergence
    between targets and decoys that biases naive permutation-based FDR
    calibration; raw chemical noise is excluded by the base-peak filter
    before the information content is computed.

    Degenerate spectra with fewer than two distinct filtered intensities are
    tapered (random 0.5–1.0 multipliers) so the decoy is never identical to
    its source; this can only occur for synthetic/uniform spectra and slightly
    perturbs entropy for those cases.

    Parameters
    ----------
    spectra : list of Spectrum
        Reference spectra to decoy.
    random_seed : int, optional
        Seed for the decoy RNG (deterministic generation).
    min_relative_intensity : float, optional
        Baseline noise floor as a fraction of the **base peak**; peaks below
        ``min_relative_intensity × base_peak`` are excluded before entropy
        computation and decoy construction.
    mz_shift_da : float, optional
        Uniform per-peak m/z jitter (Da) applied to decoy fragment positions.

    Returns
    -------
    list of Spectrum
        One decoy per input spectrum, with ``id`` and ``compound_name``
        suffixed ``_decoy``, ``is_decoy=True``, and the preserved target
        spectral entropy recorded in the ``spectral_entropy`` metadata field.

    Examples
    --------
    >>> decoys = generate_decoys(reference_spectra)

    Notes
    -----
    **Chunk-invariance:** each spectrum derives its own RNG seed from a
    stable hash of ``(random_seed, m/z array, intensity array)``, so
    ``decoy(spectrum)`` depends only on the spectrum and the master seed.
    Decoys are therefore identical whether the library is processed in one
    pass or in chunks (e.g. the workflow's streaming-library path), which
    keeps the FDR null distribution identical across execution modes. The
    hash is content-based (not id-based) so it is stable across processes
    and does not depend on ``PYTHONHASHSEED``.
    """
    decoys: List[Spectrum] = []

    for spec in spectra:
        # Independent per-spectrum RNG stream derived from the spectrum's
        # content (see Notes above): chunk boundaries cannot change the
        # decoy of any spectrum.
        mz_array = np.asarray(spec.peaks.mz, dtype=np.float64)
        intensity_array = np.asarray(spec.peaks.intensities, dtype=np.float64)
        seed_digest = hashlib.sha256(
            str(random_seed).encode("ascii")
            + mz_array.tobytes()
            + intensity_array.tobytes()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(seed_digest[:8], "little"))
        decoy_metadata = spec.metadata.copy()
        decoy_metadata["is_decoy"] = True
        decoy_id = str(spec.get("id", "unknown")) + "_decoy"
        decoy_metadata["id"] = decoy_id

        name = spec.get("compound_name") or spec.get("name")
        if name:
            decoy_metadata["compound_name"] = f"{name}_decoy"

        n_peaks = mz_array.size

        # Preserved ion information content (entropy) of the source.
        target_entropy = spectral_entropy(intensity_array, min_relative_intensity)
        decoy_metadata["spectral_entropy"] = target_entropy

        # Degenerate spectrum (no peaks / zero total intensity): nothing to
        # randomize; copy as-is.
        total_intensity = float(np.sum(intensity_array))
        if n_peaks == 0 or total_intensity <= 0.0:
            decoys.append(
                Spectrum(
                    mz=mz_array.copy(),
                    intensities=intensity_array.copy(),
                    metadata=decoy_metadata,
                )
            )
            continue

        # Strict baseline filtering BEFORE entropy computation and decoy
        # construction: peaks below ``min_relative_intensity`` × the base
        # peak are chemical noise and are excluded so they cannot skew the
        # spectral entropy estimate or leak into decoys.
        base_peak = float(np.max(intensity_array))
        if base_peak > 0.0:
            keep_mask = intensity_array >= min_relative_intensity * base_peak
        else:
            keep_mask = np.ones(n_peaks, dtype=bool)
        if int(np.count_nonzero(keep_mask)) < 2:
            # Very sparse spectra: fall back to the full peak list rather
            # than generating a degenerate single-peak decoy.
            keep_mask = np.ones(n_peaks, dtype=bool)

        filtered_mz = mz_array[keep_mask]
        filtered_intensities = intensity_array[keep_mask]
        n_filtered = filtered_intensities.size

        # Randomize the fragmentation pathways. A permutation of the filtered
        # intensity profile preserves the normalized intensity distribution
        # (and therefore the entropy) exactly.
        if np.unique(filtered_intensities).size < 2:
            # Permuting identical values is a no-op: taper instead so the
            # decoy is never identical to its source.
            taper = rng.uniform(0.5, 1.0, size=n_filtered)
            rng.shuffle(taper)
            decoy_intensities = filtered_intensities * taper
        else:
            permutation = rng.permutation(n_filtered)
            if np.array_equal(permutation, np.arange(n_filtered)):
                permutation = np.roll(permutation, 1)
            decoy_intensities = filtered_intensities[permutation]

        # Jitter fragment positions so decoys share no fragment positions
        # with their source at scoring tolerance; keep positions positive and
        # ascending (MassFlow spectra are always m/z-sorted).
        position_jitter = rng.uniform(-mz_shift_da, mz_shift_da, size=n_filtered)
        decoy_mz = np.maximum(filtered_mz + position_jitter, 0.01)
        sort_order = np.argsort(decoy_mz)

        decoys.append(
            Spectrum(
                mz=decoy_mz[sort_order],
                intensities=decoy_intensities[sort_order],
                metadata=decoy_metadata,
            )
        )

    return decoys


def calculate_empirical_p_values(
    target_scores: np.ndarray, decoy_scores: np.ndarray
) -> np.ndarray:
    """
    Calculate per-query empirical p-values against a decoy null distribution.

    COMPETITION UNIT (contract): inputs are per-query best scores, aligned
    with the target-decoy competition used by :func:`calculate_fdr`. Each
    entry of ``target_scores`` is ONE query's best (maximum) target score;
    each entry of ``decoy_scores`` is ONE query's best decoy score (only
    queries that produced at least one decoy hit).

    For a target score ``s`` the p-value is

        p = (1 + #{decoy scores >= s}) / (1 + #{decoy scores})

    i.e. the fraction of decoy competitions that matched or beat ``s``,
    with a +1 pseudo-count in numerator and denominator. Ties are counted
    against the target (conservative). When no decoy scores exist the
    null is empty and every p-value is 1.0 (no calibration possible).

    Uses binary search on sorted decoy scores for O(N log M) time and O(1)
    extra memory, avoiding the O(N x M) intermediate array that could
    exhaust RAM for very large libraries.

    Parameters
    ----------
    target_scores : np.ndarray
        Per-query best target scores. Shape: (N,), dtype: float.
    decoy_scores : np.ndarray
        Per-query best decoy scores. Shape: (M,), dtype: float.

    Returns
    -------
    np.ndarray
        Empirical p-values for each target score, same order as
        ``target_scores``.
    """
    if len(decoy_scores) == 0:
        return np.ones_like(target_scores)

    sorted_decoys = np.sort(decoy_scores)
    # For each target score, count how many decoy scores are >= it via binary search.
    positions = np.searchsorted(sorted_decoys, target_scores, side="left")
    greater_equal = len(decoy_scores) - positions

    # Apply +1 pseudo-count to numerator and denominator
    p_values = (greater_equal.astype(float) + 1.0) / (len(decoy_scores) + 1.0)
    return p_values


def calculate_fdr(
    target_scores: np.ndarray, decoy_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate per-query target-decoy competition (TDC) q-values.

    COMPETITION UNIT (contract): the unit of competition is the **query
    spectrum**, not the individual hit. ``target_scores`` and
    ``decoy_scores`` must be per-query best scores: each entry of
    ``target_scores`` is ONE query's best (maximum) score against the
    target library, and each entry of ``decoy_scores`` is ONE query's best
    score against the decoy library. Queries without a target hit do not
    appear in ``target_scores``; queries without a decoy hit do not appear
    in ``decoy_scores``. Multiple hits of the same query therefore never
    enter the competition more than once.

    Estimate: for a score threshold ``t``,

        FDR(t) = (1 + #{decoy scores >= t}) / #{target scores >= t}

    clipped to [0, 1] (conservative +1 pseudo-count prevents optimistic
    0.0 values, particularly for small libraries). The q-value of a target
    score ``s`` is the monotone closure min_{t <= s} FDR(t); it estimates
    the expected fraction of false positives among all accepted queries
    whose best target score is at least ``s``.

    TIES are handled conservatively: on equal scores, decoys are ranked
    BEFORE targets, so a tied decoy is always counted against the target.

    HETEROGENEOUS ENGINES: because each query's target hit and decoy hit
    are scored by the same engine (classical, consensus, cascade, or the
    router-assigned engine for that query), both the numerator and the
    denominator are per-query counts on the same score scale. Pooling
    across engines at the query level is therefore valid: a false-positive
    query's best target score is exchangeable with its best decoy score
    under the null, regardless of which engine scored it.

    Parameters
    ----------
    target_scores : np.ndarray
        Per-query best target scores. Shape: (N,), dtype: float.
    decoy_scores : np.ndarray
        Per-query best decoy scores. Shape: (M,), dtype: float.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - sorted_scores: Combined per-query target and decoy scores sorted
          in descending order (decoys first on ties). Shape: (N+M,).
        - q_values: q-values corresponding to each sorted score. Shape: (N+M,).
        - is_target: Boolean mask; True for target scores, False for decoys.
    """
    if len(target_scores) == 0 and len(decoy_scores) == 0:
        return np.array([]), np.array([]), np.array([], dtype=bool)

    if len(decoy_scores) == 0:
        # No null evidence: no calibration is possible. q = 1/cum_targets
        # (monotone-closed) is the conservative bound that every accepted
        # query shares when the null is empty.
        sort_idx = np.argsort(target_scores)[::-1]
        sorted_scores = target_scores[sort_idx]
        cum_targets = np.arange(1, len(sorted_scores) + 1)
        fdr = np.minimum(1.0 / cum_targets, 1.0)
        q_values = np.minimum.accumulate(fdr[::-1])[::-1]
        is_target = np.ones_like(sorted_scores, dtype=bool)
        return sorted_scores, q_values, is_target

    if len(target_scores) == 0:
        sort_idx = np.argsort(decoy_scores)[::-1]
        sorted_scores = decoy_scores[sort_idx]
        q_values = np.ones_like(sorted_scores, dtype=float)
        is_target = np.zeros_like(sorted_scores, dtype=bool)
        return sorted_scores, q_values, is_target

    scores = np.concatenate([target_scores, decoy_scores])
    is_target = np.concatenate(
        [
            np.ones(len(target_scores), dtype=bool),
            np.zeros(len(decoy_scores), dtype=bool),
        ]
    )

    # Sort descending by score; on ties, decoys (False) rank before targets
    # (True) so a tied decoy is always counted against the target.
    # lexsort uses the LAST key as the primary key: -scores descending,
    # then is_target ascending (decoys first) within ties.
    order = np.lexsort((is_target, -scores))
    sorted_scores = scores[order]
    sorted_is_target = is_target[order]

    cum_targets = np.cumsum(sorted_is_target)
    cum_decoys = np.cumsum(~sorted_is_target)

    # Conservative pseudo-count formula: FDR = (cum_decoys + 1) / cum_targets.
    # Leading decoy ranks have cum_targets == 0; their FDR is defined as 1.0.
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr_raw = (cum_decoys + 1.0) / cum_targets
    fdr = np.where(cum_targets > 0, fdr_raw, 1.0)
    fdr = np.minimum(fdr, 1.0)

    # q-values: minimum FDR over all lower-scoring ranks (monotone closure).
    q_values = np.minimum.accumulate(fdr[::-1])[::-1]

    return sorted_scores, q_values, sorted_is_target


def calibrate_query_level_fdr(
    results: Sequence[SearchResult],
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Calibrate a flat list of search results with per-query TDC.

    The competition unit is the query spectrum (see :func:`calculate_fdr`):
    each query contributes its best target hit and its best decoy hit
    exactly once, regardless of how many hits it produced. Hits of the same
    query are therefore calibrated together, and duplicate scores across
    queries map to the same q-value.

    Parameters
    ----------
    results : list of dict
        ``SearchResult`` dicts (with ``query_id``, ``score``, ``is_decoy``)
        as returned by any similarity engine.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], dict[str, int]]
        - ``q_by_query``: q-value per query id (1.0 when the query has no
          target hit and therefore no calibratable annotation).
        - ``p_by_query``: empirical p-value per query id (diagnostic; see
          :func:`calculate_empirical_p_values`).
        - ``summary``: counts for provenance reporting:
          ``n_competing_queries``, ``n_target_competitions``,
          ``n_decoy_competitions``.
    """
    best_target: dict[str, float] = {}
    best_decoy: dict[str, float] = {}
    for res in results:
        query_id = res.get("query_id")
        if query_id is None:
            continue
        score = float(res["score"])
        if res.get("is_decoy", False):
            if score > best_decoy.get(query_id, -np.inf):
                best_decoy[query_id] = score
        else:
            if score > best_target.get(query_id, -np.inf):
                best_target[query_id] = score

    query_ids = sorted(set(best_target) | set(best_decoy))
    target_scores = np.array(
        [best_target.get(q, -np.inf) for q in query_ids], dtype=np.float64
    )
    decoy_scores = np.array(
        [best_decoy.get(q, -np.inf) for q in query_ids], dtype=np.float64
    )

    # Only finite best scores enter the calibration: a query without a
    # target hit cannot be a discovery, and a query without a decoy hit
    # contributes nothing to the null.
    finite_targets = target_scores[np.isfinite(target_scores)]
    finite_decoys = decoy_scores[np.isfinite(decoy_scores)]

    q_by_query: dict[str, float] = {}
    p_by_query: dict[str, float] = {}

    if finite_targets.size > 0:
        sorted_scores, q_values, _ = calculate_fdr(finite_targets, finite_decoys)
        p_values = calculate_empirical_p_values(finite_targets, finite_decoys)

        ascending_scores = sorted_scores[::-1]
        ascending_q = q_values[::-1]

        # Conservative per-score lookup: identical scores share the q-value
        # of their LAST (lowest-ranked) occurrence, which is the largest
        # q-value within the tie block.
        q_by_score: dict[float, float] = {}
        for score in np.unique(finite_targets):
            index = int(np.searchsorted(ascending_scores, score, side="right")) - 1
            if index < 0:
                q_by_score[float(score)] = 1.0
            else:
                q_by_score[float(score)] = float(ascending_q[index])

        # p is a function of the score only; identical scores share it.
        p_by_score: dict[float, float] = {}
        for score, p_value in zip(finite_targets, p_values):
            p_by_score[float(score)] = float(p_value)

        for query_id, best in best_target.items():
            if np.isfinite(best):
                q_by_query[query_id] = q_by_score[float(best)]
                p_by_query[query_id] = p_by_score[float(best)]
            else:
                q_by_query[query_id] = 1.0
                p_by_query[query_id] = 1.0

    # Queries with no target hit cannot be annotated: uncalibrated.
    for query_id in query_ids:
        q_by_query.setdefault(query_id, 1.0)
        p_by_query.setdefault(query_id, 1.0)

    summary = {
        "n_competing_queries": len(query_ids),
        "n_target_competitions": int(finite_targets.size),
        "n_decoy_competitions": int(finite_decoys.size),
    }
    return q_by_query, p_by_query, summary


class SimilarityEngine:
    """Similarity search engine for classical spectral matching algorithms.

    Provides a unified interface to cosine and modified cosine similarity
    scoring backed by matchms. Handles MS1 pre-filtering, decoy generation,
    score calculation, and result extraction.
    """

    def __init__(self, config: SimilarityConfig):
        """Initialize the similarity search engine.

        Configures the underlying matchms similarity function based on the
        provided configuration. Supports 'cosine' and 'modified_cosine'.

        Parameters
        ----------
        config : SimilarityConfig
            The configuration object detailing the algorithm choice and
            tolerances.

        Raises
        ------
        ValueError
            If the configured algorithm is not supported.
        """
        self.config = config

        if self.config.algorithm == "cosine":
            self.similarity_function = CosineGreedy(tolerance=self.config.ms2_tolerance)
        elif self.config.algorithm == "modified_cosine":
            self.similarity_function = ModifiedCosine(
                tolerance=self.config.ms2_tolerance
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

    @_handle_lazy_reference_spectra
    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
        decoy_min_relative_intensity: float | None = None,
        decoy_mz_shift_da: float | None = None,
    ) -> List[SearchResult]:
        """Run a similarity search of query spectra against a reference library.

        This method computes a full NxM similarity matrix between the reference
        spectra and the query spectra using the configured algorithm. It leverages
        vectorized numpy operations to efficiently filter the results based on
        minimum score and minimum matched peaks. It then extracts the top N
        matches for each query and compiles the metadata into a structured list.

        Parameters
        ----------
        query_spectra : List[matchms.Spectrum]
            A list of experimental spectrum objects to be annotated.
        reference_spectra : List[matchms.Spectrum]
            A list of reference spectrum objects forming the library.
        min_score : float or None, optional
            An optional override for the minimum score threshold. If None, the
            value from the initial configuration is used.
        top_n : int or None, optional
            An optional override for the maximum number of results to return per
            query spectrum. If None, all matches exceeding the thresholds are
            returned.
        include_decoys : bool, optional
            If True, generate and search against decoy spectra for FDR calculation.
            If False, search only against the provided reference_spectra. Default
            is True.
        ref_precursor_mzs : np.ndarray or None, optional
            Pre-computed flat ``float64`` array of reference precursor m/z values,
            aligned with ``reference_spectra``. When provided, the MS1 pre-filter
            uses this array directly instead of extracting precursor_mz from
            Spectrum objects, eliminating per-call pyteomics/object overhead.
        ref_is_decoy : np.ndarray or None, optional
            Pre-computed flat ``bool`` array indicating which references are decoys.
            When provided alongside ``ref_precursor_mzs``, the array length must
            match ``len(reference_spectra)``.
        decoy_min_relative_intensity : float or None, optional
            Baseline noise floor (fraction of the base peak) used when this
            call generates decoys (``include_decoys=True``). When None, the
            module default (1% of base peak) is used. Keeps decoys identical
            to those generated by the parent workflow's config-driven call.
        decoy_mz_shift_da : float or None, optional
            Per-peak m/z jitter (Da) used when this call generates decoys.
            When None, the module default (1.0 Da) is used.

        Returns
        -------
        List[SearchResult]
            A list of dictionaries, where each dictionary represents a successful
            match and contains relevant metadata (e.g., query ID, reference name,
            score, SMILES).
        """
        if not query_spectra or not reference_spectra:
            return []

        cutoff = min_score if min_score is not None else self.config.min_score

        if include_decoys:
            ref_list = list(reference_spectra)
            decoy_spectra = generate_decoys(
                ref_list,
                min_relative_intensity=(
                    decoy_min_relative_intensity
                    if decoy_min_relative_intensity is not None
                    else _DEFAULT_DECOY_MIN_RELATIVE_INTENSITY
                ),
                mz_shift_da=(
                    decoy_mz_shift_da
                    if decoy_mz_shift_da is not None
                    else _DEFAULT_DECOY_MZ_SHIFT_DA
                ),
            )
            all_references = ref_list + decoy_spectra
            n_targets = len(ref_list)
        else:
            all_references = list(reference_spectra)
            n_targets = len(all_references)

        n_queries = len(query_spectra)

        # ------------------------------------------------------------------
        # Candidate generation & scoring
        # ------------------------------------------------------------------
        # Three scoring paths share one output contract (a dense structured
        # array of shape (n_refs, n_queries) with 'score' and 'matches'
        # columns):
        #
        # 1. Numba peak/neutral-loss prefilter (modified_cosine): skips
        #    pairs that cannot reach min_matched_peaks, then scores only the
        #    surviving pairs via matchms sparse_array.
        # 2. MS1 precursor prefilter (cosine with sparse_array): the
        #    existing v0.1 stable path.
        # 3. Full matrix scoring: exact fallback for everything else.
        #
        # All three produce identical results for the pairs they consider.
        peak_prefilter_applied = False
        if (
            self.config.algorithm == "modified_cosine"
            and self.config.enable_numba_prefilter
            and self.config.min_matched_peaks > 0
            and hasattr(self.similarity_function, "sparse_array")
        ):
            from MassFlow.acceleration import _HAS_NUMBA, prefilter_candidate_pairs

            if _HAS_NUMBA:
                idx_row, idx_col = prefilter_candidate_pairs(
                    all_references,
                    query_spectra,
                    tolerance=self.config.ms2_tolerance,
                    min_matches=self.config.min_matched_peaks,
                    algorithm="modified_cosine",
                )
                if len(idx_row) > 0:
                    sparse_results = self.similarity_function.sparse_array(
                        all_references,
                        query_spectra,
                        idx_row,
                        idx_col,
                        is_symmetric=False,
                        progress_bar=False,
                    )
                    scores_array = np.zeros(
                        (len(all_references), n_queries),
                        dtype=self.similarity_function.score_datatype,
                    )
                    scores_array[idx_row, idx_col] = sparse_results
                else:
                    scores_array = np.zeros(
                        (len(all_references), n_queries),
                        dtype=self.similarity_function.score_datatype,
                    )
                peak_prefilter_applied = True
                logger.debug(
                    "Numba peak prefilter: %d/%d pairs survived for scoring "
                    "(min_matched_peaks=%d, tolerance=%.4f).",
                    len(idx_row),
                    len(all_references) * n_queries,
                    self.config.min_matched_peaks,
                    self.config.ms2_tolerance,
                )

        # MS1 Pre-filtering for cosine with sparse array support.
        # When ref_precursor_mzs is provided (L2 cache), use it directly to
        # avoid Spectrum-object property lookups in the hot path.
        if (
            not peak_prefilter_applied
            and self.config.algorithm == "cosine"
            and hasattr(self.similarity_function, "sparse_array")
        ):
            ms1_tol = getattr(self.config, "ms1_tolerance", 0.02)
            res_ppm = getattr(self.config, "resolution_ppm", None)

            if ref_precursor_mzs is not None:
                # Optimized path: use pre-computed numpy arrays
                idx_row, idx_col = _ms1_prefilter_arrays(
                    ref_precursor_mzs, query_spectra, ms1_tol, res_ppm
                )
            else:
                idx_row, idx_col = _ms1_prefilter(
                    all_references, query_spectra, ms1_tol, res_ppm
                )

            if len(idx_row) > 0:
                sparse_results = self.similarity_function.sparse_array(
                    all_references,
                    query_spectra,
                    idx_row,
                    idx_col,
                    is_symmetric=False,
                    progress_bar=False,
                )
                scores_array = np.zeros(
                    (len(all_references), n_queries),
                    dtype=self.similarity_function.score_datatype,
                )
                scores_array[idx_row, idx_col] = sparse_results
            else:
                scores_array = np.zeros(
                    (len(all_references), n_queries),
                    dtype=self.similarity_function.score_datatype,
                )

        elif not peak_prefilter_applied:
            # Calculate scores natively for modified cosine (or cosine without sparse_array).
            # Log an informational message if the user explicitly configured PPM-based
            # MS1 pre-filtering, which is not applied during modified cosine scoring.
            # (ms1_tolerance defaults to 0.02, so only warn on the opt-in resolution_ppm.)
            if self.config.algorithm == "modified_cosine" and (
                getattr(self.config, "resolution_ppm", None) is not None
            ):
                logger.info(
                    "resolution_ppm is configured but is not applied during "
                    "modified_cosine scoring. Modified cosine uses the precursor "
                    "mass difference to align fragments, so an MS1 window is not a "
                    "strict gate. To pre-filter by precursor mass before scoring, "
                    "use 'cosine'."
                )
            try:
                scores_obj = calculate_scores(
                    references=all_references,  # type: ignore
                    queries=query_spectra,  # type: ignore
                    similarity_function=self.similarity_function,
                    is_symmetric=False,
                    array_type="sparse",
                )
            except Exception as e:
                logger.error(
                    f"Vectorized similarity calculation failed: {e}",
                    extra={
                        "step": "calculate_scores",
                        "num_queries": len(query_spectra),
                        "num_references": len(all_references),
                        "algorithm": self.config.algorithm,
                    },
                    exc_info=True,
                )
                raise

            # Safely extract numpy array (handles matchms >= 0.24 StackedSparseArray)
            scores_data = scores_obj.scores
            if hasattr(scores_data, "to_array"):
                scores_array = scores_data.to_array()
            else:
                scores_array = np.asarray(scores_data)

        results: List[SearchResult] = []

        # Extract score and matched-peaks columns from the structured array
        assert scores_array is not None, "scores_array should not be None"
        assert scores_array.dtype.names is not None, (
            "Expected structured array with named fields"
        )
        score_cols = [c for c in scores_array.dtype.names if "score" in c.lower()]
        match_cols = [c for c in scores_array.dtype.names if "matches" in c.lower()]

        if not score_cols:
            raise ValueError(
                f"Could not resolve 'score' column in array fields: {scores_array.dtype.names}"
            )
        if not match_cols:
            raise ValueError(
                f"Could not resolve 'matches' column in array fields: {scores_array.dtype.names}"
            )

        numeric_scores = scores_array[score_cols[0]].astype(float)
        matches_count = scores_array[match_cols[0]]

        # Iterate per query to apply top_n filtering
        # Note: Filtering is vectorized per query column
        for i in range(n_queries):
            query_scores = numeric_scores[:, i]
            query_matches = matches_count[:, i]

            # 1. Score threshold mask
            mask = query_scores >= cutoff

            # 2. Match count threshold mask (if applicable)
            if self.config.min_matched_peaks > 0:
                mask &= query_matches >= self.config.min_matched_peaks

            valid_indices = np.where(mask)[0]

            if valid_indices.size == 0:
                continue

            valid_scores = query_scores[valid_indices]

            # Sort descending by score
            sorted_idx_rel = np.argsort(valid_scores)[::-1]

            # Apply Top N
            if top_n is not None:
                sorted_idx_rel = sorted_idx_rel[:top_n]

            final_indices = valid_indices[sorted_idx_rel]

            # Extract Metadata
            q = query_spectra[i]
            q_id = str(q.get("id", f"query_{i}"))
            q_mz = q.get("precursor_mz")
            q_mz_val = float(q_mz) if q_mz is not None else None
            q_adduct: str | None = q.get("adduct")  # type: ignore[assignment]

            for idx in final_indices:
                ref = all_references[idx]

                # --- Adduct-mode compatibility gate ---
                ref_adduct: str | None = ref.get("adduct")  # type: ignore[assignment]
                if not _adduct_modes_compatible(ref_adduct, q_adduct):
                    continue

                # --- Retention time filtering ---
                if self.config.rt_tolerance is not None:
                    q_rt = q.get("retention_time")
                    ref_rt = ref.get("retention_time")
                    if not _is_missing(q_rt) and not _is_missing(ref_rt):
                        if abs(float(q_rt) - float(ref_rt)) > self.config.rt_tolerance:
                            continue

                score_val = float(numeric_scores[idx, i])
                match_val = int(matches_count[idx, i])

                ref_mz = ref.get("precursor_mz")
                ref_mz_val = float(ref_mz) if ref_mz is not None else None

                is_decoy = bool(ref.get("is_decoy", False)) or (idx >= n_targets)
                if ref_is_decoy is not None and idx < len(ref_is_decoy):
                    is_decoy = bool(ref_is_decoy[idx])

                results.append(
                    {
                        "query_id": q_id,
                        "query_precursor_mz": q_mz_val,
                        "reference_id": str(ref.get("id")),
                        "reference_name": str(
                            ref.get("compound_name") or ref.get("name")
                        ),
                        "reference_precursor_mz": ref_mz_val,
                        "score": score_val,
                        "matched_peaks": match_val,
                        "smiles": str(ref.get("smiles")) if ref.get("smiles") else None,
                        "inchikey": str(ref.get("inchikey"))
                        if ref.get("inchikey")
                        else None,
                        "is_decoy": is_decoy,
                        "q_value": 1.0,  # Will be updated by FDR calculation
                        "p_value": None,  # Will be updated by empirical p-value calculation
                        "annotation_tier": None,
                        "structural_similarity": None,
                        "mass_error_ppm": None,
                        "score_breakdown": None,
                    }
                )

        return results

    def batch_score(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
    ) -> np.ndarray:
        """Score pre-formed query--reference pairs in batch.

        Classical engines have no dedicated batch primitive, so each pair
        is scored individually via ``search()``.  This method exists to
        satisfy the uniform interface expected by meta-engines
        (consensus, cascade).
        """
        if len(query_spectra) != len(reference_spectra):
            raise ValueError(
                f"query_spectra and reference_spectra must have the same length; "
                f"got {len(query_spectra)} vs {len(reference_spectra)}"
            )
        scores = np.full(len(query_spectra), np.nan, dtype=np.float64)
        for i, (q, r) in enumerate(zip(query_spectra, reference_spectra)):
            results = self.search(
                query_spectra=[q],
                reference_spectra=[r],
                min_score=-1.0,
                top_n=1,
                include_decoys=False,
            )
            if results:
                scores[i] = float(results[0]["score"])
        return scores

    def load_model(self, model_path: str | Path) -> None:
        """No-op: classical engines have no trainable model weights.

        Exists for interface uniformity with ``MLEngineProtocol`` so that
        meta-engines (consensus, cascade) can call ``load_model`` on any
        sub-engine without type-checking.
        """
        logger.debug(
            "Classical engine '%s' does not use external model weights; "
            "ignoring load_model('%s').",
            self.config.algorithm,
            model_path,
        )


# =============================================================================
# Machine-learning similarity engines (require the ``[ml]`` extra)
# =============================================================================
#
# These engines leverage PyTorch, Gensim, Spec2Vec, and MS2DeepScore to provide
# deep-learning and vector-based spectral similarity scoring. They all expose a
# ``.search()`` method that returns the same ``List[SearchResult]`` type as the
# classical ``SimilarityEngine``, so downstream pipelines interact with them
# identically whether the ml extra is installed or not.
#
# When called without the required libraries, each engine raises a clear
# ``RuntimeError`` directing the user to install ``massflow[ml]``.


def _sub_engine_config(
    parent: SimilarityConfig,
    algorithm: str,
    **overrides: Any,
) -> SimilarityConfig:
    """Build a sub-engine config inheriting tolerances and remote ML settings.

    Meta-engines (consensus, cascade, router) construct child configs for
    their sub-engines.  This helper forwards the tolerances, thresholds, and
    — critically for the massflow-ml satellite boundary — the remote
    ``ml_endpoints`` and circuit-breaker settings, so a remote Spec2Vec/
    MS2DeepScore endpoint configured at the top level is also used inside
    meta-engines.

    Parameters
    ----------
    parent : SimilarityConfig
        The top-level configuration to inherit from.
    algorithm : str
        Algorithm name for the sub-engine.
    **overrides
        Fields to override on the sub-config.

    Returns
    -------
    SimilarityConfig
        Sub-engine configuration.
    """
    fields: dict[str, Any] = dict(
        algorithm=algorithm,
        ms1_tolerance=parent.ms1_tolerance,
        ms2_tolerance=parent.ms2_tolerance,
        resolution_ppm=parent.resolution_ppm,
        min_score=parent.min_score,
        min_matched_peaks=parent.min_matched_peaks,
        rt_tolerance=parent.rt_tolerance,
        ml_endpoints=parent.ml_endpoints,
        ml_request_timeout_seconds=parent.ml_request_timeout_seconds,
        ml_circuit_breaker_threshold=parent.ml_circuit_breaker_threshold,
        ml_circuit_breaker_cooldown_seconds=parent.ml_circuit_breaker_cooldown_seconds,
    )
    fields.update(overrides)
    return SimilarityConfig(**fields)


class _MLEngineBase(MLEngineProtocol):
    """Base class for ML-based similarity scoring engines.

    Implements the ``MLEngineProtocol`` contract so that any subclass can
    be registered as a ``massflow.similarity_engines`` entry point and
    discovered by ``get_similarity_engine()`` at runtime.

    Subclasses must implement ``_build_model()``, ``search()``,
    ``batch_score()``, and ``load_model()``.
    """

    def __init__(self, config: SimilarityConfig):
        self.config = config
        self._check_dependencies()
        self._build_model()

    def _check_dependencies(self) -> None:
        """Verify that required ML libraries are available."""
        if not _HAS_ML:
            raise RuntimeError(_ML_INSTALL_MSG)

    def _build_model(self) -> None:
        """Build or load the underlying ML model. Override in subclasses."""

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
        decoy_min_relative_intensity: float | None = None,
        decoy_mz_shift_da: float | None = None,
    ) -> List[SearchResult]:
        """Run similarity search. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement search()")

    def batch_score(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
    ) -> np.ndarray:
        """Score pre-formed query--reference pairs in batch.

        Default implementation delegates to ``search()`` with a
        single-query / single-reference call per pair.  Subclasses that
        can perform true vectorised batch scoring SHOULD override this.
        """
        if len(query_spectra) != len(reference_spectra):
            raise ValueError(
                f"query_spectra and reference_spectra must have the same length; "
                f"got {len(query_spectra)} vs {len(reference_spectra)}"
            )
        scores = np.full(len(query_spectra), np.nan, dtype=np.float64)
        for i, (q, r) in enumerate(zip(query_spectra, reference_spectra)):
            results = self.search(
                query_spectra=[q],
                reference_spectra=[r],
                min_score=-1.0,
                top_n=1,
                include_decoys=False,
            )
            if results:
                scores[i] = float(results[0]["score"])
        return scores

    def load_model(self, model_path: str | Path) -> None:
        """Load pre-trained model weights from disk.

        The default implementation stores the path; subclasses should
        override to perform actual model deserialization.
        """
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self._model_path}")


class Spec2VecEngine(_MLEngineBase):
    """Spec2Vec-based similarity scoring engine.

    Uses Gensim Word2Vec models trained on mass spectral peaks to compute
    spectrum-level embeddings and cosine similarity between embedded vectors.

    Requires: ``pip install massflow[ml]``
    """

    def _build_model(self) -> None:
        """Initialize the Spec2Vec model.

        The model is loaded lazily via spec2vec's ``SpectrumDocument`` and
        ``GensimToSpec2Vec`` adapters. Actual model weights are loaded on
        first use to keep import times low.
        """
        # Placeholder for Spec2Vec model initialization.
        # Model loading is deferred to the first search() call to allow the
        # factory to return quickly.  Real implementations should load the
        # pre-trained model file here or in a lazy property.
        self._model_loaded = False

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
        decoy_min_relative_intensity: float | None = None,
        decoy_mz_shift_da: float | None = None,
    ) -> List[SearchResult]:
        """Run Spec2Vec similarity search.

        Raises
        ------
        RuntimeError
            If the ``[ml]`` extra is not installed.
        """
        # This is a structured stub that produces the correct interface contract.
        # Real implementations compute Spec2Vec embeddings, perform a cosine
        # similarity search, apply min_score/top_n filtering, and return
        # ``SearchResult`` dicts.
        raise RuntimeError("Spec2Vec search is not yet implemented. " + _ML_INSTALL_MSG)

    def batch_score(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
    ) -> np.ndarray:
        """Score pre-formed query--reference pairs."""
        raise RuntimeError(
            "Spec2Vec batch scoring is not yet implemented. " + _ML_INSTALL_MSG
        )

    def load_model(self, model_path: str | Path) -> None:
        """Load a pre-trained Spec2Vec model."""
        raise RuntimeError(
            "Spec2Vec model loading is not yet implemented. " + _ML_INSTALL_MSG
        )


class MS2DeepScoreEngine(_MLEngineBase):
    """MS2DeepScore-based deep learning similarity scoring engine.

    Uses a Siamese neural network (PyTorch) trained on binned mass spectra to
    predict Tanimoto-like similarity scores directly from spectral pairs.

    Requires: ``pip install massflow[ml]``
    """

    def _build_model(self) -> None:
        """Initialize the MS2DeepScore model.

        The Siamese network weights are loaded from a pre-trained checkpoint.
        Model loading is deferred to the first ``search()`` call.
        """
        self._model_loaded = False

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
        decoy_min_relative_intensity: float | None = None,
        decoy_mz_shift_da: float | None = None,
    ) -> List[SearchResult]:
        """Run MS2DeepScore similarity search.

        Raises
        ------
        RuntimeError
            If the ``[ml]`` extra is not installed.
        """
        raise RuntimeError(
            "MS2DeepScore search is not yet implemented. " + _ML_INSTALL_MSG
        )

    def batch_score(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
    ) -> np.ndarray:
        """Score pre-formed query--reference pairs."""
        raise RuntimeError(
            "MS2DeepScore batch scoring is not yet implemented. " + _ML_INSTALL_MSG
        )

    def load_model(self, model_path: str | Path) -> None:
        """Load a pre-trained MS2DeepScore model."""
        raise RuntimeError(
            "MS2DeepScore model loading is not yet implemented. " + _ML_INSTALL_MSG
        )


class ConsensusEngine(_MLEngineBase):
    """Consensus scoring engine combining multiple similarity scores.

    Computes scores from multiple underlying engines (e.g., cosine, spec2vec,
    ms2deepscore) and produces a weighted consensus score. This is designed to
    improve annotation confidence by leveraging orthogonal scoring approaches.

    Requires: ``pip install massflow[ml]``
    """

    def _check_dependencies(self) -> None:
        """Override base check: log a warning instead of raising.

        Consensus can operate with only classical sub-engines (cosine,
        modified_cosine) even when ML extras are partially missing.
        Individual sub-engine availability is handled in ``_build_model``.
        """
        if not _HAS_ML:
            logger.warning(
                "ML extras are not fully available; consensus scoring "
                "will be limited to classical sub-engines. %s",
                _ML_INSTALL_MSG,
            )

    def _build_model(self) -> None:
        """Initialize the underlying scoring engines for the consensus.

        Creates sub-engines for each algorithm listed in
        ``config.consensus_weights``.  Classical engines (cosine,
        modified_cosine) are always available; ML engines (spec2vec,
        ms2deepscore) are only instantiated when their dependencies are
        present.  Engines that fail to build are silently skipped.
        """
        self._sub_engines: dict[
            str, SimilarityEngine | _MLEngineBase | MLEngineProtocol
        ] = {}
        self._sub_weights: dict[str, float] = {}

        weights = self.config.consensus_weights

        # Meta-algorithms cannot serve as sub-engines (prevents recursion).
        _META_ALGOS = {"consensus", "cascade"}

        for algo, weight in weights.items():
            if weight <= 0:
                continue
            if algo in _META_ALGOS:
                logger.debug("Skipping meta-algorithm '%s' as sub-engine.", algo)
                continue

            sub_cfg = _sub_engine_config(
                self.config,
                algo,
                min_score=0.0,
                min_matched_peaks=0,
            )

            try:
                engine = get_similarity_engine(sub_cfg)
                self._sub_engines[algo] = engine
                self._sub_weights[algo] = weight
            except Exception:
                logger.warning(
                    "Could not initialise sub-engine '%s' for consensus scoring.",
                    algo,
                    exc_info=True,
                )

        if not self._sub_engines:
            logger.warning(
                "No sub-engines could be initialised for consensus scoring. "
                "Searches will fall back to modified_cosine."
            )
        else:
            logger.info(
                "Consensus engine initialised with sub-engines: %s",
                list(self._sub_engines.keys()),
            )

        self._model_loaded = bool(self._sub_engines)
        # Per-search degradation record: reset at the start of every search
        # and populated whenever the engine silently downgrades what it
        # computed (sub-engine failures, total fallback). The workflow reads
        # this after search() to mark the file execution as degraded.
        self._degraded_flags: list[str] = []

    @property
    def degraded_mode_flags(self) -> list[str]:
        """Degradation flags recorded by the most recent ``search()`` call.

        Values: ``consensus_subengine_failed:<algo>`` for each sub-engine
        that raised during the last search, and
        ``consensus_all_subengines_failed`` when every sub-engine failed and
        the engine fell back to modified_cosine.
        """
        return list(self._degraded_flags)

    # ------------------------------------------------------------------
    # Classical fallback (circuit-breaker / missing-dependency safety net)
    # ------------------------------------------------------------------

    def _get_fallback_engine(self) -> SimilarityEngine:
        """Return the classical modified_cosine fallback engine.

        Used when every consensus sub-engine is unavailable (remote ML
        service unreachable with an open circuit breaker, or heavy local
        dependencies not installed).  The orchestrator applies empirical
        p-value scoring on top of these results at the workflow level.
        """
        if not hasattr(self, "_fallback_engine"):
            fallback_cfg = _sub_engine_config(
                self.config,
                "modified_cosine",
                min_score=0.0,
                min_matched_peaks=0,
            )
            self._fallback_engine = SimilarityEngine(fallback_cfg)
        return self._fallback_engine

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
        decoy_min_relative_intensity: float | None = None,
        decoy_mz_shift_da: float | None = None,
    ) -> List[SearchResult]:
        """Run consensus similarity search across configured sub-engines.

        Each sub-engine scores the query-reference pairs independently.
        Results are aggregated per ``(query_id, reference_id)`` and a
        weighted-average consensus score is computed.  Only pairs scored by
        at least ``consensus_min_engines`` sub-engines are retained.

        Parameters
        ----------
        query_spectra : List[matchms.Spectrum]
            Experimental spectra to annotate.
        reference_spectra : Iterable[matchms.Spectrum]
            Reference library spectra.
        min_score : float or None
            Override for the final consensus-score threshold.
        top_n : int or None
            Maximum hits per query after consensus aggregation.
        include_decoys : bool
            Whether sub-engines should generate and score decoys.
        ref_precursor_mzs : np.ndarray or None
            Pre-computed reference precursor m/z array (passed through).
        ref_is_decoy : np.ndarray or None
            Pre-computed reference decoy flags (passed through).
        decoy_min_relative_intensity : float or None, optional
            Decoy noise floor passed through to sub-engines that generate
            decoys (see :meth:`SimilarityEngine.search`).
        decoy_mz_shift_da : float or None, optional
            Decoy m/z jitter passed through to sub-engines that generate
            decoys.

        Returns
        -------
        List[SearchResult]
            Aggregated and weighted consensus results.
        """
        if not query_spectra or not reference_spectra:
            return []

        ref_list = list(reference_spectra)
        cutoff = min_score if min_score is not None else self.config.min_score
        min_engines = self.config.consensus_min_engines

        # Reset the per-search degradation record.
        self._degraded_flags = []

        if not self._sub_engines:
            logger.warning(
                "Consensus has no sub-engines; falling back to modified_cosine."
            )
            self._degraded_flags.append("consensus_all_subengines_failed")
            return self._get_fallback_engine().search(
                query_spectra=query_spectra,
                reference_spectra=ref_list,
                min_score=cutoff,
                top_n=top_n,
                include_decoys=include_decoys,
                ref_precursor_mzs=ref_precursor_mzs,
                ref_is_decoy=ref_is_decoy,
                decoy_min_relative_intensity=decoy_min_relative_intensity,
                decoy_mz_shift_da=decoy_mz_shift_da,
            )

        # ------------------------------------------------------------------
        # Phase 1: run every available sub-engine and collect raw results
        # ------------------------------------------------------------------
        # engine_results: list of (algo, weight, list of SearchResult)
        engine_results: list[tuple[str, float, list[SearchResult]]] = []
        failed_algos: list[str] = []

        for algo, engine in self._sub_engines.items():
            weight = self._sub_weights[algo]
            try:
                raw = engine.search(
                    query_spectra=query_spectra,
                    reference_spectra=ref_list,
                    min_score=0.0,  # collect all hits; filter later
                    top_n=None,
                    include_decoys=include_decoys,
                    ref_precursor_mzs=ref_precursor_mzs,
                    ref_is_decoy=ref_is_decoy,
                    decoy_min_relative_intensity=decoy_min_relative_intensity,
                    decoy_mz_shift_da=decoy_mz_shift_da,
                )
                engine_results.append((algo, weight, raw))
                logger.debug("Sub-engine '%s' returned %d raw results.", algo, len(raw))
            except Exception as exc:
                failed_algos.append(algo)
                self._degraded_flags.append(f"consensus_subengine_failed:{algo}")
                logger.warning(
                    "Sub-engine '%s' failed during search, skipping: %s",
                    algo,
                    exc,
                )

        if not engine_results:
            if failed_algos:
                logger.warning(
                    "All consensus sub-engines failed (%s); falling back to "
                    "modified_cosine scoring.",
                    ", ".join(failed_algos),
                )
                self._degraded_flags.append("consensus_all_subengines_failed")
                return self._get_fallback_engine().search(
                    query_spectra=query_spectra,
                    reference_spectra=ref_list,
                    min_score=cutoff,
                    top_n=top_n,
                    include_decoys=include_decoys,
                    ref_precursor_mzs=ref_precursor_mzs,
                    ref_is_decoy=ref_is_decoy,
                    decoy_min_relative_intensity=decoy_min_relative_intensity,
                    decoy_mz_shift_da=decoy_mz_shift_da,
                )
            return []

        # ------------------------------------------------------------------
        # Phase 2: aggregate by (query_id, reference_id)
        # ------------------------------------------------------------------
        # bucket[key] = {"scores": {algo: score}, "template": SearchResult}
        buckets: dict[tuple[str, str], dict] = {}

        for algo, _weight, results in engine_results:
            for res in results:
                key = (res["query_id"], res["reference_id"])
                if key not in buckets:
                    buckets[key] = {"scores": {}, "template": res}
                # Reference libraries can contain duplicate (or missing) ids;
                # keep the best per-algorithm score for the bucket rather
                # than letting later results overwrite earlier ones.
                buckets[key]["scores"].setdefault(algo, []).append(res["score"])

        # ------------------------------------------------------------------
        # Phase 3: compute weighted consensus scores
        # ------------------------------------------------------------------
        aggregated: list[SearchResult] = []

        for (_qid, _rid), bucket in buckets.items():
            scores_by_algo: dict[str, list[float]] = bucket["scores"]

            if len(scores_by_algo) < min_engines:
                continue

            total_weight = 0.0
            weighted_sum = 0.0
            for algo, scores in scores_by_algo.items():
                w = self._sub_weights.get(algo, 0.0)
                total_weight += w
                weighted_sum += max(scores) * w

            if total_weight == 0:
                continue

            consensus_score = weighted_sum / total_weight

            if consensus_score < cutoff:
                continue

            template = bucket["template"]
            best_individual = max(max(scores) for scores in scores_by_algo.values())

            aggregated.append(
                SearchResult(
                    query_id=template["query_id"],
                    query_precursor_mz=template["query_precursor_mz"],
                    reference_id=template["reference_id"],
                    reference_name=template["reference_name"],
                    reference_precursor_mz=template["reference_precursor_mz"],
                    score=float(consensus_score),
                    matched_peaks=template["matched_peaks"],
                    smiles=template["smiles"],
                    inchikey=template["inchikey"],
                    is_decoy=template["is_decoy"],
                    q_value=template["q_value"],
                    p_value=template["p_value"],
                    annotation_tier=template["annotation_tier"],
                    structural_similarity=float(best_individual),
                    mass_error_ppm=template["mass_error_ppm"],
                    score_breakdown={
                        algo: max(scores) for algo, scores in scores_by_algo.items()
                    },
                )
            )

        # ------------------------------------------------------------------
        # Phase 4: apply top_n per query (sort by consensus score desc,
        #           then by best individual score desc as tie-break)
        # ------------------------------------------------------------------
        if top_n is not None:
            by_query: dict[str, list[SearchResult]] = defaultdict(list)
            for res in aggregated:
                by_query[res["query_id"]].append(res)

            aggregated = []
            for _qid, hits in by_query.items():
                hits.sort(
                    key=lambda r: (
                        -r["score"],
                        -(r.get("structural_similarity") or 0),
                    )
                )
                aggregated.extend(hits[:top_n])

        return aggregated

    def batch_score(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
    ) -> np.ndarray:
        """Score pre-formed query--reference pairs via consensus.

        Delegates to each sub-engine and returns a weighted-consensus
        score per pair.
        """
        return super().batch_score(query_spectra, reference_spectra)

    def load_model(self, model_path: str | Path) -> None:
        """Load model weights for all sub-engines.

        Calls ``load_model`` on each sub-engine in turn.  Failures are
        logged but do not prevent other sub-engines from loading.
        """
        for algo, engine in self._sub_engines.items():
            try:
                engine.load_model(model_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load model for sub-engine '%s': %s", algo, exc
                )


class CascadeEngine(_MLEngineBase):
    """Cascaded scoring engine with hierarchical filtering.

    Applies multiple similarity filters in sequence, from fast/coarse to
    slow/precise, to efficiently narrow the candidate set before running
    expensive deep learning models.

    Requires: ``pip install massflow[ml]``
    """

    def _check_dependencies(self) -> None:
        """Override base check: log a warning instead of raising.

        Cascade can operate with only classical stages (cosine,
        modified_cosine) even when ML extras are partially missing.
        Individual stage availability is handled in ``_build_model``.
        """
        if not _HAS_ML:
            logger.warning(
                "ML extras are not fully available; cascade scoring "
                "will be limited to classical stages. %s",
                _ML_INSTALL_MSG,
            )

    def _build_model(self) -> None:
        """Initialize the cascade stages in order.

        Each stage in ``config.cascade_stages`` is instantiated as a
        sub-engine.  Classical engines (cosine, modified_cosine) are always
        available; ML engines require their respective dependencies.
        Stages that fail to build are skipped.
        """
        self._stages: list[
            tuple[str, SimilarityEngine | _MLEngineBase | MLEngineProtocol]
        ] = []

        for algo in self.config.cascade_stages:
            sub_cfg = _sub_engine_config(
                self.config,
                algo,
                min_score=0.0,
                min_matched_peaks=0,
            )

            try:
                engine = get_similarity_engine(sub_cfg)
                self._stages.append((algo, engine))
            except Exception:
                logger.warning(
                    "Could not initialise cascade stage '%s'.",
                    algo,
                    exc_info=True,
                )

        if not self._stages:
            logger.warning(
                "No cascade stages could be initialised. "
                "Searches will fall back to modified_cosine."
            )
        else:
            logger.info(
                "Cascade engine initialised with stages: %s",
                [a for a, _ in self._stages],
            )

        self._model_loaded = bool(self._stages)

        # HNSW state (built lazily at search time and cached per reference set).
        self._hnsw_index: Optional[HNSWSpectralIndex] = None
        self._hnsw_index_ref_ids: tuple[str, ...] = ()

        # Per-search degradation record (see ``degraded_mode_flags``).
        self._degraded_flags: list[str] = []

    @property
    def degraded_mode_flags(self) -> list[str]:
        """Degradation flags recorded by the most recent ``search()`` call.

        Values: ``cascade_stage_failed:<algo>`` per stage that raised,
        ``cascade_hnsw_failed`` when HNSW candidate retrieval failed, and
        ``cascade_fallback`` when the engine fell back to classical
        modified_cosine scoring.
        """
        return list(self._degraded_flags)

    # ------------------------------------------------------------------
    # Classical fallback (circuit-breaker / missing-dependency safety net)
    # ------------------------------------------------------------------

    def _get_fallback_engine(self) -> SimilarityEngine:
        """Return the classical modified_cosine fallback engine.

        Used when no cascade stage can run (remote ML endpoints unreachable
        with open circuit breakers, or heavy local dependencies missing).
        The orchestrator applies empirical p-value scoring on top of these
        results at the workflow level.
        """
        if not hasattr(self, "_fallback_engine"):
            fallback_cfg = _sub_engine_config(
                self.config,
                "modified_cosine",
                min_score=0.0,
                min_matched_peaks=0,
            )
            self._fallback_engine = SimilarityEngine(fallback_cfg)
        return self._fallback_engine

    def _run_classical_fallback(
        self,
        query_spectra: List[Spectrum],
        ref_list: List[Spectrum],
        threshold: float,
        top_n: Optional[int],
        include_decoys: bool,
        ref_precursor_mzs: Optional[np.ndarray],
        ref_is_decoy: Optional[np.ndarray],
        decoy_min_relative_intensity: Optional[float],
        decoy_mz_shift_da: Optional[float],
    ) -> List[SearchResult]:
        """Run the classical fallback engine with cascade-style thresholds."""
        logger.warning(
            "Cascade unavailable; falling back to modified_cosine scoring "
            "(threshold=%.3f).",
            threshold,
        )
        self._degraded_flags.append("cascade_fallback")
        # Decoys (when requested) are already part of ``ref_list``: they were
        # appended before the cascade stages, so the fallback must not
        # generate a second decoy set.
        results = self._get_fallback_engine().search(
            query_spectra=query_spectra,
            reference_spectra=ref_list,
            min_score=threshold,
            top_n=top_n,
            include_decoys=False,
            ref_precursor_mzs=ref_precursor_mzs,
            ref_is_decoy=ref_is_decoy,
            decoy_min_relative_intensity=decoy_min_relative_intensity,
            decoy_mz_shift_da=decoy_mz_shift_da,
        )
        return results

    # ------------------------------------------------------------------
    # HNSW candidate retrieval (approximate, sub-linear)
    # ------------------------------------------------------------------

    def _get_hnsw_index(self, ref_list: List[Spectrum]) -> Optional[HNSWSpectralIndex]:
        """Return a cached HNSW index over *ref_list*, rebuilding when needed.

        The index is cached keyed by the reference id sequence, so repeated
        searches against the same library (e.g. chunked query processing)
        amortize the construction cost. Returns ``None`` when hnswlib is
        unavailable, in which case callers fall back to exact scoring.

        Parameters
        ----------
        ref_list : list of Spectrum
            Reference library to index.

        Returns
        -------
        HNSWSpectralIndex or None
            The (possibly freshly built) index, or ``None`` when hnswlib is
            not installed.
        """
        from MassFlow import hnsw as hnsw_module

        if not hnsw_module._HAS_HNSWLIB:
            logger.warning(
                "hnsw_enabled is True but hnswlib is not installed; "
                "falling back to exact cascade scoring. %s",
                hnsw_module._HNSW_INSTALL_MSG,
            )
            return None

        ref_ids = tuple(str(ref.get("id")) for ref in ref_list)
        if self._hnsw_index is not None and self._hnsw_index_ref_ids == ref_ids:
            return self._hnsw_index

        index = hnsw_module.HNSWSpectralIndex.from_spectra(
            ref_list,
            bin_width=self.config.hnsw_bin_width,
            mz_min=self.config.hnsw_mz_min,
            mz_max=self.config.hnsw_mz_max,
            m=self.config.hnsw_m,
            ef_construction=self.config.hnsw_ef_construction,
            random_seed=self.config.hnsw_random_seed,
            max_elements=max(len(ref_list), self.config.hnsw_ef_search),
        )
        self._hnsw_index = index
        self._hnsw_index_ref_ids = ref_ids
        logger.info(
            "Built HNSW index over %d reference spectra "
            "(dim=%d, M=%d, ef_construction=%d).",
            len(ref_list),
            index.dim,
            index.m,
            index.ef_construction,
        )
        return index

    def _hnsw_candidate_ids(
        self,
        index: HNSWSpectralIndex,
        query_spectra: List[Spectrum],
        n_refs: int,
    ) -> set[str]:
        """Retrieve candidate reference ids for *query_spectra* from *index*.

        Retrieves ``hnsw_candidates_per_query`` neighbours per query spectrum
        using ``hnsw_ef_search``, then unions the per-query candidate lists.
        These candidates are approximate (the underlying similarity is
        non-metric), so exact scoring must follow.

        Parameters
        ----------
        index : HNSWSpectralIndex
            Populated index over the reference library.
        query_spectra : list of Spectrum
            Query spectra to vectorize and search.
        n_refs : int
            Total reference count (caps ``k``).

        Returns
        -------
        set[str]
            Union of reference ids retrieved for any query.
        """
        from MassFlow import hnsw as hnsw_module

        k = min(self.config.hnsw_candidates_per_query, n_refs)
        query_vectors = hnsw_module.bin_spectra(
            query_spectra,
            bin_width=self.config.hnsw_bin_width,
            mz_min=self.config.hnsw_mz_min,
            mz_max=self.config.hnsw_mz_max,
        )
        candidate_id_lists, _ = index.query(
            query_vectors,
            k=k,
            ef_search=self.config.hnsw_ef_search,
        )
        return {
            spectrum_id for per_query in candidate_id_lists for spectrum_id in per_query
        }

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
        decoy_min_relative_intensity: float | None = None,
        decoy_mz_shift_da: float | None = None,
    ) -> List[SearchResult]:
        """Run cascaded similarity search with sequential filtering.

        Each stage in the cascade scores the query-reference pairs with an
        increasingly strict threshold.  After each stage, only reference
        spectra that produced at least one hit above the stage threshold
        are passed to the next stage.  This narrows the candidate set
        progressively, reserving expensive models for the most promising
        candidates.

        When ``hnsw_enabled`` is True, an optional Phase 0 retrieves a small
        candidate set per query from a HNSW (Hierarchical Navigable Small
        World) graph built over binned reference spectra, giving sub-linear
        candidate generation for massive libraries. Because spectral
        similarity is non-metric, HNSW only *generates* candidates; the
        exact stages below always follow, and any HNSW failure falls back
        to exact scoring over the full reference set.

        **Threshold semantics**

        * ``cascade_lower_bound`` is used as the ``min_score`` for every
          stage *except the last*.
        * ``cascade_upper_bound`` (or the explicit ``min_score`` override)
          is used as ``min_score`` for the **last** stage only.

        Parameters
        ----------
        query_spectra : List[matchms.Spectrum]
            Experimental spectra to annotate.
        reference_spectra : Iterable[matchms.Spectrum]
            Reference library spectra.
        min_score : float or None
            Override for the final-stage (cascade_upper_bound) threshold.
        top_n : int or None
            Maximum hits per query in the final output.
        include_decoys : bool
            Whether sub-engines should generate and score decoys.
        ref_precursor_mzs : np.ndarray or None
            Pre-computed reference precursor m/z array (passed through).
        ref_is_decoy : np.ndarray or None
            Pre-computed reference decoy flags (passed through).
        decoy_min_relative_intensity : float or None, optional
            Decoy noise floor used when this call generates decoys.
        decoy_mz_shift_da : float or None, optional
            Decoy m/z jitter used when this call generates decoys.

        Returns
        -------
        List[SearchResult]
            Final-stage results after cascaded filtering.
        """
        if not query_spectra or not reference_spectra:
            return []

        ref_list = list(reference_spectra)
        lower_bound = self.config.cascade_lower_bound
        upper_bound = (
            min_score if min_score is not None else self.config.cascade_upper_bound
        )

        # Reset the per-search degradation record.
        self._degraded_flags = []

        # Decoys participate in the cascade exactly like targets: they are
        # generated ONCE here and then winnowed by the same stages. The
        # stages below always search with include_decoys=False so decoys are
        # never duplicated. (Without this, decoys would never be scored in
        # the single-file/streaming execution path and the FDR null would be
        # empty.)
        if include_decoys:
            generated_decoys = generate_decoys(
                ref_list,
                min_relative_intensity=(
                    decoy_min_relative_intensity
                    if decoy_min_relative_intensity is not None
                    else _DEFAULT_DECOY_MIN_RELATIVE_INTENSITY
                ),
                mz_shift_da=(
                    decoy_mz_shift_da
                    if decoy_mz_shift_da is not None
                    else _DEFAULT_DECOY_MZ_SHIFT_DA
                ),
            )
            ref_list = ref_list + generated_decoys
            if ref_precursor_mzs is not None:
                decoy_precursor_mzs = np.array(
                    [
                        float(decoy.get("precursor_mz"))
                        if decoy.get("precursor_mz") is not None
                        else 0.0
                        for decoy in generated_decoys
                    ],
                    dtype=np.float64,
                )
                ref_precursor_mzs = np.concatenate(
                    [ref_precursor_mzs, decoy_precursor_mzs]
                )
            if ref_is_decoy is not None:
                ref_is_decoy = np.concatenate(
                    [ref_is_decoy, np.ones(len(generated_decoys), dtype=bool)]
                )

        if not self._stages:
            return self._run_classical_fallback(
                query_spectra,
                ref_list,
                upper_bound,
                top_n,
                include_decoys,
                ref_precursor_mzs,
                ref_is_decoy,
                decoy_min_relative_intensity,
                decoy_mz_shift_da,
            )

        # ------------------------------------------------------------------
        # Phase 0 (optional): HNSW approximate candidate retrieval.
        #
        # The HNSW graph is built over binned reference spectra and used to
        # fetch a small candidate set per query in sub-linear time. Because
        # spectral similarity is non-metric this stage only *generates*
        # candidates — the exact cascade stages below always follow. On
        # failure (e.g. hnswlib missing) the cascade falls back to exact
        # scoring over the full reference set.
        # ------------------------------------------------------------------
        current_refs: list[Spectrum] = ref_list
        current_ref_precursor_mzs = ref_precursor_mzs
        current_ref_is_decoy = ref_is_decoy

        if self.config.hnsw_enabled:
            try:
                hnsw_index = self._get_hnsw_index(ref_list)
                if hnsw_index is not None:
                    candidate_ids = self._hnsw_candidate_ids(
                        hnsw_index, query_spectra, len(ref_list)
                    )
                    hnsw_filtered_refs: list[Spectrum] = []
                    hnsw_filtered_indices: list[int] = []
                    for ref_index, ref in enumerate(current_refs):
                        if str(ref.get("id")) in candidate_ids:
                            hnsw_filtered_refs.append(ref)
                            hnsw_filtered_indices.append(ref_index)

                    logger.debug(
                        "HNSW candidate retrieval kept %d/%d references "
                        "for exact cascade scoring.",
                        len(hnsw_filtered_refs),
                        len(current_refs),
                    )
                    if not hnsw_filtered_refs:
                        self._degraded_flags.append("cascade_hnsw_no_candidates")
                        return []

                    current_refs = hnsw_filtered_refs
                    if current_ref_precursor_mzs is not None:
                        current_ref_precursor_mzs = current_ref_precursor_mzs[
                            hnsw_filtered_indices
                        ]
                    if current_ref_is_decoy is not None:
                        current_ref_is_decoy = current_ref_is_decoy[
                            hnsw_filtered_indices
                        ]
            except Exception as exc:
                logger.warning(
                    "HNSW candidate retrieval failed (%s); falling back "
                    "to full cascade scoring.",
                    exc,
                )
                self._degraded_flags.append("cascade_hnsw_failed")

        # ------------------------------------------------------------------
        # Phase 1: run stages sequentially, winnowing the reference set
        # ------------------------------------------------------------------

        for stage_idx, (algo, engine) in enumerate(self._stages):
            is_last = stage_idx == len(self._stages) - 1
            stage_threshold = upper_bound if is_last else lower_bound

            try:
                stage_results = engine.search(
                    query_spectra=query_spectra,
                    reference_spectra=current_refs,
                    min_score=stage_threshold,
                    top_n=None,  # collect all for candidate filtering
                    include_decoys=False,  # decoys handled once at the end
                    ref_precursor_mzs=current_ref_precursor_mzs,
                    ref_is_decoy=current_ref_is_decoy,
                )
            except Exception as exc:
                logger.warning(
                    "Cascade stage %d ('%s') failed, stopping cascade: %s",
                    stage_idx + 1,
                    algo,
                    exc,
                )
                self._degraded_flags.append(f"cascade_stage_failed:{algo}")
                # A failed stage means the cascade cannot continue filtering.
                # Fall back to classical modified_cosine scoring over the
                # current candidate set so the run still produces annotations
                # (e.g. remote ML endpoint unreachable with an open circuit
                # breaker), instead of silently returning no results.
                return self._run_classical_fallback(
                    query_spectra,
                    current_refs,
                    stage_threshold,
                    top_n,
                    include_decoys,
                    current_ref_precursor_mzs,
                    current_ref_is_decoy,
                    decoy_min_relative_intensity,
                    decoy_mz_shift_da,
                )

            if is_last:
                # Final stage: these are the results to return (after
                # applying top_n).
                final_results = stage_results
            else:
                # Intermediate stage: collect reference IDs that survived
                # and use them to filter the reference set for the next
                # stage.
                surviving_ref_ids: set[str] = set()
                for res in stage_results:
                    surviving_ref_ids.add(res["reference_id"])

                if not surviving_ref_ids:
                    # No candidates passed this stage; stop early.
                    logger.debug(
                        "Cascade stage %d ('%s'): no candidates survived; "
                        "stopping cascade.",
                        stage_idx + 1,
                        algo,
                    )
                    return []

                logger.debug(
                    "Cascade stage %d ('%s'): %d candidates survived (of %d).",
                    stage_idx + 1,
                    algo,
                    len(surviving_ref_ids),
                    len(current_refs),
                )

                # Filter reference list and aligned arrays
                filtered_refs: list[Spectrum] = []
                filtered_indices: list[int] = []
                for i, ref in enumerate(current_refs):
                    ref_id = str(ref.get("id"))
                    if ref_id in surviving_ref_ids:
                        filtered_refs.append(ref)
                        filtered_indices.append(i)

                current_refs = filtered_refs

                # Align pre-computed arrays if they were provided
                if current_ref_precursor_mzs is not None:
                    current_ref_precursor_mzs = current_ref_precursor_mzs[
                        filtered_indices
                    ]
                if current_ref_is_decoy is not None:
                    current_ref_is_decoy = current_ref_is_decoy[filtered_indices]

        # ------------------------------------------------------------------
        # Phase 2: apply top_n to final results
        # ------------------------------------------------------------------
        if top_n is not None:
            by_query: dict[str, list[SearchResult]] = defaultdict(list)
            for res in final_results:
                by_query[res["query_id"]].append(res)

            final_results = []
            for _qid, hits in by_query.items():
                hits.sort(key=lambda r: -r["score"])
                final_results.extend(hits[:top_n])

        return final_results

    def batch_score(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
    ) -> np.ndarray:
        """Score pre-formed query--reference pairs via cascade.

        Uses the last cascade stage for batch scoring.
        """
        if not self._stages:
            return np.full(len(query_spectra), np.nan, dtype=np.float64)
        _, last_engine = self._stages[-1]
        return last_engine.batch_score(query_spectra, reference_spectra)

    def load_model(self, model_path: str | Path) -> None:
        """Load model weights for all cascade stages.

        Calls ``load_model`` on each stage engine in turn.  Failures are
        logged but do not prevent other stages from loading.
        """
        for algo, engine in self._stages:
            try:
                engine.load_model(model_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load model for cascade stage '%s': %s", algo, exc
                )


class MLRouter:
    """Dynamically route query spectra between classical and ML scoring engines.

    The router evaluates each query spectrum's :class:`TriageProfile` against
    user-defined thresholds in ``SimilarityConfig`` and dispatches to either a
    fast classical engine ("easy" spectra) or a more sophisticated ML engine
    ("hard" spectra such as chimeric, low-S/N, or unassigned neutral losses).

    If the ML engine fails or exceeds ``routing_ml_timeout_seconds``, the
    router automatically falls back to ``routing_fallback_engine`` and logs a
    warning without aborting the batch.

    **Multiprocessing compatibility**

    The router is instantiated inside the worker process.  Only the (small,
    pickle-safe) ``SimilarityConfig`` travels across the process boundary.
    Engine instances are built lazily on first use within each worker.

    Parameters
    ----------
    config : SimilarityConfig
        Pipeline configuration containing routing thresholds and engine choices.
    """

    def __init__(self, config: SimilarityConfig) -> None:
        self._config = config
        self._easy_engine: (
            SimilarityEngine | _MLEngineBase | MLEngineProtocol | None
        ) = None
        self._hard_engine: (
            SimilarityEngine | _MLEngineBase | MLEngineProtocol | None
        ) = None
        self._fallback_engine: SimilarityEngine | None = None
        # Per-search degradation record (see ``degraded_mode_flags``).
        self._degraded_flags: list[str] = []

    @property
    def degraded_mode_flags(self) -> list[str]:
        """Degradation flags recorded by the most recent ``route_and_search``.

        Value: ``routing_hard_fallback:<engine>`` when the hard (ML) engine
        failed or timed out and the batch was re-run with the configured
        classical fallback engine.
        """
        return list(self._degraded_flags)

    # ------------------------------------------------------------------
    # Lazy engine builders
    # ------------------------------------------------------------------

    def _get_easy_engine(self) -> SimilarityEngine | _MLEngineBase | MLEngineProtocol:
        if self._easy_engine is None:
            easy_cfg = _sub_engine_config(
                self._config,
                self._config.routing_easy_engine,
            )
            self._easy_engine = get_similarity_engine(easy_cfg)
        return self._easy_engine

    def _get_hard_engine(self) -> SimilarityEngine | _MLEngineBase | MLEngineProtocol:
        if self._hard_engine is None:
            hard_algo = self._config.routing_hard_engine
            if hard_algo not in _ML_ENGINE_REGISTRY:
                logger.warning(
                    "routing_hard_engine '%s' is not an ML engine; "
                    "using modified_cosine as hard engine instead.",
                    hard_algo,
                )
                hard_algo = "modified_cosine"  # type: ignore[assignment]

            hard_cfg = _sub_engine_config(
                self._config,
                hard_algo,
                min_score=0.0,  # collect all hits; FDR applied later
                min_matched_peaks=0,
                # Forward consensus/cascade settings if applicable
                consensus_weights=self._config.consensus_weights,
                consensus_min_engines=self._config.consensus_min_engines,
                cascade_lower_bound=self._config.cascade_lower_bound,
                cascade_upper_bound=self._config.cascade_upper_bound,
                cascade_stages=self._config.cascade_stages,
            )
            self._hard_engine = get_similarity_engine(hard_cfg)
        return self._hard_engine

    def _get_fallback_engine(self) -> SimilarityEngine:
        if self._fallback_engine is None:
            fb_cfg = _sub_engine_config(
                self._config,
                self._config.routing_fallback_engine,
            )
            eng = get_similarity_engine(fb_cfg)
            if not isinstance(eng, SimilarityEngine):
                raise RuntimeError(
                    f"routing_fallback_engine '{self._config.routing_fallback_engine}' "
                    "must be a classical engine (cosine or modified_cosine)."
                )
            self._fallback_engine = eng
        return self._fallback_engine

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, spectrum: Spectrum) -> Literal["easy", "hard"]:
        """Classify a single query spectrum as 'easy' or 'hard'.

        The decision is based on the spectrum's ``TriageProfile`` evaluated
        against the router's configured thresholds.

        Parameters
        ----------
        spectrum : matchms.Spectrum
            Query spectrum with optional ``triage_flags`` metadata.

        Returns
        -------
        Literal['easy', 'hard']
            Routing decision for this spectrum.
        """
        profile = TriageProfile.from_spectrum_metadata(spectrum.metadata)

        # Check individual thresholds ---------------------------------------------------
        reasons: list[str] = []

        # Chimeric spectra
        if profile.is_chimeric:
            action = self._config.routing_chimeric_action
            if action == "hard":
                reasons.append("chimeric")

        # Low precursor purity
        if (
            profile.precursor_purity is not None
            and profile.precursor_purity
            < self._config.routing_precursor_purity_threshold
        ):
            reasons.append("low_precursor_purity")

        # Missing MS1 purity
        if profile.missing_ms1_purity:
            reasons.append("missing_ms1_purity")

        # Low abundance precursor
        if profile.low_abundance_precursor:
            reasons.append("low_abundance_precursor")

        # Low S/N
        if (
            profile.signal_to_noise is not None
            and profile.signal_to_noise < self._config.routing_snr_threshold
        ):
            reasons.append("low_snr")
        elif profile.low_signal_to_noise:
            reasons.append("low_snr")

        # Unassigned neutral losses
        if profile.unassigned_neutral_losses:
            reasons.append("unassigned_neutral_losses")

        # Aggregate difficulty check
        if len(reasons) >= self._config.routing_min_difficulty_flags:
            logger.debug(
                "Spectrum %s routed to HARD engine — reasons: %s",
                spectrum.get("id", "<unknown>"),
                ", ".join(reasons),
            )
            return "hard"

        return "easy"

    # ------------------------------------------------------------------
    # Main search dispatch
    # ------------------------------------------------------------------

    def route_and_search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
        decoy_min_relative_intensity: float | None = None,
        decoy_mz_shift_da: float | None = None,
    ) -> List[SearchResult]:
        """Classify queries and dispatch to the appropriate engine.

        Queries classified as "easy" go to ``routing_easy_engine``;
        queries classified as "hard" go to ``routing_hard_engine``.

        If the hard engine raises an exception or times out (see
        ``routing_ml_timeout_seconds``), the hard batch is re-run through
        ``routing_fallback_engine`` and a warning is logged.

        Parameters
        ----------
        query_spectra : list of matchms.Spectrum
            Experimental query spectra.
        reference_spectra : iterable of matchms.Spectrum
            Reference library spectra.
        include_decoys : bool
            Whether engines should generate and score decoy spectra.
        ref_precursor_mzs : np.ndarray or None
            Pre-computed reference precursor m/z array.
        ref_is_decoy : np.ndarray or None
            Pre-computed reference decoy flags.
        decoy_min_relative_intensity : float or None, optional
            Decoy noise floor passed through to engines that generate decoys.
        decoy_mz_shift_da : float or None, optional
            Decoy m/z jitter passed through to engines that generate decoys.

        Returns
        -------
        list of SearchResult
            Combined results from all engines, tagged with ``routed_via``
            in the ``score_breakdown`` dict (or a new top-level entry
            ``routed_via``).
        """
        # --- Classify every query ------------------------------------------------
        easy_queries: List[Spectrum] = []
        hard_queries: List[Spectrum] = []

        # Reset the per-search degradation record.
        self._degraded_flags = []

        for q in query_spectra:
            decision = self.classify(q)
            if decision == "hard":
                hard_queries.append(q)
            else:
                easy_queries.append(q)

        n_total = len(query_spectra)
        n_easy = len(easy_queries)
        n_hard = len(hard_queries)
        logger.info(
            "MLRouter: %d/%d queries classified as EASY, %d/%d as HARD.",
            n_easy,
            n_total,
            n_hard,
            n_total,
        )

        all_results: List[SearchResult] = []

        # --- Easy batch ----------------------------------------------------------
        if easy_queries:
            easy_engine = self._get_easy_engine()
            easy_results = easy_engine.search(
                query_spectra=easy_queries,
                reference_spectra=reference_spectra,
                include_decoys=include_decoys,
                ref_precursor_mzs=ref_precursor_mzs,
                ref_is_decoy=ref_is_decoy,
                decoy_min_relative_intensity=decoy_min_relative_intensity,
                decoy_mz_shift_da=decoy_mz_shift_da,
            )
            # Tag results with routing information
            for res in easy_results:
                breakdown: dict[str, float | str] = dict(
                    res.get("score_breakdown") or {}
                )
                breakdown["routed_via"] = self._config.routing_easy_engine
                res["score_breakdown"] = breakdown  # type: ignore[arg-type,typeddict-item]
                res["routed_via"] = self._config.routing_easy_engine  # type: ignore[typeddict-unknown-key]
            all_results.extend(easy_results)
            logger.debug("Easy engine returned %d results.", len(easy_results))

        # --- Hard batch (with timeout + fallback) --------------------------------
        if hard_queries:
            hard_results = self._run_hard_with_fallback(
                hard_queries,
                reference_spectra,
                include_decoys=include_decoys,
                ref_precursor_mzs=ref_precursor_mzs,
                ref_is_decoy=ref_is_decoy,
                decoy_min_relative_intensity=decoy_min_relative_intensity,
                decoy_mz_shift_da=decoy_mz_shift_da,
            )
            # Tag results with routing information
            routed_tag: str = self._config.routing_hard_engine
            for res in hard_results:
                breakdown = dict(res.get("score_breakdown") or {})
                # Determine the actual engine that produced this result
                used_engine: str = (
                    self._config.routing_fallback_engine
                    if res.get("_fallback")
                    else routed_tag
                )
                breakdown["routed_via"] = used_engine
                res["score_breakdown"] = breakdown  # type: ignore[arg-type,typeddict-item]
                res["routed_via"] = used_engine  # type: ignore[typeddict-unknown-key]
            all_results.extend(hard_results)

        return all_results

    def _run_hard_with_fallback(
        self,
        hard_queries: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        include_decoys: bool,
        ref_precursor_mzs: np.ndarray | None,
        ref_is_decoy: np.ndarray | None,
        decoy_min_relative_intensity: float | None = None,
        decoy_mz_shift_da: float | None = None,
    ) -> List[SearchResult]:
        """Run the hard engine with a timeout; fall back on failure.

        Uses ``concurrent.futures.ThreadPoolExecutor`` to enforce the
        ``routing_ml_timeout_seconds`` deadline.  If the hard engine
        raises or times out, the fallback engine processes the batch.
        """
        timeout = self._config.routing_ml_timeout_seconds
        ref_list = list(reference_spectra)  # materialise for reuse

        hard_engine = self._get_hard_engine()

        def _hard_search() -> List[SearchResult]:
            return hard_engine.search(
                query_spectra=hard_queries,
                reference_spectra=ref_list,
                include_decoys=include_decoys,
                ref_precursor_mzs=ref_precursor_mzs,
                ref_is_decoy=ref_is_decoy,
                decoy_min_relative_intensity=decoy_min_relative_intensity,
                decoy_mz_shift_da=decoy_mz_shift_da,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_hard_search)
                results = future.result(timeout=timeout)
            logger.info(
                "Hard engine completed successfully (%d results).", len(results)
            )
            return results
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Hard engine '%s' timed out after %.1f s for %d queries; "
                "falling back to '%s'.",
                self._config.routing_hard_engine,
                timeout,
                len(hard_queries),
                self._config.routing_fallback_engine,
            )
            self._degraded_flags.append(
                f"routing_hard_fallback:{self._config.routing_fallback_engine}"
            )
        except Exception as exc:
            logger.warning(
                "Hard engine '%s' failed: %s; falling back to '%s'.",
                self._config.routing_hard_engine,
                exc,
                self._config.routing_fallback_engine,
            )
            self._degraded_flags.append(
                f"routing_hard_fallback:{self._config.routing_fallback_engine}"
            )

        # --- Fallback path -------------------------------------------------------
        fallback = self._get_fallback_engine()
        fb_results = fallback.search(
            query_spectra=hard_queries,
            reference_spectra=ref_list,
            include_decoys=include_decoys,
            ref_precursor_mzs=ref_precursor_mzs,
            ref_is_decoy=ref_is_decoy,
            decoy_min_relative_intensity=decoy_min_relative_intensity,
            decoy_mz_shift_da=decoy_mz_shift_da,
        )
        for res in fb_results:
            res["_fallback"] = True  # type: ignore[typeddict-unknown-key]
        logger.info(
            "Fallback engine '%s' returned %d results for hard queries.",
            self._config.routing_fallback_engine,
            len(fb_results),
        )
        return fb_results


# ---------------------------------------------------------------------------
# ML engine registry (populated from entry points)
# ---------------------------------------------------------------------------
# External packages (e.g. ``massflow-ml``) register their engine classes via
# the ``massflow.similarity_engines`` entry-point group.  The registry is
# populated once at module import time and cached for the lifetime of the
# process.  Built-in engines defined in this module are also registered
# through the same mechanism (see pyproject.toml).

_ML_ENGINE_REGISTRY: dict[str, type] = {}


def _discover_ml_engines() -> dict[str, type]:
    """Discover ML engine classes from registered entry points.

    Scans the ``massflow.similarity_engines`` entry-point group and returns
    a mapping of algorithm name → engine class.  Entry points that fail to
    load are logged and skipped.

    Returns
    -------
    dict[str, type]
        Algorithm name → engine class.
    """
    registry: dict[str, type] = {}
    try:
        eps = importlib.metadata.entry_points(group="massflow.similarity_engines")
    except TypeError:
        # Python < 3.12: entry_points() takes no arguments.
        # Filter the full dict manually.
        all_eps = importlib.metadata.entry_points()
        eps = all_eps.select(group="massflow.similarity_engines")  # type: ignore[union-attr]

    for ep in eps:
        try:
            engine_cls = ep.load()
        except Exception as exc:
            logger.debug(
                "Failed to load similarity engine entry point '%s': %s",
                ep.name,
                exc,
            )
            continue
        if not isinstance(engine_cls, type):
            logger.warning(
                "Entry point '%s' did not resolve to a class (got %s); skipping.",
                ep.name,
                type(engine_cls).__name__,
            )
            continue
        if not issubclass(engine_cls, MLEngineProtocol):
            logger.warning(
                "Entry point '%s' (%s) does not implement MLEngineProtocol; skipping.",
                ep.name,
                engine_cls.__name__,
            )
            continue
        registry[ep.name] = engine_cls
        logger.debug("Registered ML engine '%s' → %s", ep.name, engine_cls.__name__)

    return registry


# Populate the registry at import time.
_ML_ENGINE_REGISTRY = _discover_ml_engines()


# ---------------------------------------------------------------------------
# Legacy alias (deprecated; use ``get_similarity_engine`` or
# ``_ML_ENGINE_REGISTRY`` directly).
# ---------------------------------------------------------------------------
_ML_ENGINE_MAP = _ML_ENGINE_REGISTRY


def get_similarity_engine(
    config: SimilarityConfig,
) -> SimilarityEngine | _MLEngineBase | MLEngineProtocol:
    """Factory function to instantiate the appropriate similarity engine.

    Returns a ``SimilarityEngine`` for classical algorithms (cosine,
    modified_cosine) or an ``_MLEngineBase`` subclass for ML-based algorithms
    (spec2vec, ms2deepscore, consensus, cascade).

    ML engines are discovered dynamically from the
    ``massflow.similarity_engines`` entry-point group, enabling external
    packages (such as ``massflow-ml``) to register their own scoring engines
    without modifying core MassFlow source code.

    Parameters
    ----------
    config : SimilarityConfig
        The configuration object detailing the algorithm choice.

    Returns
    -------
    SimilarityEngine or _MLEngineBase
        The instantiated engine ready to perform ``.search()``.

    Raises
    ------
    ValueError
        If the requested algorithm is not recognised by any registered engine.
    RuntimeError
        If an ML algorithm is requested but its required dependencies
        (PyTorch, Gensim, etc.) are not installed.
    """
    algo = config.algorithm

    # Classical algorithms: always available with zero overhead.
    if algo in ("cosine", "modified_cosine"):
        return SimilarityEngine(config)

    # ── Remote ML endpoints (massflow-ml satellite boundary) ────────────
    # When an endpoint is configured for the requested algorithm, scoring is
    # delegated to the remote service through a circuit-breaker-protected
    # client instead of requiring a local installation of the heavy
    # dependencies (PyTorch, Gensim, spec2vec, ms2deepscore).
    remote_endpoint = (config.ml_endpoints or {}).get(algo)
    if remote_endpoint:
        from MassFlow.ml_client import RemoteMLEngine

        logger.info(
            "Routing algorithm '%s' to remote ML endpoint '%s'.",
            algo,
            remote_endpoint,
        )
        return RemoteMLEngine(
            algorithm=algo,
            endpoint=remote_endpoint,
            timeout_seconds=config.ml_request_timeout_seconds,
            circuit_failure_threshold=config.ml_circuit_breaker_threshold,
            circuit_cooldown_seconds=config.ml_circuit_breaker_cooldown_seconds,
        )

    # ML / meta algorithms: resolve via the entry-point registry.
    if algo in _ML_ENGINE_REGISTRY:
        engine_cls = _ML_ENGINE_REGISTRY[algo]
        try:
            return engine_cls(config)
        except RuntimeError:
            # Re-raise RuntimeError (e.g. from _check_dependencies) with a
            # clearer user-facing message.
            raise RuntimeError(
                f"Algorithm '{algo}' requires the machine-learning extras. "
                + _ML_INSTALL_MSG
            )

    raise ValueError(
        f"Unsupported algorithm: '{algo}'. "
        f"Available classical algorithms: cosine, modified_cosine. "
        f"Registered ML algorithms: {list(_ML_ENGINE_REGISTRY.keys()) or 'none'}."
    )
