"""
Spectral similarity search engine for MassFlow.

This module encapsulates the logic for comparing experimental mass spectra against
reference libraries. It provides a unified interface (`SimilarityEngine`) to classical
similarity algorithms (Cosine, Modified Cosine) backed by matchms. It handles score
calculation, result filtering/formatting, decoy generation, and FDR estimation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import wraps
from typing import Iterable, Iterator, List, Optional, TypedDict

import numpy as np
from matchms import Spectrum, calculate_scores

try:
    from matchms.similarity import (
        CosineGreedy,  # type: ignore[attr-defined]
        ModifiedCosine,  # type: ignore[attr-defined]
    )
except ImportError:
    from matchms.similarity import CosineGreedy
    from matchms.similarity import ModifiedCosineGreedy as ModifiedCosine

from MassFlow.config import SimilarityConfig

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


def generate_decoys(spectra: List[Spectrum], random_seed: int = 42) -> List[Spectrum]:
    """
    Generate a decoy library by shuffling fragment intensities.
    This mathematically breaks the structural correlation while keeping the
    precursor m/z the same and preserving monotonically increasing m/z arrays.
    """
    rng = np.random.default_rng(random_seed)
    decoys = []
    for spec in spectra:
        decoy_metadata = spec.metadata.copy()
        decoy_metadata["is_decoy"] = True
        decoy_id = str(spec.get("id", "unknown")) + "_decoy"
        decoy_metadata["id"] = decoy_id

        name = spec.get("compound_name") or spec.get("name")
        if name:
            decoy_metadata["compound_name"] = f"{name}_decoy"

        shuffled_intensities = spec.peaks.intensities.copy()
        n_peaks = len(shuffled_intensities)

        # If there are fewer than 2 unique intensity values, shuffling is ineffective.
        # Instead, apply a random taper to break structural correlation. We use
        # randomised uniform multipliers (0.5–1.0) shuffled independently so the
        # resulting pattern is not systematically correlated with m/z order.
        if len(np.unique(shuffled_intensities)) < 2 and n_peaks > 1:
            taper = rng.uniform(0.5, 1.0, size=n_peaks)
            rng.shuffle(taper)
            shuffled_intensities = shuffled_intensities * taper
        else:
            original_intensities = shuffled_intensities.copy()
            rng.shuffle(shuffled_intensities)
            # Post-shuffle check to ensure it's not identical (for low peak counts).
            # If the shuffle accidentally produced the original ordering, roll by one
            # position and add small random jitter for very sparse spectra to prevent
            # accidental structural correlation through matched m/z positions.
            if np.array_equal(shuffled_intensities, original_intensities):
                shuffled_intensities = np.roll(shuffled_intensities, 1)
            if n_peaks <= 5:
                jitter = rng.uniform(0.95, 1.05, size=n_peaks)
                shuffled_intensities = shuffled_intensities * jitter

        decoy_spec = Spectrum(
            mz=spec.peaks.mz.copy(),
            intensities=shuffled_intensities,
            metadata=decoy_metadata,
        )
        decoys.append(decoy_spec)
    return decoys


def calculate_empirical_p_values(
    target_scores: np.ndarray, decoy_scores: np.ndarray
) -> np.ndarray:
    """
    Calculate empirical p-values for target scores against a decoy null distribution.

    Uses binary search on sorted decoy scores for O(N log M) time and O(1) extra
    memory, avoiding the O(N × M) intermediate array that could exhaust RAM for
    very large libraries.
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
    """
    Calculate q-values (False Discovery Rate) for target scores.
    Uses the conservative target-decoy pseudo-count formula: FDR = (decoys + 1) / targets
    to prevent overly optimistic 0.0 FDR values, particularly for small libraries.

    Parameters
    ----------
    target_scores : np.ndarray
        Array of scores from the target library search. Shape: (N,), dtype: float.
    decoy_scores : np.ndarray
        Array of scores from the decoy library search. Shape: (M,), dtype: float.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - sorted_scores: Combined target and decoy scores sorted in descending order. Shape: (N+M,), dtype: float.
        - q_values: Calculated q-values corresponding to each score. Shape: (N+M,), dtype: float.
        - is_target: Boolean mask indicating if the score belongs to a target (True) or decoy (False). Shape: (N+M,), dtype: bool.
    """
    import polars as pl

    if len(target_scores) == 0 and len(decoy_scores) == 0:
        return np.array([]), np.array([]), np.array([], dtype=bool)

    if len(decoy_scores) == 0:
        sort_idx = np.argsort(target_scores)[::-1]
        sorted_scores = target_scores[sort_idx]
        # Conservative pseudo-count when no decoys exist: FDR = 1 / cum_targets
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

    # Use Polars for efficient sorting and cumulative calculations
    df = pl.DataFrame(
        {
            "score": np.concatenate([target_scores, decoy_scores]),
            "is_target": np.concatenate(
                [
                    np.ones_like(target_scores, dtype=bool),
                    np.zeros_like(decoy_scores, dtype=bool),
                ]
            ),
        }
    )

    # Sort descending by score, then targets first on ties
    df = df.sort(["score", "is_target"], descending=[True, True])

    df = df.with_columns(
        [
            pl.col("is_target").cast(pl.Int64).cum_sum().alias("cum_targets"),
            (~pl.col("is_target")).cast(pl.Int64).cum_sum().alias("cum_decoys"),
        ]
    )

    # Apply conservative FDR formula: (cum_decoys + 1) / cum_targets
    df = df.with_columns(
        pl.when(pl.col("cum_targets") > 0)
        .then((pl.col("cum_decoys") + 1) / pl.col("cum_targets"))
        .otherwise(1.0)
        .clip(0, 1)
        .alias("fdr")
    )

    # Calculate q-values (minimum FDR for all lower scores)
    # In Polars, we can reverse, cum_min, and reverse back
    df = df.with_columns(pl.col("fdr").reverse().cum_min().reverse().alias("q_value"))

    return (
        df.get_column("score").to_numpy(),
        df.get_column("q_value").to_numpy(),
        df.get_column("is_target").to_numpy(),
    )


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
            decoy_spectra = generate_decoys(ref_list)
            all_references = ref_list + decoy_spectra
            n_targets = len(ref_list)
        else:
            all_references = list(reference_spectra)
            n_targets = len(all_references)

        n_queries = len(query_spectra)

        # MS1 Pre-filtering for cosine with sparse array support.
        # When ref_precursor_mzs is provided (L2 cache), use it directly to
        # avoid Spectrum-object property lookups in the hot path.
        if self.config.algorithm == "cosine" and hasattr(
            self.similarity_function, "sparse_array"
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

        else:
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


class _MLEngineBase:
    """Base class for ML-based similarity scoring engines.

    Subclasses must implement ``_build_model()`` and override ``search()``.
    The public interface is intentionally compatible with ``SimilarityEngine``
    so that the factory and workflow layers can treat them uniformly.
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
    ) -> List[SearchResult]:
        """Run similarity search. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement search()")


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
        self._sub_engines: dict[str, SimilarityEngine | _MLEngineBase] = {}
        self._sub_weights: dict[str, float] = {}

        weights = self.config.consensus_weights

        for algo, weight in weights.items():
            if weight <= 0:
                continue

            sub_cfg = SimilarityConfig(
                algorithm=algo,  # type: ignore[arg-type]
                ms1_tolerance=self.config.ms1_tolerance,
                ms2_tolerance=self.config.ms2_tolerance,
                resolution_ppm=self.config.resolution_ppm,
                min_score=0.0,
                min_matched_peaks=0,
                rt_tolerance=self.config.rt_tolerance,
            )

            engine: SimilarityEngine | _MLEngineBase | None = None

            if algo in ("cosine", "modified_cosine"):
                try:
                    engine = SimilarityEngine(sub_cfg)
                except Exception as exc:
                    logger.warning(
                        "Failed to build classical sub-engine '%s': %s", algo, exc
                    )
            elif algo == "spec2vec" and _HAS_SPEC2VEC:
                try:
                    engine = Spec2VecEngine(sub_cfg)
                except Exception as exc:
                    logger.warning("Failed to build spec2vec sub-engine: %s", exc)
            elif algo == "ms2deepscore" and _HAS_MS2DEEPSCORE:
                try:
                    engine = MS2DeepScoreEngine(sub_cfg)
                except Exception as exc:
                    logger.warning("Failed to build ms2deepscore sub-engine: %s", exc)
            else:
                logger.debug(
                    "Skipping sub-engine '%s': not available or dependencies missing.",
                    algo,
                )

            if engine is not None:
                self._sub_engines[algo] = engine
                self._sub_weights[algo] = weight

        if not self._sub_engines:
            logger.warning(
                "No sub-engines could be initialised for consensus scoring. "
                "Searches will return empty results."
            )
        else:
            logger.info(
                "Consensus engine initialised with sub-engines: %s",
                list(self._sub_engines.keys()),
            )

        self._model_loaded = bool(self._sub_engines)

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
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

        Returns
        -------
        List[SearchResult]
            Aggregated and weighted consensus results.
        """
        if not self._sub_engines:
            return []

        if not query_spectra or not reference_spectra:
            return []

        ref_list = list(reference_spectra)
        cutoff = min_score if min_score is not None else self.config.min_score
        min_engines = self.config.consensus_min_engines

        # ------------------------------------------------------------------
        # Phase 1: run every available sub-engine and collect raw results
        # ------------------------------------------------------------------
        # engine_results: list of (algo, weight, list of SearchResult)
        engine_results: list[tuple[str, float, list[SearchResult]]] = []

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
                )
                engine_results.append((algo, weight, raw))
                logger.debug("Sub-engine '%s' returned %d raw results.", algo, len(raw))
            except Exception as exc:
                logger.warning(
                    "Sub-engine '%s' failed during search, skipping: %s",
                    algo,
                    exc,
                )

        if not engine_results:
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
                buckets[key]["scores"][algo] = res["score"]

        # ------------------------------------------------------------------
        # Phase 3: compute weighted consensus scores
        # ------------------------------------------------------------------
        aggregated: list[SearchResult] = []

        for (_qid, _rid), bucket in buckets.items():
            scores_by_algo: dict[str, float] = bucket["scores"]

            if len(scores_by_algo) < min_engines:
                continue

            total_weight = 0.0
            weighted_sum = 0.0
            for algo, score in scores_by_algo.items():
                w = self._sub_weights.get(algo, 0.0)
                total_weight += w
                weighted_sum += score * w

            if total_weight == 0:
                continue

            consensus_score = weighted_sum / total_weight

            if consensus_score < cutoff:
                continue

            template = bucket["template"]
            best_individual = max(scores_by_algo.values())

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
                    score_breakdown=dict(scores_by_algo),
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
        self._stages: list[tuple[str, SimilarityEngine | _MLEngineBase]] = []

        for algo in self.config.cascade_stages:
            sub_cfg = SimilarityConfig(
                algorithm=algo,  # type: ignore[arg-type]
                ms1_tolerance=self.config.ms1_tolerance,
                ms2_tolerance=self.config.ms2_tolerance,
                resolution_ppm=self.config.resolution_ppm,
                min_score=0.0,
                min_matched_peaks=0,
                rt_tolerance=self.config.rt_tolerance,
            )

            engine: SimilarityEngine | _MLEngineBase | None = None

            if algo in ("cosine", "modified_cosine"):
                try:
                    engine = SimilarityEngine(sub_cfg)
                except Exception as exc:
                    logger.warning("Failed to build cascade stage '%s': %s", algo, exc)
            elif algo == "spec2vec" and _HAS_SPEC2VEC:
                try:
                    engine = Spec2VecEngine(sub_cfg)
                except Exception as exc:
                    logger.warning("Failed to build spec2vec cascade stage: %s", exc)
            elif algo == "ms2deepscore" and _HAS_MS2DEEPSCORE:
                try:
                    engine = MS2DeepScoreEngine(sub_cfg)
                except Exception as exc:
                    logger.warning(
                        "Failed to build ms2deepscore cascade stage: %s", exc
                    )
            else:
                logger.debug(
                    "Skipping cascade stage '%s': not available or dependencies missing.",
                    algo,
                )

            if engine is not None:
                self._stages.append((algo, engine))

        if not self._stages:
            logger.warning(
                "No cascade stages could be initialised. "
                "Searches will return empty results."
            )
        else:
            logger.info(
                "Cascade engine initialised with stages: %s",
                [a for a, _ in self._stages],
            )

        self._model_loaded = bool(self._stages)

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
    ) -> List[SearchResult]:
        """Run cascaded similarity search with sequential filtering.

        Each stage in the cascade scores the query-reference pairs with an
        increasingly strict threshold.  After each stage, only reference
        spectra that produced at least one hit above the stage threshold
        are passed to the next stage.  This narrows the candidate set
        progressively, reserving expensive models for the most promising
        candidates.

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

        Returns
        -------
        List[SearchResult]
            Final-stage results after cascaded filtering.
        """
        if not self._stages:
            return []

        if not query_spectra or not reference_spectra:
            return []

        ref_list = list(reference_spectra)
        lower_bound = self.config.cascade_lower_bound
        upper_bound = (
            min_score if min_score is not None else self.config.cascade_upper_bound
        )

        # ------------------------------------------------------------------
        # Phase 1: run stages sequentially, winnowing the reference set
        # ------------------------------------------------------------------
        current_refs: list[Spectrum] = ref_list
        current_ref_precursor_mzs = ref_precursor_mzs
        current_ref_is_decoy = ref_is_decoy

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
                # If a non-final stage fails we cannot continue filtering.
                if not is_last:
                    return []
                # If the final stage fails, return what we have from the
                # previous stage (handled below).
                stage_results = []

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


_ML_ENGINE_MAP: dict[str, type] = {
    "spec2vec": Spec2VecEngine,
    "ms2deepscore": MS2DeepScoreEngine,
    "consensus": ConsensusEngine,
    "cascade": CascadeEngine,
}


def get_similarity_engine(config: SimilarityConfig) -> SimilarityEngine | _MLEngineBase:
    """Factory function to instantiate the appropriate similarity engine.

    Returns a ``SimilarityEngine`` for classical algorithms (cosine,
    modified_cosine) or an ``_MLEngineBase`` subclass for ML-based algorithms
    (spec2vec, ms2deepscore, consensus, cascade).

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
        If the requested algorithm is not recognised.
    RuntimeError
        If an ML algorithm is requested but the ``[ml]`` extra is not installed.
    """
    algo = config.algorithm

    if algo in _ML_ENGINE_MAP:
        if not _HAS_ML:
            raise RuntimeError(
                f"Algorithm '{algo}' requires the machine-learning extras. "
                + _ML_INSTALL_MSG
            )
        engine_cls = _ML_ENGINE_MAP[algo]
        return engine_cls(config)

    if algo in ("cosine", "modified_cosine"):
        return SimilarityEngine(config)

    raise ValueError(f"Unsupported algorithm: {algo}")
