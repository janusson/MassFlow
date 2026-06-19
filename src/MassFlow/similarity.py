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
    from matchms.similarity import CosineGreedy, ModifiedCosine
except ImportError:
    from matchms.similarity import CosineGreedy
    from matchms.similarity import ModifiedCosineGreedy as ModifiedCosine

from MassFlow.config import SimilarityConfig

logger = logging.getLogger(__name__)


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
    ):
        if not isinstance(reference_spectra, (list, tuple)):
            all_results = []
            processed_count = 0
            for chunk in yield_fixed_chunks(reference_spectra, chunk_size=10000):
                if not chunk:
                    continue

                processed_count += len(chunk)
                logger.info(
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
            self, query_spectra, reference_spectra, min_score, top_n, include_decoys
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
    """Perform MS1 precursor m/z pre-filtering using Da tolerance or optionally PPM resolution."""
    ref_mzs_raw = [s.get("precursor_mz") for s in all_references]
    query_mzs_raw = [q.get("precursor_mz") for q in query_spectra]

    # Track missing precursors to allow them to bypass the MS1 filter
    ref_missing = np.array([_is_missing(r) for r in ref_mzs_raw], dtype=bool)
    query_missing = np.array([_is_missing(q) for q in query_mzs_raw], dtype=bool)

    ref_mzs = np.array([float(r) if not _is_missing(r) else 0.0 for r in ref_mzs_raw])
    query_mzs = np.array(
        [float(q) if not _is_missing(q) else 0.0 for q in query_mzs_raw]
    )

    if resolution_ppm is not None:
        # For each query, find references where |ref_mz - query_mz| / query_mz <= ppm_tolerance
        # This is more efficient than a full matrix operation for sparse results.
        query_mzs_indexed = list(enumerate(query_mzs))
        ref_mzs_sorted_indices = np.argsort(ref_mzs)
        ref_mzs_sorted = ref_mzs[ref_mzs_sorted_indices]

        rows: List[int] = []
        cols: List[int] = []
        for query_idx, query_mz in query_mzs_indexed:
            if query_mz > 0:
                ppm_tol_da = resolution_ppm * query_mz / 1e6
                min_mz, max_mz = query_mz - ppm_tol_da, query_mz + ppm_tol_da

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

    else:
        # For each query, find references where |ref_mz - query_mz| <= ms1_tolerance
        # This uses binary search, which is O(Q * log(R)) and avoids O(R * Q) dense array memory allocation
        query_mzs_indexed = list(enumerate(query_mzs))
        ref_mzs_sorted_indices = np.argsort(ref_mzs)
        ref_mzs_sorted = ref_mzs[ref_mzs_sorted_indices]

        rows_abs: List[int] = []
        cols_abs: List[int] = []
        for query_idx, query_mz in query_mzs_indexed:
            if query_mz > 0:
                min_mz, max_mz = query_mz - ms1_tolerance, query_mz + ms1_tolerance

                start_idx = np.searchsorted(ref_mzs_sorted, min_mz, side="left")
                end_idx = np.searchsorted(ref_mzs_sorted, max_mz, side="right")

                original_indices = ref_mzs_sorted_indices[start_idx:end_idx]
                rows_abs.extend(original_indices)
                cols_abs.extend([query_idx] * len(original_indices))

        # Also include spectra with missing precursors, as they bypass the filter
        for i in np.where(ref_missing)[0]:
            rows_abs.extend([i] * len(query_mzs))
            cols_abs.extend(range(len(query_mzs)))
        for i in np.where(query_missing)[0]:
            rows_abs.extend(range(len(ref_mzs)))
            cols_abs.extend([i] * len(ref_mzs))

        return np.array(rows_abs), np.array(cols_abs)


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
        # If there are fewer than 2 unique intensity values, shuffling is ineffective.
        # Instead, taper the array to break structural correlation.
        if len(np.unique(shuffled_intensities)) < 2 and len(shuffled_intensities) > 1:
            shuffled_intensities = shuffled_intensities * np.linspace(
                1.0, 0.1, len(shuffled_intensities)
            )
        else:
            original_intensities = shuffled_intensities.copy()
            rng.shuffle(shuffled_intensities)
            # Post-shuffle check to ensure it's not identical (for low peak counts)
            if np.array_equal(shuffled_intensities, original_intensities):
                shuffled_intensities = np.roll(shuffled_intensities, 1)

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
    """
    if len(decoy_scores) == 0:
        return np.ones_like(target_scores)

    # Vectorized computation of instances where decoy score >= target score
    greater_equal_decoys = np.sum(decoy_scores >= target_scores[:, None], axis=1)

    # Apply +1 pseudo-count to numerator and denominator
    p_values = (greater_equal_decoys + 1) / (len(decoy_scores) + 1)
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

        # MS1 Pre-filtering for cosine with sparse array support
        if self.config.algorithm == "cosine" and hasattr(
            self.similarity_function, "sparse_array"
        ):
            ms1_tol = getattr(self.config, "ms1_tolerance", 0.02)
            res_ppm = getattr(self.config, "resolution_ppm", None)
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
            # Calculate scores natively for modified cosine
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
                    if (
                        q_rt is not None
                        and ref_rt is not None
                        and abs(float(q_rt) - float(ref_rt)) > self.config.rt_tolerance
                    ):
                        continue

                score_val = float(numeric_scores[idx, i])
                match_val = int(matches_count[idx, i])

                ref_mz = ref.get("precursor_mz")
                ref_mz_val = float(ref_mz) if ref_mz is not None else None

                is_decoy = bool(ref.get("is_decoy", False)) or (idx >= n_targets)

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
                        "score_breakdown": None,
                    }
                )

        return results


def get_similarity_engine(config: SimilarityConfig) -> SimilarityEngine:
    """Factory function to instantiate the similarity engine.

    Returns a `SimilarityEngine` configured with the algorithm specified
    in the config (cosine or modified_cosine).

    Parameters
    ----------
    config : SimilarityConfig
        The configuration object detailing the algorithm choice.

    Returns
    -------
    SimilarityEngine
        The instantiated engine ready to perform `.search()`.
    """
    return SimilarityEngine(config)
