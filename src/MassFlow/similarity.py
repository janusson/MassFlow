"""
Spectral similarity search engine for MassFlow.

This module encapsulates the logic for comparing experimental mass spectra against
reference libraries. It provides a unified interface (`SimilarityEngine`) to various
similarity algorithms (Cosine, Modified Cosine, Spec2Vec, MS2DeepScore) backed by
matchms. It handles model loading, vectorized score calculation, and result
filtering/formatting.
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
    ref_missing = np.array([r is None for r in ref_mzs_raw], dtype=bool)
    query_missing = np.array([q is None for q in query_mzs_raw], dtype=bool)

    ref_mzs = np.array([float(r) if r is not None else 0.0 for r in ref_mzs_raw])
    query_mzs = np.array([float(q) if q is not None else 0.0 for q in query_mzs_raw])

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
    def __init__(self, config: SimilarityConfig):
        """
        Initialize the similarity search engine.

        This method configures the underlying matchms similarity function based on
        the provided configuration. It supports 'cosine', 'modified_cosine',
        'spec2vec', and 'ms2deepscore'. For machine learning models (Spec2Vec,
        MS2DeepScore), it handles loading the model from the specified path.

        Parameters
        ----------
        config : SimilarityConfig
            The configuration object detailing the algorithm choice, tolerances,
            and paths to required model files.

        Raises
        ------
        ImportError
            If a required external library (e.g., 'spec2vec', 'ms2deepscore') is
            not installed.
        FileNotFoundError
            If the specified model path for ML-based algorithms does not exist.
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
        elif self.config.algorithm == "spec2vec":
            if not self.config.model_path or not self.config.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.config.model_path}"
                )

            try:
                import gensim
                from spec2vec import Spec2Vec
            except ImportError:
                raise ImportError(
                    "spec2vec is required for Spec2Vec similarity. Install it with 'pip install massflow[ml]'."
                )

            model = gensim.models.Word2Vec.load(str(self.config.model_path))
            self.similarity_function = Spec2Vec(
                model=model,
                intensity_weighting_power=0.5,
                allowed_missing_percentage=5.0,
            )

        elif self.config.algorithm == "ms2deepscore":
            if not self.config.model_path or not self.config.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.config.model_path}"
                )

            try:
                from ms2deepscore import MS2DeepScore
                from ms2deepscore.models import load_model
            except ImportError:
                raise ImportError(
                    "ms2deepscore is required for MS2DeepScore similarity. Install it with 'pip install massflow[ml]'."
                )

            model = load_model(str(self.config.model_path))
            self.similarity_function = MS2DeepScore(model=model)

        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

    def _search_spec2vec_indexed(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
    ) -> List[SearchResult]:
        """
        Run an indexed similarity search for spec2vec using a BallTree.
        """
        from sklearn.neighbors import BallTree
        from sklearn.preprocessing import normalize

        if not hasattr(self.similarity_function, "_calculate_vector"):
            raise TypeError(
                "The configured similarity function is not a valid spec2vec instance."
            )

        cutoff = min_score if min_score is not None else self.config.min_score
        radius = np.sqrt(2 - 2 * cutoff)

        if include_decoys:
            decoy_spectra = generate_decoys(reference_spectra)
            all_references = reference_spectra + decoy_spectra
        else:
            all_references = reference_spectra

        ref_embeddings = np.array(
            [self.similarity_function._calculate_vector(s) for s in all_references]
        )
        ref_embeddings_normalized = normalize(ref_embeddings, axis=1, norm="l2")
        tree = BallTree(ref_embeddings_normalized, metric="euclidean")

        results: List[SearchResult] = []

        for i, query_spec in enumerate(query_spectra):
            query_embedding = self.similarity_function._calculate_vector(
                query_spec
            ).reshape(1, -1)
            query_embedding_normalized = normalize(query_embedding, axis=1, norm="l2")

            indices, distances = tree.query_radius(
                query_embedding_normalized, r=radius, return_distance=True
            )

            indices, distances = indices[0], distances[0]

            if len(indices) == 0:
                continue

            scores = 1 - (distances**2) / 2

            sorted_indices = np.argsort(scores)[::-1]
            if top_n is not None:
                sorted_indices = sorted_indices[:top_n]

            q_id = str(query_spec.get("id", f"query_{i}"))
            q_mz = query_spec.get("precursor_mz")
            q_mz_val = float(q_mz) if q_mz is not None else None
            q_smiles = query_spec.get("smiles")

            for idx in sorted_indices:
                original_ref_idx = indices[idx]
                ref = all_references[original_ref_idx]
                score_val = scores[idx]

                ref_mz = ref.get("precursor_mz")
                ref_mz_val = float(ref_mz) if ref_mz is not None else None
                ref_smiles = ref.get("smiles")

                is_decoy = bool(
                    original_ref_idx >= len(reference_spectra)
                    if include_decoys
                    else False
                )

                structural_sim = None
                if q_smiles and ref_smiles and not is_decoy:
                    from MassFlow.cheminformatics import calculate_tanimoto_similarity

                    structural_sim = calculate_tanimoto_similarity(q_smiles, ref_smiles)

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
                        "matched_peaks": -1,  # Not applicable for spec2vec
                        "smiles": str(ref_smiles) if ref_smiles else None,
                        "inchikey": str(ref.get("inchikey"))
                        if ref.get("inchikey")
                        else None,
                        "is_decoy": is_decoy,
                        "q_value": 1.0,
                        "p_value": None,
                        "annotation_tier": None,
                        "structural_similarity": structural_sim,
                        "score_breakdown": None,
                    }
                )
        return results

    @_handle_lazy_reference_spectra
    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
    ) -> List[SearchResult]:
        """
        Run a similarity search of query spectra against a reference library.

        This method computes a full NxM similarity matrix between the reference
        spectra and the query spectra using the configured algorithm. It leverages
        vectorized numpy operations to efficiently filter the results based on
        minimum score and minimum matched peaks (for cosine-based algorithms).
        It then extracts the top N matches for each query and compiles the metadata
        into a structured list.

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
            query spectrum. If None, all matches exceeding the thresholds are returned.
        include_decoys : bool, optional
            If True, generate and search against decoy spectra for FDR calculation.
            If False, search only against the provided reference_spectra. Default is True.

        Returns
        -------
        List[SearchResult]
            A list of dictionaries, where each dictionary represents a successful match
            and contains relevant metadata (e.g., query ID, reference name, score, SMILES).
        """
        if not query_spectra or not reference_spectra:
            return []

        cutoff = min_score if min_score is not None else self.config.min_score

        if include_decoys:
            # Generate decoys and combine
            ref_list = list(reference_spectra)
            decoy_spectra = generate_decoys(ref_list)
            all_references = ref_list + decoy_spectra
            n_targets = len(ref_list)
        else:
            all_references = list(reference_spectra)
            n_targets = len(all_references)

        n_queries = len(query_spectra)

        # MS1 Pre-filtering for standard cosine
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
            # Calculate Scores natively (for Modified Cosine and ML models)
            # matchms types are invariant, so we ignore type checking for list covariance here
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

        # Deconstruct structured array (Cosine/ModifiedCosine) or standard array (ML models)
        if (
            hasattr(scores_array.dtype, "names")
            and scores_array.dtype.names is not None
        ):
            # find score and matches fields dynamically based on algo name
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
        else:
            numeric_scores = scores_array.astype(float)
            # Use -1 to indicate "not applicable" for ML models
            matches_count = np.full_like(numeric_scores, -1, dtype=int)

        n_queries = len(query_spectra)

        # Iterate per query to apply top_n filtering
        # Note: Filtering is vectorized per query column
        for i in range(n_queries):
            query_scores = numeric_scores[:, i]
            query_matches = matches_count[:, i]

            # 1. Score threshold mask
            mask = query_scores >= cutoff

            # 2. Match count threshold mask (if applicable)
            if self.config.min_matched_peaks > 0:
                # If matches_count is -1 (ML models), we don't filter by matched peaks
                mask &= (query_matches >= self.config.min_matched_peaks) | (
                    query_matches == -1
                )

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

            for idx in final_indices:
                ref = all_references[idx]
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


# EXPERIMENTAL: CascadeEngine is not part of the stable v1.0 contract.
class CascadeEngine:
    """
    Smart routing similarity engine that probabilistically routes queries to save compute.

    Tier 1 (e.g., Cosine) is run on all queries.
    If a query has a max score >= upper_bound, it is annotated via Tier 1.
    If a query has a max score < lower_bound, it is discarded (noise).
    If a query falls in the gray zone (lower_bound <= max < upper_bound), it is routed
    to Tier 2 (e.g., MS2DeepScore) for deep structural elucidation.
    """

    def __init__(self, config: SimilarityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize Tier 1 (Classical Fast)
        tier1_config = config.model_copy(update={"algorithm": config.cascade_tier1})
        self.tier1_engine = SimilarityEngine(tier1_config)

        # Initialize Tier 2 (Heavy ML)
        tier2_config = config.model_copy(update={"algorithm": config.cascade_tier2})
        self.tier2_engine = SimilarityEngine(tier2_config)

    @_handle_lazy_reference_spectra
    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
    ) -> List[SearchResult]:
        if not query_spectra or not reference_spectra:
            return []

        self.logger.info(f"Executing Tier 1 Cascade: {self.config.cascade_tier1}")
        # Run Tier 1 without strict filtering to see full distribution
        # We manually bypass the internal min_score to grab the full matrix
        original_min = self.tier1_engine.config.min_score
        self.tier1_engine.config.min_score = 0.0
        tier1_raw_results = self.tier1_engine.search(
            query_spectra,
            reference_spectra,
            min_score=0.0,
            top_n=None,
            include_decoys=include_decoys,
        )
        self.tier1_engine.config.min_score = original_min

        # Group Tier 1 results by query
        t1_grouped = defaultdict(list)
        for res in tier1_raw_results:
            t1_grouped[res["query_id"]].append(res)

        final_results = []
        gray_zone_queries = []

        # Route queries based on their MAX score in Tier 1
        for i, q in enumerate(query_spectra):
            q_id = str(q.get("id", f"query_{i}"))
            q_results = t1_grouped.get(q_id, [])

            if not q_results:
                continue

            max_t1_score = max(r["score"] for r in q_results)

            if max_t1_score >= self.config.cascade_upper_bound:
                # Exact match found. Keep the top N from Tier 1 that exceed the upper bound
                valid_t1 = [
                    r
                    for r in q_results
                    if r["score"] >= self.config.cascade_upper_bound
                ]
                valid_t1.sort(key=lambda x: x["score"], reverse=True)
                for res in valid_t1[:top_n]:
                    res["annotation_tier"] = f"Tier 1 ({self.config.cascade_tier1})"
                    final_results.append(res)
            elif max_t1_score >= self.config.cascade_lower_bound:
                # Gray zone analog. Route to Tier 2
                gray_zone_queries.append(q)
            # Else: < lower_bound. Discarded as noise.

        if gray_zone_queries:
            self.logger.info(
                f"Routing {len(gray_zone_queries)} queries to Tier 2 Cascade: {self.config.cascade_tier2}"
            )
            tier2_results = self.tier2_engine.search(
                gray_zone_queries,
                reference_spectra,
                min_score=min_score,
                top_n=top_n,
                include_decoys=include_decoys,
            )
            for res in tier2_results:
                res["annotation_tier"] = f"Tier 2 ({self.config.cascade_tier2})"
                final_results.append(res)
        else:
            self.logger.info(
                "No queries fell into the analog gray zone. Tier 2 bypassed."
            )

        return final_results


# EXPERIMENTAL: ConsensusEngine is not part of the stable v1.0 contract.
class ConsensusEngine:
    """
    Similarity search engine that aggregates scores from multiple metric algorithms.
    """

    def __init__(
        self, engines: list[tuple[SimilarityEngine, float]], min_score: float = 0.6
    ):
        """
        Initialize the consensus search engine with multiple SimilarityEngines and their weights.

        Parameters
        ----------
        engines : list of tuples
            A list where each tuple is (SimilarityEngine, weight).
            Example: [(CosineEngine, 0.4), (MS2DSEngine, 0.6)]
        min_score : float
            The minimum consensus score threshold.
        """
        self.engines = engines
        self.min_score = min_score

        total_weight = sum(w for _, w in engines)
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Engine weights sum to {total_weight}, not 1.0")

    @_handle_lazy_reference_spectra
    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
    ) -> List[SearchResult]:
        """
        Run a consensus similarity search.

        This method aggregates similarity scores from multiple underlying engines
        according to their predefined weights.
        """
        if not query_spectra or not reference_spectra:
            return []

        cutoff = min_score if min_score is not None else self.min_score

        if include_decoys:
            # Generate decoys and combine
            ref_list: List[Spectrum] = list(reference_spectra)
            decoy_spectra: List[Spectrum] = generate_decoys(ref_list)
            all_references: List[Spectrum] = ref_list + decoy_spectra
            n_targets: int = len(ref_list)
        else:
            all_references = list(reference_spectra)
            n_targets = len(all_references) // 2

        n_queries = len(query_spectra)

        # Run all engines and gather results
        for engine, weight in self.engines:
            logger.info(f"Running consensus component: {engine.config.algorithm}")
            # Run without strict filtering to ensure we have matches to aggregate
            engine.config.min_score = 0.0
            engine.search(
                query_spectra,
                reference_spectra,
                min_score=0.0,
                top_n=None,
                include_decoys=include_decoys,
            )
        n_refs = len(all_references)

        consensus_scores = np.zeros((n_refs, n_queries), dtype=float)
        consensus_matches = np.full((n_refs, n_queries), -1, dtype=int)

        # Store individual score arrays for the breakdown
        engine_score_arrays = {}

        for engine, weight in self.engines:
            # Check if this specific engine is a candidate for MS1 pre-filtering
            if engine.config.algorithm == "cosine" and hasattr(
                engine.similarity_function, "sparse_array"
            ):
                ms1_tol = getattr(engine.config, "ms1_tolerance", 0.02)
                res_ppm = getattr(engine.config, "resolution_ppm", None)
                idx_row, idx_col = _ms1_prefilter(
                    all_references, query_spectra, ms1_tol, res_ppm
                )

                if len(idx_row) > 0:
                    sparse_results = engine.similarity_function.sparse_array(
                        all_references,
                        query_spectra,
                        idx_row,
                        idx_col,
                        is_symmetric=False,
                        progress_bar=False,
                    )
                    scores_array = np.zeros(
                        (n_refs, n_queries),
                        dtype=engine.similarity_function.score_datatype,
                    )
                    scores_array[idx_row, idx_col] = sparse_results
                else:
                    scores_array = np.zeros(
                        (n_refs, n_queries),
                        dtype=engine.similarity_function.score_datatype,
                    )
            else:
                # Calculate Scores natively
                try:
                    scores_obj = calculate_scores(
                        references=all_references,  # type: ignore
                        queries=query_spectra,  # type: ignore
                        similarity_function=engine.similarity_function,
                        is_symmetric=False,
                        array_type="sparse",
                    )
                except Exception as e:
                    logger.error(
                        f"Consensus vectorized similarity calculation failed: {e}",
                        extra={
                            "step": "calculate_scores_consensus",
                            "num_queries": len(query_spectra),
                            "num_references": len(all_references),
                            "engine_algorithm": engine.config.algorithm,
                        },
                        exc_info=True,
                    )
                    raise

                scores_data = scores_obj.scores
                if hasattr(scores_data, "to_array"):
                    scores_array = scores_data.to_array()
                else:
                    scores_array = np.asarray(scores_data)

            if (
                hasattr(scores_array.dtype, "names")
                and scores_array.dtype.names is not None
            ):
                score_cols = [
                    c for c in scores_array.dtype.names if "score" in c.lower()
                ]
                match_cols = [
                    c for c in scores_array.dtype.names if "matches" in c.lower()
                ]

                numeric_scores = scores_array[score_cols[0]].astype(float)
                matches_count = scores_array[match_cols[0]]

                # Keep maximum matches count across engines if there are multiple that provide it
                mask = matches_count > consensus_matches
                consensus_matches[mask] = matches_count[mask]
            else:
                numeric_scores = scores_array.astype(float)

            engine_score_arrays[engine.config.algorithm] = numeric_scores
            consensus_scores += numeric_scores * weight

        results: List[SearchResult] = []

        for i in range(n_queries):
            query_scores = consensus_scores[:, i]
            query_matches = consensus_matches[:, i]

            mask = query_scores >= cutoff

            # Apply min_matched_peaks from the most strict engine
            min_matched = max(
                (
                    e.config.min_matched_peaks
                    for e, _ in self.engines
                    if e.config.min_matched_peaks > 0
                ),
                default=0,
            )
            if min_matched > 0:
                mask &= (query_matches >= min_matched) | (query_matches == -1)

            valid_indices = np.where(mask)[0]

            if valid_indices.size == 0:
                continue

            valid_scores = query_scores[valid_indices]
            sorted_idx_rel = np.argsort(valid_scores)[::-1]

            if top_n is not None:
                sorted_idx_rel = sorted_idx_rel[:top_n]

            final_indices = valid_indices[sorted_idx_rel]

            q = query_spectra[i]
            q_id = str(q.get("id", f"query_{i}"))
            q_mz = q.get("precursor_mz")
            q_mz_val = float(q_mz) if q_mz is not None else None

            for idx in final_indices:
                ref = all_references[idx]
                score_val = float(consensus_scores[idx, i])
                match_val = int(consensus_matches[idx, i])

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
                        "q_value": 1.0,
                        "p_value": None,
                        "annotation_tier": None,
                        "structural_similarity": None,
                        "score_breakdown": None,
                    }
                )

        return results


def get_similarity_engine(
    config: SimilarityConfig,
) -> SimilarityEngine | ConsensusEngine | CascadeEngine:
    """
    Factory function to instantiate the appropriate similarity engine.

    Depending on the configured algorithm ('cosine', 'ms2deepscore', 'consensus', or 'cascade'),
    this function returns either a standard `SimilarityEngine` or a composite
    `ConsensusEngine` or `CascadeEngine` containing multiple initialized sub-engines.

    Note: 'ms2deepscore', 'spec2vec', 'consensus', and 'cascade' engines are
    considered experimental and are not part of the stable v1.0 contract.

    Parameters
    ----------
    config : SimilarityConfig
        The configuration object detailing the primary algorithm or consensus weights.

    Returns
    -------
    SimilarityEngine or ConsensusEngine
        The instantiated engine ready to perform `.search()`.

    Raises
    ------
    ValueError
        If 'consensus' is selected but no consensus weights are provided in the config.
    """
    if config.algorithm == "cascade":
        return CascadeEngine(config)
    elif config.algorithm == "consensus":
        # If consensus_weights are missing, either fall back or raise depending
        # on the SimilarityConfig.allow_consensus_fallback flag. Falling back
        # provides a sensible runtime default (single cosine engine) for legacy
        # or minimal configs, while disabling the fallback enforces strict config.
        if not config.consensus_weights:
            # Default to permissive fallback when the config field is absent
            allow_fallback = getattr(config, "allow_consensus_fallback", True)
            if allow_fallback:
                logger.warning(
                    "consensus_weights not provided; allow_consensus_fallback is True -> falling back to a single 'cosine' engine for consensus."
                )
                # Use a single cosine engine with weight 1.0 as a sensible default
                weights_items = {"cosine": 1.0}.items()
            else:
                raise ValueError(
                    "consensus_weights must be provided for consensus algorithm. Set similarity.allow_consensus_fallback=True to enable an automatic fallback."
                )
        else:
            weights_items = config.consensus_weights.items()

        engines = []
        for algo, weight in weights_items:
            algo_config = config.model_copy(update={"algorithm": algo})
            engines.append((SimilarityEngine(algo_config), weight))

        return ConsensusEngine(engines=engines, min_score=config.min_score)
    else:
        return SimilarityEngine(config)
