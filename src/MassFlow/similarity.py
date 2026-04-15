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
from typing import List, TypedDict

import numpy as np
from matchms import Spectrum, calculate_scores

try:
    from matchms.similarity import CosineGreedy, ModifiedCosine
except ImportError:
    from matchms.similarity import CosineGreedy
    from matchms.similarity import ModifiedCosineGreedy as ModifiedCosine

from MassFlow.config import SimilarityConfig

logger = logging.getLogger(__name__)


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
    annotation_tier: str | None


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
        rng.shuffle(shuffled_intensities)

        decoy_spec = Spectrum(
            mz=spec.peaks.mz.copy(),
            intensities=shuffled_intensities,
            metadata=decoy_metadata,
        )
        decoys.append(decoy_spec)
    return decoys


def calculate_fdr(
    target_scores: np.ndarray, decoy_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate q-values (FDR) for target scores.
    Returns (sorted_scores, q_values, is_target)
    """
    if len(target_scores) == 0 and len(decoy_scores) == 0:
        return np.array([]), np.array([]), np.array([], dtype=bool)

    all_scores = np.concatenate([target_scores, decoy_scores])
    is_target = np.concatenate(
        [
            np.ones_like(target_scores, dtype=bool),
            np.zeros_like(decoy_scores, dtype=bool),
        ]
    )

    # Sort descending
    sort_idx = np.argsort(all_scores)[::-1]
    sorted_scores = all_scores[sort_idx]
    sorted_is_target = is_target[sort_idx]

    cum_targets = np.cumsum(sorted_is_target)
    cum_decoys = np.cumsum(~sorted_is_target)

    fdr = np.zeros_like(cum_targets, dtype=float)
    valid = cum_targets > 0
    fdr[valid] = cum_decoys[valid] / cum_targets[valid]

    # Q-value is the minimum FDR for all lower scores (which appear after in the descending array)
    q_values = np.minimum.accumulate(fdr[::-1])[::-1]

    return sorted_scores, q_values, sorted_is_target


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
            self.similarity_function = CosineGreedy(tolerance=self.config.tolerance)
        elif self.config.algorithm == "modified_cosine":
            self.similarity_function = ModifiedCosine(tolerance=self.config.tolerance)
        elif self.config.algorithm == "spec2vec":
            try:
                import gensim
                from spec2vec import Spec2Vec
            except ImportError:
                raise ImportError(
                    "spec2vec is required for Spec2Vec similarity. Install it with 'pip install spec2vec'."
                )

            if not self.config.model_path or not self.config.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.config.model_path}"
                )

            model = gensim.models.Word2Vec.load(str(self.config.model_path))
            self.similarity_function = Spec2Vec(
                model=model,
                intensity_weighting_power=0.5,
                allowed_missing_percentage=5.0,
            )

        elif self.config.algorithm == "ms2deepscore":
            try:
                from ms2deepscore import MS2DeepScore
                from ms2deepscore.models import load_model
            except ImportError:
                raise ImportError(
                    "ms2deepscore is required for MS2DeepScore similarity. Install it with 'pip install ms2deepscore'."
                )

            if not self.config.model_path or not self.config.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.config.model_path}"
                )

            model = load_model(str(self.config.model_path))
            self.similarity_function = MS2DeepScore(model=model)

        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
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

        Returns
        -------
        List[SearchResult]
            A list of dictionaries, where each dictionary represents a successful match
            and contains relevant metadata (e.g., query ID, reference name, score, SMILES).
        """
        if not query_spectra or not reference_spectra:
            return []

        cutoff = min_score if min_score is not None else self.config.min_score

        # Generate decoys and combine
        decoy_spectra = generate_decoys(reference_spectra)
        all_references = reference_spectra + decoy_spectra
        n_targets = len(reference_spectra)
        n_queries = len(query_spectra)

        # MS1 Pre-filtering for standard cosine
        if self.config.algorithm == "cosine" and hasattr(
            self.similarity_function, "sparse_array"
        ):
            ref_mzs = np.array(
                [float(s.get("precursor_mz") or 0.0) for s in all_references]
            )
            query_mzs = np.array(
                [float(q.get("precursor_mz") or 0.0) for q in query_spectra]
            )

            # Avoid division by zero
            safe_query_mzs = np.where(query_mzs == 0, np.nan, query_mzs)

            with np.errstate(divide="ignore", invalid="ignore"):
                ppm_diff = (
                    np.abs(ref_mzs[:, None] - safe_query_mzs[None, :])
                    / safe_query_mzs[None, :]
                    * 1e6
                )

            ms1_tol = getattr(self.config, "ms1_tolerance", 10.0)
            mask = ppm_diff <= ms1_tol
            idx_row, idx_col = np.where(mask)

            if len(idx_row) > 0:
                sparse_results = self.similarity_function.sparse_array(
                    all_references, query_spectra, idx_row, idx_col, is_symmetric=False
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
            scores_obj = calculate_scores(
                references=all_references,  # type: ignore
                queries=query_spectra,  # type: ignore
                similarity_function=self.similarity_function,
                is_symmetric=False,
            )

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

                is_decoy = idx >= n_targets

                results.append(
                    {
                        "query_id": q_id,
                        "query_precursor_mz": q_mz_val,
                        "reference_id": str(ref.get("id")),
                        "reference_name": str(
                            ref.get("compound_name") or ref.get("name")
                        ),
                        "reference_precursor_mz": ref_mz_val,
                        "score": round(score_val, 4),
                        "matched_peaks": match_val,
                        "smiles": str(ref.get("smiles")) if ref.get("smiles") else None,
                        "inchikey": (
                            str(ref.get("inchikey")) if ref.get("inchikey") else None
                        ),
                        "is_decoy": is_decoy,
                        "q_value": 1.0,  # Placeholder, computed globally later
                        "annotation_tier": None,
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

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
    ) -> List[SearchResult]:
        if not query_spectra or not reference_spectra:
            return []

        self.logger.info(f"Executing Tier 1 Cascade: {self.config.cascade_tier1}")
        # Run Tier 1 without strict filtering to see full distribution
        # We manually bypass the internal min_score to grab the full matrix
        original_min = self.tier1_engine.config.min_score
        self.tier1_engine.config.min_score = 0.0
        tier1_raw_results = self.tier1_engine.search(
            query_spectra, reference_spectra, min_score=0.0, top_n=None
        )
        self.tier1_engine.config.min_score = original_min

        # Group Tier 1 results by query
        t1_grouped = defaultdict(list)
        for res in tier1_raw_results:
            t1_grouped[res["query_id"]].append(res)

        final_results = []
        gray_zone_queries = []

        # Route queries based on their MAX score in Tier 1
        for q in query_spectra:
            q_id = str(q.get("id"))
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
                gray_zone_queries, reference_spectra, min_score=min_score, top_n=top_n
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

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
    ) -> List[SearchResult]:
        """
        Run a consensus similarity search.

        This method aggregates similarity scores from multiple underlying engines
        according to their predefined weights. For standard cosine algorithms, it
        uses MS1 precursor pre-filtering to optimize execution. Decoys are generated
        once and processed across all algorithms.

        Parameters
        ----------
        query_spectra : List[matchms.Spectrum]
            A list of experimental spectrum objects to be annotated.
        reference_spectra : List[matchms.Spectrum]
            A list of reference spectrum objects forming the library.
        min_score : float or None, optional
            An optional override for the minimum consensus score threshold.
        top_n : int or None, optional
            An optional override for the maximum number of results to return per
            query spectrum.

        Returns
        -------
        List[SearchResult]
            A list of structured search results containing the aggregated score
            and maximum matched peaks across the configured engines.
        """
        if not query_spectra or not reference_spectra:
            return []

        cutoff = min_score if min_score is not None else self.min_score

        # Generate decoys and combine
        decoy_spectra = generate_decoys(reference_spectra)
        all_references = reference_spectra + decoy_spectra
        n_targets = len(reference_spectra)
        n_queries = len(query_spectra)
        n_refs = len(all_references)

        consensus_scores = np.zeros((n_refs, n_queries), dtype=float)
        consensus_matches = np.full((n_refs, n_queries), -1, dtype=int)

        for engine, weight in self.engines:
            # Check if this specific engine is a candidate for MS1 pre-filtering
            if engine.config.algorithm == "cosine" and hasattr(
                engine.similarity_function, "sparse_array"
            ):
                ref_mzs = np.array(
                    [float(s.get("precursor_mz") or 0.0) for s in all_references]
                )
                query_mzs = np.array(
                    [float(q.get("precursor_mz") or 0.0) for q in query_spectra]
                )
                safe_query_mzs = np.where(query_mzs == 0, np.nan, query_mzs)

                with np.errstate(divide="ignore", invalid="ignore"):
                    ppm_diff = (
                        np.abs(ref_mzs[:, None] - safe_query_mzs[None, :])
                        / safe_query_mzs[None, :]
                        * 1e6
                    )

                ms1_tol = getattr(engine.config, "ms1_tolerance", 10.0)
                mask = ppm_diff <= ms1_tol
                idx_row, idx_col = np.where(mask)

                if len(idx_row) > 0:
                    sparse_results = engine.similarity_function.sparse_array(
                        all_references,
                        query_spectra,
                        idx_row,
                        idx_col,
                        is_symmetric=False,
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
                scores_obj = calculate_scores(
                    references=all_references,  # type: ignore
                    queries=query_spectra,  # type: ignore
                    similarity_function=engine.similarity_function,
                    is_symmetric=False,
                )

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

                is_decoy = idx >= n_targets

                results.append(
                    {
                        "query_id": q_id,
                        "query_precursor_mz": q_mz_val,
                        "reference_id": str(ref.get("id")),
                        "reference_name": str(
                            ref.get("compound_name") or ref.get("name")
                        ),
                        "reference_precursor_mz": ref_mz_val,
                        "score": round(score_val, 4),
                        "matched_peaks": match_val,
                        "smiles": str(ref.get("smiles")) if ref.get("smiles") else None,
                        "inchikey": (
                            str(ref.get("inchikey")) if ref.get("inchikey") else None
                        ),
                        "is_decoy": is_decoy,
                        "q_value": 1.0,
                        "annotation_tier": "Standard",
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
