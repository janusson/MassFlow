"""
Spectral similarity search engine for MassFlow.

This module encapsulates the logic for comparing experimental mass spectra against
reference libraries. It provides a unified interface (`SimilarityEngine`) to various
similarity algorithms (Cosine, Modified Cosine, Spec2Vec, MS2DeepScore) backed by
matchms. It handles model loading, vectorized score calculation, and result
filtering/formatting.
"""

from typing import Any, List, TypedDict

import numpy as np
from matchms import Spectrum, calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine

from MassFlow.config import SimilarityConfig


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

        # Calculate Scores
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
                        "inchikey": str(ref.get("inchikey"))
                        if ref.get("inchikey")
                        else None,
                        "is_decoy": is_decoy,
                        "q_value": 1.0,  # Placeholder, computed globally later
                    }
                )

        return results
