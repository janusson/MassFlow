"""
Scoring functions for Yogimass: cosine and modified cosine similarity.
"""

from types import SimpleNamespace

import numpy as np
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine
from yogimass.filters.metadata import ensure_name_metadata
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def _score_field_for_sort(scores) -> str | None:
    """
    Return the field name that should be used to sort multi-value score arrays.
    """
    score_names = getattr(scores, "score_names", None)
    if not score_names:
        return None
    for candidate in score_names:
        if candidate.endswith("_score") or candidate == "score":
            return candidate
    return score_names[0]


def _matches_field_name(scores) -> str | None:
    score_names = getattr(scores, "score_names", None)
    if not score_names:
        return None
    for candidate in score_names:
        if candidate.endswith("_matches") or candidate == "matches":
            return candidate
    return None


def _wrap_score_result(raw_match, score_field: str | None, matches_field: str | None):
    if hasattr(raw_match, "score") and hasattr(raw_match, "matches"):
        return raw_match
    if isinstance(raw_match, np.void):
        data = {"raw": raw_match}
        if score_field and score_field in raw_match.dtype.names:
            data["score"] = float(raw_match[score_field])
        if matches_field and matches_field in raw_match.dtype.names:
            data["matches"] = int(raw_match[matches_field])
        return SimpleNamespace(**data)
    return raw_match


def calculate_cosscores(reference_spectra_list, query_spectra_list, tolerance=0.005):
    """
    Calculate cosine scores between reference and query spectra collections.
    """
    similarity_measure = CosineGreedy(tolerance=tolerance)
    is_symmetric = reference_spectra_list is query_spectra_list
    cosine_scores = calculate_scores(
        reference_spectra_list,
        query_spectra_list,
        similarity_measure,
        is_symmetric=is_symmetric,
    )
    return cosine_scores


def top10_cosine_matches(reference_library, query_spectra, query_index=0, tolerance=0.005):
    """
    Return the SMILES strings for the top ten cosine matches of a query spectrum.
    """
    if not query_spectra:
        raise ValueError("query_spectra cannot be empty.")
    if query_index < 0 or query_index >= len(query_spectra):
        raise IndexError(f"query_index {query_index} is out of bounds.")

    scores = calculate_cosscores(reference_library, query_spectra, tolerance=tolerance)
    query_spectrum = query_spectra[query_index]
    score_field = _score_field_for_sort(scores)
    matches_field = _matches_field_name(scores)
    sort_kwargs = {"sort": True}
    if score_field:
        sort_kwargs["name"] = score_field
    raw_matches = scores.scores_by_query(query_spectrum, **sort_kwargs)[:10]
    best_matches = [
        (reference, _wrap_score_result(match, score_field, matches_field))
        for reference, match in raw_matches
    ]
    logger.info("Top ten best matches (cosine score, matching peaks):")
    logger.info(
        [
            (getattr(match, "score", None), getattr(match, "matches", None))
            for _, match in best_matches
        ]
    )
    for reference, _ in best_matches:
        ensure_name_metadata(reference)
    top10_smiles = [reference.get("smiles") for reference, _ in best_matches]
    logger.info(f"Top ten best matches (SMILES): {top10_smiles}")
    return top10_smiles


def threshold_matches(
    reference_library,
    check_spectra,
    min_match,
    tolerance=0.005,
    query_index=0,
    max_results=10,
):
    """
    Return matches whose number of aligned peaks exceeds ``min_match``.
    """
    if not check_spectra:
        raise ValueError("check_spectra cannot be empty.")
    if query_index < 0 or query_index >= len(check_spectra):
        raise IndexError(f"query_index {query_index} is out of bounds.")

    similarity_measure = CosineGreedy(tolerance=tolerance)
    scores = calculate_scores(
        reference_library, check_spectra, similarity_measure, is_symmetric=False
    )
    query_spectrum = check_spectra[query_index]
    score_field = _score_field_for_sort(scores)
    matches_field = _matches_field_name(scores)
    sort_kwargs = {"sort": True}
    if score_field:
        sort_kwargs["name"] = score_field
    raw_matches = scores.scores_by_query(query_spectrum, **sort_kwargs)
    sorted_matches = [
        (reference, _wrap_score_result(match, score_field, matches_field))
        for reference, match in raw_matches
    ]
    matches_over_limit = []
    for reference, match in sorted_matches:
        match_count = getattr(match, "matches", None)
        if match_count is None:
            continue
        if match_count >= min_match:
            ensure_name_metadata(reference)
            matches_over_limit.append((reference, match))
        if len(matches_over_limit) >= max_results:
            break
    matches_over_limit_smiles = [ref.get("smiles") for ref, _ in matches_over_limit]
    return matches_over_limit, matches_over_limit_smiles


def modified_cosine_scores(reference_library, check_spectra):
    similarity_measure = ModifiedCosine(tolerance=0.005)
    scores = calculate_scores(
        reference_library, check_spectra, similarity_measure, is_symmetric=False
    )
    return scores
