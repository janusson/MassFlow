"""
Compute spectra similarities: Cosine and Modified Cosine scores.
Ported from original_source/massflow_pipeline.py.
"""
from __future__ import annotations

import logging

from typing import Any, List, Tuple
from matchms import Spectrum, calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine

logger = logging.getLogger(__name__)


def calculate_cosscores(reference_spectra_list: List[Spectrum], query_spectra_list: List[Spectrum], tolerance: float = 0.005) -> Any:
    """
    Calculate cosine similarity scores for all query spectra against target library spectra.
    
    Args:
        reference_spectra_list: List of reference Spectrum objects.
        query_spectra_list: List of query Spectrum objects.
        tolerance: Tolerance for mz matching.
    
    Returns:
        matchms Scores object.
    """
    # Check is ref/query spectra are symmetric to speed up import with is_symmetric = True
    is_symmetric = (len(reference_spectra_list) == len(query_spectra_list)) and (reference_spectra_list == query_spectra_list)

    similarity_measure = CosineGreedy(tolerance)
    cosine_scores = calculate_scores(
        reference_spectra_list,
        query_spectra_list,
        similarity_measure,
        is_symmetric=is_symmetric,
    )
    return cosine_scores


def top10_cosine_matches(reference_library: List[Spectrum], query_spectra: List[Spectrum], tolerance: float = 0.005) -> Any:
    """
    Print top ten matching peaks between query spectra and reference library spectra.
    Matches are sorted by Cosine similarity.
    
    Args:
        reference_library: List of reference Spectrum objects.
        query_spectra: List of query Spectrum objects.
        tolerance: Tolerance for mz matching.
        
    Returns:
        matchms Scores object.
    """
    scores = calculate_cosscores(reference_library, query_spectra, tolerance=tolerance)
    


    # Iterate over queries and find best matches
    for i, query in enumerate(query_spectra):
        best_matches = scores.scores_by_query(query, "CosineGreedy_score", sort=True)[:10]


        
        logger.info(f"Top 10 matches for query {i} (Cosine score, matches):")
        logger.info([x[1] for x in best_matches])
        
        top10_smiles = [x[0].get("smiles") for x in best_matches]
        logger.info(f"Top 10 SMILES: {top10_smiles}")
        
    # TODO: This original function signature was a bit specific to a loop in the pipeline.
    # We might want to return a structured result here instead of just printing.
    return scores


def threshold_matches(reference_library: List[Spectrum], check_spectra: List[Spectrum], min_match: int, tolerance: float = 0.005) -> Tuple[List[Any], List[str | None]]:
    """
    Get matches with a minimum number of matching peaks.
    
    Args:
        reference_library: List of reference Spectrum objects.
        check_spectra: List of query/check Spectrum objects.
        min_match: Minimum number of matched peaks required.
        tolerance: Tolerance for mz matching.
    
    Returns:
        Tuple containing list of match objects and list of SMILES strings.
    """
    similarity_measure = CosineGreedy(tolerance)
    scores = calculate_scores(
        reference_library, check_spectra, similarity_measure, is_symmetric=False
    )

    # Note: original code hardcoded reference_library[5] for sorting, which seems like a bug or specific test case.
    # We will generalize this to return all matches over the threshold for the first query as a robust default,
    # or arguably we should return the whole scores object.
    # For fidelity to the spirit of the function, we'll try to match the "top N" behavior.
    
    if not check_spectra:
        return [], []



    # Using the first query spectrum as the target for sorting/filtering, assuming 1:N or 1:1 check context
    query = check_spectra[0]
    sorted_matches = scores.scores_by_query(query, "CosineGreedy_score", sort=True)
    
    matches_over_limit = [x for x in sorted_matches if x[1]["matches"] >= min_match][:10]


    matches_over_limit_smiles = [x[0].get("smiles") for x in matches_over_limit]

    return matches_over_limit, matches_over_limit_smiles


def modified_cosine_scores(reference_library: List[Spectrum], check_spectra: List[Spectrum], tolerance: float = 0.005) -> Any:
    """
    Computes modified cosine scores for all spectra in check_spectra against reference library spectra.
    
    Args:
        reference_library: List of reference Spectrum objects.
        check_spectra: List of query/check Spectrum objects.
        tolerance: Tolerance for mz matching.
        
    Returns:
        matchms Scores object.
    """
    similarity_measure = ModifiedCosine(tolerance=tolerance)
    scores = calculate_scores(
        reference_library, check_spectra, similarity_measure, is_symmetric=False
    )
    return scores
