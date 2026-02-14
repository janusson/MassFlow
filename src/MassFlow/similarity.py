"""
Similarity search engine for MassFlow.
Wraps matchms similarity measures for efficient querying.
"""

from typing import List, Any
from matchms import Spectrum, calculate_scores
from matchms.similarity import CosineGreedy

class SimilarityEngine:
    def __init__(self, tolerance: float = 0.01, mz_power: float = 0.0, intensity_power: float = 1.0):
        self.similarity_function = CosineGreedy(
            tolerance=tolerance, 
            mz_power=mz_power, 
            intensity_power=intensity_power
        )

    def search(
        self, 
        query_spectra: List[Spectrum], 
        reference_spectra: List[Spectrum], 
        min_score: float = 0.7,
        top_n: int = 5
    ) -> List[dict]:
        """
        Run similarity search of Query vs Reference.
        Returns a flat list of dictionaries suitable for CSV export.
        """
        if not query_spectra or not reference_spectra:
            return []

        scores = calculate_scores(
            references=reference_spectra,
            queries=query_spectra,
            similarity_function=self.similarity_function,
            is_symmetric=False
        )

        results = []

        for i, query in enumerate(query_spectra):
            query_id = query.get("id") or f"query_{i}"
            query_mz = query.get("precursor_mz")
            
            matches = scores.scores_by_query(query, sort=True)
            
            count = 0
            for reference, score_data in matches:
                score = score_data[0]
                matches_count = score_data[1]
                
                if score < min_score:
                    break
                
                if count >= top_n:
                    break

                results.append({
                    "query_id": query_id,
                    "query_precursor_mz": query_mz,
                    "reference_id": reference.get("id"),
                    "reference_name": reference.get("compound_name") or reference.get("name"),
                    "reference_precursor_mz": reference.get("precursor_mz"),
                    "score": round(float(score), 4),
                    "matched_peaks": int(matches_count),
                    "smiles": reference.get("smiles"),
                    "inchikey": reference.get("inchikey")
                })
                count += 1

        return results
