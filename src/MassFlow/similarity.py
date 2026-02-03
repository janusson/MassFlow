"""
Compute spectra similarities using Strategy pattern.
Supports Cosine and Modified Cosine scores.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from matchms import Scores, Spectrum, calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine

from MassFlow.config import SimilarityConfig

logger = logging.getLogger(__name__)


class SimilarityCalculator(ABC):
    """Abstract base class for similarity calculation strategies."""

    @abstractmethod
    def calculate(self, references: list[Spectrum], queries: list[Spectrum]) -> Scores:
        """
        Calculate similarity scores between reference and query spectra.

        Args:
            references: List of reference Spectrum objects.
            queries: List of query Spectrum objects.

        Returns:
            Scores: matchms Scores object containing the results.
        """
        pass


class CosineSimilarity(SimilarityCalculator):
    """Strategy for Cosine Similarity (CosineGreedy)."""

    def __init__(self, tolerance: float, min_matched_peaks: int):
        """
        Initialize the Cosine Similarity calculator.

        Args:
            tolerance: Tolerance for m/z matching.
            min_matched_peaks: Minimum number of matched peaks.
        """
        self.tolerance = tolerance
        self.min_matched_peaks = min_matched_peaks
        self.similarity_measure = CosineGreedy(tolerance=self.tolerance)

    def calculate(self, references: list[Spectrum], queries: list[Spectrum]) -> Scores:
        """Execute calculation using CosineGreedy."""
        is_symmetric = references is queries
        return calculate_scores(
            references,
            queries,
            self.similarity_measure,
            is_symmetric=is_symmetric,
        )


class ModifiedCosineSimilarity(SimilarityCalculator):
    """Strategy for Modified Cosine Similarity."""

    def __init__(self, tolerance: float, min_matched_peaks: int):
        """
        Initialize the Modified Cosine Similarity calculator.

        Args:
            tolerance: Tolerance for m/z matching.
            min_matched_peaks: Minimum number of matched peaks.
        """
        self.tolerance = tolerance
        self.min_matched_peaks = min_matched_peaks
        self.similarity_measure = ModifiedCosine(tolerance=self.tolerance)

    def calculate(self, references: list[Spectrum], queries: list[Spectrum]) -> Scores:
        """Execute calculation using ModifiedCosine."""
        is_symmetric = references is queries
        return calculate_scores(
            references,
            queries,
            self.similarity_measure,
            is_symmetric=is_symmetric,
        )


def get_similarity_calculator(config: SimilarityConfig) -> SimilarityCalculator:
    """
    Factory function to get the appropriate similarity calculator based on configuration.

    Args:
        config: SimilarityConfig object.

    Returns:
        SimilarityCalculator: An instance of a concrete SimilarityCalculator strategy.

    Raises:
        ValueError: If the algorithm specified in config is unknown.
    """
    if config.algorithm == "cosine":
        return CosineSimilarity(
            tolerance=config.tolerance,
            min_matched_peaks=config.min_matched_peaks,
        )
    elif config.algorithm == "modified_cosine":
        return ModifiedCosineSimilarity(
            tolerance=config.tolerance,
            min_matched_peaks=config.min_matched_peaks,
        )
    else:
        # Pydantic validation should normally prevent this, but we raise for safety.
        raise ValueError(f"Unknown similarity algorithm: {config.algorithm}")
