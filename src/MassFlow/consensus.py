"""
Consensus orchestration logic for MassFlow.

This module provides the `ConsensusEngine` class, which aggregates and resolves
annotation results from multiple independent spectral similarity engines. It is
engine-agnostic and relies on the standardized data structures defined in
`MassFlow.models`. It features weighted score aggregation, configurable
tie-breaking strategies, and scientific credibility checks to flag
discordance between different scoring algorithms.
"""

from typing import Dict, List, Optional, Tuple

from MassFlow.models import (
    AggregatedCandidate,
    AnnotationHit,
    ConsensusConfig,
    ConsensusInput,
    ConsensusResult,
)


class ConsensusEngine:
    """
    Orchestrates multiple annotation hits into a single consensus result.

    This class is engine-agnostic. It relies entirely on the `ConsensusInput`
    contract to perform weighted score aggregation, tie-breaking, and discordance
    flagging.
    """

    def __init__(self, config: ConsensusConfig) -> None:
        """
        Initialize the ConsensusEngine.

        Parameters
        ----------
        config : ConsensusConfig
            Configuration detailing engine weights, tie-breakers, and flagging thresholds.

        Raises
        ------
        ValueError
            If the sum of all engine weights is not greater than zero.
        """
        self.config = config
        self._total_weight = sum(self.config.engine_weights.values())

        if self._total_weight <= 0:
            raise ValueError("The sum of engine weights must be greater than 0.")

    def resolve(self, consensus_input: ConsensusInput) -> ConsensusResult:
        """
        Execute the consensus logic for a single query spectrum.

        Parameters
        ----------
        consensus_input : ConsensusInput
            The collection of engine hits for the query.

        Returns
        -------
        ConsensusResult
            The final orchestrated result, including the winning candidate and review flags.
        """
        if not consensus_input.hits:
            return ConsensusResult(query_id=consensus_input.query_id)

        # 1. Group hits by reference candidate
        candidate_map: Dict[str, AggregatedCandidate] = {}
        for hit in consensus_input.hits:
            if hit.reference_id not in candidate_map:
                candidate_map[hit.reference_id] = AggregatedCandidate(
                    reference_id=hit.reference_id,
                    inchikey=hit.inchikey,
                    smiles=hit.smiles,
                )

            candidate = candidate_map[hit.reference_id]
            candidate.engine_scores[hit.engine_id] = hit.score
            candidate.engine_ranks[hit.engine_id] = hit.rank

        # 2. Calculate Weighted Consensus Scores
        for candidate in candidate_map.values():
            weighted_sum = 0.0
            for engine_id, weight in self.config.engine_weights.items():
                # Treat missing scores from an engine as 0.0
                score = candidate.engine_scores.get(engine_id, 0.0)
                weighted_sum += score * weight

            candidate.consensus_score = weighted_sum / self._total_weight

        # Sort candidates descending by consensus score
        sorted_candidates = sorted(
            list(candidate_map.values()),
            key=lambda c: c.consensus_score,
            reverse=True,
        )

        # 3. Handle Tie-Breaking for the Top Position
        top_score = sorted_candidates[0].consensus_score
        tied_candidates = [
            c for c in sorted_candidates if c.consensus_score == top_score
        ]

        best_candidate = tied_candidates[0]
        if len(tied_candidates) > 1:
            best_candidate = self._apply_tie_breaker(tied_candidates)
            # Re-sort to put the tie-breaker winner at index 0
            sorted_candidates.remove(best_candidate)
            sorted_candidates.insert(0, best_candidate)

        # 4. Perform Scientific Credibility Check
        flagged, reason = self._check_scientific_credibility(consensus_input.hits)

        return ConsensusResult(
            query_id=consensus_input.query_id,
            best_reference_id=best_candidate.reference_id,
            best_consensus_score=best_candidate.consensus_score,
            flagged_for_review=flagged,
            review_reason=reason,
            candidates=sorted_candidates,
        )

    def _apply_tie_breaker(
        self, tied_candidates: List[AggregatedCandidate]
    ) -> AggregatedCandidate:
        """
        Resolve exact consensus score ties between top candidates.

        Parameters
        ----------
        tied_candidates : List[AggregatedCandidate]
            List of candidates that share the exact same top consensus score.

        Returns
        -------
        AggregatedCandidate
            The winning candidate based on the configured tie-breaker strategy.

        Raises
        ------
        ValueError
            If the 'validator_engine' strategy is chosen but no validator engine is configured.
        """
        strategy = self.config.tie_breaker_strategy

        if strategy == "highest_rank":
            # Winner is the one with the lowest (best) sum of ranks across all configured engines
            # Missing ranks are heavily penalized (assigned an arbitrary high rank, e.g., 999)
            def rank_sum(c: AggregatedCandidate) -> int:
                return sum(
                    c.engine_ranks.get(eng, 999)
                    for eng in self.config.engine_weights.keys()
                )

            return min(tied_candidates, key=rank_sum)

        elif strategy == "average_score":
            # Winner is the one with the highest unweighted average score
            def avg_score(c: AggregatedCandidate) -> float:
                scores = list(c.engine_scores.values())
                return sum(scores) / len(scores) if scores else 0.0

            return max(tied_candidates, key=avg_score)

        elif strategy == "validator_engine":
            # Trust the score of the specific validator engine
            validator = self.config.validator_engine
            if not validator:
                raise ValueError(
                    "validator_engine must be set when using 'validator_engine' tie_breaker."
                )
            return max(
                tied_candidates, key=lambda c: c.engine_scores.get(validator, 0.0)
            )

        # Fallback
        return tied_candidates[0]

    def _check_scientific_credibility(
        self, all_hits: List[AnnotationHit]
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if models disagree significantly on the top candidates.

        If Engine A's #1 hit is ranked worse than `flag_rank_discrepancy_threshold`
        by Engine B (or is entirely missing from Engine B's results), it indicates
        structural or algorithmic discordance requiring human review.

        Parameters
        ----------
        all_hits : List[AnnotationHit]
            All raw hits provided to the consensus engine.

        Returns
        -------
        tuple[bool, str or None]
            A boolean indicating if the result is flagged, and an optional reason string.
        """
        # Find the #1 ranked hit for each configured engine
        top_hits_by_engine: Dict[str, AnnotationHit] = {}
        for hit in all_hits:
            if hit.engine_id not in self.config.engine_weights:
                continue
            if hit.rank == 1:
                top_hits_by_engine[hit.engine_id] = hit

        threshold = self.config.flag_rank_discrepancy_threshold

        # Compare each engine's #1 hit against the other engines' rankings
        for engine_a, top_hit_a in top_hits_by_engine.items():
            for engine_b in self.config.engine_weights.keys():
                if engine_a == engine_b:
                    continue

                # Find how Engine B ranked Engine A's top hit
                rank_in_b = None
                for hit in all_hits:
                    if (
                        hit.engine_id == engine_b
                        and hit.reference_id == top_hit_a.reference_id
                    ):
                        rank_in_b = hit.rank
                        break

                if rank_in_b is None:
                    return True, (
                        f"Discordance: {engine_a}'s top hit ({top_hit_a.reference_id}) "
                        f"was completely unranked by {engine_b}."
                    )

                if rank_in_b > threshold:
                    return True, (
                        f"Discordance: {engine_a}'s top hit ({top_hit_a.reference_id}) "
                        f"was ranked #{rank_in_b} by {engine_b} (Threshold: {threshold})."
                    )

        return False, None
