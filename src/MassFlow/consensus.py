"""
Consensus orchestration logic for MassFlow.

This module provides the `generate_consensus` function and `ConsensusEngine` class,
which aggregate and resolve annotation results from multiple independent spectral
similarity engines. It is engine-agnostic and relies on the standardized data
structures defined in `MassFlow.models`. It features probabilistically weighted score
aggregation, configurable tie-breaking strategies, and scientific credibility checks
to flag orthogonal agreement failures.
"""

import logging
from typing import Dict

from MassFlow.models import (
    AggregatedCandidate,
    ConsensusConfig,
    ConsensusInput,
    ConsensusResult,
)

logger = logging.getLogger(__name__)


def generate_consensus(
    input_data: ConsensusInput, config: ConsensusConfig
) -> ConsensusResult:
    """
    Generate a consensus result from a collection of annotation hits.

    Parameters
    ----------
    input_data : ConsensusInput
        The collection of engine hits for the query.
    config : ConsensusConfig
        Configuration detailing engine weights, tie-breakers, and flagging thresholds.

    Returns
    -------
    ConsensusResult
        The final orchestrated result, including the winning candidate and review flags.
    """
    if not input_data.hits:
        return ConsensusResult(query_id=input_data.query_id)

    total_weight = sum(config.engine_weights.values())
    if total_weight <= 0:
        raise ValueError("The sum of engine weights must be greater than 0.")

    # 1. Group hits by reference candidate
    candidate_map: Dict[str, AggregatedCandidate] = {}
    for hit in input_data.hits:
        if hit.reference_id not in candidate_map:
            candidate_map[hit.reference_id] = AggregatedCandidate(
                reference_id=hit.reference_id,
                inchikey=hit.inchikey,
                smiles=hit.smiles,
            )

        candidate = candidate_map[hit.reference_id]
        candidate.engine_scores[hit.engine_id] = hit.score
        candidate.engine_ranks[hit.engine_id] = hit.rank

    # 2. Calculate Weighted Average Scores
    for candidate in candidate_map.values():
        weighted_sum = 0.0
        for engine_id, weight in config.engine_weights.items():
            # Treat missing scores from an engine as 0.0
            score = candidate.engine_scores.get(engine_id, 0.0)
            weighted_sum += score * weight

        candidate.consensus_score = weighted_sum / total_weight

    # Sort candidates descending by consensus score
    sorted_candidates = sorted(
        list(candidate_map.values()),
        key=lambda c: c.consensus_score,
        reverse=True,
    )

    # Log the expert guide: top 3 candidates
    logger.debug(
        f"Calculating weighted consensus for query '{input_data.query_id}'. Top 3 results:"
    )
    for i, cand in enumerate(sorted_candidates[:3], start=1):
        scores_str = ", ".join([f"{e}: {s:.4f}" for e, s in cand.engine_scores.items()])
        logger.debug(
            f"  #{i} | Ref: {cand.reference_id} | Score: {cand.consensus_score:.4f} (Breakdown: {scores_str})"
        )

    # 3. Handle Tie-Breaking for the Top Position
    top_score = sorted_candidates[0].consensus_score
    tied_candidates = [c for c in sorted_candidates if c.consensus_score == top_score]

    best_candidate = tied_candidates[0]
    if len(tied_candidates) > 1:
        logger.info(
            f"Tie detected for query '{input_data.query_id}' (Score: {top_score:.4f}). Applying tie-breaker: {config.tie_breaker_strategy}"
        )
        strategy = config.tie_breaker_strategy

        if strategy == "highest_rank":

            def rank_sum(c: AggregatedCandidate) -> int:
                return sum(
                    c.engine_ranks.get(eng, 999) for eng in config.engine_weights.keys()
                )

            best_candidate = min(tied_candidates, key=rank_sum)
        elif strategy == "average_score":

            def avg_score(c: AggregatedCandidate) -> float:
                scores = list(c.engine_scores.values())
                return sum(scores) / len(scores) if scores else 0.0

            best_candidate = max(tied_candidates, key=avg_score)
        elif strategy == "validator_engine":
            validator = config.validator_engine
            if not validator:
                raise ValueError(
                    "validator_engine must be set when using 'validator_engine' tie_breaker."
                )
            best_candidate = max(
                tied_candidates, key=lambda c: c.engine_scores.get(validator, 0.0)
            )

        # Re-sort to put the tie-breaker winner at index 0
        if best_candidate in sorted_candidates:
            sorted_candidates.remove(best_candidate)
            sorted_candidates.insert(0, best_candidate)

        logger.debug(f"Tie-breaker winner: {best_candidate.reference_id}")

    # 4. Perform Scientific Credibility Check: Flag Rank Discrepancy
    flagged = False
    reason = None

    # Identify the 'primary engine' (highest weight)
    primary_engine = max(config.engine_weights.items(), key=lambda x: x[1])[0]

    # Find the #1 ranked hit for the primary engine
    primary_top_hit = None
    for hit in input_data.hits:
        if hit.engine_id == primary_engine and hit.rank == 1:
            primary_top_hit = hit
            break

    if primary_top_hit:
        threshold = config.flag_rank_discrepancy_threshold
        # Check against all other configured engines
        for engine_b in config.engine_weights.keys():
            if primary_engine == engine_b:
                continue

            # Find how Engine B ranked the primary engine's top hit
            rank_in_b = None
            for hit in input_data.hits:
                if (
                    hit.engine_id == engine_b
                    and hit.reference_id == primary_top_hit.reference_id
                ):
                    rank_in_b = hit.rank
                    break

            if rank_in_b is None:
                flagged = True
                reason = (
                    f"Orthogonal Agreement Failure: Primary engine '{primary_engine}' top hit "
                    f"({primary_top_hit.reference_id}) was completely unranked by {engine_b}."
                )
                logger.warning(f"Consensus Warning [{input_data.query_id}]: {reason}")
                break
            elif rank_in_b > threshold:
                flagged = True
                reason = (
                    f"Orthogonal Agreement Failure: Primary engine '{primary_engine}' top hit "
                    f"({primary_top_hit.reference_id}) was ranked #{rank_in_b} by {engine_b} "
                    f"(Threshold: {threshold})."
                )
                logger.warning(f"Consensus Warning [{input_data.query_id}]: {reason}")
                break

    return ConsensusResult(
        query_id=input_data.query_id,
        best_reference_id=best_candidate.reference_id,
        best_consensus_score=best_candidate.consensus_score,
        flagged_for_review=flagged,
        review_reason=reason,
        candidates=sorted_candidates,
    )


class ConsensusEngine:
    """
    Orchestrates multiple annotation hits into a single consensus result.

    This class is engine-agnostic. It relies entirely on the `generate_consensus`
    function to perform weighted score aggregation, tie-breaking, and discordance
    flagging.
    """

    def __init__(self, config: ConsensusConfig) -> None:
        """
        Initialize the ConsensusEngine.

        Parameters
        ----------
        config : ConsensusConfig
            Configuration detailing engine weights, tie-breakers, and flagging thresholds.
        """
        self.config = config
        # Validation
        if sum(self.config.engine_weights.values()) <= 0:
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
        return generate_consensus(consensus_input, self.config)
