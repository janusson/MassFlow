"""
Data contracts and shared Pydantic models for the MassFlow Orchestrator API.

This module defines the engine-agnostic structures used to communicate between
the core `MassFlow` pipeline and external Machine Learning similarity modules.
These contracts enforce a uniform shape for annotation results, consensus groupings,
and orchestration logic configuration, ensuring type safety without adding external
dependencies like PyTorch or TensorFlow.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AnnotationHit(BaseModel):
    """
    A single spectral annotation result from a specific similarity engine.

    This structure represents one potential match between an experimental query
    and a reference library entry, agnostic of the underlying algorithm used.
    """

    engine_id: str = Field(
        ..., description="Identifier for the engine (e.g., 'cosine', 'ms2deepscore')."
    )
    reference_id: str = Field(
        ..., description="Unique identifier for the reference candidate."
    )
    score: float = Field(
        ...,
        description="The similarity score calculated by the engine (typically 0.0 to 1.0).",
    )
    rank: int = Field(
        ..., description="The rank of this hit within the engine's specific result set."
    )
    inchikey: Optional[str] = Field(
        default=None, description="InChIKey for structure-level aggregation."
    )
    smiles: Optional[str] = Field(
        default=None, description="SMILES string of the candidate."
    )


class ConsensusInput(BaseModel):
    """
    A collection of all annotation hits for a single experimental query spectrum.

    This contract groups all competing engine outputs for a specific query,
    serving as the primary input payload for the `ConsensusEngine`.
    """

    query_id: str = Field(..., description="Unique identifier for the query spectrum.")
    hits: List[AnnotationHit] = Field(
        default_factory=list, description="All hits across all engines for this query."
    )


class AggregatedCandidate(BaseModel):
    """
    Internal orchestration structure mapping a specific reference candidate
    to its scores and ranks across multiple engines.
    """

    reference_id: str
    inchikey: Optional[str]
    smiles: Optional[str]
    consensus_score: float = 0.0
    engine_scores: Dict[str, float] = Field(default_factory=dict)
    engine_ranks: Dict[str, int] = Field(default_factory=dict)


class ConsensusResult(BaseModel):
    """
    The final orchestrated output summarizing the consensus agreement across engines.
    """

    query_id: str
    best_reference_id: Optional[str] = Field(
        default=None,
        description="The winning reference ID after consensus and tie-breaking.",
    )
    best_consensus_score: Optional[float] = Field(
        default=None,
        description="The final aggregated score for the winning candidate.",
    )
    flagged_for_review: bool = Field(
        default=False,
        description="True if top engines strongly disagree on the candidate.",
    )
    review_reason: Optional[str] = Field(
        default=None, description="Explanation of the scientific credibility flag."
    )
    candidates: List[AggregatedCandidate] = Field(
        default_factory=list,
        description="List of all evaluated candidates sorted by score.",
    )


class ConsensusConfig(BaseModel):
    """
    Configuration for the consensus weighting and tie-breaking logic.
    """

    engine_weights: Dict[str, float] = Field(
        ...,
        description="Mapping of engine_id to its relative weight (e.g., {'cosine': 0.6, 'ms2deepscore': 0.4}).",
    )
    tie_breaker_strategy: Literal[
        "highest_rank", "average_score", "validator_engine"
    ] = Field(
        default="highest_rank",
        description="Strategy to resolve exact consensus score ties.",
    )
    validator_engine: Optional[str] = Field(
        default=None,
        description="Engine ID to trust during a 'validator_engine' tie-break.",
    )
    flag_rank_discrepancy_threshold: int = Field(
        default=5,
        description="Flag result if Engine A's top hit is ranked worse than this in Engine B.",
    )
