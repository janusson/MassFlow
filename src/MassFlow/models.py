"""
Data contracts and shared Pydantic models for the MassFlow Orchestrator API.

This module defines the engine-agnostic structures used to communicate between
the core `MassFlow` pipeline and external Machine Learning similarity modules.
These contracts enforce a uniform shape for annotation results, consensus groupings,
and orchestration logic configuration, ensuring type safety without adding external
dependencies like PyTorch or TensorFlow.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import Chem
from rdkit.Chem import Descriptors

from MassFlow.cheminformatics import ADDUCT_OFFSETS, calculate_isotopic_envelope


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

    This configuration dictates how individual similarity engine outputs are aggregated into
    a single consensus score. Tuning these parameters alters the balance between false positives
    and false negatives in structural identification.
    """

    engine_weights: Dict[str, float] = Field(
        ...,
        description=(
            "Mapping of engine_id to its relative weight (e.g., {'exact_mass': 0.6, 'ms2deepscore': 0.4}).\n\n"
            "Scientific Justification:\n"
            "These weights represent the prior probability of engine accuracy within the ensemble. "
            "Weighting should reflect the known precision-recall trade-offs of the underlying engines. "
            "For example, exact-mass-based or isotopic-envelope engines provide high precision (low false-positive rate) "
            "and should receive higher priors for confident exact-structure identification. Conversely, "
            "fragmentation-pattern-based Machine Learning models (e.g., MS2DeepScore, Spec2Vec) offer high recall "
            "(sensitivity) for structural analogs but may suffer from lower precision for exact molecular matches. "
            "Tuning these weights fundamentally defines the operating point on the Receiver Operating Characteristic (ROC) curve."
        ),
    )
    tie_breaker_strategy: Literal[
        "highest_rank", "average_score", "validator_engine"
    ] = Field(
        default="highest_rank",
        description=(
            "Strategy to resolve exact consensus score ties between competing candidate structures.\n\n"
            "Scientific Justification:\n"
            "- 'highest_rank': Defers to the candidate that achieved the single best rank across any individual engine, "
            "leveraging the hypothesis that at least one scoring modality has captured the true orthogonal signal.\n"
            "- 'average_score': Assumes ensemble stability; averages the raw scores, favoring candidates with broad but "
            "moderate empirical agreement across all underlying feature spaces.\n"
            "- 'validator_engine': Employs orthogonal validation by deferring to a specific, highly-trusted engine "
            "(e.g., retreating to exact mass limits when fragmentation spectra yield ambiguous ties)."
        ),
    )
    validator_engine: Optional[str] = Field(
        default=None,
        description="Engine ID to trust during a 'validator_engine' tie-break (e.g., 'exact_mass').",
    )
    flag_rank_discrepancy_threshold: int = Field(
        default=5,
        description=(
            "Threshold for flagging a consensus result due to algorithmic disagreement.\n\n"
            "Scientific Justification:\n"
            "This parameter serves as a heuristic for 'orthogonal agreement failure' across engines. If a candidate is highly ranked "
            "by one engine (e.g., rank 1) but falls below this threshold in another (e.g., rank > 5), it indicates severe "
            "disagreement across orthogonal feature spaces (e.g., functional group similarity vs. backbone fragmentation). "
            "These cases represent low-confidence hits that require manual expert review to prevent false discoveries "
            "from propagating through automated downstream pipelines."
        ),
    )


class IsotopicDistribution(BaseModel):
    """Schema for a molecule's theoretical isotopic distribution."""

    model_config = ConfigDict(extra="forbid")

    peaks: List[tuple[float, float]] = Field(
        ...,
        description="List of (centroid_mass, relative_abundance) tuples representing the M, M+1, M+2... isotopic peaks.",
    )


class MolecularStructure(BaseModel):
    """Schema for molecular metadata and structure validation."""

    # Optimize for JSON: strip whitespace, forbid extra fields
    model_config = ConfigDict(
        str_strip_whitespace=True, extra="forbid", validate_assignment=True
    )

    smiles: Optional[str] = Field(None, description="Canonical SMILES")
    inchi: Optional[str] = Field(None, description="Standard InChI")
    formula: Optional[str] = Field(None, description="Chemical formula")
    exact_mass: Optional[float] = Field(
        None, ge=0, description="Monoisotopic exact mass"
    )
    isotopic_distribution: Optional[IsotopicDistribution] = Field(
        None, description="Theoretical isotopic distribution (mass, abundance) pairs."
    )
    isotopic_envelope: Optional[List[tuple[float, float]]] = Field(
        None, description="Theoretical isotopic envelope (M, M+1, M+2...)"
    )
    is_physically_valid: bool = Field(
        default=True,
        description="False if strict 5 ppm mass validation fails or SMILES is unparseable.",
    )

    @model_validator(mode="after")
    def validate_and_compute_mass(self) -> "MolecularStructure":
        """
        Validates the chemical structure via RDKit.
        Calculates exact mass if missing, or validates it against the provided mass.
        """
        mol = None

        # 1. Parse Structure
        if self.smiles:
            mol = Chem.MolFromSmiles(self.smiles)
            if not mol:
                # GRACEFUL FALLBACK: Flag as invalid, do not crash
                self.__dict__["is_physically_valid"] = False
        elif self.inchi:
            mol = Chem.MolFromInchi(self.inchi)
            if not mol:
                # GRACEFUL FALLBACK: Flag as invalid, do not crash
                self.__dict__["is_physically_valid"] = False

        # 2. Validate / Compute Exact Mass
        if mol:
            calculated_mass = Descriptors.ExactMolWt(mol)

            if self.exact_mass is not None:
                # Enforce strict 5 ppm mass error threshold for structural integrity
                ppm_error = (
                    abs(self.exact_mass - calculated_mass) / calculated_mass * 1e6
                )
                if ppm_error > 5.0:
                    # GRACEFUL FALLBACK: Bypass 5 ppm crash
                    self.__dict__["is_physically_valid"] = False
            else:
                # Auto-fill missing exact mass
                self.exact_mass = calculated_mass

            # Auto-fill missing formula
            if not self.formula:
                self.formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

            # Auto-fill isotopic envelope
            if self.smiles and not self.isotopic_envelope and self.is_physically_valid:
                self.isotopic_envelope = calculate_isotopic_envelope(self.smiles)

        return self


class SpectrumMetadata(BaseModel):
    """Schema for LC-MS/MS specific metadata."""

    model_config = ConfigDict(extra="ignore")

    spectrum_id: str
    precursor_mz: float = Field(..., gt=0)
    retention_time: Optional[float] = Field(None, ge=0, description="RT in seconds")
    charge: Optional[int] = Field(None, description="Ion charge state (e.g., 1, -1, 2)")
    ion_mode: Optional[Literal["positive", "negative", "neutral"]] = None
    collision_energy: Optional[float] = None
    adduct: Optional[str] = Field(
        None, description="Ionization adduct (e.g., [M+H]+, [M-H]-)"
    )
    molecule: Optional[MolecularStructure] = None
    is_physically_valid: bool = Field(
        default=True,
        description="False if adduct is unknown or theoretical m/z deviates by >5 ppm.",
    )

    @model_validator(mode="after")
    def validate_precursor_mass_logic(self) -> "SpectrumMetadata":
        """
        Ensures the precursor m/z physically aligns with the exact mass, charge,
        and specific adduct within a strict 5 ppm tolerance.
        """
        # Assign default adduct based on ion_mode if not provided
        if not self.adduct:
            if self.ion_mode == "positive":
                self.adduct = "[M+H]+"
            elif self.ion_mode == "negative":
                self.adduct = "[M-H]-"

        # Cascade failure from molecule layer
        if self.molecule and not self.molecule.is_physically_valid:
            self.__dict__["is_physically_valid"] = False

        # Graceful fallback: Skip strict validation if structural data is missing
        if not (
            self.molecule and self.molecule.exact_mass and self.charge and self.adduct
        ):
            return self

        # Graceful fallback: Bypass exact mass validation for non-standard adducts
        if self.adduct not in ADDUCT_OFFSETS:
            self.__dict__["is_physically_valid"] = False
            return self

        exact_mass = self.molecule.exact_mass
        charge = self.charge
        offset = ADDUCT_OFFSETS[self.adduct]

        # Theoretical m/z calculation. ADDUCT_OFFSETS assume z=+/-1 for the offset mass.
        theoretical_mz = (exact_mass + offset) / abs(charge)

        # Enforce strict 5 ppm tolerance
        ppm_error = abs(self.precursor_mz - theoretical_mz) / theoretical_mz * 1e6
        if ppm_error > 5.0:
            # GRACEFUL FALLBACK: Bypass 5 ppm crash
            self.__dict__["is_physically_valid"] = False

        return self


class SpectralPeaks(BaseModel):
    """Schema for the raw spectral arrays."""

    model_config = ConfigDict(
        # Rust core optimizes `list[float]` serialization drastically
        ser_json_bytes="utf8"
    )

    mz_array: List[float]
    intensity_array: List[float]

    @model_validator(mode="after")
    def validate_arrays(self) -> "SpectralPeaks":
        """Ensures array integrity."""
        length = len(self.mz_array)
        if length != len(self.intensity_array):
            raise ValueError(
                f"Array length mismatch: mz ({length}) vs intensity ({len(self.intensity_array)})"
            )

        # Ensure m/z is sorted (crucial for fast similarity searches later)
        if not all(self.mz_array[i] <= self.mz_array[i + 1] for i in range(length - 1)):
            raise ValueError("mz_array must be monotonically increasing.")

        return self


class MassFlowSpectrum(BaseModel):
    """Top-level schema representing a complete MS/MS spectral record."""

    metadata: SpectrumMetadata
    peaks: SpectralPeaks
