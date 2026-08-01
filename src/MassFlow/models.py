"""
Data contracts and shared Pydantic models for MassFlow.

This module defines the core scientific data structures used across the
pipeline: molecular structure validation, spectral metadata with adduct-aware
precursor mass verification, and theoretical isotopic distributions.

RDKit is an **optional** dependency.  When unavailable the models fall back
to ``formula``-based mass validation via pyteomics, and the classical
cosine-scoring pipeline continues without interruption.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Optional RDKit import
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover -- tested via the "no rdkit" CI variant
    _HAS_RDKIT = False

from MassFlow.cheminformatics import (
    _formula_to_isotopic_envelope,
    _formula_to_monoisotopic_mass,
    _smiles_to_formula,
    compute_adduct_offset,
)

logger = logging.getLogger(__name__)


class IsotopicDistribution(BaseModel):
    """Schema for a molecule's theoretical isotopic distribution."""

    model_config = ConfigDict(extra="forbid")

    peaks: List[tuple[float, float]] = Field(
        ...,
        description=(
            "List of (centroid_mass, relative_abundance) tuples representing "
            "the M, M+1, M+2... isotopic peaks."
        ),
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
        description=(
            "False if strict 5 ppm mass validation fails, SMILES is "
            "unparseable, or the formula cannot be resolved."
        ),
    )

    @model_validator(mode="after")
    def validate_and_compute_mass(self) -> "MolecularStructure":
        """
        Validate the chemical structure and compute/verify exact mass.

        **Priority order for formula resolution:**

        1. If ``formula`` is present, use it directly with pyteomics
           (works with or without RDKit).
        2. If ``smiles`` or ``inchi`` is present and RDKit is available,
           derive the formula from the structural representation.
        3. If neither a formula nor a parsable structure is available,
           set ``is_physically_valid = True`` and skip the 5 ppm check
           (the spectrum can still participate in classical cosine scoring).
        """
        resolved_formula: Optional[str] = self.formula
        mol = None

        # ── Resolve formula ────────────────────────────────────────────
        if not resolved_formula and _HAS_RDKIT:
            if self.smiles:
                mol = Chem.MolFromSmiles(self.smiles)
                if mol:
                    resolved_formula = _smiles_to_formula(self.smiles)
                else:
                    self.__dict__["is_physically_valid"] = False
            elif self.inchi:
                mol = Chem.MolFromInchi(self.inchi)
                if mol:
                    from rdkit.Chem import rdMolDescriptors

                    resolved_formula = rdMolDescriptors.CalcMolFormula(mol)
                else:
                    self.__dict__["is_physically_valid"] = False

        # ── No formula resolvable: skip 5 ppm check, allow spectrum ────
        if resolved_formula is None:
            logger.debug(
                "No formula or parsable structure for spectrum; "
                "skipping 5 ppm mass validation."
            )
            # Still try to auto-fill formula from molecule if already parsed
            if mol is not None and _HAS_RDKIT:
                from rdkit.Chem import rdMolDescriptors

                self.formula = rdMolDescriptors.CalcMolFormula(mol)
            return self

        # ── Compute / validate exact mass via pyteomics SSOT ───────────
        calculated_mass = _formula_to_monoisotopic_mass(resolved_formula)

        if self.exact_mass is not None:
            ppm_error = abs(self.exact_mass - calculated_mass) / calculated_mass * 1e6
            if ppm_error > 5.0:
                self.__dict__["is_physically_valid"] = False
        else:
            # Auto-fill missing exact mass
            self.exact_mass = calculated_mass

        # Auto-fill missing formula
        if not self.formula:
            self.formula = resolved_formula

        # Auto-fill isotopic envelope from formula (works without RDKit)
        if not self.isotopic_envelope and self.is_physically_valid:
            self.isotopic_envelope = _formula_to_isotopic_envelope(resolved_formula)

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
    experimental_isotopic_envelope: Optional[List[tuple[float, float]]] = Field(
        default=None,
        description="Experimental MS1 isotopic envelope (mz, abundance) pairs if available.",
    )
    is_physically_valid: bool = Field(
        default=True,
        description="False if adduct is unknown or theoretical m/z deviates by >5 ppm.",
    )

    @model_validator(mode="after")
    def validate_precursor_mass_logic(self) -> "SpectrumMetadata":
        """
        Ensure the precursor m/z physically aligns with the exact mass, charge,
        and specific adduct within a strict 5 ppm tolerance.

        **Fallback behaviour when RDKit is unavailable:**

        - If the molecule's ``formula`` field is populated, the 5 ppm check
          proceeds using pyteomics directly.
        - If neither a formula nor RDKit is available, the validation is
          skipped gracefully and the spectrum is allowed to proceed to
          classical cosine scoring.
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

        # Skip strict validation if structural data is missing
        if not (
            self.molecule and self.molecule.exact_mass and self.charge and self.adduct
        ):
            return self

        # Graceful fallback: Bypass exact mass validation for non-standard adducts
        offset = compute_adduct_offset(self.adduct)
        if offset is None:
            self.__dict__["is_physically_valid"] = False
            return self

        exact_mass = self.molecule.exact_mass
        charge = self.charge

        # Theoretical m/z calculation using pyteomics Composition-derived offset.
        theoretical_mz = (exact_mass + offset) / abs(charge)

        # Enforce strict 5 ppm tolerance
        ppm_error = abs(self.precursor_mz - theoretical_mz) / theoretical_mz * 1e6
        if ppm_error > 5.0:
            self.__dict__["is_physically_valid"] = False

        return self


# ---------------------------------------------------------------------------
# Triage / Quality-flag model for spectral difficulty classification
# ---------------------------------------------------------------------------


class TriageProfile(BaseModel):
    """Structured quality flags computed during spectral pre-processing.

    This model parses the JSON ``triage_flags`` column stored alongside each
    spectrum in the SQLite database. It provides a stable, typed interface for
    the :class:`MLRouter` to decide whether a spectrum should be routed to a
    classical scoring engine ("easy") or an ML engine ("hard").

    **Semantics of individual flags**

    * ``is_chimeric`` – the precursor isolation window contained co-eluting ions,
      so the fragmentation spectrum is a mixture of two or more precursors.
    * ``low_abundance_precursor`` – the MS1 precursor ion intensity fell below
      the configurable signal threshold, increasing the chance that low-intensity
      fragment peaks are noise.
    * ``missing_ms1_purity`` – no MS1 isolation-purity metrics were recorded by
      the instrument method (common for data-dependent acquisition without
      survey scans).
    * ``low_signal_to_noise`` – the median or mean fragment S/N is below the
      acceptable threshold, making peak-picking less reliable.
    * ``unassigned_neutral_losses`` – the de-novo or rule-based neutral-loss
      assignment step left a non-trivial fraction of fragment differences
      unexplained, suggesting an unusual or unexpected fragmentation pathway.

    **Default / absent flags**

    When the ``triage_flags`` column is ``NULL`` or ``"{}"``, all boolean flags
    default to ``False`` and quantitative fields to ``None``, which the router
    interprets as "no evidence of difficulty" → route to the easy engine.
    """

    model_config = ConfigDict(extra="allow")

    # --- Boolean quality flags -----------------------------------------------
    is_chimeric: bool = Field(
        default=False,
        description="Co-eluting precursor ions in the isolation window.",
    )
    low_abundance_precursor: bool = Field(
        default=False,
        description="MS1 precursor intensity below configured noise threshold.",
    )
    missing_ms1_purity: bool = Field(
        default=False,
        description="No MS1 isolation-purity data recorded.",
    )
    low_signal_to_noise: bool = Field(
        default=False,
        description="Fragment spectrum S/N below acceptable threshold.",
    )
    unassigned_neutral_losses: bool = Field(
        default=False,
        description="Non-trivial fraction of neutral losses unexplained.",
    )

    # --- Quantitative metrics (None = not measured) --------------------------
    precursor_purity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quantitative MS1 isolation purity (0.0 – 1.0).",
    )
    signal_to_noise: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Measured fragment-level signal-to-noise ratio.",
    )
    num_peaks: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total number of fragment peaks in the spectrum.",
    )

    # --- Aggregate difficulty score (derived, not stored) --------------------
    @property
    def difficulty_flags_active(self) -> int:
        """Count of active boolean difficulty flags.

        Returns
        -------
        int
            Number of boolean triage flags currently set to ``True``.
        """
        flags: list[bool] = [
            self.is_chimeric,
            self.low_abundance_precursor,
            self.missing_ms1_purity,
            self.low_signal_to_noise,
            self.unassigned_neutral_losses,
        ]
        return sum(1 for f in flags if f)

    @classmethod
    def from_spectrum_metadata(cls, metadata: Dict[str, Any]) -> "TriageProfile":
        """Extract a ``TriageProfile`` from a spectrum's metadata dictionary.

        The ``triage_flags`` key (if present) is parsed as JSON. Top-level
        keys in the metadata that collide with TriageProfile field names are
        also consumed, so that spectra loaded from a SQLite database that
        flattened triage_flags into the metadata (as done by
        ``SpectralDatabase._row_to_spectrum``) are handled transparently.

        Parameters
        ----------
        metadata : dict
            The spectrum ``.metadata`` dict (or ``.get_metadata()``).

        Returns
        -------
        TriageProfile
            Populated profile; absent / unparseable data yields default flags.
        """
        # 1. Try the canonical "triage_flags" JSON sub-dict.
        raw: Optional[str] = metadata.get("triage_flags")
        if isinstance(raw, dict):
            # Already deserialised by _row_to_spectrum or equivalent.
            raw_dict = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                raw_dict = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                raw_dict = {}
        else:
            raw_dict = {}

        # 2. Also pull top-level metadata keys that match our field names.
        #    This handles the case where _row_to_spectrum called
        #    ``metadata.update(triage)``, so e.g. `is_chimeric` is a top-level
        #    metadata key rather than nested under `triage_flags`.
        field_names = {
            "is_chimeric",
            "low_abundance_precursor",
            "missing_ms1_purity",
            "low_signal_to_noise",
            "unassigned_neutral_losses",
            "precursor_purity",
            "signal_to_noise",
            "num_peaks",
        }
        merged: Dict[str, Any] = {}
        for name in field_names:
            if name in metadata and name not in raw_dict:
                merged[name] = metadata[name]
        merged.update(raw_dict)

        # Filter to only our known fields (extra keys pass via extra="allow")
        filtered = {k: v for k, v in merged.items() if k in field_names}
        return cls(**filtered)

    @classmethod
    def empty(cls) -> "TriageProfile":
        """Return a profile with all flags set to their safe defaults.

        This is the profile used when no triage data is available at all.
        """
        return cls()
