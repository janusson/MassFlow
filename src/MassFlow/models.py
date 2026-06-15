"""
Data contracts and shared Pydantic models for MassFlow.

This module defines the core scientific data structures used across the
pipeline: molecular structure validation, spectral metadata with adduct-aware
precursor mass verification, and theoretical isotopic distributions.
"""

from typing import Annotated, Any, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PlainValidator, model_validator

def _cast_float32(v: Any) -> float:
    return float(np.float32(v)) if v is not None else None

Float32 = Annotated[float, PlainValidator(_cast_float32)]
from rdkit import Chem
from rdkit.Chem import Descriptors

from MassFlow.cheminformatics import ADDUCT_OFFSETS, calculate_isotopic_envelope


class IsotopicDistribution(BaseModel):
    """Schema for a molecule's theoretical isotopic distribution."""

    model_config = ConfigDict(extra="forbid")

    peaks: List[tuple[Float32, Float32]] = Field(
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
    exact_mass: Optional[Float32] = Field(
        None, ge=0, description="Monoisotopic exact mass"
    )
    isotopic_distribution: Optional[IsotopicDistribution] = Field(
        None, description="Theoretical isotopic distribution (mass, abundance) pairs."
    )
    isotopic_envelope: Optional[List[tuple[Float32, Float32]]] = Field(
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
            calculated_mass = Descriptors.ExactMolWt(mol)  # type: ignore[attr-defined]

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
    precursor_mz: Float32 = Field(..., gt=0)
    retention_time: Optional[Float32] = Field(None, ge=0, description="RT in seconds")
    charge: Optional[int] = Field(None, description="Ion charge state (e.g., 1, -1, 2)")
    ion_mode: Optional[Literal["positive", "negative", "neutral"]] = None
    collision_energy: Optional[Float32] = None
    adduct: Optional[str] = Field(
        None, description="Ionization adduct (e.g., [M+H]+, [M-H]-)"
    )
    molecule: Optional[MolecularStructure] = None
    experimental_isotopic_envelope: Optional[List[tuple[Float32, Float32]]] = Field(
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
