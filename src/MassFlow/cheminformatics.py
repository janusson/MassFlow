"""
Cheminformatics utilities bridging MassFlow and RDKit.

This module provides high-performance, memoized calculations for structural properties,
including exact masses with adduct offsets and Tanimoto similarity scores using
Morgan fingerprints.
"""

from collections import defaultdict
from functools import lru_cache
from typing import Optional

import pyteomics.mass as pmass
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

# Standard monoisotopic exact masses (NIST)
H_MASS = 1.0078250322
C_MASS = 12.0000000000
N_MASS = 14.0030740044
O_MASS = 15.9949146196
NA_MASS = 22.9897692809
K_MASS = 38.9637064864
CL_MASS = 34.96885268
ELECTRON_MASS = 0.0005485799
PROTON_MASS = H_MASS - ELECTRON_MASS

# Theoretical shifts for common LC-MS adducts (m/z offsets for z=1 or z=-1)
# All values are calculated to 6 decimal places using monoisotopic exact masses.
ADDUCT_OFFSETS = {
    # Positive Ion Mode
    "[M+H]+": 1.007276,
    "[M+NH4]+": 18.033826,
    "[M+Na]+": 22.989221,
    "[M+K]+": 38.963158,
    "[M]+": -0.000549,
    # Negative Ion Mode
    "[M-H]-": -1.007276,
    "[M+Cl]-": 34.969401,
    "[M+HCOO]-": 44.998203,
    "[M+CH3COO]-": 59.013853,
    "[M+FA-H]-": 44.998203,  # Formate (common LC-MS alias)
    "[M]-": 0.000549,
}


@lru_cache(maxsize=16384)
def _get_morgan_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048):
    """
    Generate and cache a Morgan fingerprint (as a bit vector) from a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> Optional[float]:
    """
    Calculate the Tanimoto similarity score between two SMILES strings using
    cached Morgan fingerprints.

    Parameters
    ----------
    smiles1 : str
        First SMILES string.
    smiles2 : str
        Second SMILES string.

    Returns
    -------
    float or None
        Tanimoto similarity score (0.0 to 1.0), or None if either SMILES is invalid
        or cannot be parsed.
    """
    if not smiles1 or not smiles2:
        return None

    fp1 = _get_morgan_fingerprint(smiles1)
    fp2 = _get_morgan_fingerprint(smiles2)

    if not fp1 or not fp2:
        return None

    return DataStructs.TanimotoSimilarity(fp1, fp2)


@lru_cache(maxsize=16384)
def get_isotopic_distribution(
    smiles: str, threshold: float = 0.001
) -> list[tuple[float, float]]:
    """
    Calculate the theoretical isotopic distribution for a molecule.

    Uses pyteomics to generate precise isotopologues, grouping them by nominal
    mass offset (M, M+1, M+2, etc.) to return abundance-weighted centroid masses
    and their relative abundances.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecule.
    threshold : float
        The relative abundance threshold (0.0 to 1.0) below which isotopic peaks are ignored.

    Returns
    -------
    list of tuple of (float, float)
        A list of (centroid_mass, relative_abundance) tuples, sorted by mass.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []

    formula = rdMolDescriptors.CalcMolFormula(mol)
    try:
        comp = pmass.Composition(formula=formula)
        mono_mass = pmass.calculate_mass(formula=formula)
    except Exception:
        # Fallback if formula cannot be parsed
        return []

    # Calculate fine isotopologues with a tight internal threshold to capture M, M+1, M+2 accurately
    isotopologues = pmass.isotopologues(
        comp, report_abundance=True, overall_threshold=1e-8
    )

    bins = defaultdict(list)
    for iso_comp, abundance in isotopologues:
        iso_mass = pmass.calculate_mass(iso_comp)
        offset = round(iso_mass - mono_mass)
        bins[offset].append((iso_mass, abundance))

    res = []
    total_abundances = {
        offset: sum(a for m, a in items) for offset, items in bins.items()
    }
    if not total_abundances:
        return []

    max_abund = max(total_abundances.values())

    for offset, items in sorted(bins.items()):
        total_a = sum(a for m, a in items)
        rel_a = total_a / max_abund
        if rel_a >= threshold:
            centroid_mass = sum(m * a for m, a in items) / total_a
            res.append((round(centroid_mass, 6), round(rel_a, 6)))

    return res


@lru_cache(maxsize=16384)
def calculate_isotopic_envelope(
    smiles: str, max_isopeaks: int = 4
) -> list[tuple[float, float]]:
    """
    Calculate the theoretical isotopic envelope for a molecule, returning up to
    `max_isopeaks` (e.g., M, M+1, M+2, M+3) normalized such that the base peak is 1.0.

    Attempts to use RDKit's `rdMolDescriptors.GetIsotopicDistribution` if available.
    Otherwise, gracefully falls back to the high-precision pyteomics centroid approach.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecule.
    max_isopeaks : int
        The maximum number of isotopic peaks to return (default 4).

    Returns
    -------
    list of tuple of (float, float)
        A list of (centroid_mass, normalized_abundance) tuples, sorted by mass.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []

    # Attempt to use RDKit's native method if available in the installed version
    if hasattr(rdMolDescriptors, "GetIsotopicDistribution"):
        try:
            # Assuming rdMolDescriptors.GetIsotopicDistribution(mol) returns something iterable
            # producing objects with .mass and .abundance (or similar tuple structure).
            # We will coerce it to our needs.
            dist = rdMolDescriptors.GetIsotopicDistribution(mol)

            # Extract tuples (mass, abundance) - handling potential RDKit object types
            peaks = []
            for item in dist:
                if hasattr(item, "mass") and hasattr(item, "abundance"):
                    peaks.append((item.mass, item.abundance))
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    peaks.append((item[0], item[1]))

            if not peaks:
                return []

            # Normalize to base peak = 1.0
            max_abund = max(p[1] for p in peaks)
            normalized = [(round(m, 6), round(a / max_abund, 6)) for m, a in peaks]

            # Sort by mass and limit to max_isopeaks
            normalized.sort(key=lambda x: x[0])
            return normalized[:max_isopeaks]
        except Exception:
            pass  # Fall back to pyteomics

    # Fallback: Pyteomics centroiding approach
    formula = rdMolDescriptors.CalcMolFormula(mol)
    try:
        comp = pmass.Composition(formula=formula)
        mono_mass = pmass.calculate_mass(formula=formula)
    except Exception:
        return []

    # Get fine isotopologues
    isotopologues = pmass.isotopologues(
        comp, report_abundance=True, overall_threshold=1e-8
    )

    # Bin by nominal mass offset
    bins = defaultdict(list)
    for iso_comp, abundance in isotopologues:
        iso_mass = pmass.calculate_mass(iso_comp)
        offset = round(iso_mass - mono_mass)
        bins[offset].append((iso_mass, abundance))

    total_abundances = {
        offset: sum(a for m, a in items) for offset, items in bins.items()
    }
    if not total_abundances:
        return []

    max_abund = max(total_abundances.values())

    res = []
    # Only keep the lowest `max_isopeaks` offsets (0, 1, 2, 3)
    for offset, items in sorted(bins.items())[:max_isopeaks]:
        total_a = sum(a for m, a in items)
        if total_a > 0:
            rel_a = total_a / max_abund
            centroid_mass = sum(m * a for m, a in items) / total_a
            res.append((round(centroid_mass, 6), round(rel_a, 6)))

    return res


@lru_cache(maxsize=16384)
def calculate_theoretical_mass(smiles: str, adduct: str = "[M+H]+") -> Optional[float]:
    """
    Calculate the theoretical exact mass for a chemical structure given an adduct.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecule.
    adduct : str
        The adduct notation (e.g., "[M+H]+", "[M+Na]+").

    Returns
    -------
    float or None
        The theoretical m/z of the adduct, or None if the SMILES is invalid.

    Raises
    ------
    ValueError
        If the adduct is not recognized in the internal offset registry.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    exact_mass = Descriptors.ExactMolWt(mol)

    offset = ADDUCT_OFFSETS.get(adduct)
    if offset is None:
        raise ValueError(
            f"Adduct '{adduct}' is not supported. Supported adducts: {list(ADDUCT_OFFSETS.keys())}"
        )

    return exact_mass + offset
