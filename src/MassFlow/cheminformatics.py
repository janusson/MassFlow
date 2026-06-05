"""
Cheminformatics utilities bridging MassFlow and RDKit.

This module provides high-performance, memoized calculations for structural properties,
including exact masses with adduct offsets and Tanimoto similarity scores using
Morgan fingerprints.
"""

import math
import re
from collections import defaultdict
from functools import lru_cache
from typing import Optional, Set

import pyteomics.mass as pmass
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

# Standard monoisotopic exact masses (NIST)
H_MASS = 1.0078250322
C_MASS = 12.0000000000
N_MASS = 14.0030740044
O_MASS = 15.9949146196
S_MASS = 31.9720711374
P_MASS = 30.9737616320
F_MASS = 18.9984031627
BR_MASS = 78.9183371000
CL_MASS = 34.96885268
NA_MASS = 22.9897692809
K_MASS = 38.9637064864
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

    # RDKit's CalcMolFormula for SMILES like [13C] may just return C6.
    # We must explicitly use atoms to get the correct formula or mass if isotopes are present.
    # Pyteomics prefers its own isotope notation.
    try:
        # Check for isotopes in the RDKit molecule
        has_isotope = any(atom.GetIsotope() > 0 for atom in mol.GetAtoms())

        if has_isotope:
            # Construct formula manually for pyteomics if isotopes are present.
            # Use RDKit's CalcMolFormula to obtain base counts (including implicit H),
            # then overlay isotopic labels from explicit atoms so we produce something like C[13]6H6.
            # Try to compute a base formula from a molecule copy with isotopes zeroed so implicit H counts are preserved
            try:
                mol_no_iso = Chem.Mol(mol)
                for atom in mol_no_iso.GetAtoms():
                    atom.SetIsotope(0)
                base_formula = rdMolDescriptors.CalcMolFormula(mol_no_iso)
            except Exception:
                base_formula = rdMolDescriptors.CalcMolFormula(mol)

            base_matches = re.findall(r"([A-Z][a-z]*)(\d*)", base_formula)
            base_counts = {
                element: int(count) if count else 1 for element, count in base_matches
            }

            counts: defaultdict[str, defaultdict[int, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                isotope = atom.GetIsotope()
                counts[symbol][isotope] += 1

            parts = []
            for symbol in sorted(base_counts.keys()):
                total = base_counts[symbol]
                if symbol in counts:
                    # Add isotopically labeled atoms first (e.g., C[13]6)
                    isotopic_items = [
                        (iso, counts[symbol][iso])
                        for iso in sorted(counts[symbol].keys())
                        if iso > 0
                    ]
                    for iso, cnt in isotopic_items:
                        parts.append(f"{symbol}[{iso}]{cnt}")

                    # Non-labeled atoms present explicitly
                    non_iso_count = counts[symbol].get(0, 0)
                    if non_iso_count > 0:
                        parts.append(f"{symbol}{non_iso_count}")

                    # If explicit atom counts sum to less than the base total (rare), append the remainder
                    sum_counts = non_iso_count + sum(cnt for _, cnt in isotopic_items)
                    if sum_counts < total:
                        rem = total - sum_counts
                        parts.append(f"{symbol}{rem}")
                else:
                    # No isotopic labels for this element; use base count
                    parts.append(f"{symbol}{total}")

            formula_py = "".join(parts)

            comp = pmass.Composition(formula=formula_py)
            mono_mass = pmass.calculate_mass(formula=formula_py)
        else:
            formula = rdMolDescriptors.CalcMolFormula(mol)
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


# Common exact neutral losses and the elements they physically require
COMMON_NEUTRAL_LOSSES = [
    (18.0106, {"O"}),  # H2O
    (17.0265, {"N"}),  # NH3
    (27.9949, {"O"}),  # CO
    (43.9898, {"O"}),  # CO2
    (33.9877, {"S"}),  # H2S  (2 × H_MASS + S_MASS = 33.9877; 34.9956 was H₃S sulfonium)
    (63.9619, {"S", "O"}),  # SO2
    (78.9585, {"P", "O"}),  # PO3
    (35.9767, {"Cl"}),  # HCl
    (79.9262, {"Br"}),  # HBr
    (20.0062, {"F"}),  # HF
]


def calculate_isotopic_similarity(
    exp_env: list[tuple[float, float]],
    theor_env: list[tuple[float, float]],
    mz_tolerance: float = 0.05,
) -> float:
    """
    Calculate the cosine similarity between an experimental and theoretical isotopic envelope.

    Parameters
    ----------
    exp_env : list of tuple
        Experimental MS1 isotopic envelope as (mz, abundance) pairs.
    theor_env : list of tuple
        Theoretical MS1 isotopic envelope as (mz, abundance) pairs.
    mz_tolerance : float
        Tolerance in Da to match an experimental peak to a theoretical one.

    Returns
    -------
    float
        Cosine similarity score (0.0 to 1.0).
    """
    if not exp_env or not theor_env:
        return 0.0

    # Align peaks greedily
    aligned_exp = []
    aligned_theor = []

    used_exp = set()
    for t_mz, t_int in theor_env:
        best_match = None
        best_diff = mz_tolerance

        for i, (e_mz, e_int) in enumerate(exp_env):
            if i in used_exp:
                continue
            diff = abs(t_mz - e_mz)
            if diff <= best_diff:
                best_diff = diff
                best_match = (i, e_int)

        if best_match is not None:
            used_exp.add(best_match[0])
            aligned_exp.append(best_match[1])
            aligned_theor.append(t_int)

    if not aligned_exp:
        return 0.0

    dot_product = sum(e * t for e, t in zip(aligned_exp, aligned_theor))
    norm_exp = math.sqrt(sum(e * e for _, e in exp_env))
    norm_theor = math.sqrt(sum(t * t for _, t in theor_env))

    if norm_exp == 0.0 or norm_theor == 0.0:
        return 0.0

    return dot_product / (norm_exp * norm_theor)


def parse_elements_from_smiles(smiles: str) -> Set[str]:
    """Extract a set of element symbols present in a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return set()
    formula = rdMolDescriptors.CalcMolFormula(mol)
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    return {element for element, _ in matches}


def find_impossible_neutral_losses(
    mz_array: list[float],
    int_array: list[float],
    precursor_mz: float,
    smiles: str,
    tolerance: float = 0.02,
    intensity_threshold: float = 0.05,
) -> list[tuple[float, float, Set[str]]]:
    """
    Identify observed neutral losses that are physically impossible given the molecular formula.

    Parameters
    ----------
    mz_array : list of float
        Fragment m/z array.
    int_array : list of float
        Fragment intensity array.
    precursor_mz : float
        Precursor m/z.
    smiles : str
        SMILES string of the candidate structure.
    tolerance : float
        m/z tolerance for matching neutral losses.
    intensity_threshold : float
        Minimum relative intensity (0.0 to 1.0) to consider a fragment peak.

    Returns
    -------
    list of tuple
        A list of impossible neutral losses detected, structured as:
        (observed_loss, exact_mass_matched, required_atoms_missing)
    """
    atoms = parse_elements_from_smiles(smiles)
    if not atoms:
        return []

    max_int = max(int_array) if len(int_array) > 0 else 0.0
    if max_int == 0.0:
        return []

    impossible_losses = []

    for mz, intensity in zip(mz_array, int_array):
        if intensity / max_int < intensity_threshold:
            continue

        nl = precursor_mz - mz
        if nl <= 0:
            continue

        for exact_mass, required_atoms in COMMON_NEUTRAL_LOSSES:
            if abs(nl - exact_mass) <= tolerance:
                if not required_atoms.issubset(atoms):
                    impossible_losses.append((nl, exact_mass, required_atoms))

    return impossible_losses
