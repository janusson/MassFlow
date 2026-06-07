"""
Cheminformatics utilities bridging MassFlow and RDKit.

This module provides high-performance, memoized calculations for structural properties,
including exact masses with adduct offsets and Tanimoto similarity scores using
Morgan fingerprints.
"""

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Optional

import pyteomics.mass as pmass
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

# Electron rest mass (Da). pyteomics deliberately omits the electron from neutral
# atomic compositions, so we apply it explicitly when an ion gains or loses charge.
# This is the one mass constant we keep: every element mass now comes from pyteomics.
ELECTRON_MASS = 0.0005485799

# Adduct definitions: each adduct maps to (component_spec, charge).
#
# component_spec is a sequence of signed chemical formulas describing the atoms
# added to (+) or removed from (-) the neutral molecule M, e.g. "+H", "-H",
# "+CHO2" (formate). An empty spec means no atoms change (e.g. "[M]+", a radical
# cation formed purely by electron loss).
#
# charge is the signed integer ion charge. The net m/z shift is the component
# mass minus charge × ELECTRON_MASS (losing electrons to form a cation lowers the
# mass; gaining electrons to form an anion raises it). calculate_theoretical_mass
# divides the neutral-plus-shift mass by abs(charge) to yield the observed m/z.
_ADDUCT_DEFS: dict[str, tuple[str, int]] = {
    # Positive Ion Mode
    "[M+H]+": ("+H", 1),
    "[M+NH4]+": ("+NH4", 1),
    "[M+Na]+": ("+Na", 1),
    "[M+K]+": ("+K", 1),
    "[M]+": ("", 1),
    "[M+2H]2+": ("+H2", 2),  # doubly protonated, m/z = (M + 2 protons) / 2
    # Negative Ion Mode
    "[M-H]-": ("-H", -1),
    "[M+Cl]-": ("+Cl", -1),
    "[M+HCOO]-": ("+CHO2", -1),
    "[M+CH3COO]-": ("+C2H3O2", -1),
    "[M+FA-H]-": ("+CHO2", -1),  # Formate (common LC-MS alias)
    "[M]-": ("", -1),
}


def _component_mass(component_spec: str) -> float:
    """
    Sum the signed monoisotopic masses of the formulas in an adduct component spec.

    Parameters
    ----------
    component_spec : str
        A run of signed chemical formulas, e.g. "+H", "-H", "+CHO2". An empty
        string contributes zero mass.

    Returns
    -------
    float
        Net monoisotopic mass added (positive) or removed (negative), in Da.
    """
    total_mass = 0.0
    for sign, formula in re.findall(r"([+-])([A-Za-z0-9]+)", component_spec):
        formula_mass = pmass.calculate_mass(formula=formula)
        total_mass += formula_mass if sign == "+" else -formula_mass
    return total_mass


# Theoretical neutral-mass shifts for common LC-MS adducts, derived from
# _ADDUCT_DEFS via pyteomics. Each value is the component mass minus the electron
# mass change for the ion's charge. For z=±1 these reproduce the historical
# hardcoded offsets to sub-mDa; multiply-charged shifts are divided by abs(charge)
# downstream in calculate_theoretical_mass.
ADDUCT_OFFSETS: dict[str, float] = {
    adduct: _component_mass(component_spec) - charge * ELECTRON_MASS
    for adduct, (component_spec, charge) in _ADDUCT_DEFS.items()
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

    exact_mass = Descriptors.ExactMolWt(mol)  # type: ignore[attr-defined]

    definition = _ADDUCT_DEFS.get(adduct)
    if definition is None:
        raise ValueError(
            f"Adduct '{adduct}' is not supported. Supported adducts: {list(_ADDUCT_DEFS.keys())}"
        )

    _component_spec, charge = definition
    offset = ADDUCT_OFFSETS[adduct]

    # Divide by the absolute charge so multiply-charged adducts (e.g. [M+2H]2+)
    # report the observed m/z rather than the neutral mass.
    return (exact_mass + offset) / abs(charge)


# Common neutral losses expressed as chemical formulas. The monoisotopic mass and
# the {element: minimum_count} requirement are both derived from each formula at
# module load via pyteomics, so there is a single source of truth and no parallel
# hand-maintained dicts to drift out of sync.
#
# The element-count requirement (not mere presence) prevents false negatives such as
# CO₂ loss (needs 2 O) passing on a 1-O molecule. For example, CO₂ requires at least
# 2 oxygen atoms and PO₃ requires at least 3 oxygen atoms in the candidate's formula.
_NEUTRAL_LOSS_FORMULAS = [
    "H2O",
    "NH3",
    "CO",
    "CO2",
    "H2S",
    "SO2",
    "PO3",
    "HCl",
    "HBr",
    "HF",
]

# Each entry is (monoisotopic_mass_da, {element: minimum_count}).
# pmass.Composition(formula=f) yields a dict-like {element: count} mapping, which
# doubles as both the exact-mass input and the element-count requirement.
COMMON_NEUTRAL_LOSSES: list[tuple[float, dict[str, int]]] = [
    (pmass.calculate_mass(formula=formula), dict(pmass.Composition(formula=formula)))
    for formula in _NEUTRAL_LOSS_FORMULAS
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


def parse_elements_from_smiles(smiles: str) -> Counter:
    """
    Parse a SMILES string and return a Counter mapping each element symbol to
    its atom count in the molecular formula (implicit H included).

    Parameters
    ----------
    smiles : str
        A valid SMILES string.

    Returns
    -------
    collections.Counter
        Element-to-count mapping, e.g. Counter({'C': 3, 'H': 8, 'O': 3}) for glycerol.
        Returns an empty Counter if the SMILES is invalid or unparseable.

    Examples
    --------
    >>> parse_elements_from_smiles("OCC(O)CO")  # glycerol C3H8O3
    Counter({'H': 8, 'C': 3, 'O': 3})
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return Counter()
    formula = rdMolDescriptors.CalcMolFormula(mol)
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    return Counter({element: int(count) if count else 1 for element, count in matches})


def find_impossible_neutral_losses(
    mz_array: list[float],
    int_array: list[float],
    precursor_mz: float,
    smiles: str,
    tolerance: float = 0.02,
    intensity_threshold: float = 0.05,
) -> list[tuple[float, float, dict[str, int]]]:
    """
    Identify observed neutral losses that are physically impossible given the molecular formula.

    Element counts (not just presence) are compared against the minimum counts required by
    each neutral loss fragment.  For example, CO₂ loss requires at least 2 oxygen atoms;
    a candidate with only 1 oxygen is correctly flagged even though oxygen is present.

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
        A list of impossible neutral losses detected, each as:
        (observed_loss_da, exact_neutral_loss_da, required_element_counts)
    """
    element_counts = parse_elements_from_smiles(smiles)
    if not element_counts:
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

        for exact_mass, required_counts in COMMON_NEUTRAL_LOSSES:
            if abs(nl - exact_mass) <= tolerance:
                # Check that the candidate has at least the required count of each element.
                if not all(
                    element_counts.get(elem, 0) >= count
                    for elem, count in required_counts.items()
                ):
                    impossible_losses.append((nl, exact_mass, required_counts))

    return impossible_losses
