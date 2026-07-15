"""
Cheminformatics utilities for MassFlow.

This module provides high-performance, memoized calculations for structural
properties including exact masses with adduct offsets, isotopic envelope
generation, and Tanimoto similarity scores using Morgan fingerprints.

RDKit is an **optional** dependency (installed via the ``[chem]`` extra).
When RDKit is unavailable:

- Morgan fingerprinting and Tanimoto similarity return ``None``.
- Mass calculation and isotopic envelope generation fall back to
  pyteomics if a ``formula`` string is supplied directly.
- SMILES-based operations that require molecular parsing are skipped
  gracefully, and the pipeline continues with classical cosine scoring.
"""

import logging
import math
from collections import defaultdict
from functools import lru_cache
from typing import Optional

import pyteomics.mass as pmass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional RDKit import
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem, DataStructs  # noqa: F811
    from rdkit.Chem import rdMolDescriptors

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover -- tested via the "no rdkit" CI variant
    _HAS_RDKIT = False
    logger.info(
        "RDKit is not installed. Install the 'chem' extra for structural "
        "cheminformatics features (Tanimoto similarity, SMILES parsing). "
        "Classical cosine scoring remains fully functional."
    )

# Electron rest mass (Da). pyteomics deliberately omits the electron from neutral
# atomic compositions, so we apply it explicitly when an ion gains or loses charge.
# This is the one mass constant we keep: every element mass now comes from pyteomics.
ELECTRON_MASS = 0.0005485799

# Adduct specifications: each adduct maps to (atoms_formula, charge).
#
# atoms_formula is a pyteomics-compatible chemical formula describing the atoms
# added to the neutral molecule M. Use positive element counts for additions
# and negative counts (e.g. "H-1") for subtractions. An empty string means no
# atoms change (radical ions formed purely by electron loss/gain).
#
# charge is the signed integer ion charge. The net m/z shift is computed via
# Composition arithmetic:
#     offset = pmass.calculate_mass(formula=atoms_formula) - charge × ELECTRON_MASS
# and the observed m/z is (neutral_mass + offset) / abs(charge).
_ADDUCT_SPECS: dict[str, tuple[str, int]] = {
    # Positive Ion Mode
    "[M+H]+": ("H", 1),
    "[M+NH4]+": ("NH4", 1),
    "[M+Na]+": ("Na", 1),
    "[M+K]+": ("K", 1),
    "[M]+": ("", 1),
    "[M+2H]2+": ("H2", 2),  # doubly protonated, m/z = (M + 2 protons) / 2
    # Negative Ion Mode
    "[M-H]-": ("H-1", -1),
    "[M+Cl]-": ("Cl", -1),
    "[M+HCOO]-": ("CHO2", -1),
    "[M+CH3COO]-": ("C2H3O2", -1),
    "[M+FA-H]-": ("CHO2", -1),  # Formate (common LC-MS alias)
    "[M]-": ("", -1),
}


# =============================================================================
# Public API: adduct offset (pure pyteomics, no RDKit dependency)
# =============================================================================


@lru_cache(maxsize=256)
def compute_adduct_offset(adduct: str) -> float | None:
    """
    Compute the mass offset for an adduct using pyteomics Composition arithmetic.

    The offset represents the net mass change when a neutral molecule forms
    the specified adduct ion: the mass of added/removed atoms minus the electron
    mass correction for the ion's charge.

    Parameters
    ----------
    adduct : str
        The adduct notation (e.g., ``"[M+H]+"``, ``"[M-H]-"``).

    Returns
    -------
    float or None
        The mass offset in Da, or None if the adduct is not recognized.
    """
    spec = _ADDUCT_SPECS.get(adduct)
    if spec is None:
        return None
    atoms_formula, charge = spec
    if atoms_formula:
        comp = pmass.Composition(formula=atoms_formula)
        atoms_mass = pmass.calculate_mass(composition=comp)
    else:
        atoms_mass = 0.0
    return atoms_mass - charge * ELECTRON_MASS


# =============================================================================
# Formula helpers (work with or without RDKit)
# =============================================================================


def _formula_to_monoisotopic_mass(formula: str) -> float:
    """Compute the monoisotopic mass from a pyteomics-compatible formula string.

    Parameters
    ----------
    formula : str
        A chemical formula parsable by ``pyteomics.mass.Composition``
        (e.g. ``"C8H10N4O2"``).

    Returns
    -------
    float
        The monoisotopic exact mass in Da.
    """
    return pmass.calculate_mass(formula=formula)


def _formula_to_isotopic_envelope(
    formula: str, max_isopeaks: int = 4
) -> list[tuple[float, float]]:
    """Compute the theoretical isotopic envelope from a formula string using pyteomics.

    Parameters
    ----------
    formula : str
        A pyteomics-compatible chemical formula.
    max_isopeaks : int
        Maximum number of isotopic peaks to return (M, M+1, M+2, …).

    Returns
    -------
    list of tuple of (float, float)
        Each tuple is ``(centroid_mass, relative_abundance)``, sorted by mass.
    """
    try:
        comp = pmass.Composition(formula=formula)
        mono_mass = pmass.calculate_mass(formula=formula)
    except Exception:
        logger.debug("Could not parse formula '%s' for isotopic envelope.", formula)
        return []

    isotopologues = pmass.isotopologues(
        comp, report_abundance=True, overall_threshold=1e-8
    )

    bins: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    for iso_comp, abundance in isotopologues:
        iso_mass = pmass.calculate_mass(iso_comp)
        offset = round(iso_mass - mono_mass)
        bins[offset].append((iso_mass, abundance))

    total_abundances = {
        offset: sum(a for _m, a in items) for offset, items in bins.items()
    }
    if not total_abundances:
        return []

    max_abund = max(total_abundances.values())
    result: list[tuple[float, float]] = []

    for _offset, items in sorted(bins.items())[:max_isopeaks]:
        total_a = sum(a for _m, a in items)
        if total_a > 0:
            rel_a = total_a / max_abund
            centroid_mass = sum(m * a for m, a in items) / total_a
            result.append((round(centroid_mass, 6), round(rel_a, 6)))

    return result


# =============================================================================
# RDKit-dependent helpers (only defined when RDKit is available)
# =============================================================================


if _HAS_RDKIT:

    @lru_cache(maxsize=16384)
    def _get_morgan_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048):
        """
        Generate and cache a Morgan fingerprint (as a bit vector) from a SMILES string.

        Only available when RDKit is installed. Returns ``None`` otherwise.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

    def _mol_to_pyteomics_formula(mol) -> str:
        """
        Convert an RDKit molecule to a formula string compatible with pyteomics.

        For standard (non-isotopically-labeled) molecules this simply delegates to
        ``rdMolDescriptors.CalcMolFormula``. For isotope-labeled molecules the
        formula is constructed manually with pyteomics' ``{ELEMENT}[MASS]COUNT``
        notation (e.g. ``C[13]6H6``) so that ``pmass.calculate_mass`` returns the
        correct isotopically-weighted monoisotopic mass.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            A parsed RDKit molecule.

        Returns
        -------
        str
            A pyteomics-compatible chemical formula.
        """
        has_isotope = any(atom.GetIsotope() > 0 for atom in mol.GetAtoms())
        if not has_isotope:
            return rdMolDescriptors.CalcMolFormula(mol)

        # Isotope-labelled: build the formula with pyteomics isotope notation.
        # Get the base formula from an isotope-zeroed copy to pick up implicit H.
        mol_no_iso = Chem.Mol(mol)
        for atom in mol_no_iso.GetAtoms():
            atom.SetIsotope(0)
        base_formula = rdMolDescriptors.CalcMolFormula(mol_no_iso)

        # Use pyteomics Composition to parse the base formula instead of regex.
        base_comp = pmass.Composition(formula=base_formula)
        base_counts: dict[str, int] = dict(base_comp)

        counts: defaultdict[str, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            isotope = atom.GetIsotope()
            counts[symbol][isotope] += 1

        parts: list[str] = []
        for symbol in sorted(base_counts.keys()):
            total = base_counts[symbol]
            if symbol in counts:
                # Add isotopically-labeled atoms first: C[13]6
                iso_items = [
                    (iso, counts[symbol][iso])
                    for iso in sorted(counts[symbol].keys())
                    if iso > 0
                ]
                for iso, cnt in iso_items:
                    parts.append(f"{symbol}[{iso}]{cnt}")
                # Non-labeled atoms
                non_iso_count = counts[symbol].get(0, 0)
                if non_iso_count > 0:
                    parts.append(f"{symbol}{non_iso_count}")
                # Remainder from base formula (e.g. implicit H not captured explicitly)
                sum_counts = non_iso_count + sum(cnt for _, cnt in iso_items)
                if sum_counts < total:
                    parts.append(f"{symbol}{total - sum_counts}")
            else:
                parts.append(f"{symbol}{total}")

        return "".join(parts)

    def _smiles_to_formula(smiles: str) -> Optional[str]:
        """Parse a SMILES string and return a pyteomics-compatible formula.

        Returns ``None`` if the SMILES cannot be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        return _mol_to_pyteomics_formula(mol)

else:
    # -----------------------------------------------------------------------
    # Stubs when RDKit is absent: all RDKit-dependent functions return
    # a safe sentinel (``None`` or empty container).
    # -----------------------------------------------------------------------

    def _get_morgan_fingerprint(  # type: ignore[misc]
        smiles: str, radius: int = 2, nBits: int = 2048
    ):
        """Morgan fingerprinting requires RDKit.  Returns ``None``."""
        return None

    # Make the lru_cache decorator work on the stub as well.
    _get_morgan_fingerprint = lru_cache(maxsize=16384)(_get_morgan_fingerprint)

    def _mol_to_pyteomics_formula(mol) -> str:  # type: ignore[no-redef]
        """Requires RDKit. Raises RuntimeError if called without RDKit."""
        raise RuntimeError(
            "_mol_to_pyteomics_formula requires RDKit. Use formula-based APIs instead."
        )

    def _smiles_to_formula(smiles: str) -> Optional[str]:
        """SMILES-to-formula conversion requires RDKit. Returns ``None``."""
        return None


# =============================================================================
# Tanimoto similarity (requires RDKit)
# =============================================================================


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> Optional[float]:
    """
    Calculate the Tanimoto similarity score between two SMILES strings using
    cached Morgan fingerprints.

    Requires RDKit (the ``[chem]`` extra).  Returns ``None`` if RDKit is not
    installed, or if either SMILES is invalid.

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
        or RDKit is unavailable.
    """
    if not _HAS_RDKIT:
        logger.debug("Tanimoto similarity skipped: RDKit not installed.")
        return None

    if not smiles1 or not smiles2:
        return None

    fp1 = _get_morgan_fingerprint(smiles1)
    fp2 = _get_morgan_fingerprint(smiles2)

    if not fp1 or not fp2:
        return None

    return DataStructs.TanimotoSimilarity(fp1, fp2)


# =============================================================================
# Isotopic distribution & envelope
# =============================================================================


@lru_cache(maxsize=16384)
def get_isotopic_distribution(
    smiles: str = "",
    threshold: float = 0.001,
    *,
    formula: str = "",
) -> list[tuple[float, float]]:
    """
    Calculate the theoretical isotopic distribution for a molecule.

    Uses pyteomics to generate precise isotopologues, grouped by nominal
    mass offset (M, M+1, M+2, etc.).  When RDKit is available the molecule
    is parsed from ``smiles``; otherwise a ``formula`` string must be supplied.

    Parameters
    ----------
    smiles : str
        SMILES string (requires RDKit).
    formula : str
        Chemical formula parsable by pyteomics (e.g. ``"C8H10N4O2"``).
        Takes precedence over ``smiles`` when both are provided.
    threshold : float
        Relative abundance threshold (0.0 to 1.0) below which isotopic peaks
        are ignored.

    Returns
    -------
    list of tuple of (float, float)
        Sorted list of ``(centroid_mass, relative_abundance)`` tuples.
    """
    # Prefer formula when given (works without RDKit).
    if formula:
        try:
            comp = pmass.Composition(formula=formula)
            mono_mass = pmass.calculate_mass(formula=formula)
        except Exception:
            return []
        isotopologues = pmass.isotopologues(
            comp, report_abundance=True, overall_threshold=1e-8
        )
    elif _HAS_RDKIT and smiles:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        formula_rd = rdMolDescriptors.CalcMolFormula(mol)
        try:
            comp = pmass.Composition(formula=formula_rd)
            mono_mass = pmass.calculate_mass(formula=formula_rd)
        except Exception:
            return []
        isotopologues = pmass.isotopologues(
            comp, report_abundance=True, overall_threshold=1e-8
        )
    else:
        logger.debug(
            "Isotopic distribution requires either formula= or smiles= "
            "(with RDKit installed)."
        )
        return []

    bins: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    for iso_comp, abundance in isotopologues:
        iso_mass = pmass.calculate_mass(iso_comp)
        offset = round(iso_mass - mono_mass)
        bins[offset].append((iso_mass, abundance))

    total_abundances = {
        offset: sum(a for _m, a in items) for offset, items in bins.items()
    }
    if not total_abundances:
        return []

    max_abund = max(total_abundances.values())
    result: list[tuple[float, float]] = []

    for _offset, items in sorted(bins.items()):
        total_a = sum(a for _m, a in items)
        rel_a = total_a / max_abund
        if rel_a >= threshold:
            centroid_mass = sum(m * a for m, a in items) / total_a
            result.append((round(centroid_mass, 6), round(rel_a, 6)))

    return result


@lru_cache(maxsize=16384)
def calculate_isotopic_envelope(
    smiles: str = "",
    max_isopeaks: int = 4,
    *,
    formula: str = "",
) -> list[tuple[float, float]]:
    """
    Calculate the theoretical isotopic envelope (M, M+1, M+2, …), normalized
    such that the base peak is 1.0.

    When RDKit is available, the molecule is parsed from ``smiles`` (including
    isotope-labelled SMILES like ``[13CH3]C(=O)O``).  Without RDKit, a
    ``formula`` string can be supplied directly; pyteomics will compute the
    envelope from the atomic composition.

    Parameters
    ----------
    smiles : str
        SMILES string (requires RDKit).
    formula : str
        Chemical formula parsable by pyteomics (e.g. ``"C8H10N4O2"``).
        Takes precedence over ``smiles`` when both are provided.
    max_isopeaks : int
        Maximum number of isotopic peaks to return (default 4).

    Returns
    -------
    list of tuple of (float, float)
        Sorted list of ``(centroid_mass, relative_abundance)`` pairs.
    """
    # ── Path A: formula provided (works with or without RDKit) ──────────
    if formula:
        return _formula_to_isotopic_envelope(formula, max_isopeaks)

    # ── Path B: SMILES via RDKit ────────────────────────────────────────
    if not _HAS_RDKIT:
        logger.debug(
            "Isotopic envelope from SMILES requires RDKit. "
            "Provide a 'formula' string instead."
        )
        return []

    if not smiles:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []

    # Attempt to use RDKit's native method if available.
    if hasattr(rdMolDescriptors, "GetIsotopicDistribution"):
        try:
            dist = rdMolDescriptors.GetIsotopicDistribution(mol)
            peaks: list[tuple[float, float]] = []
            for item in dist:
                if hasattr(item, "mass") and hasattr(item, "abundance"):
                    peaks.append((item.mass, item.abundance))
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    peaks.append((item[0], item[1]))

            if peaks:
                max_abund = max(p[1] for p in peaks)
                normalized = [(round(m, 6), round(a / max_abund, 6)) for m, a in peaks]
                normalized.sort(key=lambda x: x[0])
                return normalized[:max_isopeaks]
        except Exception:
            pass  # Fall through to pyteomics

    # ── Pyteomics fallback for SMILES (may lose isotope-labelling precision) ──
    try:
        has_isotope = any(atom.GetIsotope() > 0 for atom in mol.GetAtoms())
        if has_isotope:
            try:
                mol_no_iso = Chem.Mol(mol)
                for atom in mol_no_iso.GetAtoms():
                    atom.SetIsotope(0)
                base_formula = rdMolDescriptors.CalcMolFormula(mol_no_iso)
            except Exception:
                base_formula = rdMolDescriptors.CalcMolFormula(mol)

            base_comp = pmass.Composition(formula=base_formula)
            base_counts: dict[str, int] = dict(base_comp)

            counts: defaultdict[str, defaultdict[int, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                isotope = atom.GetIsotope()
                counts[symbol][isotope] += 1

            parts: list[str] = []
            for symbol in sorted(base_counts.keys()):
                total = base_counts[symbol]
                if symbol in counts:
                    isotopic_items = [
                        (iso, counts[symbol][iso])
                        for iso in sorted(counts[symbol].keys())
                        if iso > 0
                    ]
                    for iso, cnt in isotopic_items:
                        parts.append(f"{symbol}[{iso}]{cnt}")
                    non_iso_count = counts[symbol].get(0, 0)
                    if non_iso_count > 0:
                        parts.append(f"{symbol}{non_iso_count}")
                    sum_counts = non_iso_count + sum(cnt for _, cnt in isotopic_items)
                    if sum_counts < total:
                        parts.append(f"{symbol}{total - sum_counts}")
                else:
                    parts.append(f"{symbol}{total}")

            formula_py = "".join(parts)
            comp = pmass.Composition(formula=formula_py)
            mono_mass = pmass.calculate_mass(formula=formula_py)
        else:
            formula_rd = rdMolDescriptors.CalcMolFormula(mol)
            comp = pmass.Composition(formula=formula_rd)
            mono_mass = pmass.calculate_mass(formula=formula_rd)
    except Exception:
        return []

    isotopologues = pmass.isotopologues(
        comp, report_abundance=True, overall_threshold=1e-8
    )

    bins: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    for iso_comp, abundance in isotopologues:
        iso_mass = pmass.calculate_mass(iso_comp)
        offset = round(iso_mass - mono_mass)
        bins[offset].append((iso_mass, abundance))

    total_abundances = {
        offset: sum(a for _m, a in items) for offset, items in bins.items()
    }
    if not total_abundances:
        return []

    max_abund = max(total_abundances.values())
    result: list[tuple[float, float]] = []

    for _offset, items in sorted(bins.items())[:max_isopeaks]:
        total_a = sum(a for _m, a in items)
        if total_a > 0:
            rel_a = total_a / max_abund
            centroid_mass = sum(m * a for m, a in items) / total_a
            result.append((round(centroid_mass, 6), round(rel_a, 6)))

    return result


# =============================================================================
# Theoretical mass calculation (formula-first, SMILES as fallback)
# =============================================================================


@lru_cache(maxsize=16384)
def calculate_theoretical_mass(
    smiles: str = "",
    adduct: str = "[M+H]+",
    *,
    formula: str = "",
) -> Optional[float]:
    """
    Calculate the theoretical m/z for a chemical structure given an adduct.

    The neutral monoisotopic mass is computed strictly via ``pyteomics.mass``.
    When RDKit is available, the formula is derived from ``smiles``.  Without
    RDKit, a ``formula`` string must be supplied.

    Parameters
    ----------
    smiles : str
        SMILES string (requires RDKit).
    formula : str
        Chemical formula parsable by pyteomics (e.g. ``"C8H10N4O2"``).
        Takes precedence over ``smiles`` when both are provided.
    adduct : str
        The adduct notation (e.g., ``"[M+H]+"``, ``"[M+Na]+"``).

    Returns
    -------
    float or None
        The theoretical m/z of the adduct, or None if the structure cannot
        be resolved.

    Raises
    ------
    ValueError
        If the adduct is not recognized in the internal offset registry.
    """
    # ── Resolve neutral formula ────────────────────────────────────────
    neutral_formula: Optional[str] = None

    if formula:
        neutral_formula = formula
    elif _HAS_RDKIT and smiles:
        neutral_formula = _smiles_to_formula(smiles)

    if neutral_formula is None:
        logger.debug(
            "Cannot compute theoretical mass: no formula available. "
            "Provide 'formula=' or install RDKit for SMILES support."
        )
        return None

    neutral_mass = pmass.calculate_mass(formula=neutral_formula)

    # ── Compute adduct offset ──────────────────────────────────────────
    offset = compute_adduct_offset(adduct)
    if offset is None:
        raise ValueError(
            f"Adduct '{adduct}' is not supported. "
            f"Supported adducts: {list(_ADDUCT_SPECS.keys())}"
        )

    _atoms_formula, spec_charge = _ADDUCT_SPECS[adduct]
    return (neutral_mass + offset) / abs(spec_charge)


# =============================================================================
# Isotopic envelope similarity (pure math, no RDKit dependency)
# =============================================================================


def calculate_isotopic_similarity(
    exp_env: list[tuple[float, float]],
    theor_env: list[tuple[float, float]],
    mz_tolerance: float = 0.05,
) -> float:
    """
    Calculate the cosine similarity between an experimental and theoretical
    isotopic envelope.

    Parameters
    ----------
    exp_env : list of tuple
        Experimental MS1 isotopic envelope as ``(mz, abundance)`` pairs.
    theor_env : list of tuple
        Theoretical MS1 isotopic envelope as ``(mz, abundance)`` pairs.
    mz_tolerance : float
        Tolerance in Da to match an experimental peak to a theoretical one.

    Returns
    -------
    float
        Cosine similarity score between 0.0 and 1.0.
    """
    if not exp_env or not theor_env:
        return 0.0

    # Align peaks greedily
    aligned_exp: list[float] = []
    aligned_theor: list[float] = []

    used_exp: set[int] = set()
    for t_mz, t_int in theor_env:
        best_match: Optional[tuple[int, float]] = None
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
    norm_exp = math.sqrt(sum(e * e for _m, e in exp_env))
    norm_theor = math.sqrt(sum(t * t for _m, t in theor_env))

    if norm_exp == 0.0 or norm_theor == 0.0:
        return 0.0

    return dot_product / (norm_exp * norm_theor)
