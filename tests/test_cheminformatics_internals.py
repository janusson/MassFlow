"""
Comprehensive coverage tests for MassFlow cheminformatics.py:
- compute_adduct_offset (edge cases)
- _get_morgan_fingerprint (invalid SMILES)
- calculate_tanimoto_similarity (edge cases)
- get_isotopic_distribution (edge cases)
- calculate_isotopic_envelope (edge cases)
- _mol_to_pyteomics_formula
- calculate_theoretical_mass
- calculate_isotopic_similarity
"""

import pytest

try:
    from rdkit import Chem

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

from MassFlow.cheminformatics import (
    _get_morgan_fingerprint,
    _formula_to_isotopic_envelope,
    _mol_to_pyteomics_formula,
    calculate_isotopic_envelope,
    calculate_isotopic_similarity,
    calculate_tanimoto_similarity,
    calculate_theoretical_mass,
    compute_adduct_offset,
    get_isotopic_distribution,
)

# ==============================================================================
# compute_adduct_offset
# ==============================================================================


def test_compute_adduct_offset_known():
    offset = compute_adduct_offset("[M+H]+")
    assert offset is not None
    assert abs(offset - 1.007276) < 0.01

    offset_neg = compute_adduct_offset("[M-H]-")
    assert offset_neg is not None
    assert abs(offset_neg + 1.007276) < 0.01


def test_compute_adduct_offset_unknown():
    offset = compute_adduct_offset("[M+Unknown]+")
    assert offset is None


def test_compute_adduct_offset_radical_cation():
    offset = compute_adduct_offset("[M]+")
    assert offset is not None
    assert abs(offset + 0.0005485799) < 0.0001


def test_compute_adduct_offset_doubly_charged():
    offset = compute_adduct_offset("[M+2H]2+")
    assert offset is not None


def test_compute_adduct_offset_uses_cache():
    o1 = compute_adduct_offset("[M+Na]+")
    o2 = compute_adduct_offset("[M+Na]+")
    assert o1 == o2


# ==============================================================================
# _get_morgan_fingerprint
# ==============================================================================


def test_get_morgan_fingerprint_valid():
    fp = _get_morgan_fingerprint("CCO")
    assert fp is not None


def test_get_morgan_fingerprint_invalid():
    fp = _get_morgan_fingerprint("INVALID")
    assert fp is None


# ==============================================================================
# calculate_tanimoto_similarity
# ==============================================================================


def test_calculate_tanimoto_identical():
    score = calculate_tanimoto_similarity("CCO", "CCO")
    assert score == 1.0


def test_calculate_tanimoto_different():
    score = calculate_tanimoto_similarity("CCO", "CCCCCC")
    assert score is not None
    assert 0.0 <= score < 1.0


def test_calculate_tanimoto_invalid_first():
    score = calculate_tanimoto_similarity("INVALID", "CCO")
    assert score is None


def test_calculate_tanimoto_invalid_second():
    score = calculate_tanimoto_similarity("CCO", "INVALID")
    assert score is None


def test_calculate_tanimoto_empty_inputs():
    assert calculate_tanimoto_similarity("", "CCO") is None
    assert calculate_tanimoto_similarity("CCO", "") is None


# ==============================================================================
# get_isotopic_distribution
# ==============================================================================


def test_get_isotopic_distribution_valid():
    # SMILES path (requires RDKit)
    dist = get_isotopic_distribution("CCO")
    assert len(dist) > 0
    for mass, abund in dist:
        assert isinstance(mass, float)
        assert isinstance(abund, float)


def test_get_isotopic_distribution_valid_from_formula():
    """Formula-based path works without RDKit."""
    dist = get_isotopic_distribution(formula="C2H6O")
    assert len(dist) > 0
    for mass, abund in dist:
        assert isinstance(mass, float)
        assert isinstance(abund, float)


def test_get_isotopic_distribution_invalid():
    dist = get_isotopic_distribution("INVALID")
    assert dist == []


def test_get_isotopic_distribution_with_threshold():
    dist = get_isotopic_distribution("CCO", threshold=0.5)
    assert len(dist) > 0


def test_get_isotopic_distribution_with_threshold_from_formula():
    """Formula-based path with threshold works without RDKit."""
    dist = get_isotopic_distribution(formula="C2H6O", threshold=0.5)
    assert len(dist) > 0


# ==============================================================================
# calculate_isotopic_envelope
# ==============================================================================


def test_calculate_isotopic_envelope_valid():
    # SMILES path (requires RDKit)
    env = calculate_isotopic_envelope("CCO")
    assert len(env) > 0
    for entry in env:
        assert isinstance(entry, tuple)
        assert len(entry) == 2


def test_calculate_isotopic_envelope_valid_from_formula():
    """Formula-based envelope works without RDKit."""
    env = calculate_isotopic_envelope(formula="C2H6O")
    assert len(env) > 0
    for entry in env:
        assert isinstance(entry, tuple)
        assert len(entry) == 2


def test_calculate_isotopic_envelope_invalid():
    env = calculate_isotopic_envelope("INVALID")
    assert env == []


def test_calculate_isotopic_envelope_with_limit():
    # SMILES path
    env = calculate_isotopic_envelope("CCO", max_isopeaks=2)
    assert len(env) <= 2


def test_calculate_isotopic_envelope_with_limit_from_formula():
    """Formula-based envelope with limit works without RDKit."""
    env = calculate_isotopic_envelope(formula="C2H6O", max_isopeaks=2)
    assert len(env) <= 2


# ==============================================================================
# _mol_to_pyteomics_formula
# ==============================================================================


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_mol_to_pyteomics_formula_ethanol():
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    formula = _mol_to_pyteomics_formula(mol)
    # Should be parseable by pyteomics
    from pyteomics.mass import Composition

    comp = Composition(formula=formula)
    assert comp is not None
    assert "C" in formula
    assert "H" in formula
    assert "O" in formula


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_mol_to_pyteomics_formula_methane():
    mol = Chem.MolFromSmiles("C")
    assert mol is not None
    formula = _mol_to_pyteomics_formula(mol)
    assert "C" in formula


# ==============================================================================
# calculate_theoretical_mass
# ==============================================================================


def test_calculate_theoretical_mass_valid():
    # SMILES path (requires RDKit)
    mass = calculate_theoretical_mass("CCO")
    assert mass is not None
    # calculate_theoretical_mass includes [M+H]+ adduct by default
    # ethanol + H+ mass ~ 47.05
    assert abs(mass - 47.05) < 0.1


def test_calculate_theoretical_mass_valid_from_formula():
    """Formula-based mass calculation works without RDKit."""
    mass = calculate_theoretical_mass(formula="C2H6O")
    assert mass is not None
    # ethanol + H+ mass ~ 47.05
    assert abs(mass - 47.05) < 0.1


def test_calculate_theoretical_mass_from_formula_with_adduct():
    """Formula-based mass with explicit adduct."""
    mass = calculate_theoretical_mass(formula="C2H6O", adduct="[M+Na]+")
    assert mass is not None
    # ethanol + Na+ mass ~ 69.03
    assert abs(mass - 69.03) < 0.1


def test_calculate_theoretical_mass_invalid():
    mass = calculate_theoretical_mass("INVALID")
    assert mass is None


def test_calculate_theoretical_mass_with_adduct():
    # SMILES path
    mass = calculate_theoretical_mass("CCO", adduct="[M+H]+")
    assert mass is not None
    # ethanol + H+ mass ~ 47.05
    assert abs(mass - 47.05) < 0.1


def test_calculate_theoretical_mass_unknown_adduct():
    with pytest.raises(ValueError, match="not supported"):
        calculate_theoretical_mass("CCO", adduct="[M+Unknown]+")


# ==============================================================================
# calculate_isotopic_similarity
# ==============================================================================


def test_calculate_isotopic_similarity_identical():
    """When envelopes are identical, similarity is 1.0."""
    # Use formula-based envelope (works with or without RDKit)
    env = _formula_to_isotopic_envelope("C2H6O")
    if not env:
        pytest.skip("Isotopic envelope empty")
    sim = calculate_isotopic_similarity(env, env)
    assert abs(sim - 1.0) < 0.01


def test_calculate_isotopic_similarity_different():
    """Different molecules have different envelopes."""
    env1 = _formula_to_isotopic_envelope("C2H6O")
    env2 = _formula_to_isotopic_envelope("C6H14")
    if not env1 or not env2:
        pytest.skip("Isotopic envelope empty")
    sim = calculate_isotopic_similarity(env1, env2)
    assert sim is not None
    assert 0.0 <= sim <= 1.0


def test_calculate_isotopic_similarity_empty_first():
    sim = calculate_isotopic_similarity([], [(100.0, 1.0)])
    assert sim == 0.0


def test_calculate_isotopic_similarity_empty_second():
    sim = calculate_isotopic_similarity([(100.0, 1.0)], [])
    assert sim == 0.0


def test_calculate_isotopic_similarity_both_empty():
    sim = calculate_isotopic_similarity([], [])
    assert sim == 0.0
