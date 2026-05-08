import pytest

from MassFlow.cheminformatics import (
    _get_morgan_fingerprint,
    calculate_tanimoto_similarity,
    calculate_theoretical_mass,
)


def test_tanimoto_similarity():
    # Aspirin
    smiles1 = "CC(=O)OC1=CC=CC=C1C(=O)O"
    # Salicylic Acid (structurally similar, differs by acetyl group)
    smiles2 = "C1=CC=C(C(=C1)C(=O)O)O"

    score = calculate_tanimoto_similarity(smiles1, smiles2)
    assert score is not None
    assert 0.4 < score < 1.0  # They share a significant substructure


def test_tanimoto_invalid_smiles():
    score = calculate_tanimoto_similarity("INVALID_SMILES", "C")
    assert score is None


def test_theoretical_mass_protonated():
    # Caffeine: C8H10N4O2
    # Exact mass is ~194.08038
    # Protonated [M+H]+ should be ~195.087
    mass = calculate_theoretical_mass("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "[M+H]+")
    assert mass is not None
    assert abs(mass - 195.08766) < 0.001


def test_theoretical_mass_sodiated():
    # Caffeine [M+Na]+ should be ~217.069
    mass = calculate_theoretical_mass("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "[M+Na]+")
    assert mass is not None
    assert abs(mass - 217.0696) < 0.001


def test_theoretical_mass_invalid_adduct():
    with pytest.raises(ValueError, match="is not supported"):
        calculate_theoretical_mass("C", "[M+UNKNOWN]+")


def test_caching_behavior():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    # Run once to populate cache
    _get_morgan_fingerprint(smiles)

    # Check cache info
    info = _get_morgan_fingerprint.cache_info()
    assert info.hits == 0
    assert info.misses >= 1

    # Run again, should hit cache
    _get_morgan_fingerprint(smiles)
    info2 = _get_morgan_fingerprint.cache_info()
    assert info2.hits == 1
