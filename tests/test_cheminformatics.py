import pytest

from MassFlow.cheminformatics import (
    BR_MASS,
    C_MASS,
    CL_MASS,
    COMMON_NEUTRAL_LOSSES,
    F_MASS,
    H_MASS,
    N_MASS,
    O_MASS,
    P_MASS,
    S_MASS,
    _get_morgan_fingerprint,
    calculate_tanimoto_similarity,
    calculate_theoretical_mass,
    find_impossible_neutral_losses,
    parse_elements_from_smiles,
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


# First-principles expected masses for every entry in COMMON_NEUTRAL_LOSSES,
# computed from the element constants defined in cheminformatics.py.
# Table index must stay in sync with the declaration order in that file.
_NEUTRAL_LOSS_FIRST_PRINCIPLES = [
    ("H2O", 0, 2 * H_MASS + 1 * O_MASS),
    ("NH3", 1, 1 * N_MASS + 3 * H_MASS),
    ("CO", 2, 1 * C_MASS + 1 * O_MASS),
    ("CO2", 3, 1 * C_MASS + 2 * O_MASS),
    ("H2S", 4, 2 * H_MASS + 1 * S_MASS),
    ("SO2", 5, 1 * S_MASS + 2 * O_MASS),
    ("PO3", 6, 1 * P_MASS + 3 * O_MASS),
    ("HCl", 7, 1 * H_MASS + 1 * CL_MASS),
    ("HBr", 8, 1 * H_MASS + 1 * BR_MASS),
    ("HF", 9, 1 * H_MASS + 1 * F_MASS),
]


@pytest.mark.parametrize(
    "name,table_index,expected_mass", _NEUTRAL_LOSS_FIRST_PRINCIPLES
)
def test_neutral_loss_mass_first_principles(name, table_index, expected_mass):
    """
    Regression test: every COMMON_NEUTRAL_LOSSES mass must agree with a
    first-principles calculation (element exact masses from NIST) to within
    0.5 mDa.  A failure here means the tabulated value is wrong for the named
    neutral loss fragment.
    """
    recorded_mass = COMMON_NEUTRAL_LOSSES[table_index][0]
    deviation_da = abs(recorded_mass - expected_mass)
    assert deviation_da < 5e-4, (
        f"{name}: recorded mass {recorded_mass:.4f} differs from "
        f"first-principles {expected_mass:.4f} by {deviation_da * 1000:.2f} mDa"
    )


def test_co2_loss_on_one_oxygen_molecule_is_impossible():
    """CO₂ loss requires ≥ 2 oxygen atoms; methanol (1 O) must be flagged."""
    # methanol CH4O: precursor 100, fragment at 56.0102 → NL = 43.9898 (CO2)
    precursor_mz = 100.0
    fragment_mz = precursor_mz - 43.9898
    losses = find_impossible_neutral_losses(
        mz_array=[fragment_mz],
        int_array=[100.0],
        precursor_mz=precursor_mz,
        smiles="CO",  # methanol: C1H4O1
    )
    assert len(losses) == 1
    nl, exact, req = losses[0]
    assert abs(nl - 43.9898) < 0.01
    assert req == {"C": 1, "O": 2}


def test_co2_loss_on_three_oxygen_glycerol_is_possible():
    """CO₂ loss on glycerol (3 O) must NOT be flagged — 3 ≥ 2 required."""
    precursor_mz = 100.0
    fragment_mz = precursor_mz - 43.9898
    losses = find_impossible_neutral_losses(
        mz_array=[fragment_mz],
        int_array=[100.0],
        precursor_mz=precursor_mz,
        smiles="OCC(O)CO",  # glycerol: C3H8O3
    )
    assert losses == []


def test_po3_loss_on_one_oxygen_phospho_compound_is_impossible():
    """PO₃ loss requires ≥ 3 oxygen atoms; trimethylphosphine oxide (1 O) must be flagged."""
    # trimethylphosphine oxide: O=P(C)(C)C → C3H9OP (1 O, 1 P)
    precursor_mz = 150.0
    fragment_mz = precursor_mz - 78.9585  # PO3 mass
    losses = find_impossible_neutral_losses(
        mz_array=[fragment_mz],
        int_array=[100.0],
        precursor_mz=precursor_mz,
        smiles="O=P(C)(C)C",  # 1 O, 1 P → PO3 impossible
    )
    assert len(losses) == 1
    _, _, req = losses[0]
    assert req == {"P": 1, "O": 3}


def test_h2o_loss_on_oxygen_containing_molecule_is_possible():
    """Regression: H₂O loss on a molecule with O must not be flagged."""
    # ethanol CCO: C2H6O → H2O requires H ≥ 2, O ≥ 1 → possible
    precursor_mz = 100.0
    fragment_mz = precursor_mz - 18.0106
    losses = find_impossible_neutral_losses(
        mz_array=[fragment_mz],
        int_array=[100.0],
        precursor_mz=precursor_mz,
        smiles="CCO",  # ethanol: C2H6O
    )
    assert losses == []


def test_parse_elements_from_smiles_returns_counts():
    """parse_elements_from_smiles must return element counts, not just a set."""
    counts = parse_elements_from_smiles("OCC(O)CO")  # glycerol C3H8O3
    assert counts["C"] == 3
    assert counts["O"] == 3
    assert counts["H"] == 8


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
