import pytest

try:
    from rdkit import Chem

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

from MassFlow.cheminformatics import (
    _formula_to_monoisotopic_mass,
    _mol_to_pyteomics_formula,
    calculate_isotopic_envelope,
    calculate_theoretical_mass,
)
from MassFlow.models import MolecularStructure, SpectrumMetadata


# Helper to calculate theoretical m/z for [M+H]+ using pyteomics SSOT.
def get_theoretical_mz(smiles):
    return calculate_theoretical_mass(smiles, "[M+H]+")


# Helper: calculate mass directly from formula (no RDKit needed).
def get_theoretical_mz_from_formula(formula, adduct="[M+H]+"):
    return calculate_theoretical_mass(formula=formula, adduct=adduct)


def test_boundary_5ppm_validation():
    """
    Explicitly test the 5.0 ppm threshold.
    4.9 ppm should be valid, 5.1 ppm should be invalid.

    Uses formula-based mass calculation (no RDKit required).
    """
    formula = "C6H6"  # Benzene formula
    exact_mass = _formula_to_monoisotopic_mass(formula)

    # 4.9 ppm shift
    mass_4_9_ppm = exact_mass * (1 + 4.9 / 1e6)
    struct_valid = MolecularStructure(formula=formula, exact_mass=mass_4_9_ppm)
    assert struct_valid.is_physically_valid is True

    # 5.1 ppm shift
    mass_5_1_ppm = exact_mass * (1 + 5.1 / 1e6)
    struct_invalid = MolecularStructure(formula=formula, exact_mass=mass_5_1_ppm)
    assert struct_invalid.is_physically_valid is False


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_boundary_5ppm_validation_smiles():
    """
    Same as above but using SMILES (requires RDKit).
    """
    smiles = "C1=CC=CC=C1"  # Benzene
    mol = Chem.MolFromSmiles(smiles)
    formula = _mol_to_pyteomics_formula(mol)
    import pyteomics.mass as pmass

    exact_mass = pmass.calculate_mass(formula=formula)

    # 4.9 ppm shift
    mass_4_9_ppm = exact_mass * (1 + 4.9 / 1e6)
    struct_valid = MolecularStructure(smiles=smiles, exact_mass=mass_4_9_ppm)
    assert struct_valid.is_physically_valid is True

    # 5.1 ppm shift
    mass_5_1_ppm = exact_mass * (1 + 5.1 / 1e6)
    struct_invalid = MolecularStructure(smiles=smiles, exact_mass=mass_5_1_ppm)
    assert struct_invalid.is_physically_valid is False


def test_radical_cations_anions():
    """
    Test radical species [M]+. and [M]-.
    Uses formula-based mass calculation (no RDKit required).
    """
    from MassFlow.cheminformatics import compute_adduct_offset

    formula = "C10H8"  # Naphthalene formula
    exact_mass = _formula_to_monoisotopic_mass(formula)

    # Theoretical [M]+. is exact_mass - electron_mass
    offset_pos = compute_adduct_offset("[M]+")
    assert offset_pos is not None
    theoretical_mz = exact_mass + offset_pos

    # Correct m/z
    meta_pos = SpectrumMetadata(
        spectrum_id="rad_pos",
        precursor_mz=round(theoretical_mz, 6),
        charge=1,
        adduct="[M]+",
        molecule=MolecularStructure(formula=formula),
    )
    assert meta_pos.is_physically_valid is True

    # Radical anion [M]-.
    offset_neg = compute_adduct_offset("[M]-")
    assert offset_neg is not None
    theoretical_mz_neg = exact_mass + offset_neg
    meta_neg = SpectrumMetadata(
        spectrum_id="rad_neg",
        precursor_mz=round(theoretical_mz_neg, 6),
        charge=-1,
        adduct="[M]-",
        molecule=MolecularStructure(formula=formula),
    )
    assert meta_neg.is_physically_valid is True


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_radical_cations_anions_smiles():
    """
    Same as above but using SMILES (requires RDKit).
    """
    import pyteomics.mass as pmass

    from MassFlow.cheminformatics import compute_adduct_offset

    # Naphthalene radical cation
    smiles = "C1=CC=C2C=CC=CC2=C1"
    mol = Chem.MolFromSmiles(smiles)
    formula = _mol_to_pyteomics_formula(mol)
    exact_mass = pmass.calculate_mass(formula=formula)

    # Theoretical [M]+. is exact_mass - electron_mass
    offset_pos = compute_adduct_offset("[M]+")
    assert offset_pos is not None
    theoretical_mz = exact_mass + offset_pos

    # Correct m/z
    meta_pos = SpectrumMetadata(
        spectrum_id="rad_pos",
        precursor_mz=round(theoretical_mz, 6),
        charge=1,
        adduct="[M]+",
        molecule=MolecularStructure(smiles=smiles),
    )
    assert meta_pos.is_physically_valid is True

    # Radical anion [M]-.
    offset_neg = compute_adduct_offset("[M]-")
    assert offset_neg is not None
    theoretical_mz_neg = exact_mass + offset_neg
    meta_neg = SpectrumMetadata(
        spectrum_id="rad_neg",
        precursor_mz=round(theoretical_mz_neg, 6),
        charge=-1,
        adduct="[M]-",
        molecule=MolecularStructure(smiles=smiles),
    )
    assert meta_neg.is_physically_valid is True


def test_highly_halogenated_isotopic_envelope():
    """
    Test isotopic envelope for molecules with high Chlorine/Bromine content.
    Uses formula-based calculation (no RDKit required).
    """
    # Hexachlorobenzene formula: C6Cl6
    envelope = calculate_isotopic_envelope(formula="C6Cl6", max_isopeaks=5)

    abundances = [p[1] for p in envelope]

    # Base peak should be the second or third peak (M+2 or M+4)
    # But definitely not M (index 0)
    assert len(abundances) > 1
    assert abundances[0] < 1.0
    assert max(abundances) == 1.0
    assert any(a > abundances[0] for a in abundances[1:])


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_highly_halogenated_isotopic_envelope_smiles():
    """
    Same as above but using SMILES (requires RDKit).
    """
    smiles = "C1(Cl)=C(Cl)C(Cl)=C(Cl)C(Cl)=C1Cl"
    envelope = calculate_isotopic_envelope(smiles, max_isopeaks=5)

    abundances = [p[1] for p in envelope]

    assert len(abundances) > 1
    assert abundances[0] < 1.0
    assert max(abundances) == 1.0
    assert any(a > abundances[0] for a in abundances[1:])


def test_precursor_mass_validation_edge_cases():
    """
    Test SpectrumMetadata validation with various charges and offsets.
    Uses formula-based mass calculation.
    """
    formula = "C2H6O"  # Ethanol
    theo_mz = get_theoretical_mz_from_formula(formula)

    # 1. Unknown adduct should flag as invalid
    meta_unknown = SpectrumMetadata(
        spectrum_id="unknown",
        precursor_mz=theo_mz,
        charge=1,
        adduct="[M+GHOST]+",
        molecule=MolecularStructure(formula=formula),
    )
    assert meta_unknown.is_physically_valid is False

    # 2. Charge mismatch
    meta_charge_mismatch = SpectrumMetadata(
        spectrum_id="charge_mismatch",
        precursor_mz=theo_mz,
        charge=2,
        adduct="[M+H]+",
        molecule=MolecularStructure(formula=formula),
    )
    assert meta_charge_mismatch.is_physically_valid is False


def test_invalid_smiles_graceful_failure():
    """
    Ensure invalid SMILES flags is_physically_valid=False instead of crashing.
    """
    struct = MolecularStructure(smiles="NOT_A_SMILES")
    assert struct.is_physically_valid is False
