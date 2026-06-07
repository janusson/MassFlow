import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from MassFlow.cheminformatics import calculate_isotopic_envelope
from MassFlow.models import MolecularStructure, SpectralPeaks, SpectrumMetadata


# Helper to calculate theoretical m/z for [M+H]+
def get_theoretical_mz(smiles):
    mol = Chem.MolFromSmiles(smiles)
    exact_mass = Descriptors.ExactMolWt(mol)
    return (exact_mass + 1.007276) / 1.0


def test_boundary_5ppm_validation():
    """
    Explicitly test the 5.0 ppm threshold.
    4.9 ppm should be valid, 5.1 ppm should be invalid.
    """
    smiles = "C1=CC=CC=C1"  # Benzene
    mol = Chem.MolFromSmiles(smiles)
    exact_mass = Descriptors.ExactMolWt(mol)

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
    These use the ELECTRON_MASS offset (~0.000549) in ADDUCT_OFFSETS.
    """
    # Naphthalene radical cation
    smiles = "C1=CC=C2C=CC=CC2=C1"
    mol = Chem.MolFromSmiles(smiles)
    exact_mass = Descriptors.ExactMolWt(mol)

    # Theoretical [M]+. is exact_mass - electron_mass
    # ADDUCT_OFFSETS["[M]+"] = -0.000549
    theoretical_mz = exact_mass - 0.0005485799

    # Correct m/z
    meta_pos = SpectrumMetadata(
        spectrum_id="rad_pos",
        precursor_mz=theoretical_mz,
        charge=1,
        adduct="[M]+",
        molecule=MolecularStructure(smiles=smiles),
    )
    assert meta_pos.is_physically_valid is True

    # Radical anion [M]-.
    # Theoretical [M]-. is exact_mass + electron_mass
    # ADDUCT_OFFSETS["[M]-"] = 0.000549
    theoretical_mz_neg = exact_mass + 0.0005485799
    meta_neg = SpectrumMetadata(
        spectrum_id="rad_neg",
        precursor_mz=theoretical_mz_neg,
        charge=-1,
        adduct="[M]-",
        molecule=MolecularStructure(smiles=smiles),
    )
    # Note: SpectrumMetadata uses abs(charge) in denominator.
    # theoretical_mz = (exact_mass + offset) / abs(charge)
    assert meta_neg.is_physically_valid is True


def test_highly_halogenated_isotopic_envelope():
    """
    Test isotopic envelope for molecules with high Chlorine/Bromine content.
    These have very distinct M+2 peaks.
    """
    # Hexachlorobenzene
    smiles = "C1(Cl)=C(Cl)C(Cl)=C(Cl)C(Cl)=C1Cl"
    envelope = calculate_isotopic_envelope(smiles, max_isopeaks=5)

    # Cl6 should have a very strong M+2 and M+4
    # Relative abundances for Cl6: M=100, M+2~195, M+4~160, M+6~65...
    # The code normalizes to base peak = 1.0.
    # For Cl6, the base peak is M+2.

    abundances = [p[1] for p in envelope]

    # Base peak should be the second or third peak (M+2 or M+4 depending on exact stats)
    # But definitely not M (index 0)
    assert abundances[0] < 1.0
    assert max(abundances) == 1.0
    assert any(a > abundances[0] for a in abundances[1:])


def test_precursor_mass_validation_edge_cases():
    """
    Test SpectrumMetadata validation with various charges and offsets.
    """
    smiles = "CCO"  # Ethanol
    theo_mz = get_theoretical_mz(smiles)

    # 1. Unknown adduct should flag as invalid
    meta_unknown = SpectrumMetadata(
        spectrum_id="unknown",
        precursor_mz=theo_mz,
        charge=1,
        adduct="[M+GHOST]+",
        molecule=MolecularStructure(smiles=smiles),
    )
    assert meta_unknown.is_physically_valid is False

    # 2. Charge mismatch
    # If m/z is calculated for z=1 but we claim z=2
    meta_charge_mismatch = SpectrumMetadata(
        spectrum_id="charge_mismatch",
        precursor_mz=theo_mz,
        charge=2,
        adduct="[M+H]+",
        molecule=MolecularStructure(smiles=smiles),
    )
    # theo_mz (z=2) = (exact_mass + 1.007276) / 2
    # This will be ~23 vs ~47, definitely > 5ppm
    assert meta_charge_mismatch.is_physically_valid is False


def test_invalid_smiles_graceful_failure():
    """
    Ensure invalid SMILES flags is_physically_valid=False instead of crashing.
    """
    struct = MolecularStructure(smiles="NOT_A_SMILES")
    assert struct.is_physically_valid is False

    # Peaks validation check (already exists but good for coverage)
    with pytest.raises(ValueError, match="Array length mismatch"):
        SpectralPeaks(mz_array=[100.0], intensity_array=[1.0, 2.0])
