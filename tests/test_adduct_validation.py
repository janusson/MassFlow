"""Adduct validation tests using pyteomics as the single source of truth."""

from MassFlow.cheminformatics import calculate_theoretical_mass
from MassFlow.models import MolecularStructure, SpectrumMetadata

# Caffeine is used as the base molecule for these tests.
CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CAFFEINE_FORMULA = "C8H10N4O2"


def get_caffeine() -> MolecularStructure:
    return MolecularStructure(smiles=CAFFEINE_SMILES)


def get_caffeine_from_formula() -> MolecularStructure:
    return MolecularStructure(formula=CAFFEINE_FORMULA)


def test_adduct_m_plus_na_valid():
    # Theoretical m/z derived from pyteomics via calculate_theoretical_mass.
    theo_mz = calculate_theoretical_mass(CAFFEINE_SMILES, "[M+Na]+")
    assert theo_mz is not None
    SpectrumMetadata(
        spectrum_id="test_na_valid",
        precursor_mz=round(theo_mz, 6),
        charge=1,
        adduct="[M+Na]+",
        molecule=get_caffeine(),
    )


def test_adduct_m_plus_na_valid_from_formula():
    """Formula-based adduct validation (no RDKit required)."""
    theo_mz = calculate_theoretical_mass(formula=CAFFEINE_FORMULA, adduct="[M+Na]+")
    assert theo_mz is not None
    SpectrumMetadata(
        spectrum_id="test_na_valid_formula",
        precursor_mz=round(theo_mz, 6),
        charge=1,
        adduct="[M+Na]+",
        molecule=get_caffeine_from_formula(),
    )


def test_adduct_m_plus_nh4_valid():
    theo_mz = calculate_theoretical_mass(CAFFEINE_SMILES, "[M+NH4]+")
    assert theo_mz is not None
    SpectrumMetadata(
        spectrum_id="test_nh4_valid",
        precursor_mz=round(theo_mz, 6),
        charge=1,
        adduct="[M+NH4]+",
        molecule=get_caffeine(),
    )


def test_adduct_m_plus_nh4_valid_from_formula():
    """Formula-based."""
    theo_mz = calculate_theoretical_mass(formula=CAFFEINE_FORMULA, adduct="[M+NH4]+")
    assert theo_mz is not None
    SpectrumMetadata(
        spectrum_id="test_nh4_valid_formula",
        precursor_mz=round(theo_mz, 6),
        charge=1,
        adduct="[M+NH4]+",
        molecule=get_caffeine_from_formula(),
    )


def test_adduct_m_minus_h_valid():
    theo_mz = calculate_theoretical_mass(CAFFEINE_SMILES, "[M-H]-")
    assert theo_mz is not None
    SpectrumMetadata(
        spectrum_id="test_m_minus_h_valid",
        precursor_mz=round(theo_mz, 6),
        charge=-1,
        adduct="[M-H]-",
        molecule=get_caffeine(),
    )


def test_adduct_m_minus_h_valid_from_formula():
    """Formula-based."""
    theo_mz = calculate_theoretical_mass(formula=CAFFEINE_FORMULA, adduct="[M-H]-")
    assert theo_mz is not None
    SpectrumMetadata(
        spectrum_id="test_m_minus_h_valid_formula",
        precursor_mz=round(theo_mz, 6),
        charge=-1,
        adduct="[M-H]-",
        molecule=get_caffeine_from_formula(),
    )


def test_adduct_m_plus_na_invalid_ppm():
    theo_mz = calculate_theoretical_mass(CAFFEINE_SMILES, "[M+Na]+")
    assert theo_mz is not None
    # Adding ~5 ppm error to theoretical m/z pushes past the tolerance.
    invalid_mz = theo_mz * (1 + 5.1 / 1e6)
    meta = SpectrumMetadata(
        spectrum_id="spec_1",
        precursor_mz=invalid_mz,
        charge=1,
        adduct="[M+Na]+",
        molecule=get_caffeine(),
    )
    assert meta.is_physically_valid is False


def test_adduct_m_plus_na_invalid_ppm_from_formula():
    """Formula-based."""
    theo_mz = calculate_theoretical_mass(formula=CAFFEINE_FORMULA, adduct="[M+Na]+")
    assert theo_mz is not None
    invalid_mz = theo_mz * (1 + 5.1 / 1e6)
    meta = SpectrumMetadata(
        spectrum_id="spec_1_formula",
        precursor_mz=invalid_mz,
        charge=1,
        adduct="[M+Na]+",
        molecule=get_caffeine_from_formula(),
    )
    assert meta.is_physically_valid is False


def test_adduct_m_plus_nh4_invalid_ppm():
    theo_mz = calculate_theoretical_mass(CAFFEINE_SMILES, "[M+NH4]+")
    assert theo_mz is not None
    invalid_mz = theo_mz * (1 + 5.1 / 1e6)
    meta = SpectrumMetadata(
        spectrum_id="spec_1",
        precursor_mz=invalid_mz,
        charge=1,
        adduct="[M+NH4]+",
        molecule=get_caffeine(),
    )
    assert meta.is_physically_valid is False


def test_adduct_m_plus_nh4_invalid_ppm_from_formula():
    """Formula-based."""
    theo_mz = calculate_theoretical_mass(formula=CAFFEINE_FORMULA, adduct="[M+NH4]+")
    assert theo_mz is not None
    invalid_mz = theo_mz * (1 + 5.1 / 1e6)
    meta = SpectrumMetadata(
        spectrum_id="spec_1_formula",
        precursor_mz=invalid_mz,
        charge=1,
        adduct="[M+NH4]+",
        molecule=get_caffeine_from_formula(),
    )
    assert meta.is_physically_valid is False


def test_adduct_default_positive_mode():
    theo_mz = calculate_theoretical_mass(CAFFEINE_SMILES, "[M+H]+")
    assert theo_mz is not None
    meta = SpectrumMetadata(
        spectrum_id="test_default_pos",
        precursor_mz=round(theo_mz, 6),
        charge=1,
        ion_mode="positive",
        molecule=get_caffeine(),
    )
    assert meta.adduct == "[M+H]+"


def test_adduct_default_positive_mode_from_formula():
    """Formula-based."""
    theo_mz = calculate_theoretical_mass(formula=CAFFEINE_FORMULA, adduct="[M+H]+")
    assert theo_mz is not None
    meta = SpectrumMetadata(
        spectrum_id="test_default_pos_formula",
        precursor_mz=round(theo_mz, 6),
        charge=1,
        ion_mode="positive",
        molecule=get_caffeine_from_formula(),
    )
    assert meta.adduct == "[M+H]+"


def test_adduct_default_negative_mode():
    theo_mz = calculate_theoretical_mass(CAFFEINE_SMILES, "[M-H]-")
    assert theo_mz is not None
    meta = SpectrumMetadata(
        spectrum_id="test_default_neg",
        precursor_mz=round(theo_mz, 6),
        charge=-1,
        ion_mode="negative",
        molecule=get_caffeine(),
    )
    assert meta.adduct == "[M-H]-"


def test_adduct_default_negative_mode_from_formula():
    """Formula-based."""
    theo_mz = calculate_theoretical_mass(formula=CAFFEINE_FORMULA, adduct="[M-H]-")
    assert theo_mz is not None
    meta = SpectrumMetadata(
        spectrum_id="test_default_neg_formula",
        precursor_mz=round(theo_mz, 6),
        charge=-1,
        ion_mode="negative",
        molecule=get_caffeine_from_formula(),
    )
    assert meta.adduct == "[M-H]-"


def test_unsupported_adduct():
    meta = SpectrumMetadata(
        spectrum_id="spec_1",
        precursor_mz=200.0,
        charge=1,
        adduct="[M+Weird]+",
        molecule=get_caffeine(),
    )
    assert meta.is_physically_valid is False


def test_unsupported_adduct_from_formula():
    """Formula-based."""
    meta = SpectrumMetadata(
        spectrum_id="spec_1_formula",
        precursor_mz=200.0,
        charge=1,
        adduct="[M+Weird]+",
        molecule=get_caffeine_from_formula(),
    )
    assert meta.is_physically_valid is False
