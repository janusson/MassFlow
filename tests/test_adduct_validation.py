from MassFlow.models import MolecularStructure, SpectrumMetadata

# Caffeine is used as the base molecule for these tests.
# Its monoisotopic exact mass is ~194.080376.
CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def get_caffeine() -> MolecularStructure:
    return MolecularStructure(smiles=CAFFEINE_SMILES)


def test_adduct_m_plus_na_valid():
    # Exact mass: 194.080376 + Na offset: 22.989221 = 217.069597
    SpectrumMetadata(
        spectrum_id="test_na_valid",
        precursor_mz=217.069597,
        charge=1,
        adduct="[M+Na]+",
        molecule=get_caffeine(),
    )


def test_adduct_m_plus_nh4_valid():
    # Exact mass: 194.080376 + NH4 offset: 18.033826 = 212.114202
    SpectrumMetadata(
        spectrum_id="test_nh4_valid",
        precursor_mz=212.114202,
        charge=1,
        adduct="[M+NH4]+",
        molecule=get_caffeine(),
    )


def test_adduct_m_minus_h_valid():
    # Exact mass: 194.080376 + [M-H]- offset: -1.007276 = 193.073100
    SpectrumMetadata(
        spectrum_id="test_m_minus_h_valid",
        precursor_mz=193.073100,
        charge=-1,
        adduct="[M-H]-",
        molecule=get_caffeine(),
    )


def test_adduct_m_plus_na_invalid_ppm():
    # Theoretical m/z is 217.069597.
    # Adding 0.005 to mz deviates by ~23 ppm, which should fail the strict 5 ppm tolerance.
    invalid_mz = 217.069597 + 0.005
    meta = SpectrumMetadata(
        spectrum_id="spec_1",
        precursor_mz=invalid_mz,
        charge=1,
        adduct="[M+Na]+",
        molecule=get_caffeine(),
    )
    assert meta.is_physically_valid is False


def test_adduct_m_plus_nh4_invalid_ppm():
    # Theoretical m/z is 212.114202
    # Subtracting 0.005 is ~23.5 ppm error
    invalid_mz = 212.114202 - 0.005
    meta = SpectrumMetadata(
        spectrum_id="spec_1",
        precursor_mz=invalid_mz,
        charge=1,
        adduct="[M+NH4]+",
        molecule=get_caffeine(),
    )
    assert meta.is_physically_valid is False


def test_adduct_default_positive_mode():
    # Should default to [M+H]+ (offset 1.007276, mz 195.087652)
    meta = SpectrumMetadata(
        spectrum_id="test_default_pos",
        precursor_mz=195.087652,
        charge=1,
        ion_mode="positive",
        molecule=get_caffeine(),
    )
    assert meta.adduct == "[M+H]+"


def test_adduct_default_negative_mode():
    # Should default to [M-H]- (offset -1.007276, mz 193.073100)
    meta = SpectrumMetadata(
        spectrum_id="test_default_neg",
        precursor_mz=193.073100,
        charge=-1,
        ion_mode="negative",
        molecule=get_caffeine(),
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
