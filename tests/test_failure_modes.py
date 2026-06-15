from MassFlow.cheminformatics import (
    calculate_isotopic_envelope,
    calculate_isotopic_similarity,
    calculate_theoretical_mass,
)
from MassFlow.models import MolecularStructure, SpectrumMetadata


def test_precursor_mz_just_outside_5ppm_flags_invalid():
    """
    Ensure SpectrumMetadata enforces the strict 5.0 ppm tolerance.
    A precursor mz shifted by just under 5 ppm should be valid; just over should be invalid.
    """
    smiles = "C1=CC=CC=C1"  # Benzene

    theo_mz = calculate_theoretical_mass(smiles, "[M+H]+")
    assert theo_mz is not None

    # Slightly inside the 5 ppm threshold -> should be considered valid
    inside_ppm = 4.999
    precursor_inside = theo_mz * (1 + inside_ppm / 1e6)

    meta_inside = SpectrumMetadata(
        spectrum_id="benzene_inside",
        precursor_mz=precursor_inside,
        charge=1,
        adduct="[M+H]+",
        molecule=MolecularStructure(smiles=smiles),
    )
    assert meta_inside.is_physically_valid is True

    # Slightly outside the 5 ppm threshold -> should be flagged invalid
    outside_ppm = 5.5
    precursor_outside = theo_mz * (1 + outside_ppm / 1e6)

    meta_outside = SpectrumMetadata(
        spectrum_id="benzene_outside",
        precursor_mz=precursor_outside,
        charge=1,
        adduct="[M+H]+",
        molecule=MolecularStructure(smiles=smiles),
    )
    assert meta_outside.is_physically_valid is False


def test_radical_ion_precursor_shifted_beyond_5ppm_are_invalid():
    """
    Edge-case: radical cations/anions use very small electron-mass offsets.
    Verify that tiny mass shifts that push error >5 ppm are still detected.
    """
    smiles = "C1=CC=C2C=CC=CC2=C1"  # Naphthalene

    for adduct, charge in (("[M]+", 1), ("[M]-", -1)):
        theo_mz = calculate_theoretical_mass(smiles, adduct)
        assert theo_mz is not None

        # Move slightly beyond the 5 ppm threshold
        outside_ppm = 5.5
        precursor = theo_mz * (1 + outside_ppm / 1e6)

        meta = SpectrumMetadata(
            spectrum_id=f"rad_{adduct}",
            precursor_mz=precursor,
            charge=charge,
            adduct=adduct,
            molecule=MolecularStructure(smiles=smiles),
        )
        assert meta.is_physically_valid is False


def test_isotopic_similarity_shifted_beyond_tolerance_returns_zero():
    """
    Highly-halogenated molecules (e.g., many Cl/Br) have distinctive M+2/M+4 peaks.
    If experimental peaks are uniformly shifted beyond the mz_tolerance used for
    envelope alignment, the isotopic similarity should fall to 0.0.
    """
    smiles = "C1(Cl)=C(Cl)C(Cl)=C(Cl)C(Cl)=C1Cl"  # Hexachlorobenzene

    theor_env = calculate_isotopic_envelope(smiles, max_isopeaks=4)
    assert len(theor_env) > 0

    # Shift all experimental peaks by +0.06 Da which is just outside the default 0.05 Da tolerance
    exp_env_shifted = [(mz + 0.06, abund) for mz, abund in theor_env]

    sim = calculate_isotopic_similarity(exp_env_shifted, theor_env)

    # No peaks should align within the default tolerance, resulting in 0.0 similarity
    assert sim == 0.0

    # Sanity check: a small shift within the tolerance should produce a non-zero similarity
    exp_env_small_shift = [(mz + 0.049, abund) for mz, abund in theor_env]
    sim_small = calculate_isotopic_similarity(exp_env_small_shift, theor_env)
    assert sim_small > 0.0
