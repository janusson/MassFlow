from MassFlow.models import MolecularStructure


def test_molecular_structure_ethanol_envelope():
    # Ethanol SMILES
    smiles = "CCO"

    # Instantiating the model should auto-populate the isotopic_envelope
    mol = MolecularStructure(smiles=smiles)

    envelope = mol.isotopic_envelope
    assert envelope is not None
    assert len(envelope) >= 2  # At least M and M+1

    # Check M peak (Ethanol exact mass is ~46.041865)
    m_peak = envelope[0]
    assert abs(m_peak[0] - 46.04) < 0.01
    assert m_peak[1] == 1.0  # Base peak must be normalized to 1.0

    # Check M+1 peak (~13C contribution, mostly)
    m1_peak = envelope[1]
    assert abs(m1_peak[0] - 47.04) < 0.01

    # Two carbons -> ~2.2% chance of a 13C
    assert 0.015 < m1_peak[1] < 0.03
