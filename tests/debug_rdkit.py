from rdkit.Chem import rdMolDescriptors
from MassFlow.cheminformatics import calculate_isotopic_envelope


def test_check_rdkit_features():
    print(
        f"\nRDKit Has GetIsotopicDistribution: {hasattr(rdMolDescriptors, 'GetIsotopicDistribution')}"
    )
    smiles_c13 = "[13C]1=[13C][13C]=[13C][13C]=[13C]1"
    envelope = calculate_isotopic_envelope(smiles_c13)
    print(f"Envelope for {smiles_c13}: {envelope}")
