from MassFlow.cheminformatics import (
    calculate_isotopic_envelope,
    get_isotopic_distribution,
)
from MassFlow.models import IsotopicDistribution, MolecularStructure


def test_isotopic_distribution_caffeine():
    # Caffeine SMILES
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    dist = get_isotopic_distribution(smiles, threshold=0.01)

    assert len(dist) >= 2
    # M peak
    assert abs(dist[0][0] - 194.08) < 0.01
    assert dist[0][1] == 1.0  # Normalized to 1.0

    # M+1 peak
    assert abs(dist[1][0] - 195.08) < 0.01
    assert 0.05 < dist[1][1] < 0.15  # roughly 10% abundance


def test_model_integration():
    dist = IsotopicDistribution(peaks=[(194.08, 1.0), (195.08, 0.1)])
    mol = MolecularStructure(
        smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C", isotopic_distribution=dist
    )
    assert mol.isotopic_distribution is not None
    assert mol.isotopic_distribution.peaks[0][0] == 194.08


def test_calculate_isotopic_envelope():
    # Caffeine
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    envelope = calculate_isotopic_envelope(smiles, max_isopeaks=3)

    assert len(envelope) == 3
    # Check normalization
    assert envelope[0][1] == 1.0

    # Check masses
    assert abs(envelope[0][0] - 194.08) < 0.01
    assert abs(envelope[1][0] - 195.08) < 0.01
    assert abs(envelope[2][0] - 196.08) < 0.01

    # Check relative abundances (M+1 ~10%, M+2 <1%)
    assert 0.05 < envelope[1][1] < 0.15
    assert 0.005 < envelope[2][1] < 0.02
