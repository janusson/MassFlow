from MassFlow.cheminformatics import (
    calculate_isotopic_envelope,
    get_isotopic_distribution,
)
from MassFlow.models import IsotopicDistribution, MolecularStructure

# Caffeine SMILES and formula for dual-path testing
CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CAFFEINE_FORMULA = "C8H10N4O2"


def test_isotopic_distribution_caffeine():
    dist = get_isotopic_distribution(CAFFEINE_SMILES, threshold=0.01)

    assert len(dist) >= 2
    # M peak
    assert abs(dist[0][0] - 194.08) < 0.01
    assert dist[0][1] == 1.0  # Normalized to 1.0

    # M+1 peak
    assert abs(dist[1][0] - 195.08) < 0.01
    assert 0.05 < dist[1][1] < 0.15  # roughly 10% abundance


def test_isotopic_distribution_caffeine_from_formula():
    """Formula-based isotopic distribution (no RDKit required)."""
    dist = get_isotopic_distribution(formula=CAFFEINE_FORMULA, threshold=0.01)

    assert len(dist) >= 2
    assert abs(dist[0][0] - 194.08) < 0.01
    assert dist[0][1] == 1.0
    assert abs(dist[1][0] - 195.08) < 0.01
    assert 0.05 < dist[1][1] < 0.15


def test_model_integration():
    dist = IsotopicDistribution(peaks=[(194.08, 1.0), (195.08, 0.1)])
    mol = MolecularStructure(
        smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C", isotopic_distribution=dist
    )
    assert mol.isotopic_distribution is not None
    assert mol.isotopic_distribution.peaks[0][0] == 194.08


def test_calculate_isotopic_envelope():
    # Caffeine SMILES (requires RDKit)
    envelope = calculate_isotopic_envelope(CAFFEINE_SMILES, max_isopeaks=3)

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


def test_calculate_isotopic_envelope_from_formula():
    """Formula-based isotopic envelope (no RDKit required)."""
    envelope = calculate_isotopic_envelope(formula=CAFFEINE_FORMULA, max_isopeaks=3)

    assert len(envelope) == 3
    assert envelope[0][1] == 1.0

    assert abs(envelope[0][0] - 194.08) < 0.01
    assert abs(envelope[1][0] - 195.08) < 0.01
    assert abs(envelope[2][0] - 196.08) < 0.01

    assert 0.05 < envelope[1][1] < 0.15
    assert 0.005 < envelope[2][1] < 0.02
