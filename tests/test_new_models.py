import pytest
from pydantic import ValidationError

from MassFlow.models import (
    MassFlowSpectrum,
    MolecularStructure,
    SpectralPeaks,
    SpectrumMetadata,
)


def test_spectrum_validation():
    # Valid Case
    s = MassFlowSpectrum(
        metadata=SpectrumMetadata(
            spectrum_id="test_001",
            precursor_mz=195.08765,
            charge=1,
            ion_mode="positive",
            molecule=MolecularStructure(
                smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
            ),  # Caffeine
        ),
        peaks=SpectralPeaks(mz_array=[138.1, 195.1], intensity_array=[0.5, 1.0]),
    )
    assert s.metadata.molecule.formula == "C8H10N4O2"
    assert round(s.metadata.molecule.exact_mass, 2) == 194.08


def test_invalid_smiles():
    mol = MolecularStructure(smiles="NOT_A_SMILES")
    assert mol.is_physically_valid is False


def test_mass_mismatch():
    # Caffeine mass is ~194.08. Providing 300.0 should fail 5ppm threshold.
    mol = MolecularStructure(smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C", exact_mass=300.0)
    assert mol.is_physically_valid is False


def test_array_mismatch():
    with pytest.raises(ValidationError):
        SpectralPeaks(mz_array=[100.0], intensity_array=[0.5, 1.0])


if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    test_spectrum_validation()
    test_invalid_smiles()
    test_mass_mismatch()
    test_array_mismatch()
    console.print("[bold green]All Pydantic v2 validation tests passed![/bold green]")
