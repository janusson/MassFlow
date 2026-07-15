"""
Generate a minimal SQLite database of test spectra for use in comprehensive tests.

This script creates a small, deterministic set of matchms.Spectrum objects with
diverse metadata and peak patterns, writes them to an MGF file and a MassFlow
SQLite database, both of which are consumed by the comprehensive test suite.

Usage:
    uv run python tests/generate_minimal_db.py
"""

from pathlib import Path

import numpy as np
from matchms import Spectrum


OUTPUT_DIR = Path(__file__).parent / "data"
DB_PATH = OUTPUT_DIR / "minimal_test_library.db"
MGF_PATH = OUTPUT_DIR / "minimal_test_library.mgf"


def _build_spectra():
    """Generate a minimal but diverse set of test spectra."""
    spectra = []

    # Spectrum 1: Simple, well-behaved spectrum with all metadata
    s1 = Spectrum(
        mz=np.array([100.0, 150.0, 200.0, 250.0, 300.0], dtype=np.float64),
        intensities=np.array([0.1, 0.3, 1.0, 0.5, 0.2], dtype=np.float64),
        metadata={
            "id": "ref_001",
            "compound_name": "Caffeine",
            "precursor_mz": 195.0877,
            "charge": 1,
            "ionmode": "positive",
            "adduct": "[M+H]+",
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
            "formula": "C8H10N4O2",
            "retention_time": 120.0,
        },
    )
    spectra.append(s1)

    # Spectrum 2: Negative mode, different adduct
    s2 = Spectrum(
        mz=np.array([80.0, 120.0, 180.0, 220.0], dtype=np.float64),
        intensities=np.array([0.5, 0.8, 1.0, 0.3], dtype=np.float64),
        metadata={
            "id": "ref_002",
            "compound_name": "Salicylic Acid",
            "precursor_mz": 137.0244,
            "charge": -1,
            "ionmode": "negative",
            "adduct": "[M-H]-",
            "smiles": "C1=CC=C(C(=C1)C(=O)O)O",
            "inchikey": "YGSDEFSMJLZEOE-UHFFFAOYSA-N",
            "formula": "C7H6O3",
            "retention_time": 200.0,
        },
    )
    spectra.append(s2)

    # Spectrum 3: Missing some metadata (adduct, retention_time)
    s3 = Spectrum(
        mz=np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0], dtype=np.float64),
        intensities=np.array([0.1, 0.2, 0.3, 1.0, 0.8, 0.5, 0.1], dtype=np.float64),
        metadata={
            "id": "ref_003",
            "compound_name": "Glucose",
            "precursor_mz": 203.0526,
            "charge": 1,
            "ionmode": "positive",
            "smiles": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
            "formula": "C6H12O6",
        },
    )
    spectra.append(s3)

    # Spectrum 4: High m/z, many peaks
    rng = np.random.default_rng(42)
    n_peaks = 50
    s4_mz = np.sort(rng.uniform(100.0, 2000.0, n_peaks)).astype(np.float64)
    s4_int = rng.uniform(0.01, 1.0, n_peaks).astype(np.float64)
    s4 = Spectrum(
        mz=s4_mz,
        intensities=s4_int,
        metadata={
            "id": "ref_004",
            "compound_name": "Reserpine",
            "precursor_mz": 609.2807,
            "charge": 1,
            "ionmode": "positive",
            "adduct": "[M+H]+",
            "smiles": "CO[C@H]1[C@@H](C[C@@H]2CN3CCC4=C([C@H]3C[C@@H]2C1)NC5=C4C=CC(=C5)OC)C(=O)OC",
            "formula": "C33H40N2O9",
        },
    )
    spectra.append(s4)

    # Spectrum 5: Very few peaks (minimal)
    s5 = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([0.5, 1.0], dtype=np.float64),
        metadata={
            "id": "ref_005",
            "compound_name": "Minimal Compound",
            "precursor_mz": 150.0,
            "charge": 1,
            "ionmode": "positive",
            "adduct": "[M+H]+",
        },
    )
    spectra.append(s5)

    # Spectrum 6: Same m/z as s1 precursor but different compound (for MS1 filter testing)
    s6 = Spectrum(
        mz=np.array([195.0, 196.0, 197.0], dtype=np.float64),
        intensities=np.array([1.0, 0.5, 0.2], dtype=np.float64),
        metadata={
            "id": "ref_006",
            "compound_name": "Close Match",
            "precursor_mz": 195.1,
            "charge": 1,
            "ionmode": "positive",
            "adduct": "[M+H]+",
        },
    )
    spectra.append(s6)

    # Spectrum 7: Decoy-like metadata (for testing decoy generation)
    s7 = Spectrum(
        mz=np.array([400.0, 500.0, 600.0], dtype=np.float64),
        intensities=np.array([1.0, 0.8, 0.5], dtype=np.float64),
        metadata={
            "id": "ref_007",
            "compound_name": "Test Decoy Target",
            "precursor_mz": 500.0,
            "charge": 2,
            "ionmode": "positive",
            "adduct": "[M+2H]2+",
            "is_decoy": False,
        },
    )
    spectra.append(s7)

    return spectra


def main():
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from MassFlow.database import SpectralDatabase
    from MassFlow.io import save_spectra_to_mgf

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spectra = _build_spectra()

    # Write MGF file (human-readable, supported by load_spectra)
    print(f"Writing {len(spectra)} spectra to {MGF_PATH}...")
    save_spectra_to_mgf(spectra, MGF_PATH)

    # Write SQLite database
    print(f"Writing {len(spectra)} spectra to {DB_PATH}...")
    if DB_PATH.exists():
        DB_PATH.unlink()
    db = SpectralDatabase(DB_PATH)
    count = db.add_spectra(iter(spectra), category="test_library")
    db.close()
    print(f"Added {count} spectra to database.")

    print("Done. Test data is ready.")
    print(f"  MGF:  {MGF_PATH}")
    print(f"  DB:   {DB_PATH}")


if __name__ == "__main__":
    main()
