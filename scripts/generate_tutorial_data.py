"""Generate synthetic tutorial data for MassFlow onboarding.

Creates a self-contained ``tutorial/`` directory with:
- ``tutorial_library.msp`` – reference steroid spectra (Testosterone, Progesterone, Cortisol)
- ``tutorial_experimental.mgf`` – experimental queries (matches, analogues, and noise)
- ``tutorial_config.yaml`` – pre-configured analysis parameters

Usage (standalone)::

    uv run python scripts/generate_tutorial_data.py

Usage (via CLI)::

    uv run massflow tutorial
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TypedDict

import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp


class SteroidSpec(TypedDict):
    """Typed descriptor for a single steroid reference spectrum."""

    name: str
    precursor_mz: float
    peaks: list[tuple[float, float]]
    inchikey: str
    smiles: str


# ── Spectrum builder ──────────────────────────────────────────────────────────


def create_steroid_spectrum(
    name: str,
    precursor_mz: float,
    peaks: list[tuple[float, float]],
    inchikey: str,
    smiles: str,
) -> Spectrum:
    """Build a matchms ``Spectrum`` with steroid metadata."""
    mz, intensities = zip(*peaks)
    return Spectrum(
        mz=np.array(mz, dtype="float"),
        intensities=np.array(intensities, dtype="float"),
        metadata={
            "compound_name": name,
            "precursor_mz": precursor_mz,
            "inchikey": inchikey,
            "smiles": smiles,
            "ionmode": "positive",
            "charge": 1,
            "adduct": "[M+H]+",
        },
    )


# ── Main generator ────────────────────────────────────────────────────────────


def main(clean_first: bool = False) -> dict[str, Path]:
    """Generate the tutorial data set.

    Parameters
    ----------
    clean_first : bool
        If ``True``, delete any existing ``tutorial/`` directory before
        regenerating.

    Returns
    -------
    dict[str, Path]
        Mapping of logical names to their absolute ``Path`` objects:
        ``library``, ``experimental``, ``config``.
    """
    tutorial_dir = Path("tutorial")
    results_dir = tutorial_dir / "results"

    if clean_first and tutorial_dir.exists():
        shutil.rmtree(tutorial_dir)

    tutorial_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # ── Reference library: steroid standards ──────────────────────────────
    steroids_lib: list[SteroidSpec] = [
        {
            "name": "Testosterone",
            "precursor_mz": 289.216,
            "peaks": [
                (109.065, 0.4),
                (147.117, 0.2),
                (253.195, 0.3),
                (271.206, 0.6),
                (289.216, 1.0),
            ],
            "inchikey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N",
            "smiles": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        },
        {
            "name": "Progesterone",
            "precursor_mz": 315.232,
            "peaks": [
                (97.065, 0.3),
                (109.065, 0.5),
                (124.089, 0.4),
                (297.221, 0.2),
                (315.232, 1.0),
            ],
            "inchikey": "OSYQYABOXYAWRY-UHFFFAOYSA-N",
            "smiles": "CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C",
        },
        {
            "name": "Cortisol",
            "precursor_mz": 363.217,
            "peaks": [
                (97.065, 0.4),
                (121.065, 0.8),
                (309.185, 0.2),
                (327.196, 0.5),
                (345.206, 0.3),
                (363.217, 1.0),
            ],
            "inchikey": "JLYXYYPFDSBSJV-UHFFFAOYSA-N",
            "smiles": "CC12CC(C(C1(CCC3C2CCC4=CC(=O)CCC34C)O)C(=O)CO)O",
        },
    ]

    lib_spectra = [create_steroid_spectrum(**s) for s in steroids_lib]

    library_path = tutorial_dir / "tutorial_library.msp"
    save_as_msp(lib_spectra, str(library_path))

    # ── Experimental queries ──────────────────────────────────────────────
    # 1. Exact match for Testosterone (slight noise)
    # 2. Match for Progesterone (noise and mild m/z shifts)
    # 3. "Modified" Cortisol (shifted peaks to exercise modified cosine)
    # 4. Noise spectrum (should not match anything)

    exp_spectra: list[Spectrum] = []

    # 1. Testosterone Match
    exp_spectra.append(
        Spectrum(
            mz=lib_spectra[0].peaks.mz
            + np.random.normal(0, 0.001, len(lib_spectra[0].peaks.mz)),
            intensities=lib_spectra[0].peaks.intensities
            * np.random.uniform(0.9, 1.1, len(lib_spectra[0].peaks.mz)),
            metadata={
                "precursor_mz": 289.216,
                "scan_id": "scan_101",
                "retention_time": 12.5,
            },
        )
    )

    # 2. Progesterone Match
    exp_spectra.append(
        Spectrum(
            mz=lib_spectra[1].peaks.mz
            + np.random.normal(0, 0.002, len(lib_spectra[1].peaks.mz)),
            intensities=lib_spectra[1].peaks.intensities
            * np.random.uniform(0.8, 1.2, len(lib_spectra[1].peaks.mz)),
            metadata={
                "precursor_mz": 315.232,
                "scan_id": "scan_102",
                "retention_time": 14.2,
            },
        )
    )

    # 3. Modified Cortisol (Cortisone-like, shift +2.016 Da for demo)
    exp_spectra.append(
        Spectrum(
            mz=lib_spectra[2].peaks.mz + 2.016,
            intensities=lib_spectra[2].peaks.intensities,
            metadata={
                "precursor_mz": 363.217 + 2.016,
                "scan_id": "scan_103",
                "retention_time": 11.8,
            },
        )
    )

    # 4. Noise
    exp_spectra.append(
        Spectrum(
            mz=np.array([50.0, 100.0, 150.0, 200.0]),
            intensities=np.array([0.1, 0.05, 0.1, 0.05]),
            metadata={
                "precursor_mz": 400.0,
                "scan_id": "scan_104",
                "retention_time": 5.0,
            },
        )
    )

    experimental_path = tutorial_dir / "tutorial_experimental.mgf"
    save_as_mgf(exp_spectra, str(experimental_path))

    # ── Tutorial configuration ────────────────────────────────────────────
    config_content = """project:
  name: "MassFlow_Tutorial"
  output_directory: "tutorial/results"

input:
  input_path: "tutorial/tutorial_experimental.mgf"
  library_path: "tutorial/tutorial_library.msp"
  format: "mgf"

processing:
  clean_metadata: true
  normalize_intensity: true
  filter_min_peaks: true
  min_peaks: 3

similarity:
  algorithm: "cosine"
  ms1_tolerance: 0.02
  ms2_tolerance: 0.02
  tolerance_unit: "Da"
  min_score: 0.1
  fdr_threshold: 1.0

export:
  format: "csv"
"""
    config_path = tutorial_dir / "tutorial_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    return {
        "library": library_path.resolve(),
        "experimental": experimental_path.resolve(),
        "config": config_path.resolve(),
    }


# ── Display helpers ───────────────────────────────────────────────────────────


def _print_next_steps(
    library_path: Path, experimental_path: Path, config_path: Path
) -> None:
    """Print a formatted "Next Steps" block so users know what to run next."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  ✓  Tutorial data generated successfully!")
    print(f"{sep}")
    print(f"  Reference library : {library_path}")
    print(f"  Experimental data : {experimental_path}")
    print(f"  Configuration     : {config_path}")
    print(f"{sep}")
    print()
    print("  Next Steps — copy and run these commands:")
    print()
    print("  # 1. Build the SQLite reference database")
    print("  uv run massflow db build \\")
    print("      --input tutorial/tutorial_library.msp \\")
    print("      --output tutorial/results/compiled_library.db \\")
    print("      --config tutorial/tutorial_config.yaml \\")
    print("      --category library")
    print()
    print("  # 2. Annotate experimental spectra against the database")
    print("  uv run massflow annotate --config tutorial/tutorial_config.yaml")
    print()
    print(f"{sep}")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    paths = main()
    _print_next_steps(
        library_path=paths["library"],
        experimental_path=paths["experimental"],
        config_path=paths["config"],
    )
