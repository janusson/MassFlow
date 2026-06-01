from pathlib import Path

import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp


def create_steroid_spectrum(name, precursor_mz, peaks, inchikey, smiles):
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


def main():
    tutorial_dir = Path("tutorial")
    tutorial_dir.mkdir(exist_ok=True)

    # Define some "real" steroid data
    # Peaks are (m/z, intensity)
    steroids_lib = [
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
    save_as_msp(lib_spectra, str(tutorial_dir / "tutorial_library.msp"))

    # Create experimental data
    # 1. Exact match for Testosterone
    # 2. Match for Progesterone with some noise and slight m/z shifts
    # 3. A "modified" Cortisol (e.g. Cortisone, but we'll just shift peaks to show modified cosine)
    # 4. An unknown noise spectrum

    exp_spectra = []

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

    # 3. Modified Cortisol (Cortisone-like, shift +2 Da or similar for demo)
    # We'll just take cortisol and shift it to simulate a related molecule
    exp_spectra.append(
        Spectrum(
            mz=lib_spectra[2].peaks.mz + 2.016,  # Simulating a small modification
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

    save_as_mgf(exp_spectra, str(tutorial_dir / "tutorial_experimental.mgf"))

    # Create tutorial config
    config_content = """project:
  name: "MassFlow_Tutorial"
  output_directory: "tutorial/results"

input:
  file_path: "tutorial/tutorial_experimental.mgf"
  library_path: "tutorial/tutorial_library.msp"

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
"""
    with open(tutorial_dir / "tutorial_config.yaml", "w") as f:
        f.write(config_content)

    print("Tutorial data and config generated in /tutorial")


if __name__ == "__main__":
    main()
