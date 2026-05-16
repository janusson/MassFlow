"""
Script to generate synthetic mass spectrometry data for testing MassFlow.
"""

import random
from pathlib import Path

import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp


def generate_spectrum(
    precursor_mz: float,
    peaks_count: int = 10,
    noise_level: float = 0.05,
    shift: float = 0.0,
    compound_name: str = "Unknown",
):
    """
    Generate a random spectrum.
    """
    # Create random m/z values lower than precursor
    mz = np.sort(np.random.uniform(50, max(50.1, precursor_mz - 10), peaks_count))
    mz += shift
    intensities = np.random.uniform(0.1, 1.0, peaks_count)

    # Add precursor peak
    mz = np.append(mz, precursor_mz + shift)
    intensities = np.append(intensities, 1.0)  # Base peak

    # Sort
    idx = np.argsort(mz)
    mz = mz[idx]
    intensities = intensities[idx]

    # Add noise to intensities
    if noise_level > 0:
        intensities += np.random.normal(0, noise_level, len(intensities))
        intensities = np.clip(intensities, 0.0, None)

    return Spectrum(
        mz=mz,
        intensities=intensities,
        metadata={
            "precursor_mz": precursor_mz + shift,
            "compound_name": compound_name,
            "ionmode": "positive",
            "charge": 1,
        },
    )


def main():
    from rich.console import Console

    console = Console()
    data_root = Path("data")
    raw_dir = data_root / "raw"
    ref_dir = data_root / "reference"

    raw_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Generating Reference Library...[/cyan]")
    reference_spectra = []
    num_library = 100
    for i in range(num_library):
        mz = random.uniform(200, 800)
        # Random number of peaks between 5 and 20
        n_peaks = random.randint(5, 20)
        spec = generate_spectrum(mz, peaks_count=n_peaks, compound_name=f"Compound_{i}")
        reference_spectra.append(spec)

    save_as_msp(reference_spectra, str(ref_dir / "synthetic_library.msp"))
    console.print(
        f"[green]Saved {num_library} spectra to {ref_dir / 'synthetic_library.msp'}[/green]"
    )

    console.print("[cyan]Generating Experimental Runs...[/cyan]")
    num_runs = 50  # 50 files
    spectra_per_run = 20  # 20 spectra each => 1000 total queries

    for run_idx in range(num_runs):
        run_spectra = []
        for j in range(spectra_per_run):
            # 50% match, 50% noise
            if random.random() < 0.5:
                # Pick a random reference
                ref_spec = random.choice(reference_spectra)

                # Clone and modify
                mz = ref_spec.peaks.mz.copy()
                intensities = ref_spec.peaks.intensities.copy()

                # Add noise to intensities
                intensities += np.random.normal(0, 0.05, len(intensities))
                intensities = np.clip(intensities, 0.0, None)

                # Sometimes shift precursor (Modified Cosine test)
                shift = 0.0
                if random.random() < 0.2:
                    shift = 16.0  # +O
                    mz += shift

                spec = Spectrum(
                    mz=mz,
                    intensities=intensities,
                    metadata={
                        "precursor_mz": float(ref_spec.get("precursor_mz")) + shift,
                        "compound_name": "Unknown_Query",
                        "scan_id": f"Run_{run_idx}_Scan_{j}",
                        "charge": 1,
                        "ionmode": "positive",
                    },
                )
            else:
                # Generate noise spectrum
                mz_noise = random.uniform(100, 1000)
                n_peaks = random.randint(5, 50)
                spec = generate_spectrum(
                    mz_noise,
                    peaks_count=n_peaks,
                    compound_name="Noise",
                    noise_level=0.1,
                )
                spec.set("scan_id", f"Run_{run_idx}_Scan_{j}")

            run_spectra.append(spec)

        filename = raw_dir / f"synthetic_run_{run_idx:03d}.mgf"
        save_as_mgf(run_spectra, str(filename))

    console.print(f"[green]Generated {num_runs} MGF files in {raw_dir}[/green]")


if __name__ == "__main__":
    main()
