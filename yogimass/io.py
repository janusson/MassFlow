"""
I/O functions for SpectralMetricMS: import/export of libraries and spectra.
Ported from original_source/io/mgf.py and yogimass_pipeline.py.
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path
import pandas as pd
from matchms.importing import load_from_mgf, load_from_msp
from matchms.exporting import save_as_mgf, save_as_msp, save_as_json

# Configure basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def list_msp_libraries(directory: str) -> list[str]:
    """List all .msp files in a directory."""
    msp_libraries_list = glob.glob(os.path.join(directory, "*.msp"), recursive=True)
    logger.info(f"{len(msp_libraries_list)} MSP libraries found in {directory}.")
    return msp_libraries_list


def list_mgf_libraries(directory: str) -> list[str]:
    """List all .mgf files in a directory."""
    mgf_libraries_list = glob.glob(os.path.join(directory, "*.mgf"), recursive=True)
    logger.info(f"{len(mgf_libraries_list)} MGF libraries found in {directory}.")
    return mgf_libraries_list


def list_available_libraries(mgf_libraries_list: list[str], msp_libraries_list: list[str]) -> dict[str, list[str]]:
    """Log available libraries and return a summary dict."""
    summary = {
        "mgf": [Path(item).name for item in mgf_libraries_list],
        "msp": [Path(item).name for item in msp_libraries_list],
    }
    for library_type, library_names in summary.items():
        header = library_type.upper()
        logger.info(f"Available {header} libraries ({len(library_names)}):")
        if library_names:
            for library in library_names:
                logger.info(f"  - {library}")
        else:
            logger.info("  (none found)")
    return summary


def fetch_mgflib_spectrum(library_filepath: str, spectrum_number: int):
    """
    Load MS spectrum peak and meta data from a library file.
    Returns: spectrum_xy_data (DataFrame), spectrum_metadata, spectrum_chemical
    """
    spectra_list = list(load_from_mgf(library_filepath))
    if spectrum_number >= len(spectra_list):
        raise IndexError(f"Spectrum number {spectrum_number} out of range for library {library_filepath} (size: {len(spectra_list)})")

    spectrum = spectra_list[spectrum_number]
    spectrum_peaks = spectrum.peaks.mz
    spectrum_counts = spectrum.peaks.intensities
    
    # Handle empty spectrum case to avoid division by zero
    if len(spectrum_counts) > 0:
        normalized_counts = spectrum_counts / max(spectrum_counts)
    else:
        normalized_counts = spectrum_counts

    percent_abundance = normalized_counts * 100
    spectrum_dict = dict(zip(spectrum_peaks, percent_abundance))
    
    spectrum_xy_data = pd.DataFrame.from_dict(
        spectrum_dict, orient="index", columns=["Abundance (%)"]
    )
    spectrum_xy_data.reset_index(inplace=True)
    spectrum_xy_data.rename(columns={"index": "m/z"}, inplace=True)
    spectrum_xy_data.sort_values(by="m/z", inplace=True)
    
    spectrum_metadata = spectrum.metadata
    spectrum_chemical = spectrum.metadata.get("name", "Unknown")
    
    return spectrum_xy_data, spectrum_metadata, spectrum_chemical


def save_spectra_to_mgf(spectra_list: list, export_filepath: str, export_name: str) -> None:
    export_mgf_path = os.path.join(export_filepath, export_name + ".mgf")
    save_as_mgf(spectra_list, export_mgf_path)
    logger.info(f"{len(spectra_list)} spectra saved to MGF: {export_mgf_path}")


def save_spectra_to_msp(spectra_list: list, export_filepath: str, export_name: str) -> None:
    export_msp_path = os.path.join(export_filepath, export_name + ".msp")
    save_as_msp(spectra_list, export_msp_path)
    logger.info(f"{len(spectra_list)} spectra saved to MSP: {export_msp_path}")


def save_spectra_to_json(spectra_list: list, export_filepath: str, export_name: str) -> None:
    export_json_path = os.path.join(export_filepath, export_name + ".json")
    save_as_json(spectra_list, export_json_path)
    logger.info(f"{len(spectra_list)} spectra saved to JSON: {export_json_path}")


def save_spectra_to_pickle(spectra_list: list, export_filepath: str, export_name: str) -> None:
    file_export_pickle = os.path.join(export_filepath, export_name + ".pickle")
    with open(file_export_pickle, "wb") as f:
        pickle.dump(spectra_list, f)
    logger.info(f"{len(spectra_list)} spectra saved to pickle: {file_export_pickle}")
