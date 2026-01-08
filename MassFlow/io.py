"""
I/O functions for MassFlow: import/export of libraries and spectra.
Ported from original_source/io/mgf.py and massflow_pipeline.py.
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
logger = logging.getLogger(__name__)




def list_msp_libraries(directory: str) -> list[str]:
    """
    List all .msp files in a directory.

    Args:
        directory: Path to the directory to search.

    Returns:
        List of absolute file paths to .msp files.
    """
    msp_libraries_list = glob.glob(os.path.join(directory, "*.msp"), recursive=True)
    logger.info(f"{len(msp_libraries_list)} MSP libraries found in {directory}.")
    return msp_libraries_list


def list_mgf_libraries(directory: str) -> list[str]:
    """
    List all .mgf files in a directory.

    Args:
        directory: Path to the directory to search.

    Returns:
        List of absolute file paths to .mgf files.
    """
    mgf_libraries_list = glob.glob(os.path.join(directory, "*.mgf"), recursive=True)
    logger.info(f"{len(mgf_libraries_list)} MGF libraries found in {directory}.")
    return mgf_libraries_list


def list_available_libraries(mgf_libraries_list: list[str], msp_libraries_list: list[str]) -> dict[str, list[str]]:
    """
    Log available libraries and return a summary dict.

    Args:
        mgf_libraries_list: List of MGF file paths.
        msp_libraries_list: List of MSP file paths.

    Returns:
        Dictionary with keys 'mgf' and 'msp' containing lists of filenames.
    """
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



from itertools import islice

def fetch_mgflib_spectrum(library_filepath: str, spectrum_number: int) -> tuple[pd.DataFrame, dict, str]:
    """
    Load MS spectrum peak and meta data from a library file.

    Args:
        library_filepath: Path to the MGF library file.
        spectrum_number: Index of the spectrum to fetch (0-based).

    Returns:
        tuple: (spectrum_xy_data (DataFrame), spectrum_metadata (dict), spectrum_chemical (str))

    Raises:
        IndexError: If spectrum_number is out of range.
    """
    # Optimized to not load the full list
    spectrum_generator = load_from_mgf(library_filepath)
    
    # Advance to the desired index
    try:
        spectrum = next(islice(spectrum_generator, spectrum_number, spectrum_number + 1))
    except StopIteration:
        raise IndexError(f"Spectrum number {spectrum_number} out of range for library {library_filepath}")

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
    # matchms may standardize 'name' to 'compound_name'
    spectrum_chemical = spectrum.get("compound_name") or spectrum.get("name") or "Unknown"
    
    return spectrum_xy_data, spectrum_metadata, spectrum_chemical



def save_spectra_to_mgf(spectra_list: list, export_filepath: str, export_name: str) -> None:
    """Save spectra to MGF format."""
    export_mgf_path = os.path.join(export_filepath, export_name + ".mgf")
    save_as_mgf(spectra_list, export_mgf_path)
    logger.info(f"{len(spectra_list)} spectra saved to MGF: {export_mgf_path}")


def save_spectra_to_msp(spectra_list: list, export_filepath: str, export_name: str) -> None:
    """Save spectra to MSP format."""
    export_msp_path = os.path.join(export_filepath, export_name + ".msp")
    save_as_msp(spectra_list, export_msp_path)
    logger.info(f"{len(spectra_list)} spectra saved to MSP: {export_msp_path}")


def save_spectra_to_json(spectra_list: list, export_filepath: str, export_name: str) -> None:
    """Save spectra to JSON format."""
    export_json_path = os.path.join(export_filepath, export_name + ".json")
    save_as_json(spectra_list, export_json_path)
    logger.info(f"{len(spectra_list)} spectra saved to JSON: {export_json_path}")


def save_spectra_to_pickle(spectra_list: list, export_filepath: str, export_name: str) -> None:
    """Save spectra to pickle format."""
    file_export_pickle = os.path.join(export_filepath, export_name + ".pickle")
    with open(file_export_pickle, "wb") as f:
        pickle.dump(spectra_list, f)
    logger.info(f"{len(spectra_list)} spectra saved to pickle: {file_export_pickle}")
