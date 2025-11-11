"""
I/O functions for Yogimass: import/export of libraries and spectra.
"""

import glob
import os
from os import path
from pathlib import Path
import pandas as pd
from matchms.importing import load_from_mgf, load_from_msp
from matchms.exporting import save_as_mgf, save_as_msp, save_as_json
import pickle
from yogimass.utils.logging import get_logger
logger = get_logger(__name__)

def list_msp_libraries(directory):
    msp_libraries_list = glob.glob(os.path.join(directory, "*.msp"), recursive=True)
    logger.info(f"{len(msp_libraries_list)} MSP libraries found.")
    return msp_libraries_list

def list_mgf_libraries(directory):
    mgf_libraries_list = glob.glob(path.join(directory, "*.mgf"), recursive=True)
    logger.info(f"{len(mgf_libraries_list)} MGF libraries found.")
    return mgf_libraries_list

def list_available_libraries(mgf_libraries_list, msp_libraries_list):
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
    logger.info("MGF/MSP Libraries listed.")
    return summary

def fetch_mgflib_spectrum(library_filepath, spectrum_number):
    spectra_list = list(load_from_mgf(library_filepath))
    spectrum_peaks = spectra_list[spectrum_number].peaks.mz
    spectrum_counts = spectra_list[spectrum_number].peaks.intensities
    normalized_counts = spectrum_counts / max(spectrum_counts)
    percent_abundance = normalized_counts * 100
    spectrum_dict = dict(zip(spectrum_peaks, percent_abundance))
    spectrum_xy_data = pd.DataFrame.from_dict(
        spectrum_dict, orient="index", columns=["Abundance (%)"]
    )
    spectrum_xy_data.reset_index(inplace=True)
    spectrum_xy_data.rename(columns={"index": "m/z"}, inplace=True)
    spectrum_xy_data.sort_values(by="m/z", inplace=True)
    spectrum_metadata = spectra_list[spectrum_number].metadata
    spectrum_chemical = spectra_list[spectrum_number].metadata["name"]
    return spectrum_xy_data, spectrum_metadata, spectrum_chemical

def save_spectra_to_mgf(spectra_list, export_filepath, export_name):
    export_mgf_path = path.join(export_filepath, export_name + ".mgf")
    save_as_mgf(spectra_list, export_mgf_path)
    logger.info(f"{len(spectra_list)} spectra saved to MGF: {export_mgf_path}")

def save_spectra_to_msp(spectra_list, export_filepath, export_name):
    export_msp_path = path.join(export_filepath, export_name + ".msp")
    save_as_msp(spectra_list, export_msp_path)
    logger.info(f"{len(spectra_list)} spectra saved to MSP: {export_msp_path}")

def save_spectra_to_json(spectra_list, export_filepath, export_name):
    export_json_path = path.join(export_filepath, export_name + ".json")
    save_as_json(spectra_list, export_json_path)
    logger.info(f"{len(spectra_list)} spectra saved to JSON: {export_json_path}")

def save_spectra_to_pickle(spectra_list, export_filepath, export_name):
    file_export_pickle = path.join(export_filepath, export_name + ".pickle")
    pickle.dump(spectra_list, open(file_export_pickle, "wb"))
    logger.info(f"{len(spectra_list)} spectra saved to pickle: {file_export_pickle}")
