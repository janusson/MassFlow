"""
Processing filter functions for cleaning library spectra metadata and peaks.
Ported from original_source/yogimass_pipeline.py.
"""
from __future__ import annotations

import logging
from tqdm import tqdm
from matchms.importing import load_from_mgf, load_from_msp
from matchms.filtering import (
    default_filters,
    repair_inchi_inchikey_smiles,
    derive_adduct_from_name,
    derive_formula_from_name,
    harmonize_undefined_smiles,
    harmonize_undefined_inchi,
    harmonize_undefined_inchikey,
    clean_compound_name,
    derive_ionmode,
    make_charge_int,
    normalize_intensities,
    select_by_intensity,
    select_by_relative_intensity,
    select_by_mz,
)

logger = logging.getLogger(__name__)


def metadata_processing(spectrum):
    """
    Repairs spectrum metadata and return as new spectrum.
    """
    if spectrum is None:
        return None

    # Apply default filters
    spectrum = default_filters(spectrum)
    
    # Metadata repairs and derivations
    spectrum = repair_inchi_inchikey_smiles(spectrum)
    spectrum = derive_adduct_from_name(spectrum)
    spectrum = derive_formula_from_name(spectrum)

    # Harmonization
    spectrum = harmonize_undefined_smiles(spectrum)
    spectrum = harmonize_undefined_inchi(spectrum)
    spectrum = harmonize_undefined_inchikey(spectrum)

    # Standardization
    spectrum = clean_compound_name(spectrum)
    spectrum = derive_ionmode(spectrum)
    spectrum = make_charge_int(spectrum)

    return spectrum


def peak_processing(spectrum):
    """
    Process mass spectrum peaks: filtering and normalization.
    """
    if spectrum is None:
        return None

    spectrum = default_filters(spectrum)
    spectrum = select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.08)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum


def clean_mgf_library(mgf_path: str) -> list:
    """
    Main data processing pipeline. Clean up spectra metadata and peaks for an MGF library.
    """
    logger.info(f"Cleaning {mgf_path} library spectra...")
    library_list = list(load_from_mgf(mgf_path))
    
    # Apply filters sequentially
    processed_spectra = []
    for s in tqdm(library_list, desc="Processing MGF library", unit="spectra"):
        meta_processed = metadata_processing(s)
        if meta_processed:
            peak_processed = peak_processing(meta_processed)
            if peak_processed:
                processed_spectra.append(peak_processed)
    
    logger.info(f"Retained {len(processed_spectra)} spectra after cleaning.")
    return processed_spectra


def clean_msp_library(msp_path: str) -> list:
    """
    Cleans an MSP library given its path using main data processing pipeline.
    """
    logger.info(f"Cleaning {msp_path} library spectra...")
    library_list = list(load_from_msp(msp_path))
    
    processed_spectra = []
    for s in tqdm(library_list, desc="Processing MSP library", unit="spectra"):
        meta_processed = metadata_processing(s)
        if meta_processed:
            peak_processed = peak_processing(meta_processed)
            if peak_processed:
                processed_spectra.append(peak_processed)
                
    logger.info(f"Retained {len(processed_spectra)} spectra after cleaning.")
    return processed_spectra
