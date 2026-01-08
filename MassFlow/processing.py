"""
Processing filter functions for cleaning library spectra metadata and peaks.
Ported from original_source/massflow_pipeline.py.
"""
from __future__ import annotations

import logging
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


from typing import Any, Optional
from matchms import Spectrum

logger = logging.getLogger(__name__)

# Interval for progress logging
LOG_INTERVAL = 1000


def metadata_processing(spectrum: Spectrum) -> Optional[Spectrum]:
    """
    Repair spectrum metadata and return as new spectrum.
    
    Args:
        spectrum: The input matchms Spectrum object.
        
    Returns:
        The processed Spectrum, or None if input was None.
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


def peak_processing(spectrum: Spectrum) -> Optional[Spectrum]:
    """
    Process mass spectrum peaks: filtering and normalization.
    
    Args:
        spectrum: The input matchms Spectrum object.
        
    Returns:
        The processed Spectrum, or None if input was None.
    """
    if spectrum is None:
        return None

    spectrum = default_filters(spectrum)
    spectrum = select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.08)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum



from typing import Any, Optional, Iterator, Iterable

def process_spectra(spectra_iterable: Iterable[Spectrum]) -> Iterator[Spectrum]:
    """
    Apply metadata and peak processing to an iterable of spectra.
    Yields processed spectra one by one.
    
    Args:
        spectra_iterable: Iterable of matchms Spectrum objects.
        
    Yields:
        Processed Spectrum objects.
    """
    for i, s in enumerate(spectra_iterable):
        if (i + 1) % LOG_INTERVAL == 0:
            logger.info(f"Processing spectrum {i + 1}...")

        meta_processed = metadata_processing(s)
        if meta_processed:
            peak_processed = peak_processing(meta_processed)
            if peak_processed:
                yield peak_processed


def clean_mgf_library(mgf_path: str) -> list[Spectrum]:
    """
    Main data processing pipeline. Clean up spectra metadata and peaks for an MGF library.
    
    Args:
        mgf_path: Path to the MGF file.
        
    Returns:
        List of processed Spectrum objects.
    """
    logger.info(f"Cleaning {mgf_path} library spectra...")
    # matchms load_from_mgf returns a generator, do not cast to list immediately
    library_iterable = load_from_mgf(mgf_path)
    
    # We consume the generator into a list here because the specific requirement 
    # (e.g. CLI export) often expects a full list, esp. for pickle.
    # If purely streaming export is needed, this can be adapted.
    processed_spectra = list(process_spectra(library_iterable))
    
    logger.info(f"Retained {len(processed_spectra)} spectra after cleaning.")
    return processed_spectra


def clean_msp_library(msp_path: str) -> list[Spectrum]:
    """
    Cleans an MSP library given its path using main data processing pipeline.
    
    Args:
        msp_path: Path to the MSP file.
        
    Returns:
        List of processed Spectrum objects.
    """
    logger.info(f"Cleaning {msp_path} library spectra...")
    library_iterable = load_from_msp(msp_path)
    
    processed_spectra = list(process_spectra(library_iterable))
                
    logger.info(f"Retained {len(processed_spectra)} spectra after cleaning.")
    return processed_spectra
