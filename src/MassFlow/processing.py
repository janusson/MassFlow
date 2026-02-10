'''
Processing module for spectral cleaning and filtering using matchms.
Acts as a facade for matchms filtering operations with integrated logging.
'''

import logging
from typing import Iterator, Optional

from matchms import Spectrum
from matchms.filtering import (
    clean_compound_name,
    default_filters,
    derive_adduct_from_name,
    derive_formula_from_name,
    derive_ionmode,
    harmonize_undefined_inchi,
    harmonize_undefined_inchikey,
    harmonize_undefined_smiles,
    make_charge_int,
    normalize_intensities,
    repair_inchi_inchikey_smiles,
    require_minimum_number_of_peaks,
    select_by_intensity,
)

from MassFlow.config import ProcessingConfig

logger = logging.getLogger(__name__)


def metadata_processing(spectrum: Spectrum) -> Optional[Spectrum]:
    '''
    Standardize and repair spectrum metadata using matchms filters.

    Args:
        spectrum: The input matchms Spectrum object.

    Returns:
        The processed Spectrum, or None if the spectrum was invalidated.
    '''
    if spectrum is None:
        return None

    # Apply default filters (handles common metadata issues)
    spectrum = default_filters(spectrum)

    # Metadata repairs and derivations
    spectrum = repair_inchi_inchikey_smiles(spectrum)
    spectrum = derive_adduct_from_name(spectrum)
    spectrum = derive_formula_from_name(spectrum)

    # Harmonization of missing values
    spectrum = harmonize_undefined_smiles(spectrum)
    spectrum = harmonize_undefined_inchi(spectrum)
    spectrum = harmonize_undefined_inchikey(spectrum)

    # Final standardization
    spectrum = clean_compound_name(spectrum)
    spectrum = derive_ionmode(spectrum)
    spectrum = make_charge_int(spectrum)

    return spectrum


def peak_processing(spectrum: Spectrum, config: ProcessingConfig) -> Optional[Spectrum]:
    '''
    Apply peak-level filters and normalization based on configuration.
    Follows the order: Filter Intensity -> Filter Peak Count -> Normalize.

    Args:
        spectrum: The input matchms Spectrum object.
        config: ProcessingConfig containing filter thresholds.

    Returns:
        The processed Spectrum, or None if it fails to meet criteria.
    '''
    if spectrum is None:
        return None

    spec_id = spectrum.get("id", "Unknown ID")

    # 1. Filter Noise (Absolute Intensity)
    # Important: Do this before normalization to avoid scaling noise.
    spectrum = select_by_intensity(spectrum, intensity_from=config.min_intensity)
    if spectrum is None:
        logger.debug(
            f"Spectrum {spec_id} dropped: all peaks below min_intensity {config.min_intensity}"
        )
        return None

    # 2. Filter Peak Count
    spectrum = require_minimum_number_of_peaks(spectrum, n_min=config.min_peaks)
    if spectrum is None:
        logger.debug(f"Spectrum {spec_id} dropped: fewer than {config.min_peaks} peaks")
        return None

    # 3. Normalize Intensities
    if config.normalize_intensity:
        spectrum = normalize_intensities(spectrum)

    return spectrum


def process_spectra(
    spectra: Iterator[Spectrum], config: ProcessingConfig
) -> Iterator[Spectrum]:
    '''
    Orchestrate the spectral processing pipeline.
    Iterates through input spectra, applies processing, and logs dropped items.

    Args:
        spectra: Iterator of raw matchms Spectrum objects.
        config: Processing configuration object.

    Yields:
        Cleaned and filtered Spectrum objects.
    '''
    for i, spectrum in enumerate(spectra):
        if spectrum is None:
            continue

        try:
            # Step A: Metadata Processing
            processed_spec = metadata_processing(spectrum)
            if processed_spec is None:
                continue

            # Step B: Peak Processing
            processed_spec = peak_processing(processed_spec, config)
            if processed_spec is None:
                continue

            yield processed_spec

        except Exception as e:
            spec_id = spectrum.get("id", f"index_{i}")
            logger.error(f"Unexpected error processing spectrum {spec_id}: {str(e)}")
            # Continue to next spectrum instead of crashing
            continue
