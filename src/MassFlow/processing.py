"""
Spectral processing and filtering module for MassFlow.

This module serves as a facade for the ``matchms`` library, providing a streamlined
interface for cleaning, filtering, and normalizing mass spectral data. It implements
a two-stage processing pipeline: metadata standardization (e.g., repairing InChIKeys,
deriving formulas) and peak-level filtering (e.g., noise removal, m/z range truncation).
It is designed to fail fast on invalid data while logging detailed diagnostics.
"""

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
    reduce_to_number_of_peaks,
    repair_inchi_inchikey_smiles,
    require_minimum_number_of_peaks,
    select_by_intensity,
    select_by_mz,
)

from MassFlow.config import ProcessingConfig

logger = logging.getLogger(__name__)


def metadata_processing(
    spectrum: Spectrum, config: Optional[ProcessingConfig] = None
) -> Optional[Spectrum]:
    """
    Standardize and repair spectrum metadata using matchms filters.

    This function applies a sequence of metadata cleaning operations to ensure consistency
    across spectral records. It handles tasks such as repairing InChIKeys, deriving
    formulas and adducts from compound names, harmonizing missing identifiers (SMILES,
    InChI), and standardizing charge and ion mode information. If a configuration
    object is provided, it can also inject instrument-specific metadata.

    Parameters
    ----------
    spectrum : matchms.Spectrum
        The input spectrum object to process.
    config : ProcessingConfig, optional
        Configuration object containing instrument metadata (e.g., instrument name,
        ionization mode) to be injected into the spectrum. Default is None.

    Returns
    -------
    matchms.Spectrum or None
        The processed spectrum with standardized metadata, or None if the spectrum
        is invalidated during processing (e.g., due to critical missing info).
    """
    if spectrum is None:
        return None

    # Apply default filters (handles common metadata issues)
    # Explicit casting or assignment to handle Optional[Spectrum] return types
    s: Optional[Spectrum] = default_filters(spectrum)
    if s is None:
        return None

    # Metadata repairs and derivations
    s = repair_inchi_inchikey_smiles(s)
    if s is None:
        return None

    s = derive_adduct_from_name(s)
    if s is None:
        return None

    s = derive_formula_from_name(s)
    if s is None:
        return None

    # Harmonization of missing values
    s = harmonize_undefined_smiles(s)
    if s is None:
        return None

    s = harmonize_undefined_inchi(s)
    if s is None:
        return None

    s = harmonize_undefined_inchikey(s)
    if s is None:
        return None

    # Final standardization
    s = clean_compound_name(s)
    if s is None:
        return None

    s = derive_ionmode(s)
    if s is None:
        return None

    s = make_charge_int(s)
    if s is None:
        return None

    # Inject instrument metadata if provided in config
    if config:
        if config.instrument:
            s.set("instrument", config.instrument)
        if config.mode:
            s.set("ionmode", config.mode)

    return s


def peak_processing(spectrum: Spectrum, config: ProcessingConfig) -> Optional[Spectrum]:
    """
    Apply peak-level filters and normalization based on configuration.

    This function filters spectral peaks based on intensity thresholds (noise removal),
    peak counts (minimum required peaks), and m/z ranges. It can also reduce the
    spectrum to the top-N peaks and normalize intensities. The operations are performed
    in a specific order:
    1. Filter by intensity (Noise Threshold)
    2. Filter by minimum peak count
    3. Filter by m/z range
    4. Reduce to Top-N peaks
    5. Normalize intensities

    Parameters
    ----------
    spectrum : matchms.Spectrum
        The input spectrum object containing peak data.
    config : ProcessingConfig
        The configuration object defining filter thresholds (e.g., `noise_threshold`,
        `min_peaks`, `mz_min`, `mz_max`, `n_max`, `normalize_intensity`).

    Returns
    -------
    matchms.Spectrum or None
        The processed spectrum with filtered/normalized peaks, or None if the
        spectrum fails any of the filtering criteria (e.g., too few peaks remaining).
    """
    if spectrum is None:
        return None

    spec_id = spectrum.get("id", "Unknown ID")

    # 1. Filter Noise (Absolute Intensity)
    # Use noise_threshold from config if available (and positive), otherwise fallback to min_intensity
    threshold = (
        config.noise_threshold if config.noise_threshold > 0 else config.min_intensity
    )

    s: Optional[Spectrum] = select_by_intensity(
        spectrum, intensity_from=threshold, intensity_to=float("inf")
    )
    if s is None:
        logger.debug(
            f"Spectrum {spec_id} dropped: all peaks below noise threshold {threshold}"
        )
        return None

    # 2. Filter Peak Count
    s = require_minimum_number_of_peaks(s, n_required=config.min_peaks)
    if s is None:
        logger.debug(f"Spectrum {spec_id} dropped: fewer than {config.min_peaks} peaks")
        return None

    # 3. M/Z Range Truncation
    # Defaults to 0-1000 Da if not specified in config
    mz_from = getattr(config, "mz_min", 0.0)
    mz_to = getattr(config, "mz_max", 1000.0)
    s = select_by_mz(s, mz_from=mz_from, mz_to=mz_to)
    if s is None:
        logger.debug(
            f"Spectrum {spec_id} dropped: no peaks in m/z range {mz_from}-{mz_to}"
        )
        return None

    # 4. Max-Peak Restriction (Top-N)
    n_max = getattr(config, "n_max", 0)
    if n_max and n_max > 0:
        s = reduce_to_number_of_peaks(s, n_max=n_max)
        if s is None:
            return None

    # 5. Normalize Intensities
    if config.normalize_intensity:
        s = normalize_intensities(s)
        if s is None:
            return None

    return s


def process_spectra(
    spectra: Iterator[Spectrum], config: ProcessingConfig
) -> Iterator[Spectrum]:
    """
    Orchestrate the full spectral processing pipeline.

    This generator function iterates through a collection of raw spectra and applies
    both metadata processing (`metadata_processing`) and peak processing (`peak_processing`)
    sequentially. It adheres to a 'fail-fast' approach where invalid spectra (returning None
    from any step) are silently skipped (logged within the sub-functions).

    Parameters
    ----------
    spectra : Iterator[matchms.Spectrum]
        An iterator yielding raw spectrum objects to be processed.
    config : ProcessingConfig
        The configuration object governing all processing steps.

    Yields
    ------
    matchms.Spectrum
        Fully processed, cleaned, and filtered spectrum objects ready for analysis.
    """
    for spectrum in spectra:
        if spectrum is None:
            continue

        # Step A: Metadata Processing
        processed_spec = metadata_processing(spectrum, config)
        if processed_spec is None:
            continue

        # Step B: Peak Processing
        processed_spec = peak_processing(processed_spec, config)
        if processed_spec is None:
            continue

        yield processed_spec
