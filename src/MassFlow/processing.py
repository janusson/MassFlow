"""
Spectral processing and filtering module for MassFlow.

This module serves as a facade for the ``matchms`` library, providing a streamlined
interface for cleaning, filtering, and normalizing mass spectral data. It implements
a two-stage processing pipeline: metadata standardization (e.g., repairing InChIKeys,
deriving formulas) and peak-level filtering (e.g., noise removal, m/z range truncation).
It is designed to fail fast on invalid data while logging detailed diagnostics.
"""

import logging
from typing import Iterator, List, Optional, Tuple

import numpy as np
import polars as pl
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
    require_minimum_number_of_peaks,
    select_by_intensity,
    select_by_mz,
)

from MassFlow.config import ProcessingConfig

logger = logging.getLogger(__name__)


def compute_spectral_metrics(
    mz_array: np.ndarray, precursor_mz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute vectorized spectral metrics: neutral losses and m/z offsets.

    Parameters
    ----------
    mz_array : np.ndarray
        Array of peak m/z values.
    precursor_mz : float
        The precursor m/z for the spectrum.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing (neutral_losses, mz_offsets).
    """
    if precursor_mz is None or precursor_mz <= 0:
        return np.array([]), np.array([])

    neutral_losses = precursor_mz - mz_array
    mz_offsets = mz_array - precursor_mz

    return neutral_losses, mz_offsets


def metadata_processing(
    spectrum: Spectrum, config: Optional[ProcessingConfig] = None
) -> Optional[Spectrum]:
    """
    Standardize and repair spectrum metadata using matchms filters.
    """
    if spectrum is None:
        return None

    s: Spectrum = spectrum

    # Pre-emptively fix ionmode to prevent matchms default_filters from raising AssertionError
    ionmode = s.get("ionmode")
    if ionmode is not None:
        if isinstance(ionmode, (list, tuple)):
            ionmode = ionmode[0] if len(ionmode) > 0 else None
        if ionmode is not None:
            s.set("ionmode", str(ionmode).lower())

    # Apply default filters (handles common metadata issues)
    if config is None or getattr(config, "clean_metadata", True):
        # matchms default_filters may return None
        s_opt = default_filters(s)
        if s_opt is None:
            return None
        s = s_opt

    # Skip matchms add_retention_time because we manually parse it safely in the batch extraction
    # and matchms's internal regexes throw extremely slow WARNING logs for 'CCS:' strings.
    # if config is None or getattr(config, "add_retention_time", True):
    #     s = add_retention_time(s)
    #     if s is None:
    #         return None

    # Metadata repairs and derivations
    if config is None or getattr(config, "repair_inchi_inchikey_smiles", True):
        # matchms filters return a new spectrum or None
        from matchms.filtering import repair_inchi_inchikey_smiles

        s = repair_inchi_inchikey_smiles(s)
        if s is None:
            return None
        s = harmonize_undefined_smiles(s)
        if s is None:
            return None
        s = harmonize_undefined_inchi(s)
        if s is None:
            return None
        s = harmonize_undefined_inchikey(s)
        if s is None:
            return None

    if config is None or getattr(config, "derive_adduct_from_name", True):
        s = derive_adduct_from_name(s)
        if s is None:
            return None

    if config is None or getattr(config, "derive_formula_from_name", True):
        s = derive_formula_from_name(s)
        if s is None:
            return None

    if config is None or getattr(config, "clean_compound_name", True):
        s = clean_compound_name(s)
        if s is None:
            return None

    if config is None or getattr(config, "derive_ionmode", True):
        s = derive_ionmode(s)
        if s is None:
            return None

    if config is None or getattr(config, "make_charge_int", True):
        s = make_charge_int(s)
        if s is None:
            return None

    # Inject instrument metadata if provided in config
    if config and s is not None:
        if config.instrument:
            s.set("instrument", config.instrument)
        if config.mode:
            s.set("ionmode", config.mode)

    return s


def calculate_triage_flags(spectrum: Spectrum) -> Spectrum:
    """
    Compute structural triage flags (e.g., specific immonium ions) and attach
    them to the spectrum metadata for downstream ML routing.
    """
    if spectrum is None or spectrum.peaks is None:
        return spectrum

    mz_array = spectrum.peaks.mz
    intensity_array = spectrum.peaks.intensities

    triage_flags = {}

    # Fast NumPy-based Tyrosine immonium ion check
    target_mz = 136.076
    mz_tolerance = 0.05

    # Ensure mz_array is not empty
    if len(mz_array) > 0:
        idx = np.searchsorted(mz_array, target_mz)
        has_tyrosine = False

        # Check the closest indices
        for i in [idx - 1, idx]:
            if 0 <= i < len(mz_array):
                if abs(mz_array[i] - target_mz) <= mz_tolerance:
                    if intensity_array[i] > 0:
                        has_tyrosine = True
                        break

        if has_tyrosine:
            triage_flags["has_tyrosine_fragment"] = True

    if triage_flags:
        spectrum.set("triage_flags", triage_flags)

    return spectrum


def peak_processing(spectrum: Spectrum, config: ProcessingConfig) -> Optional[Spectrum]:
    """
    Apply peak-level filters and normalization based on configuration.
    """
    if spectrum is None:
        return None

    s: Optional[Spectrum] = spectrum

    # 1. Filter Noise (Absolute Intensity)
    if getattr(config, "filter_by_intensity", True):
        threshold = (
            config.noise_threshold
            if getattr(config, "noise_threshold", 0) > 0
            else getattr(config, "min_intensity", 0.0)
        )
        s = select_by_intensity(s, intensity_from=threshold, intensity_to=float("inf"))
        if s is None:
            return None

    # 2. Filter Peak Count
    if getattr(config, "filter_min_peaks", True):
        s = require_minimum_number_of_peaks(s, n_required=config.min_peaks)
        if s is None:
            return None

    # 3. M/Z Range Truncation
    if getattr(config, "filter_by_mz", True):
        mz_from = getattr(config, "mz_min", 0.0)
        mz_to = getattr(config, "mz_max", 1000.0)
        s = select_by_mz(s, mz_from=mz_from, mz_to=mz_to)
        if s is None:
            return None

    # 4. Max-Peak Restriction (Top-N)
    if getattr(config, "reduce_to_top_n_peaks", False):
        n_max = getattr(config, "n_max", 0)
        if n_max and n_max > 0:
            s = reduce_to_number_of_peaks(s, n_max=n_max)
            if s is None:
                return None

    # 5. Normalize Intensities
    if getattr(config, "normalize_intensity", True):
        s = normalize_intensities(s)
        if s is None:
            return None

    # Compute and attach structural triage flags
    if s is not None:
        s = calculate_triage_flags(s)

    return s


def process_spectra_batch(
    spectra: List[Spectrum], config: ProcessingConfig
) -> List[Spectrum]:
    """
    Process a batch of spectra using Polars for high-performance metadata operations.
    """
    if not spectra:
        return []

    # 1. Extract metadata into a Polars LazyFrame for fast batch validation
    metadata_rows = []
    for i, s in enumerate(spectra):
        # Handle cases where charge might be a list or non-numeric
        raw_charge = s.get("charge", 0)
        if isinstance(raw_charge, (list, tuple)) and len(raw_charge) > 0:
            charge = int(raw_charge[0])
        elif hasattr(raw_charge, "__iter__") and not isinstance(
            raw_charge, (str, bytes)
        ):
            # Handle matchms ChargeList or similar iterables
            try:
                charge = int(next(iter(raw_charge)))
            except (StopIteration, ValueError, TypeError):
                charge = 0
        else:
            try:
                charge = int(raw_charge) if raw_charge is not None else 0
            except (ValueError, TypeError):
                charge = 0

        try:
            rt = float(s.get("retention_time", 0.0))
        except (ValueError, TypeError):
            rt = 0.0

        metadata_rows.append(
            {
                "batch_index": i,
                "id": s.get("id"),
                "precursor_mz": float(s.get("precursor_mz", 0.0)),
                "retention_time": rt,
                "charge": charge,
                "ionmode": s.get("ionmode"),
                "peak_count": len(s.peaks) if s.peaks else 0,
            }
        )

    # Use LazyFrame to prepare filters without immediate execution
    lf = pl.LazyFrame(metadata_rows)

    # Apply batch-level metadata filters (e.g. minimum peaks, m/z range)
    mz_min = getattr(config, "mz_min", 0.0)
    mz_max = getattr(config, "mz_max", 1000.0)
    min_peaks = getattr(config, "min_peaks", 1)

    filtered_lf = lf.filter(
        (pl.col("precursor_mz") >= mz_min)
        & (pl.col("precursor_mz") <= mz_max)
        & (pl.col("peak_count") >= min_peaks)
    )

    # Compute vectorized m/z offsets and neutral losses for the entire batch metadata table
    # (This represents the relationship of the PRECURSOR to nominal mass ranges)
    filtered_lf = filtered_lf.with_columns(
        [
            (pl.col("precursor_mz") % 1).alias("mz_nominal_offset"),
            (pl.col("precursor_mz") - 18.01).alias("theoretical_water_loss"),
        ]
    )

    # Materialize the filtered metadata
    valid_metadata = filtered_lf.collect()
    valid_indices = valid_metadata.get_column("batch_index").to_list()

    processed_batch = []
    for idx in valid_indices:
        spec = spectra[idx]

        try:
            # Apply standard matchms metadata repairs
            spec = metadata_processing(spec, config)
            if spec is None:
                continue
        except Exception as e:
            logger.error(
                f"Skipping spectrum due to metadata processing error: {e}",
                extra={
                    "spectrum_id": spec.get("id"),
                    "precursor_mz": spec.get("precursor_mz"),
                    "compound_name": spec.get("compound_name"),
                    "step": "metadata_processing",
                },
                exc_info=True,
            )
            continue

        try:
            # Apply peak-level processing
            spec = peak_processing(spec, config)
            if spec is None:
                continue
        except Exception as e:
            logger.error(
                f"Skipping spectrum due to peak processing error: {e}",
                extra={
                    "spectrum_id": spec.get("id"),
                    "precursor_mz": spec.get("precursor_mz"),
                    "compound_name": spec.get("compound_name"),
                    "step": "peak_processing",
                },
                exc_info=True,
            )
            continue

        # Vectorized Peak-level calculations (Neutral Loss & Offset)
        # We attach these to the Spectrum metadata for use by similarity engines
        nl, offsets = compute_spectral_metrics(spec.peaks.mz, spec.get("precursor_mz"))
        spec.set("neutral_losses", nl)
        spec.set("mz_offsets", offsets)

        processed_batch.append(spec)

    return processed_batch


def process_spectra(
    spectra: Iterator[Spectrum], config: ProcessingConfig
) -> Iterator[Spectrum]:
    """
    Orchestrate the full spectral processing pipeline.
    Processes in chunks to optimize Polars batch performance while staying memory-aware.
    """
    chunk_size = 5000
    chunk = []

    for spectrum in spectra:
        if spectrum is None:
            continue
        chunk.append(spectrum)

        if len(chunk) >= chunk_size:
            yield from process_spectra_batch(chunk, config)
            chunk.clear()

    if chunk:
        yield from process_spectra_batch(chunk, config)
