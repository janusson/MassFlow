"""
Spectral processing and filtering module for MassFlow.

This module serves as a facade for the ``matchms`` library, providing a streamlined
interface for cleaning, filtering, and normalizing mass spectral data. It implements
a three-stage processing pipeline: metadata standardization (e.g., repairing InChIKeys,
deriving formulas), strict physical-integrity validation of declared chemistry
(5 ppm precursor gate, see ``physical_integrity_reason``), and peak-level filtering
(e.g., noise removal, m/z range truncation).
It is designed to fail fast on invalid data while logging detailed diagnostics.
"""

import logging
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import polars as pl
import pyteomics.mass as pmass
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

from MassFlow.cheminformatics import normalize_adduct
from MassFlow.config import ProcessingConfig

# Neutral monoisotopic mass of water (H2O), computed once at import time from
# pyteomics to guarantee SSOT consistency across the codebase.
_WATER_MASS: float = pmass.calculate_mass(formula="H2O")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strict physical-integrity gate (5 ppm precursor validation)
# ---------------------------------------------------------------------------
#
# The classical batch pipeline constructs the ``SpectrumMetadata`` /
# ``MolecularStructure`` contracts (MassFlow.models) for every spectrum that
# declares a structural claim (formula, smiles, or inchi) and rejects spectra
# whose declared chemistry is malformed or physically impossible (> 5 ppm
# precursor deviation). Spectra without structural claims are exempt and pass
# through without any model construction (the dominant case for raw
# experimental query files, so the gate adds no measurable cost there).

# ``MolecularStructure`` auto-fills a theoretical isotopic envelope on every
# construction (~3.4 ms/spectrum: pyteomics isotopologue enumeration). The
# classical annotation pipeline never consumes isotopic envelopes (they are
# used only by the model layer / experimental ML surfaces), so the gate pins
# a non-empty placeholder to suppress the auto-fill. The 5 ppm verdicts are
# unaffected: the envelope is never part of the physical-validity logic.
_ISOTOPIC_ENVELOPE_SKIP_MARKER = [(0.0, 0.0)]

# Quarantine logger: the same dedicated logger io.py uses for spectra rejected
# by the I/O validation layer, so gate rejections appear in the quarantine log
# tail surfaced by the diagnostics surface (tui/diagnostics.py).
quarantine_logger = logging.getLogger("quarantine")


class PhysicalIntegrityError(Exception):
    """Raised when a reference library fails the strict physical-integrity gate.

    A library spectrum that declares a molecular structure whose metadata is
    malformed or whose precursor m/z contradicts the declared chemistry by
    more than 5 ppm is a data-integrity failure: silently searching a shrunk
    target pool would change every FDR calibration. The annotate path
    therefore aborts the run and reports this error instead of proceeding
    with a silently altered library.
    """

    def __init__(self, message: str, rejection_reasons: List[str]) -> None:
        super().__init__(message)
        self.rejection_reasons = rejection_reasons


def _spectrum_identifier(spectrum: Spectrum) -> str:
    """Best-effort human-readable identifier for a spectrum in messages."""
    for key in ("id", "spectrum_id", "scans", "compound_name", "name"):
        value = spectrum.get(key)
        if value not in (None, ""):
            return str(value)
    return "<unknown>"


def _clean_optional_str(value: Any) -> Optional[str]:
    """Return a stripped string, or None for empty / non-string values."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _short_exception_message(exc: Exception) -> str:
    """One-line, truncated representation of a validation exception."""
    text = " ".join(str(exc).split())
    return text[:200] or exc.__class__.__name__


def physical_integrity_reason(spectrum: Spectrum) -> Optional[str]:
    """Verdict of the strict 5 ppm physical-integrity gate for one spectrum.

    This is the classical-annotate counterpart of the streaming ingestion
    gate (``MassFlow.streaming.engine.validate_streaming_spectrum``): it runs
    the existing ``SpectrumMetadata`` / ``MolecularStructure`` contracts from
    :mod:`MassFlow.models` against harmonized spectrum metadata and returns a
    human-readable rejection reason when the spectrum must not enter the
    search pool.

    The check activates **only** for spectra that declare a structural claim
    (``formula``, ``smiles``, or ``inchi``). Spectra without such a claim are
    exempt (``None``) and pay no construction cost. Within a claim, the
    verdicts follow the model contracts exactly:

    * an unparseable SMILES/InChI claim is rejected;
    * a declared ``exact_mass`` conflicting with the formula-derived mass by
      more than 5 ppm is rejected;
    * with a complete context (structure, charge, and an adduct -- explicit
      or imputed from ``ionmode``), a precursor m/z deviating from the
      theoretical m/z by more than 5 ppm is rejected;
    * a non-registry adduct with an otherwise complete context is rejected;
    * malformed structural metadata (e.g. an unparseable formula) is
      rejected with the underlying error.

    Missing context (no charge, no adduct, no derivable ion mode) disables
    the strict mass check for that spectrum, exactly as documented for
    ``SpectrumMetadata``; without RDKit, smiles/inchi-only claims degrade to
    the documented formula-only fallback.

    Parameters
    ----------
    spectrum : matchms.Spectrum
        A spectrum whose metadata has already been harmonized by
        :func:`metadata_processing`.

    Returns
    -------
    str or None
        A rejection reason, or ``None`` when the spectrum passes the gate
        (or is exempt from it).
    """
    # Import lazily: models.py pulls in the optional RDKit stack, which must
    # stay out of the import graph for environments without the [chem] extra.
    from MassFlow.models import MolecularStructure, SpectrumMetadata

    precursor_raw = spectrum.get("precursor_mz")
    if precursor_raw is None:
        return None
    try:
        precursor_mz = float(precursor_raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(precursor_mz) or precursor_mz <= 0:
        return None

    formula = _clean_optional_str(spectrum.get("formula"))
    smiles = _clean_optional_str(spectrum.get("smiles"))
    inchi = _clean_optional_str(spectrum.get("inchi"))
    if not (formula or smiles or inchi):
        # Fast path: no structural claim -> nothing to verify. This is the
        # dominant case for raw experimental query files.
        return None

    spec_id = _spectrum_identifier(spectrum)

    # exact_mass is only meaningful alongside a structural claim; keep it
    # None when absent so the models auto-fill it from the formula. Note:
    # matchms normalizes the ``exact_mass`` metadata key to ``parent_mass``
    # on Spectrum construction, so read the canonical key first.
    exact_mass: Optional[float] = None
    exact_mass_raw = spectrum.get("parent_mass") or spectrum.get("exact_mass")
    if exact_mass_raw not in (None, ""):
        try:
            exact_mass = float(exact_mass_raw)
        except (TypeError, ValueError):
            return (
                f"spectrum {spec_id}: malformed exact_mass metadata "
                f"({exact_mass_raw!r}) cannot be parsed as a number"
            )
        if not np.isfinite(exact_mass):
            return (
                f"spectrum {spec_id}: malformed exact_mass metadata "
                f"({exact_mass_raw!r}) is not a finite number"
            )

    # Charge: 0 / missing is treated as unknown (mirroring the streaming
    # gate), which disables the strict mass check per the model contract.
    # Normalize container-style charge values (mzML/MGF loaders can produce
    # lists/iterables) the same way the batch metadata extraction does.
    charge_raw = spectrum.get("charge")
    if isinstance(charge_raw, (list, tuple)) and len(charge_raw) > 0:
        charge_raw = charge_raw[0]
    elif hasattr(charge_raw, "__iter__") and not isinstance(charge_raw, (str, bytes)):
        try:
            charge_raw = next(iter(charge_raw))
        except (StopIteration, TypeError):
            charge_raw = None
    charge: Optional[int] = None
    if charge_raw not in (None, 0, "", "0"):
        try:
            charge = int(charge_raw)
        except (TypeError, ValueError):
            charge = None

    # ionmode participates in the model via adduct imputation and the strict
    # positive/negative/neutral literal. An unparseable value means "unknown
    # ion mode": pass None so the model does not impute an adduct.
    ion_mode: Optional[str] = None
    ionmode_raw = _clean_optional_str(spectrum.get("ionmode"))
    if ionmode_raw in ("positive", "negative", "neutral"):
        ion_mode = ionmode_raw

    adduct = _clean_optional_str(spectrum.get("adduct"))

    try:
        molecule = MolecularStructure(
            formula=formula,
            smiles=smiles,
            inchi=inchi,
            exact_mass=exact_mass,
            # Suppress the ~3.4 ms/spectrum isotopic-envelope auto-fill; the
            # envelope is not consumed by the classical annotation path and
            # the physical verdicts are unaffected (see module notes).
            isotopic_envelope=_ISOTOPIC_ENVELOPE_SKIP_MARKER,
        )
    except Exception as exc:  # malformed structural metadata (e.g. formula)
        return (
            f"spectrum {spec_id}: malformed structural metadata "
            f"({_short_exception_message(exc)})"
        )

    if not molecule.is_physically_valid:
        if molecule.formula and molecule.exact_mass is not None:
            # The formula was resolved (declared or auto-filled), so an
            # invalid verdict with an exact mass present is a mass conflict.
            try:
                from MassFlow.cheminformatics import _formula_to_monoisotopic_mass

                formula_mass = _formula_to_monoisotopic_mass(molecule.formula)
                ppm_error = abs(molecule.exact_mass - formula_mass) / formula_mass * 1e6
                return (
                    f"spectrum {spec_id}: declared exact mass "
                    f"{molecule.exact_mass} conflicts with the formula-derived "
                    f"mass {formula_mass} of {molecule.formula} by "
                    f"{ppm_error:.2f} ppm (>5.0 ppm limit)"
                )
            except Exception:
                pass
        return (
            f"spectrum {spec_id}: declared SMILES/InChI could not be parsed "
            "as a valid chemical structure"
        )

    try:
        metadata = SpectrumMetadata(
            spectrum_id=spec_id,
            precursor_mz=precursor_mz,
            charge=charge,
            ion_mode=ion_mode,
            adduct=adduct,
            molecule=molecule,
        )
    except Exception as exc:  # malformed metadata (field-level constraints)
        return (
            f"spectrum {spec_id}: malformed metadata for strict precursor "
            f"validation ({_short_exception_message(exc)})"
        )

    if metadata.is_physically_valid:
        return None

    # The molecule is valid, so an invalid spectrum verdict is either an
    # unsupported adduct or a >5 ppm precursor deviation. Recompute the
    # display values (identical arithmetic to the model validator).
    from MassFlow.cheminformatics import compute_adduct_offset

    effective_adduct = metadata.adduct
    offset = compute_adduct_offset(effective_adduct) if effective_adduct else None
    if offset is None:
        return (
            f"spectrum {spec_id}: adduct {effective_adduct!r} is not supported "
            "by the MassFlow adduct registry; strict precursor validation "
            "cannot confirm the declared structure"
        )
    if metadata.charge and molecule.exact_mass is not None:
        theoretical_mz = (molecule.exact_mass + offset) / abs(metadata.charge)
        ppm_error = abs(precursor_mz - theoretical_mz) / theoretical_mz * 1e6
        return (
            f"spectrum {spec_id}: precursor m/z {precursor_mz:.4f} deviates "
            f"{ppm_error:.2f} ppm from the theoretical m/z "
            f"{theoretical_mz:.4f} of the declared structure "
            f"(formula {molecule.formula or 'unknown'}, adduct "
            f"{effective_adduct}); limit is 5.0 ppm"
        )
    return (
        f"spectrum {spec_id}: declared structure is not physically consistent "
        "with the precursor m/z (strict 5 ppm validation)"
    )


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
    spectrum: Optional[Spectrum], config: Optional[ProcessingConfig] = None
) -> Optional[Spectrum]:
    """
    Standardize and repair spectrum metadata using matchms filters.
    """
    if spectrum is None:
        return None

    s: Spectrum = spectrum

    # Pre-emptively fix ionmode to prevent matchms default_filters from raising AssertionError
    ionmode = s.get("ionmode")
    if isinstance(ionmode, str):
        s.set("ionmode", ionmode.lower())

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

    # Canonicalise adduct notation (e.g. "M+H", "[M+H]1+") so the stored
    # metadata, the physics gate below, and the search engines all see the
    # same spelling. Unknown labels are left untouched and fail the gate.
    canonical_adduct = normalize_adduct(_clean_optional_str(s.get("adduct")))
    if canonical_adduct is not None:
        s.set("adduct", canonical_adduct)

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


def peak_processing(
    spectrum: Optional[Spectrum], config: ProcessingConfig
) -> Optional[Spectrum]:
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

    return s


def process_spectra_batch(
    spectra: List[Spectrum],
    config: ProcessingConfig,
    rejection_reporter: Optional[Callable[[str], None]] = None,
) -> List[Spectrum]:
    """
    Process a batch of spectra using Polars for high-performance metadata operations.

    Parameters
    ----------
    spectra : list of matchms.Spectrum
        Spectra to process.
    config : ProcessingConfig
        Processing parameters.
    rejection_reporter : Callable[[str], None] or None
        Optional callback invoked with the rejection reason for every
        spectrum dropped by the strict 5 ppm physical-integrity gate. Used by
        the workflow to make chemistry rejections observable in the per-file
        execution result (a scientific analysis must never silently drop
        data). Peak-filter drops are *not* reported here; they remain
        config-driven and are counted by the caller via the length delta.
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

    # Apply batch-level metadata filters, gated on the same documented
    # toggles as the per-spectrum path (peak_processing):
    #   - filter_min_peaks controls the minimum-peak-count rejection;
    #   - filter_by_mz controls the precursor m/z window.
    # Without this gating, the batch path silently dropped spectra even when
    # the toggles were disabled (a configuration that said "do not filter"
    # filtered anyway).
    mz_min = getattr(config, "mz_min", 0.0)
    mz_max = getattr(config, "mz_max", 1000.0)
    min_peaks = getattr(config, "min_peaks", 1)

    batch_filters = []
    if getattr(config, "filter_by_mz", False):
        batch_filters.append(
            (pl.col("precursor_mz") >= mz_min) & (pl.col("precursor_mz") <= mz_max)
        )
    if getattr(config, "filter_min_peaks", False):
        batch_filters.append(pl.col("peak_count") >= min_peaks)

    if batch_filters:
        filtered_lf = lf.filter(batch_filters)
    else:
        filtered_lf = lf

    # Compute vectorized m/z offsets and neutral losses for the entire batch metadata table
    # (This represents the relationship of the PRECURSOR to nominal mass ranges)
    filtered_lf = filtered_lf.with_columns(
        [
            (pl.col("precursor_mz") % 1).alias("mz_nominal_offset"),
            (pl.col("precursor_mz") - _WATER_MASS).alias("theoretical_water_loss"),
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

        # Strict 5 ppm physical-integrity gate: a spectrum that declares a
        # molecular structure whose metadata is malformed or whose precursor
        # m/z contradicts the declared chemistry must never enter the search
        # pool (mirrors the streaming ingestion gate).
        physics_reason = physical_integrity_reason(spec)
        if physics_reason is not None:
            quarantine_logger.warning(
                "Quarantined Spectrum | Source: processing gate | "
                f"ID: {_spectrum_identifier(spec)} | Reason: {physics_reason}"
            )
            if rejection_reporter is not None:
                rejection_reporter(physics_reason)
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

        processed_batch.append(spec)

    return processed_batch


def process_spectra(
    spectra: Iterable[Spectrum],
    config: ProcessingConfig,
    rejection_reporter: Optional[Callable[[str], None]] = None,
) -> Iterator[Spectrum]:
    """
    Orchestrate the full spectral processing pipeline.
    Processes in chunks to optimize Polars batch performance while staying memory-aware.

    Parameters
    ----------
    spectra : Iterable[matchms.Spectrum]
        Spectra to process.
    config : ProcessingConfig
        Processing parameters.
    rejection_reporter : Callable[[str], None] or None
        Optional callback invoked with the rejection reason for every
        spectrum dropped by the strict 5 ppm physical-integrity gate (see
        :func:`process_spectra_batch`).
    """
    chunk_size = 5000
    chunk = []

    for spectrum in spectra:
        if spectrum is None:
            continue
        chunk.append(spectrum)

        if len(chunk) >= chunk_size:
            yield from process_spectra_batch(chunk, config, rejection_reporter)
            chunk.clear()

    if chunk:
        yield from process_spectra_batch(chunk, config, rejection_reporter)
