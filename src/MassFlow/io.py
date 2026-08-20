"""
I/O helpers for MassFlow spectral data.

This module is the file-system boundary for MassFlow. It loads spectra from
open interchange formats such as mzML, mzXML, MGF, and MSP, as well as
SQLite databases. It exports search results to CSV and mzTab-M formats and
provides utilities for writing provenance reports and open-format spectra.

The loader is intentionally narrow in scope: it does not auto-convert
proprietary vendor files and it does not sanitize metadata during import.
Vendor raw formats are rejected with an actionable error, and metadata
harmonization is deferred to :mod:`MassFlow.processing` after import.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import polars as pl
import yaml
from matchms import Spectrum
from matchms.importing import (
    load_from_mgf,
    load_from_msp,
    load_from_mzml,
    load_from_mzxml,
)

logger = logging.getLogger(__name__)

PROPRIETARY_FORMATS = {".raw", ".d", ".wiff", ".lcd", ".t2d", ".baf"}


class UnsupportedVendorFormatError(Exception):
    """Raised when ``load_spectra`` receives a vendor-specific raw data format."""

    pass


quarantine_logger = logging.getLogger("quarantine")


def _validate_spectra_iterator(
    spectra: Iterable[Spectrum], source_path: Path
) -> Iterator[Spectrum]:
    """
    A high-performance validation layer for spectrum iterators.

    This generator function intercepts each spectrum and performs a series of
    strict checks. If a spectrum is invalid, it is logged to a dedicated
    quarantine log and skipped, never reaching the downstream processing engine.

    Checks:
    1. Precursor MZ must exist, be numeric, and be positive.
    2. Peak arrays must not be empty.
    3. M/Z and intensity arrays must have matching lengths.
    4. All intensity values must be positive.
    5. M/Z values must be monotonically increasing.

    Parameters
    ----------
    spectra : Iterable[Spectrum]
        An iterator of raw spectrum objects from a loader.
    source_path : Path
        The file path the spectrum was loaded from (for logging).

    Yields
    ------
    Spectrum
        A valid spectrum object that has passed all checks.
    """
    for spectrum in spectra:
        if spectrum is None:
            continue

        # Manually handle PEPMASS for MGF files when harmonization is off
        if spectrum.get("precursor_mz") is None and spectrum.get("pepmass") is not None:
            pepmass = spectrum.get("pepmass")
            if isinstance(pepmass, (tuple, list)) and len(pepmass) > 0:
                spectrum.set("precursor_mz", pepmass[0])
            else:
                spectrum.set("precursor_mz", pepmass)

        spec_id = "Unknown"
        if spectrum.get("id"):
            spec_id = f"ID={spectrum.get('id')}"
        elif spectrum.get("spectrum_id"):
            spec_id = f"ID={spectrum.get('spectrum_id')}"
        elif spectrum.get("scans"):
            spec_id = f"SCANS={spectrum.get('scans')}"

        is_valid = True
        rejection_reason = ""

        # 1. Precursor Check
        precursor_mz = spectrum.get("precursor_mz")
        if precursor_mz is None:
            is_valid = False
            rejection_reason = "Missing precursor_mz"
        else:
            try:
                if float(precursor_mz) <= 0:
                    is_valid = False
                    rejection_reason = f"Non-positive precursor_mz: {precursor_mz}"
            except (ValueError, TypeError):
                is_valid = False
                rejection_reason = f"Non-numeric precursor_mz: {precursor_mz}"

        # 2. Peaks Check
        if is_valid:
            if spectrum.peaks is None or len(spectrum.peaks.mz) == 0:
                is_valid = False
                rejection_reason = "Empty peak arrays"
            elif len(spectrum.peaks.mz) != len(spectrum.peaks.intensities):
                is_valid = False
                rejection_reason = (
                    f"Mismatched array lengths (mz={len(spectrum.peaks.mz)}, "
                    f"int={len(spectrum.peaks.intensities)})"
                )
            elif not np.all(np.diff(spectrum.peaks.mz) >= 0):
                is_valid = False
                rejection_reason = "M/Z values are not monotonically increasing"
            elif not np.all(spectrum.peaks.intensities > 0):
                is_valid = False
                rejection_reason = "Contains non-positive intensity values"

        if is_valid:
            yield spectrum
        else:
            quarantine_logger.warning(
                f"Quarantined Spectrum | Source: {source_path.name} | "
                f"ID: {spec_id} | Reason: {rejection_reason}"
            )


def load_spectra(
    file_path: Path, file_format: Optional[str] = None
) -> Iterator[Spectrum]:
    """
    Load spectra from an open spectral file or a MassFlow-native store.

    The loader infers the format from ``file_path`` unless ``file_format`` is
    provided explicitly. Supported formats are mzML, mzXML, MGF, MSP, and SQLite
    (``db``/``sqlite``). Imported spectra are yielded exactly as the backend
    loader returns them; MassFlow's metadata cleaning and peak filtering happen
    later in :func:`MassFlow.processing.process_spectra`.

    Parameters
    ----------
    file_path : Path
        Path to the input file or directory. Bruker ``.d`` directories are
        treated as vendor raw inputs and rejected unless pre-converted.
    file_format : str, optional
        Explicit format override. Accepted values are ``mzml``, ``mzxml``,
        ``mgf``, ``msp``, ``db``, and ``sqlite``. If omitted, the format is
        inferred from the file extension.

    Yields
    ------
    matchms.Spectrum
        Raw spectrum objects yielded one at a time.

    Raises
    ------
    UnsupportedVendorFormatError
        If ``file_path`` points to a vendor-specific raw format that must be
        converted to an open format before MassFlow ingestion.
    ValueError
        If the file format is unsupported.

    Notes
    -----
    ``matchms`` importers are called with ``metadata_harmonization=False`` so
    that MassFlow can apply its own processing pipeline in
    :mod:`MassFlow.processing`.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    ext = path.suffix.lower()

    # Step 1: Check for unsupported vendor formats
    if ext in PROPRIETARY_FORMATS or (path.is_dir() and ext == ".d"):
        raise UnsupportedVendorFormatError(
            "MassFlow requires open data formats. Please convert vendor files to .mzML or .mgf using ProteoWizard or MS-DIAL prior to pipeline ingestion."
        )

    # Step 2: Determine loading function
    fmt = (file_format or ext.lstrip(".")).lower()
    loader = None
    # Disable matchms internal harmonization to allow MassFlow's processing module to handle it

    if fmt == "mzml":
        loader = load_from_mzml(str(path), metadata_harmonization=False)
    elif fmt == "msp":
        loader = load_from_msp(str(path), metadata_harmonization=False)
    elif fmt == "mgf":
        loader = load_from_mgf(str(path), metadata_harmonization=False)
    elif fmt == "mzxml":
        loader = load_from_mzxml(str(path), metadata_harmonization=False)
    elif fmt in ["db", "sqlite"]:
        from MassFlow.storage import create_spectral_store

        # A hybrid database is a SQLite file whose peak arrays live in a
        # sibling `<stem>.zarr` store; attach it so metadata-only reads
        # still resolve fragment arrays.
        if path.with_suffix(".zarr").is_dir():
            store = create_spectral_store(path, backend="hybrid")
        else:
            store = create_spectral_store(path, backend="sqlite")
        loader = store.get_spectra()
    elif fmt == "zarr":
        from MassFlow.storage import create_spectral_store

        store = create_spectral_store(path, backend="zarr")
        loader = store.get_spectra()
    else:
        raise ValueError(f"Format '{fmt}' is not supported by MassFlow.")

    # Step 3: Yield spectra through the validation layer
    yield from _validate_spectra_iterator(loader, path)


def _build_results_dataframe(
    results: list[dict[str, Any]],
    query_spectra: Optional[Iterable[Spectrum]] = None,
) -> Optional[pl.DataFrame]:
    """
    Construct a results DataFrame from match results and optional query spectra.

    This is an internal helper shared by the various export functions to ensure
    consistent data shaping, merging, and status labeling.
    """
    # Sanitize any numpy scalars in result dicts before they enter Polars.
    # numpy bool/int/float scalars trigger DeprecationWarnings (and future
    # errors) when interpreted as indices during DataFrame construction.
    clean_results = []
    for r in results:
        clean_r = r.copy()
        for key in ("is_decoy",):
            if key in clean_r and hasattr(clean_r[key], "item"):
                clean_r[key] = clean_r[key].item()
        clean_results.append(clean_r)

    if query_spectra is not None:
        q_ids, q_mzs, q_rts = [], [], []
        for q in query_spectra:
            q_ids.append(str(q.get("id")))
            q_mz = q.get("precursor_mz")
            q_rt = q.get("retention_time")
            q_mzs.append(float(q_mz) if q_mz is not None else None)
            q_rts.append(float(q_rt) if q_rt is not None else None)

        base_df = pl.DataFrame(
            {
                "query_id": q_ids,
                "query_precursor_mz": q_mzs,
                "query_retention_time": q_rts,
            },
            schema={
                "query_id": pl.Utf8,
                "query_precursor_mz": pl.Float64,
                "query_retention_time": pl.Float64,
            },
        )

        if clean_results:
            results_df = pl.DataFrame(clean_results)
            if "is_decoy" in results_df.columns:
                results_df = results_df.with_columns(
                    pl.col("is_decoy").cast(pl.Boolean)
                )

            df = base_df.join(results_df, on="query_id", how="left")
            if "query_precursor_mz_right" in df.columns:
                df = df.drop("query_precursor_mz_right")
        else:
            df = base_df
    else:
        if not clean_results:
            return None
        df = pl.DataFrame(clean_results)

    # Add Annotation_Status
    if "score" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("score").is_null())
            .then(pl.lit("Unknown"))
            .when(pl.col("score") >= 0.9)
            .then(pl.lit("Matched"))
            .otherwise(pl.lit("Putative"))
            .alias("Annotation_Status")
        )
    else:
        df = df.with_columns(pl.lit("Unknown").alias("Annotation_Status"))

    return df


def save_match_results(
    results: list[dict[str, Any]],
    output_path: Path,
    query_spectra: Optional[Iterable[Spectrum]] = None,
) -> None:
    """
    Save annotation results to a CSV report.

    If ``query_spectra`` is provided, the output contains one row per query
    spectrum, including unmatched queries. Match rows are left-joined onto that
    base table using ``query_id``. An ``Annotation_Status`` column is added with
    the values ``Matched`` for scores of at least 0.9, ``Putative`` for lower
    non-null scores, and ``Unknown`` when no score is available.

    Parameters
    ----------
    results : list of dict
        Match result rows to export. Each row is expected to contain at least a
        ``query_id`` key when ``query_spectra`` is provided.
    output_path : Path
        The destination file path for the CSV output. Parent directories will be
        created if they do not exist.
    query_spectra : Optional[Iterable[Spectrum]]
        Full set of experimental query spectra. When provided, unmatched queries
        are still represented in the CSV output.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the output directory cannot be created or the CSV file cannot be
        written.

    Notes
    -----
    If both ``results`` is empty and ``query_spectra`` is ``None``, the
    function logs a warning and returns without writing a file.
    """
    df = _build_results_dataframe(results, query_spectra)
    if df is None:
        logger.warning("No results to save and no query_spectra provided.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def save_analysis_report(
    output_path: Path,
    report_data: dict[str, Any],
) -> None:
    """
    Save a YAML sidecar report describing the provenance of an analysis output.

    The report is intended to accompany a CSV results file and capture the
    configuration and runtime context that produced it. This helps keep result
    tables lightweight while preserving the scientific and procedural details
    needed to reproduce or interpret a run.

    Parameters
    ----------
    output_path : Path
        Destination path for the YAML report file.
    report_data : dict[str, Any]
        Serializable report content describing the analysis provenance.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the output directory cannot be created or the file cannot be
        written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "report_created_at": datetime.now(timezone.utc).isoformat(),
        **report_data,
    }

    with open(output_path, "w") as file_handle:
        yaml.safe_dump(
            report_payload,
            file_handle,
            sort_keys=False,
            allow_unicode=True,
        )

    logger.info(f"Analysis report saved to {output_path}")


def save_spectra_to_msp(spectra: Iterable[Spectrum], export_path: Path) -> None:
    """
    Export spectra to an MSP file.

    Parameters
    ----------
    spectra : Iterable[matchms.Spectrum]
        Spectra to export. The iterable is materialized into memory before
        passing it to the MSP writer.
    export_path : Path
        The file path for the resulting MSP file.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the output directory cannot be created or the MSP file cannot be
        written.
    """
    from matchms.exporting import save_as_msp

    export_path.parent.mkdir(parents=True, exist_ok=True)
    save_as_msp(list(spectra), str(export_path))


def save_spectra_to_mgf(spectra: Iterable[Spectrum], export_path: Path) -> None:
    """
    Export spectra to an MGF file.

    Parameters
    ----------
    spectra : Iterable[matchms.Spectrum]
        Spectra to export.
    export_path : Path
        The file path for the resulting MGF file.
    """
    from matchms.exporting import save_as_mgf

    export_path.parent.mkdir(parents=True, exist_ok=True)
    save_as_mgf(list(spectra), str(export_path))


def save_match_results_to_mztab(
    results: list[dict[str, Any]],
    output_path: Path,
    query_spectra: Optional[Iterable[Spectrum]] = None,
) -> None:
    """
    Save annotation results in a minimal mzTab-M format.

    This exporter produces a tab-separated file with MTD (metadata) and
    SML (small molecule list) sections compatible with GNPS and other
    metabolomics tools.

    Parameters
    ----------
    results : list of dict
        Match result rows to export.
    output_path : Path
        The destination file path for the mzTab-M output.
    query_spectra : Optional[Iterable[Spectrum]]
        Full set of experimental query spectra.
    """
    df = _build_results_dataframe(results, query_spectra)
    if df is None:
        logger.warning("No results to save and no query_spectra provided.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # 1. Metadata Section (MTD)
        f.write("MTD\tmzTab-version\t2.0.0-M\n")
        f.write("MTD\tmzTab-ID\tMassFlow_Export\n")
        f.write(
            f"MTD\ttitle\tMassFlow Annotation Results - {datetime.now().isoformat()}\n"
        )
        f.write("MTD\tdescription\tAutomated spectral annotation via MassFlow\n")

        # 2. Small Molecule Header (SMH)
        cols = df.columns
        header = "SMH\t" + "\t".join(cols) + "\n"
        f.write(header)

        # 3. Small Molecule List (SML)
        for row in df.iter_rows():
            row_vals = [str(v) if v is not None else "" for v in row]
            f.write("SML\t" + "\t".join(row_vals) + "\n")

    logger.info(f"mzTab-M results saved to {output_path}")
