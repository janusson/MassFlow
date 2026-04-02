"""
I/O helpers for MassFlow spectral data.

This module is the file-system boundary for MassFlow. It loads spectra from
open interchange formats such as mzML, mzXML, MGF, and MSP, as well as
MassFlow-native serialized stores such as SQLite databases and pickle files. It
also exports search results and spectra to flat-file formats for downstream use.

The loader is intentionally narrow in scope: it does not auto-convert
proprietary vendor files and it does not sanitize metadata during import.
Vendor raw formats are rejected with an actionable error, and metadata
harmonization is deferred to :mod:`MassFlow.processing` after import.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pandas as pd
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


def load_spectra(
    file_path: Path, file_format: Optional[str] = None
) -> Iterator[Spectrum]:
    """
    Load spectra from an open spectral file or a MassFlow-native store.

    The loader infers the format from ``file_path`` unless ``file_format`` is
    provided explicitly. Supported formats are mzML, mzXML, MGF, MSP, SQLite
    (``db``/``sqlite``), and pickle. Imported spectra are yielded exactly as the
    backend loader returns them; MassFlow's metadata cleaning and peak filtering
    happen later in :func:`MassFlow.processing.process_spectra`.

    Parameters
    ----------
    file_path : Path
        Path to the input file or directory. Bruker ``.d`` directories are
        treated as vendor raw inputs and rejected unless pre-converted.
    file_format : str, optional
        Explicit format override. Accepted values are ``mzml``, ``mzxml``,
        ``mgf``, ``msp``, ``db``, ``sqlite``, and ``pickle``. If omitted, the
        format is inferred from the file extension.

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
    ext = path.suffix.lower()

    # Step 1: Check for unsupported vendor formats
    if ext in PROPRIETARY_FORMATS or (path.is_dir() and ext == ".d"):
        raise UnsupportedVendorFormatError(
            "MassFlow requires open data formats. Please convert vendor files to .mzML or .mgf using ProteoWizard or MS-DIAL prior to pipeline ingestion."
        )

    # Step 2: Determine loading function
    fmt = (file_format or ext.lstrip(".")).lower()

    # Disable matchms internal harmonization to allow MassFlow's processing module to handle it
    args = {"metadata_harmonization": False}

    if fmt == "mzml":
        loader = load_from_mzml
    elif fmt == "msp":
        loader = load_from_msp
    elif fmt == "mgf":
        loader = load_from_mgf
    elif fmt == "mzxml":
        loader = load_from_mzxml
    elif fmt in ["db", "sqlite"]:
        from MassFlow.database import SpectralDatabase

        db = SpectralDatabase(path)
        yield from db.get_spectra()
        return
    elif fmt == "pickle":
        with open(path, "rb") as f:
            yield from iter(pickle.load(f))
        return
    else:
        raise ValueError(f"Format '{fmt}' is not supported by MassFlow.")

    # Step 3: Yield spectra using the selected loader
    yield from loader(str(path), **args)


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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if query_spectra is not None:
        # Build base dataframe from all queries
        base_rows = []
        for q in query_spectra:
            q_id = str(q.get("id"))
            q_mz = q.get("precursor_mz")
            q_rt = q.get("retention_time")
            base_rows.append(
                {
                    "query_id": q_id,
                    "query_precursor_mz": float(q_mz) if q_mz is not None else None,
                    "query_retention_time": float(q_rt) if q_rt is not None else None,
                }
            )
        base_df = pd.DataFrame(base_rows)

        if results:
            results_df = pd.DataFrame(results)
            # Left join results onto the base query dataframe
            df = pd.merge(
                base_df,
                results_df,
                on="query_id",
                how="left",
                suffixes=("", "_matched"),
            )
            # Drop duplicated columns if any exist from the merge
            if "query_precursor_mz_matched" in df.columns:
                df = df.drop(columns=["query_precursor_mz_matched"])
        else:
            df = base_df

        # Add Annotation_Status
        if "score" in df.columns:
            df["Annotation_Status"] = df["score"].apply(
                lambda x: (
                    "Unknown" if pd.isna(x) else ("Matched" if x >= 0.9 else "Putative")
                )
            )
        else:
            df["Annotation_Status"] = "Unknown"
    else:
        if not results:
            logger.warning("No results to save and no query_spectra provided.")
            return
        df = pd.DataFrame(results)
        if "score" in df.columns:
            df["Annotation_Status"] = df["score"].apply(
                lambda x: (
                    "Unknown" if pd.isna(x) else ("Matched" if x >= 0.9 else "Putative")
                )
            )
        else:
            df["Annotation_Status"] = "Unknown"

    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


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


def save_spectra_to_pickle(spectra: Iterable[Spectrum], export_path: Path) -> None:
    """
    Serialize spectra to a pickle file.

    Parameters
    ----------
    spectra : Iterable[matchms.Spectrum]
        Spectra to serialize. The iterable is materialized into memory before
        serialization.
    export_path : Path
        The file path for the resulting pickle file.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the output directory cannot be created or the pickle file cannot be
        written.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "wb") as f:
        pickle.dump(list(spectra), f)
