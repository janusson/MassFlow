"""
Input/Output (I/O) operations for MassFlow.

This module handles the loading and saving of spectral data, including automated
conversion of proprietary formats (e.g., .raw, .d) to mzML using ProteoWizard's
msconvert. It implements robust metadata sanitization to ensure compatibility
with downstream processing tools like matchms, preventing crashes due to
malformed or non-standard metadata fields.
"""

from __future__ import annotations

import logging
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pandas as pd
from matchms import Spectrum
from matchms.filtering import default_filters
from matchms.importing import (
    load_from_mgf,
    load_from_msp,
    load_from_mzml,
    load_from_mzxml,
)

logger = logging.getLogger(__name__)

PROPRIETARY_FORMATS = {".raw", ".d", ".wiff", ".lcd", ".t2d"}


def _run_msconvert(input_path: Path) -> Path:
    """
    Attempt to convert a proprietary mass spectrometry file to mzML format.

    This function invokes ProteoWizard's ``msconvert`` utility as a subprocess to
    perform the conversion. The converted file is placed in a ``converted`` subdirectory
    relative to the input file's location.

    Parameters
    ----------
    input_path : Path
        The file path of the proprietary raw data file (e.g., .raw, .d).

    Returns
    -------
    Path
        The file path to the resulting .mzML file.

    Raises
    ------
    RuntimeError
        If ``msconvert`` is not found in the system PATH or if the conversion process fails.
    FileNotFoundError
        If the expected output file is not found after the process completes.
    """
    if shutil.which("msconvert") is None:
        raise RuntimeError(
            f"Detected proprietary format {input_path.suffix}, but 'msconvert' was not found in your PATH. "
            "Please install ProteoWizard (https://proteowizard.sourceforge.io/) to enable auto-conversion."
        )

    output_dir = input_path.parent / "converted"
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Auto-converting {input_path.name} to mzML...")

    try:
        # Run msconvert: --mzML flag ensures standard output
        subprocess.run(
            ["msconvert", str(input_path), "--mzML", "-o", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        # msconvert replaces extension with .mzML
        expected_output = output_dir / (input_path.stem + ".mzML")
        if expected_output.exists():
            return expected_output
        raise FileNotFoundError("msconvert finished but output file was not found.")
    except subprocess.CalledProcessError as e:
        logger.error(f"msconvert failed: {e.stderr}")
        raise RuntimeError(
            f"Failed to convert {input_path.name}. Ensure the file is not corrupted."
        )


def _sanitize_metadata(spectrum: Spectrum) -> Optional[Spectrum]:
    """
    Clean and repair critical metadata fields in a spectrum.

    This internal utility aggressively filters specific metadata keys (like
    retention time and CCS) that often contain garbage strings in public datasets
    (e.g., "CCS:", "N/A"), which can cause downstream crashes in ``matchms``.
    It ensures these fields are strictly numeric or set to None.

    Parameters
    ----------
    spectrum : matchms.Spectrum
        The input spectrum object to sanitize.

    Returns
    -------
    matchms.Spectrum or None
        The sanitized spectrum object. Returns None if the input is None.
    """
    if spectrum is None:
        return None

    # Fields that MUST be numeric
    numeric_keys = [
        "retention_time",
        "retentiontime",
        "RETENTIONTIME",
        "ccs",
        "CCS",
        "precursor_mz",
    ]

    for key in numeric_keys:
        val = spectrum.get(key)
        if val is not None:
            # If value is string and contains garbage (like "CCS:"), nullify it
            if isinstance(val, str):
                v_str = val.strip().upper()
                if not v_str or any(x in v_str for x in ["CCS", "N/A", "NONE", "NAN"]):
                    spectrum.set(key, None)
                    continue
            # Ensure it can actually be a float
            try:
                spectrum.set(key, float(val))
            except (ValueError, TypeError):
                spectrum.set(key, None)

    return spectrum


def _apply_filters(spectrum: Spectrum) -> Optional[Spectrum]:
    """
    Apply sanitization and default matchms filters to a spectrum.

    Parameters
    ----------
    spectrum : matchms.Spectrum
        The raw spectrum object.

    Returns
    -------
    matchms.Spectrum or None
        The processed spectrum, or None if it fails sanitization or filtering.
    """
    spec = _sanitize_metadata(spectrum)
    if spec:
        return default_filters(spec)
    return None


def load_spectra(
    file_path: Path, file_format: Optional[str] = None
) -> Iterator[Spectrum]:
    """
    Load mass spectra from a file, handling multiple formats and auto-conversion.

    This comprehensive loader identifies the file format from the extension or
    provided argument. It supports standard formats (mzML, mzXML, MGF, MSP) and
    proprietary formats (via auto-conversion using ``msconvert``). It also supports
    loading from a local SQLite database or a pickle file. Loaded spectra undergo
    immediate metadata sanitization to ensure data integrity.

    Parameters
    ----------
    file_path : Path
        The path to the input file or directory (for .d folders).
    file_format : str, optional
        Explicitly specify the file format (e.g., 'mzml', 'mgf'). If None,
        it is inferred from the file extension.

    Yields
    ------
    matchms.Spectrum
        Yields sanitized and basic-filtered spectrum objects one by one.

    Raises
    ------
    ValueError
        If the file format is unsupported.
    RuntimeError
        If conversion of a proprietary format fails.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Step 1: Auto-conversion
    if ext in PROPRIETARY_FORMATS or (path.is_dir() and ext == ".d"):
        path = _run_msconvert(path)
        ext = ".mzml"

    # Step 2: Determine loading function
    fmt = (file_format or ext.lstrip(".")).lower()

    # Disable internal harmonization to avoid early crashes on dirty strings
    args = {"metadata_harmonization": False}

    if fmt == "mzml":
        gen = load_from_mzml(str(path), **args)
    elif fmt == "msp":
        gen = load_from_msp(str(path), **args)
    elif fmt == "mgf":
        gen = load_from_mgf(str(path), **args)
    elif fmt == "mzxml":
        gen = load_from_mzxml(str(path), **args)
    elif fmt in ["db", "sqlite"]:
        from MassFlow.database import SpectralDatabase

        db = SpectralDatabase(path)
        gen = db.get_spectra()
    elif fmt == "pickle":
        with open(path, "rb") as f:
            gen = iter(pickle.load(f))
    else:
        raise ValueError(f"Format '{fmt}' is not supported by MassFlow.")

    # Step 3: Yield sanitized spectra
    for spectrum in gen:
        processed = _apply_filters(spectrum)
        if processed:
            yield processed


def save_match_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """
    Save annotation matching results to a CSV file.

    Parameters
    ----------
    results : list of dict
        A list of dictionaries, where each dictionary represents a row of matching results.
    output_path : Path
        The destination file path for the CSV output. Parent directories will be
        created if they do not exist.

    Returns
    -------
    None
    """
    if not results:
        logger.warning("No results to save.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


def save_spectra_to_msp(spectra: Iterable[Spectrum], export_path: Path) -> None:
    """
    Export a collection of spectra to an MSP file.

    Parameters
    ----------
    spectra : Iterable[matchms.Spectrum]
        The spectra to export.
    export_path : Path
        The file path for the resulting MSP file.

    Returns
    -------
    None
    """
    from matchms.exporting import save_as_msp

    export_path.parent.mkdir(parents=True, exist_ok=True)
    save_as_msp(list(spectra), str(export_path))


def save_spectra_to_pickle(spectra: Iterable[Spectrum], export_path: Path) -> None:
    """
    Serialize a collection of spectra to a pickle file.

    Parameters
    ----------
    spectra : Iterable[matchms.Spectrum]
        The spectra to serialize.
    export_path : Path
        The file path for the resulting pickle file.

    Returns
    -------
    None
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "wb") as f:
        pickle.dump(list(spectra), f)
