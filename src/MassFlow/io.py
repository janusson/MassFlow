"""
I/O functions for MassFlow: unified spectral loading and result export.
"""

from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, Iterator

from matchms import Spectrum
from matchms.exporting import save_as_json, save_as_mgf, save_as_msp
from matchms.importing import load_from_mgf, load_from_msp, load_from_mzml

logger = logging.getLogger(__name__)


def load_spectra(file_path: Path, file_format: str) -> Iterator[Spectrum]:
    """
    Unified loader for spectral data files.

    Args:
        file_path: Path to the spectral data file.
        file_format: String indicating the format ('mgf', 'msp', 'mzml').

    Returns:
        Iterator of matchms Spectrum objects.

    Raises:
        ValueError: If the format is not supported.
    """
    fmt = file_format.lower().strip(".")
    path_str = str(file_path)

    if fmt == "mgf":
        return load_from_mgf(path_str)
    elif fmt == "msp":
        return load_from_msp(path_str)
    elif fmt == "mzml":
        return load_from_mzml(path_str)
    else:
        raise ValueError(
            f"Unsupported file format: {fmt}. Supported formats: 'mgf', 'msp', 'mzml'."
        )


def save_match_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """
    Save similarity search results to a CSV file.

    Args:
        results: List of result dictionaries.
        output_path: Full path to the output CSV file.
    """
    if not results:
        logger.warning("No results to save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use keys from the first dictionary as headers
    fieldnames = list(results[0].keys())

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results successfully saved to: {output_path}")
    except IOError as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise


def save_spectra_to_mgf(spectra: Iterable[Spectrum], export_path: Path) -> None:
    """
    Save spectra to MGF format.

    Args:
        spectra: Iterable of Spectrum objects.
        export_path: Full path to the output .mgf file.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    save_as_mgf(spectra, str(export_path))
    logger.info(f"Spectra saved to MGF: {export_path}")


def save_spectra_to_msp(spectra: Iterable[Spectrum], export_path: Path) -> None:
    """
    Save spectra to MSP format.

    Args:
        spectra: Iterable of Spectrum objects.
        export_path: Full path to the output .msp file.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    save_as_msp(spectra, str(export_path))
    logger.info(f"Spectra saved to MSP: {export_path}")


def save_spectra_to_json(spectra: Iterable[Spectrum], export_path: Path) -> None:
    """
    Save spectra to JSON format.

    Args:
        spectra: Iterable of Spectrum objects.
        export_path: Full path to the output .json file.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    save_as_json(spectra, str(export_path))
    logger.info(f"Spectra saved to JSON: {export_path}")


def save_spectra_to_pickle(spectra: Iterable[Spectrum], export_path: Path) -> None:
    """
    Save spectra to pickle format.

    Args:
        spectra: Iterable of Spectrum objects.
        export_path: Full path to the output .pickle file.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Pickle requires the full list to be materialized
    spectra_list = list(spectra)

    with open(export_path, "wb") as f:
        pickle.dump(spectra_list, f)
    logger.info(f"{len(spectra_list)} spectra saved to pickle: {export_path}")


def list_files_by_extension(directory: Path, extension: str) -> list[Path]:
    """
    Utility to list files in a directory with a specific extension.

    Args:
        directory: Directory to search.
        extension: Extension to look for (e.g., 'mgf').

    Returns:
        List of Path objects.
    """
    if not directory.is_dir():
        logger.warning(f"Directory not found: {directory}")
        return []

    ext = extension.lstrip(".")
    pattern = f"*.{ext}"
    files = list(directory.glob(pattern))
    logger.info(f"Found {len(files)} .{ext} files in {directory}")
    return files
