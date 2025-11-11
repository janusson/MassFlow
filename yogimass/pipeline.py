"""
Yogimass main pipeline module.
Contains high-level API for library import, cleaning, export, and scoring.
"""

from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from matchms.importing import load_from_mgf, load_from_msp

from .filters import metadata_processing, peak_processing
from .filters.metadata import ensure_name_metadata
from .io import (
    list_msp_libraries,
    list_mgf_libraries,
    list_available_libraries,
    fetch_mgflib_spectrum,
    save_spectra_to_mgf,
    save_spectra_to_msp,
    save_spectra_to_json,
    save_spectra_to_pickle,
)
from .scoring import (
    calculate_cosscores,
    top10_cosine_matches,
    threshold_matches,
    modified_cosine_scores,
)
from .utils import get_spectrum_inchi
from .utils.logging import get_logger


def load_settings(libraries_path="./database", reference_filename="eric_lib.msp"):
    """
    Return absolute paths for the libraries directory and the reference MSP file.
    """
    libraries_dir = Path(libraries_path).expanduser().resolve()
    reference_path = libraries_dir / reference_filename
    return str(libraries_dir), str(reference_path)


def clean_mgf_library(library_path):
    """
    Load an MGF library and apply Yogimass metadata/peak filters.
    """
    return _clean_library(library_path, load_from_mgf)


def clean_msp_library(library_path):
    """
    Load an MSP library and apply Yogimass metadata/peak filters.
    """
    return _clean_library(library_path, load_from_msp)


def batch_clean_mgf_libraries(
    data_dir,
    output_dir,
    *,
    export_formats=("mgf",),
):
    """
    Clean every MGF file under ``data_dir`` and export cleaned spectra to ``output_dir``.
    """
    mgf_files = list_mgf_libraries(data_dir)
    return _batch_clean_libraries(
        mgf_files,
        clean_mgf_library,
        output_dir,
        export_formats,
    )


def batch_clean_msp_libraries(
    data_dir,
    output_dir,
    *,
    export_formats=("msp",),
):
    """
    Clean every MSP file under ``data_dir`` and export cleaned spectra to ``output_dir``.
    """
    msp_files = list_msp_libraries(data_dir)
    return _batch_clean_libraries(
        msp_files,
        clean_msp_library,
        output_dir,
        export_formats,
    )


def _clean_library(library_path, loader: Callable[[str], Iterable]):
    library_path = Path(library_path)
    if not library_path.is_file():
        raise FileNotFoundError(f"Library does not exist: {library_path}")
    spectra = list(loader(str(library_path)))
    cleaned = []
    for spectrum in spectra:
        spectrum = metadata_processing(spectrum)
        spectrum = peak_processing(spectrum)
        spectrum = ensure_name_metadata(spectrum)
        cleaned.append(spectrum)
    logger.info("Cleaned %d spectra from %s", len(cleaned), library_path.name)
    return cleaned


def _batch_clean_libraries(
    library_paths: Sequence[str],
    cleaner: Callable[[str], List],
    output_dir,
    export_formats,
):
    export_formats = tuple(fmt.lower() for fmt in export_formats)
    if not export_formats:
        raise ValueError("At least one export format must be specified.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files = []
    for library in library_paths:
        cleaned_spectra = cleaner(library)
        export_name = f"{Path(library).stem}_cleaned"
        for fmt in export_formats:
            exporter = _EXPORTERS.get(fmt)
            extension = _EXTENSIONS.get(fmt)
            if exporter is None or extension is None:
                raise ValueError(f"Unsupported export format: {fmt}")
            exporter(cleaned_spectra, str(output_dir), export_name)
            exported_files.append(output_dir / f"{export_name}{extension}")
    if not library_paths:
        logger.warning("No libraries found to clean.")
    return exported_files

logger = get_logger("yogimass.pipeline")
logger.info("Yogimass pipeline loaded.")

__all__ = [
    "load_settings",
    "clean_mgf_library",
    "clean_msp_library",
    "batch_clean_mgf_libraries",
    "batch_clean_msp_libraries",
    "list_msp_libraries",
    "list_mgf_libraries",
    "list_available_libraries",
    "peak_processing",
    "get_spectrum_inchi",
    "calculate_cosscores",
    "top10_cosine_matches",
    "threshold_matches",
    "modified_cosine_scores",
]

_EXPORTERS = {
    "mgf": save_spectra_to_mgf,
    "msp": save_spectra_to_msp,
    "json": save_spectra_to_json,
    "pickle": save_spectra_to_pickle,
}

_EXTENSIONS = {
    "mgf": ".mgf",
    "msp": ".msp",
    "json": ".json",
    "pickle": ".pickle",
}
