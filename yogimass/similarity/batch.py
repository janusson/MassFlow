"""
Batch processing helpers for similarity workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from matchms import Spectrum
from matchms.importing import load_from_mgf, load_from_mzml

from yogimass.similarity.library import LocalSpectralLibrary
from yogimass.similarity.processing import SpectrumProcessor
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def batch_process_folder(
    input_dir: str | Path,
    processor: SpectrumProcessor,
    *,
    library: LocalSpectralLibrary | None = None,
    recursive: bool = False,
) -> dict[Path, list[Spectrum]]:
    """
    Process every MGF/mzML file in ``input_dir`` and optionally add to ``library``.
    """
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base}")
    pattern = "**/*" if recursive else "*"
    processed: dict[Path, list[Spectrum]] = {}
    for filepath in base.glob(pattern):
        if not filepath.is_file():
            continue
        loader = _loader_for_suffix(filepath.suffix.lower())
        if loader is None:
            continue
        spectra = list(loader(str(filepath)))
        processed_spectra = [processor.process(spectrum) for spectrum in spectra]
        processed[filepath] = processed_spectra
        if library is not None:
            for spectrum in processed_spectra:
                library.add_spectrum(spectrum, overwrite=True)
        logger.info(
            "Processed %d spectra from %s", len(processed_spectra), filepath.name
        )
    return processed


def _loader_for_suffix(suffix: str) -> Callable[[str], Iterable[Spectrum]] | None:
    if suffix == ".mgf":
        return load_from_mgf
    if suffix == ".mzml":
        return load_from_mzml
    return None
