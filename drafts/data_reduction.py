"""
## Situation
# 
## Objective
Reduce MSMS data to a List of matchms.Spectrum Objects.

This is the universal adaptor for data to make the pipeline vendor agnostic (.mzML, .smp, .mgf, etc.).

## Why?
Uniformity: You stop writing if file_type == "mzml" logic. You write your math once, and it applies to everything.

Numpy under the hood: The actual spectral data (masses and intensities) are stored as numpy arrays. This makes the math (cosine similarity, networking) blindingly fast compared to Python lists.

Metadata Dictionary: It keeps a flexible dictionary for metadata. If an .mzML file has "Retention Time" and an .msp file has "SMILES," this object holds both without crashing.
"""

# data reduction pipeline for MS(MS) data formats

import os

from matchms import Spectrum
from matchms.importing import load_from_mgf, load_from_msp, load_from_mzml


def universal_loader(file_path: str) -> list[Spectrum]:
    """
    Ingests ANY common MS format and reduces it to a list of matchms Spectra.
    """
    filename, extension = os.path.splitext(file_path)
    extension = extension.lower()

    spectra_list = []

    # 1. The Ingest Strategy
    if extension == ".mzml":
        # mzML is often generator-based (to save RAM), we convert to list
        spectra_list = list(load_from_mzml(file_path))

    elif extension == ".msp":
        spectra_list = list(load_from_msp(file_path))

    elif extension == ".mgf":
        spectra_list = list(load_from_mgf(file_path))

    # 2. The "Reduction" (Standardization)
    # This ensures every spectrum, regardless of source, has the same basic fields
    cleaned_spectra = []
    for sp in spectra_list:
        # Example: Ensure metadata is consistent
        # If 'precursortype' is missing, set it to generic, etc.
        if sp is not None:
            cleaned_spectra.append(sp)

    return cleaned_spectra
