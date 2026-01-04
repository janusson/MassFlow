#!/usr/bin/python
# -*- coding: utf-8 -*-
#   Python 3.9
"""
⌬ Description:
Tandem Mass Spectrometry data analysis pipeline.
"""
from unicodedata import name
from log import slab_logging
import matchms

__all__ = ['load_settings', 'list_msp_libraries', 'list_mgf_libraries', 'list_available_libraries', 'fetch_mgflib_spectrum', 'save_spectra_to_mgf', 'get_spectrum_inchi']

# * Settings
def load_settings():
    """
    IO settings. Adjust to local filepaths.

    Returns:
        paths: Database path, library path.
    """
    from os import path as path

    libraries_path = path.abspath(r"./database/msp databases") # SpectralMetricMS library directory
    reference_msp_lib = path.join(libraries_path, "eric_lib.msp") # reference debugging library
    return libraries_path, reference_msp_lib

load_settings() # Load 

###############################################################################

# ? 1. Import functions for MSMS information

# * Get library information
def list_msp_libraries(directory) -> list:
    """
    Import MS libraries from .msp file(s).
    Returns the library directory, a list of .msp libraries.
    """
    import glob
    import os

    msp_libraries_list = glob.glob(
        os.path.join(directory, "*.msp"), recursive=True
    )  # list of .msp (NIST text-type libraries) files
    print(f"{len(msp_libraries_list)} MSP libraries found.")
    return msp_libraries_list


def list_mgf_libraries(directory) -> list:
    """
    Import MS libraries from .mgf file(s) and .msp file(s).
    Returns the library directory, a list of .mgf libraries.
    """
    import glob
    from os import path as path

    mgf_libraries_list = glob.glob(
        path.join(directory, "*.mgf"), recursive=True
    )  # list of .mgf (GNPS style libraries) files
    print(f"{len(mgf_libraries_list)} MGF libraries found.")
    return mgf_libraries_list


# * List available libraries (MGF and MSP)
def list_available_libraries(mgf_libraries_list, msp_libraries_list):
    """
    #? Debug function to list names of available libraries.
    Prints available libraries in libraries_path by MGF and MSP format:
    library_name: library_filepath
    """
    from os import path as path
    import logging

    print("Available MGF libraries:")
    print("------------------------")
    mgf_library_names = [i.split("\\")[-1] for i in mgf_libraries_list]
    for library in mgf_library_names:
        print(path.basename(library))
    print("------------------------\n")

    print("Available MSP libraries:")
    print("------------------------")
    msp_library_names = [i.split("\\")[-1] for i in msp_libraries_list]
    for library in msp_library_names:
        print(path.basename(library))
    print("------------------------\n")

    return print("MGF/MSP Libraries listed.")


# * Load spectrum metadata from .mgf library file
def fetch_mgflib_spectrum(library_filepath, spectrum_number):
    """
    Load MS spectrum peak and meta data from a library file (given spectrum number and
    library filepath).

    Args:
        library_filepath (pathlike, optional): match_ms spectrum object. Defaults to DEFAULT_LIB.
        spectrum_number (int, optional): match_ms spectrum object. Defaults to 0.

    Returns:
        DataFrame, metadata: Mass spectrum dataframe ['m/z', 'Abundance (%)']
        and metadata.
    """
    import pandas as pd
    from matchms.importing import load_from_mgf

    # Load MS .mgf library file with matchms:
    spectra_list = list(load_from_mgf(library_filepath))

    # Get spectrum information, normalize spectrum, return as DataFrame
    spectrum_peaks = spectra_list[spectrum_number].peaks.mz
    spectrum_counts = spectra_list[spectrum_number].peaks.intensities
    normalized_counts = spectrum_counts / max(spectrum_counts)
    percent_abundance = normalized_counts * 100
    spectrum_dict = dict(zip(spectrum_peaks, percent_abundance))
    spectrum_xy_data = pd.DataFrame.from_dict(
        spectrum_dict, orient="index", columns=["Abundance (%)"]
    )
    spectrum_xy_data.reset_index(inplace=True)
    spectrum_xy_data.rename(columns={"index": "m/z"}, inplace=True)
    spectrum_xy_data.sort_values(by="m/z", inplace=True)

    spectrum_metadata = spectra_list[spectrum_number].metadata  # get spectrum metadata
    spectrum_chemical = spectra_list[spectrum_number].metadata[
        "name"
    ]  # get spectrum metadata

    return spectrum_xy_data, spectrum_metadata, spectrum_chemical


###############################################################################
# ? 2. Library data processing pipeline:

# * Processing filter functions for cleaning library spectra metadata and peaks.
"""
From matchms documentation:
In general, `matchms.filtering` contains filters that either will
- harmonize, clean, extend, and/or check the spectrum metadata
- process or filter spectrum peaks (e.g. remove low intensity peaks, or normalize peak intensities)
"""

# * FILTER: Fix spectra metadata with spectrum filters
def metadata_processing(spectrum):
    """
    Repairs spectrum metadata and return as new spectrum.

    Args:
        spectrum (spectrum object): match_ms spectrum object

    Returns:
        spectrum object: filtered match_ms spectrum object
    """
    from matchms.filtering import default_filters
    from matchms.filtering import repair_inchi_inchikey_smiles
    from matchms.filtering import derive_inchikey_from_inchi
    from matchms.filtering import derive_smiles_from_inchi
    from matchms.filtering import derive_inchi_from_smiles
    from matchms.filtering import harmonize_undefined_inchi
    from matchms.filtering import harmonize_undefined_inchikey
    from matchms.filtering import harmonize_undefined_smiles
    from matchms.filtering import clean_compound_name
    from matchms.filtering import derive_adduct_from_name
    from matchms.filtering import derive_formula_from_name
    from matchms.filtering import derive_ionmode
    from matchms.filtering import make_charge_int
    from matchms.filtering import set_ionmode_na_when_missing

    # from matchms.filtering import add_compound_name # included in default_filters
    # from matchms.filtering import add_precursor_mz # included in default_filters

    spectrum = default_filters(spectrum)  # apply default filters
    spectrum = repair_inchi_inchikey_smiles(spectrum)
    spectrum = derive_inchi_from_smiles(spectrum)
    spectrum = derive_smiles_from_inchi(spectrum)
    spectrum = derive_inchikey_from_inchi(spectrum)
    spectrum = derive_adduct_from_name(spectrum)
    spectrum = derive_formula_from_name(spectrum)

    spectrum = harmonize_undefined_smiles(spectrum)
    spectrum = harmonize_undefined_inchi(spectrum)
    spectrum = harmonize_undefined_inchikey(spectrum)

    spectrum = clean_compound_name(spectrum)
    spectrum = derive_ionmode(spectrum)
    spectrum = make_charge_int(spectrum)
    spectrum = set_ionmode_na_when_missing(spectrum)

    return spectrum


# * FILTER: Process mass spectrum peaks
def peak_processing(spectrum):
    from matchms.filtering import default_filters
    from matchms.filtering import normalize_intensities
    from matchms.filtering import select_by_intensity
    from matchms.filtering import select_by_relative_intensity
    from matchms.filtering import select_by_mz

    spectrum = default_filters(spectrum)
    spectrum = select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.08)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum


# * PIPELINE: Clean up spectra metadata and peaks for MGF library
def clean_mgf_library(mgf_path):
    """
    Main data processing pipeline. Clean up spectra metadata and peaks for an MGF library.

    Args:
        mgf_path (path): Path to mgf library

    Returns:
        list: List of cleaned and processed spectra. #? or generator?
    """
    from matchms.importing import load_from_mgf

    print(f"Cleaning {mgf_path} library spectra...")
    library_list = list(load_from_mgf(mgf_path))
    # First process metadata, then clean MSMS spectrum peaks
    processed_spectra = [metadata_processing(i) for i in library_list]
    processed_spectra = [peak_processing(s) for s in library_list]
    # # TODO Can this operation be vectorised?
    # library_array = np.array(
    #     load_from_mgf(mgf_path)).tolist()
    return processed_spectra


def clean_msp_library(msp_path):
    """
    Cleans an MSP library given it's path using main data processing pipeline.

    Args:
        msp_path (path): Path to MSP library

    Returns:
        list: List of cleaned and processed spectra. #? generator?
    """
    from matchms.importing import load_from_msp

    print(f"Cleaning {msp_path} library spectra...")
    # library_array = np.array(
    #     load_from_mgf(mgf_path)).tolist()  # TODO Can this operation be vectorised?
    library_list = list(load_from_msp(msp_path))
    processed_spectra = [metadata_processing(i) for i in library_list]
    processed_spectra = [peak_processing(s) for s in library_list]
    return processed_spectra


###############################################################################
# ? 3. Spectrum export functions and properties:

# * Exporting functions for spectrum data.


def save_spectra_to_mgf(spectra_list, export_filepath, export_name):
    """
    Save spectra to MGF library file.
    """
    from os import path
    from matchms.exporting import save_as_mgf

    export_mgf_path = path.join(export_filepath, export_name + ".mgf")
    for spectrum in spectra_list:
        save_as_mgf(spectra_list, export_mgf_path)
    return print(f"{len(spectra_list)} spectra saved to MGF: {export_mgf_path}")


def save_spectra_to_msp(spectra_list, export_filepath, export_name):
    """
    Save spectra to MSP library file.
    """
    from os import path
    from matchms.exporting import save_as_msp

    export_msp_path = path.join(export_filepath, export_name + ".msp")
    for spectrum in spectra_list:
        save_as_msp(spectra_list, export_msp_path)
    return print(f"{len(spectra_list)} spectra saved to MSP: {export_msp_path}")


def save_spectra_to_json(spectra_list, export_filepath, export_name):
    """
    Save spectra to JSON library file.
    """
    from os import path
    from matchms.exporting import save_as_json

    export_json_path = path.join(export_filepath, export_name + ".json")
    for spectrum in spectra_list:
        save_as_json(spectra_list, export_filepath)
    return print(f"{len(spectra_list)} spectra saved to JSON: {export_json_path}")


def save_spectra_to_pickle(spectra_list, export_filepath, export_name):
    """
    For exporting many thousands of spectra, save spectra to pickle file for pythonic use.
    """
    import pickle
    from os import path as path

    file_export_pickle = path.join(export_filepath, export_name + ".pickle")
    pickle.dump(spectra_list, open(file_export_pickle, "wb"))
    return print(f"{len(spectra_list)} spectra saved to pickle: {file_export_pickle}")


# * Get spectra INCHI key from MGF library
def get_spectrum_inchi(spectrum):
    """
    Get inchi from spectrum metadata.
    """
    return spectrum.get("inchikey")

###############################################################################

# ? 4. Compute spectra similarities : Cosine score

# * Calculate cosine similarity scores for all given spectra against target library spectra.


def calculate_cosscores(reference_spectra_list, query_spectra_list, tolerance=0.005):
    """
    Calculate cosine similarity scores for all query spectra against target library spectra.
    Returns a scores object with cosine match scores and number of peaks matched.
    Tolerance (float)
    Args:
        reference_spectra_list (matchms_spectra): Reference spectra list.
        query_spectra_list (spectra): Spectra to be compared against reference spectra.
        tolerance (float, optional): Cosine metch algorith tolerance. Defaults to 0.005.

    Returns:
        scores: matchms_scores object representing cosine similarity scores.
    """

    from matchms import calculate_scores
    from matchms.similarity import CosineGreedy

    # Check is ref/query spectra are symmetric to speed up import with is_symmetric = True
    asymmetric_test = (
        True if len(reference_spectra_list) == len(query_spectra_list) else False
    )
    similarity_measure = CosineGreedy(tolerance)  # Set similarity measure
    cosine_scores = calculate_scores(
        reference_spectra_list,
        query_spectra_list,
        similarity_measure,
        is_symmetric=asymmetric_test,
    )
    return cosine_scores


# * Print top ten sorted best matches by Cosine similarity and min. number of matching peaks


def top10_cosine_matches(reference_library, query_spectra, spectrum_number=1):
    """
    Print top ten matching peaks between query spectra and reference library spectra.

    Matches are sorted by Cosine similarity.
    """
    # select a spectrum number from the query list and find best 10 matches

    # make a scores object with spectra and similarity scores
    scores = calculate_cosscores(reference_library, query_spectra, tolerance=0.005)
    for spectrum_number in scores:
        best_matches = scores.scores_by_query(
            query_spectra[spectrum_number], sort=True
        )[
            :10
        ]  # List only top 10 matches
        print("Top ten best matches (Cosine score, # of matching peaks):")
        print([x[1] for x in best_matches])
    # Get SMILES for top ten candidate matches
    top10_smiles = [
        x[0].get("smiles") for x in best_matches
    ]  # error with matchms scores object
    print(f"Top ten best matches (SMILES): {top10_smiles}")
    return None  # TODO: return top10_smiles


# Show best matches with minimum number of matching peaks
def threshold_matches(reference_library, check_spectra, min_match, tolerance=0.005):
    """
    Get matches with a minimum number of matching peaks.

    Returns matches as matchms spectra and the smiles of the matched spectra.
    Default greedy cosine similarity matching algorithm tolerance is 0.005.
    Returns:
        spectrum, list: matchms spectra and the smiles of the matched spectra.

    Usage:
    # threshold_matches(reference_library, check_spectra, 6) # match a reference library with 6 or more matching peaks to check_spectra
    """
    from matchms import calculate_scores
    from matchms.similarity import CosineGreedy

    similarity_measure = CosineGreedy(tolerance)  # Set similarity measure
    scores = calculate_scores(
        reference_library, check_spectra, similarity_measure, is_symmetric=False
    )

    sorted_matches = scores.scores_by_query(
        reference_library[5], sort=True
    )  # for match 5
    matches_over_limit = [x for x in sorted_matches if x[1]["matches"] >= min_match][
        :10
    ]
    # [x[1] for x in matches_over_limit] #? list of matching spectra?
    matches_over_limit_smiles = [x[0].get("smiles") for x in matches_over_limit]

    return matches_over_limit, matches_over_limit_smiles


###############################################################################

# TODO, finish spec2vec and modified cosine score reports
# ? 4. Compute spectra similarities: Modified Cosine score
def modified_cosine_scores(reference_library, check_spectra):
    """
    Computes modified cosine scores for all spectra in check_spectra against reference library spectra.

    Args:
        reference_library (match_ms_spectrum):
        check_spectra (match_ms_spectrum):

    Returns:
        Scores: mstchms_scores object with modified cosine scores between reference and check spectra.
    """
    from matchms import calculate_scores
    from matchms.similarity import ModifiedCosine

    # note, made edits to 'C:\Users\ericj\anaconda3\lib\site-packages\matchms\similarity\ModifiedCosine.py', line ~98
    # issue resolved with pull request:

    similarity_measure = ModifiedCosine(tolerance=0.005)
    scores = calculate_scores(
        reference_library, check_spectra, similarity_measure, is_symmetric=False
    )
    return scores


def main():
    # load_settings()
    print("Started SLAB MSMS Processing. \n\n\n ")

if __name__ == "__main__":
    main()