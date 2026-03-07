"""
Integration test suite for MassFlow SimilarityEngine, focusing on ModifiedCosine.
"""

from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import io
from MassFlow.config import SimilarityConfig
from MassFlow.similarity import SimilarityEngine


@pytest.fixture(scope="module")
def cocaine_spectrum() -> Spectrum:
    """
    Ingests the example library and isolates the Cocaine spectrum.

    Returns
    -------
    matchms.Spectrum
        The unshifted Cocaine reference standard spectrum.
    """
    library_path = Path("data/reference/example_library.msp")
    if not library_path.exists():
        pytest.skip(f"Library file not found at {library_path}")

    # Load spectra without harmonization to avoid dropping custom fields during test
    spectra = list(io.load_spectra(library_path, "msp"))

    # Isolate Cocaine
    for spec in spectra:
        name = spec.get("compound_name") or spec.get("name") or ""
        if "cocaine" in name.lower():
            if spec.get("precursor_mz") is None:
                pytest.fail("Found Cocaine spectrum, but missing precursor_mz.")
            return spec

    pytest.fail("Cocaine spectrum not found in the example library.")


@pytest.fixture
def methylene_homolog(cocaine_spectrum: Spectrum) -> Spectrum:
    """
    Creates a mass-shifted test fixture (+14.0156 Da) simulating a methylene homolog.

    Parameters
    ----------
    cocaine_spectrum : matchms.Spectrum
        The original Cocaine reference spectrum.

    Returns
    -------
    matchms.Spectrum
        A synthesized spectrum with all peaks and precursor shifted by +14.0156 Da.
    """
    shift = 14.0156

    orig_mz = cocaine_spectrum.peaks.mz
    orig_ints = cocaine_spectrum.peaks.intensities
    orig_precursor = float(cocaine_spectrum.get("precursor_mz"))

    shifted_mz = orig_mz + shift
    shifted_precursor = orig_precursor + shift

    # Clone metadata and update precursor
    new_metadata = cocaine_spectrum.metadata.copy()
    new_metadata["precursor_mz"] = shifted_precursor
    # Update ID to avoid confusion in results mapping
    new_metadata["id"] = "shifted_cocaine"

    return Spectrum(mz=shifted_mz, intensities=orig_ints, metadata=new_metadata)


def test_modified_cosine_integration(
    cocaine_spectrum: Spectrum, methylene_homolog: Spectrum
) -> None:
    """
    Verify correct neutral loss handling against known reference standard pairs.

    Mathematical Constraint Validation:
    The modified cosine score computes the normalized dot product of matched peak
    intensities, natively incorporating neutral losses.
    |m/z_A - m/z_B - delta_M| <= tolerance

    Since the test fixture shifts both the precursor and the fragment peaks by
    exactly +14.0156 Da, the mass difference (delta_M) matches the peak shift.
    ModifiedCosine must score this analog highly, while CosineGreedy must fail
    to align the shifted fragments and score it poorly.

    Parameters
    ----------
    cocaine_spectrum : matchms.Spectrum
        The reference spectrum.
    methylene_homolog : matchms.Spectrum
        The mass-shifted query spectrum.
    """
    # Configuration for CosineGreedy
    cosine_config = SimilarityConfig(
        algorithm="cosine",
        tolerance=0.1,
        min_score=0.0,  # Zero threshold to capture the failure score
    )
    cosine_engine = SimilarityEngine(cosine_config)

    # Configuration for ModifiedCosine
    mod_cosine_config = SimilarityConfig(
        algorithm="modified_cosine",
        tolerance=0.1,
        min_score=0.0,
    )
    mod_cosine_engine = SimilarityEngine(mod_cosine_config)

    # Run searches
    cosine_results = cosine_engine.search(
        query_spectra=[methylene_homolog], reference_spectra=[cocaine_spectrum]
    )
    mod_cosine_results = mod_cosine_engine.search(
        query_spectra=[methylene_homolog], reference_spectra=[cocaine_spectrum]
    )

    assert len(cosine_results) == 1, "CosineGreedy search returned no result object."
    assert len(mod_cosine_results) == 1, (
        "ModifiedCosine search returned no result object."
    )

    cosine_score = cosine_results[0]["score"]
    mod_cosine_score = mod_cosine_results[0]["score"]

    # Assertions
    # Modified Cosine should be effectively 1.0 because all shifted peaks
    # satisfy the delta_M constraint exactly.
    assert mod_cosine_score > 0.95, (
        f"ModifiedCosine failed to match mass-shifted analog. "
        f"Expected score > 0.95, got {mod_cosine_score}"
    )

    # CosineGreedy should be very low because the peaks are shifted by 14.0156 Da
    # which exceeds the strict 0.1 Da tolerance.
    assert cosine_score < 0.3, (
        f"CosineGreedy incorrectly matched mass-shifted peaks. "
        f"Expected score < 0.3, got {cosine_score}"
    )
