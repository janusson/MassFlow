"""
TDD fixtures for v0.2 features: Retention Time (RT) filtering and Mass Error (ppm).

These tests define the expected behaviour of two upcoming engine capabilities
before the production code exists. They are designed to fail (Red phase) until
``MassFlow.similarity.calculate_mass_error_ppm`` and the ``rt_tolerance``
parameter on ``SimilarityConfig`` are implemented.
"""

from __future__ import annotations

import math

import numpy as np
from matchms import Spectrum

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import SimilarityEngine, calculate_mass_error_ppm

# ---------------------------------------------------------------------------
# Mass Error (ppm) calculation
# ---------------------------------------------------------------------------


def test_mass_error_ppm_calculation() -> None:
    """
    Verify that the mass error in ppm is calculated exactly.

    Mass error (ppm) = |query_mz - ref_mz| / ref_mz * 1e6

    Given a query precursor of 400.0020 Da and a reference precursor of
    400.0000 Da, the expected mass error is exactly 5.0 ppm.
    """
    query_precursor_mz = 400.0020
    reference_precursor_mz = 400.0000

    result = calculate_mass_error_ppm(query_precursor_mz, reference_precursor_mz)

    assert math.isclose(result, 5.0, rel_tol=1e-9), (
        f"Expected mass error 5.0 ppm, got {result} "
        f"(query={query_precursor_mz}, ref={reference_precursor_mz})"
    )


# ---------------------------------------------------------------------------
# Retention Time filtering
# ---------------------------------------------------------------------------


def test_retention_time_filtering() -> None:
    """Reject a synthetic pair assigned times 36 seconds apart at 0.2 minutes.

    This tests unit-aware filtering, not experimental separation of isomers.
    """
    # --- Build spectra for two structural isomers -------------------------
    # Identical peaks and precursor for both isomers
    mz = np.array([86.09, 132.10], dtype="float")
    intensities = np.array([100.0, 200.0], dtype="float")
    precursor_mz = 132.10

    # Reference: Leucine (elutes earlier)
    leucine_ref = Spectrum(
        mz=mz.copy(),
        intensities=intensities.copy(),
        metadata={
            "id": "leucine_ref",
            "compound_name": "Leucine",
            "precursor_mz": precursor_mz,
            "retention_time": 150.0,
        },
    )

    # Query: Isoleucine (elutes later)
    isoleucine_query = Spectrum(
        mz=mz.copy(),
        intensities=intensities.copy(),
        metadata={
            "id": "isoleucine_query",
            "compound_name": "Isoleucine",
            "precursor_mz": precursor_mz,
            "retention_time": 186.0,
        },
    )

    # --- Configure engine with a narrow RT tolerance ----------------------
    config = SimilarityConfig(
        algorithm="cosine",
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        rt_tolerance=0.2,
        rt_tolerance_unit="minutes",
    )
    engine = SimilarityEngine(config)

    # --- Execute search ---------------------------------------------------
    results = engine.search(
        query_spectra=[isoleucine_query],
        reference_spectra=[leucine_ref],
    )

    # Exclude decoy hits from the assertion
    target_results = [r for r in results if not r.get("is_decoy")]

    assert len(target_results) == 0, (
        f"RT filtering failed: isomers separated by 0.6 min should be "
        f"rejected at tolerance 0.2 min, but {len(target_results)} "
        f"match(es) were returned."
    )
