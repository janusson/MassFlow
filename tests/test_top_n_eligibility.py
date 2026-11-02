"""Synthetic evidence for limiting eligible classical-search results.

These are constructed caffeine ions, not measured reference spectra. Fragment
formulas specify atom inventories and charge, not a validated fragmentation
pathway. Reapplying the limit before either metadata gate loses valid hits.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pytest
from matchms import Spectrum
from pyteomics import mass

from MassFlow.cheminformatics import ELECTRON_MASS, compute_adduct_offset
from MassFlow.config import SimilarityConfig
from MassFlow.models import MolecularStructure, SpectrumMetadata
from MassFlow.similarity import SimilarityEngine

pytestmark = [pytest.mark.core, pytest.mark.scientific]

Algorithm = Literal["cosine", "modified_cosine"]
Gate = Literal["retention_time", "polarity"]
ReferenceMode = Literal["list", "iterator", "cached"]


def _caffeine_spectrum(
    spectrum_id: str,
    intensities: tuple[float, float, float],
    retention_time_seconds: float | None = 60.0,
    charge: int = 1,
) -> Spectrum:
    """Build a formula-validated caffeine ion with hypothetical fragments."""
    adduct = "[M+H]+" if charge == 1 else "[M-H]-"
    offset = compute_adduct_offset(adduct)
    assert offset is not None
    precursor_mz = float(mass.calculate_mass(formula="C8H10N4O2")) + offset
    molecule = MolecularStructure(formula="C8H10N4O2")
    metadata = SpectrumMetadata(
        spectrum_id=spectrum_id,
        precursor_mz=precursor_mz,
        charge=charge,
        adduct=adduct,
        molecule=molecule,
    )
    # Exercise the actual formula/5-ppm validators, without mocking chemistry.
    assert molecule.is_physically_valid
    assert metadata.is_physically_valid
    # Three metadata-sized formula calculations; no loop over spectral arrays.
    fragment_mzs = (
        np.array(
            [
                mass.calculate_mass(formula=formula)
                for formula in ("C2H4N", "C4H7N2", "C6H8N3O")
            ],
            dtype=np.float64,
        )
        - charge * ELECTRON_MASS
    )
    return Spectrum(
        mz=fragment_mzs,
        intensities=np.array(intensities, dtype=np.float64),
        metadata={
            "id": spectrum_id,
            "compound_name": spectrum_id,
            "formula": "C8H10N4O2",
            "precursor_mz": precursor_mz,
            "adduct": adduct,
            "charge": charge,
            "retention_time": retention_time_seconds,
        },
    )


def _engine(algorithm: Algorithm, gate: Gate) -> SimilarityEngine:
    """Isolate each downstream gate from the precursor candidacy window."""
    return SimilarityEngine(
        SimilarityConfig(
            algorithm=algorithm,
            ms1_tolerance=3.0 if gate == "polarity" else 0.02,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
            rt_tolerance=5.0,
        )
    )


@pytest.mark.parametrize("algorithm", ["cosine", "modified_cosine"])
@pytest.mark.parametrize("gate", ["retention_time", "polarity"])
@pytest.mark.parametrize("top_n", [1, 2])
@pytest.mark.parametrize("reference_mode", ["list", "iterator", "cached"])
def test_top_n_counts_only_eligible_results(
    algorithm: Algorithm, gate: Gate, top_n: int, reference_mode: ReferenceMode
) -> None:
    """An ineligible perfect score must not displace eligible near-matches."""
    query = _caffeine_spectrum("query", (1.0, 1.0, 1.0))
    rejected = _caffeine_spectrum(
        "rejected",
        (1.0, 1.0, 1.0),
        retention_time_seconds=600.0 if gate == "retention_time" else 60.0,
        charge=-1 if gate == "polarity" else 1,
    )
    first = _caffeine_spectrum("eligible_first", (1.0, 1.0, 0.5), 61.0)
    second = _caffeine_spectrum("eligible_second", (1.0, 0.5, 0.5), 64.0)
    references = [second, rejected, first]  # Deliberately not score ordered.
    engine = _engine(algorithm, gate)

    # Independent normalized dot products: 3/3, 2.5/sqrt(3*2.25),
    # 2/sqrt(3*1.5). All fragment edges are isolated at this tolerance.
    for reference, expected_score in (
        (rejected, 1.0),
        (first, 0.9622504486493763),
        (second, 0.9428090415820634),
    ):
        score = engine.similarity_function.pair(reference, query)
        assert float(score["score"]) == pytest.approx(expected_score, abs=1e-12)
        assert int(score["matches"]) == 3

    unlimited = engine.search([query], references, include_decoys=False)
    assert [hit["reference_id"] for hit in unlimited] == [
        "eligible_first",
        "eligible_second",
    ]
    assert [hit["score"] for hit in unlimited] == pytest.approx(
        [0.9622504486493763, 0.9428090415820634], abs=1e-12
    )
    selected_references: Iterable[Spectrum] = references
    if reference_mode == "iterator":
        selected_references = iter(references)
    precursor_mzs = (
        np.array(
            [reference.get("precursor_mz") for reference in references],
            dtype=np.float64,
        )
        if reference_mode == "cached"
        else None
    )
    capped = engine.search(
        [query],
        selected_references,
        top_n=top_n,
        include_decoys=False,
        ref_precursor_mzs=precursor_mzs,
    )
    assert capped == unlimited[:top_n]


@pytest.mark.parametrize("algorithm", ["cosine", "modified_cosine"])
@pytest.mark.parametrize("retention_time_seconds", [None, 65.0, 65.0001])
def test_top_n_preserves_retention_time_boundaries(
    algorithm: Algorithm, retention_time_seconds: float | None
) -> None:
    """Missing RT bypasses; the exact five-second boundary is inclusive."""
    query = _caffeine_spectrum("query", (1.0, 1.0, 1.0))
    reference = _caffeine_spectrum("reference", (1.0, 1.0, 1.0), retention_time_seconds)
    results = _engine(algorithm, "retention_time").search(
        [query], [reference], top_n=1, include_decoys=False
    )
    assert len(results) == (0 if retention_time_seconds == 65.0001 else 1)


@pytest.mark.parametrize("algorithm", ["cosine", "modified_cosine"])
def test_top_n_budget_is_per_query(algorithm: Algorithm) -> None:
    """Each query receives its own best eligible result after RT filtering."""
    early_query = _caffeine_spectrum("early_query", (1.0, 1.0, 1.0), 60.0)
    late_query = _caffeine_spectrum("late_query", (1.0, 1.0, 0.5), 600.0)
    early_reference = _caffeine_spectrum("early_reference", (1.0, 1.0, 0.5), 61.0)
    late_reference = _caffeine_spectrum("late_reference", (1.0, 1.0, 1.0), 601.0)
    results = _engine(algorithm, "retention_time").search(
        [early_query, late_query],
        [late_reference, early_reference],
        top_n=1,
        include_decoys=False,
    )
    assert [(hit["query_id"], hit["reference_id"]) for hit in results] == [
        ("early_query", "early_reference"),
        ("late_query", "late_reference"),
    ]


@pytest.mark.parametrize("algorithm", ["cosine", "modified_cosine"])
def test_top_n_zero_returns_no_results(algorithm: Algorithm) -> None:
    """A zero result budget remains empty even when every candidate qualifies."""
    query = _caffeine_spectrum("query", (1.0, 1.0, 1.0))
    assert (
        _engine(algorithm, "retention_time").search(
            [query], [query], top_n=0, include_decoys=False
        )
        == []
    )
