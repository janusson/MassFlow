"""Synthetic proof that acceleration preserves mixed-frame modified-cosine hits.

These are constructed spectra, not measured caffeine/theobromine fragmentation.
Formula-derived, singly protonated precursors and illustrative fragments keep
the example in a realistic small-molecule mass range. Equal intensities make
the expected three-pair cosine hand-computable: 3 / (sqrt(3) * sqrt(3)) = 1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from matchms import Spectrum
from matchms.similarity import ModifiedCosineGreedy
from pyteomics import mass

from MassFlow.acceleration import (
    _HAS_NUMBA,
    _count_tolerance_matches_python,
    _prefilter_pairs_numba,
    _prefilter_pairs_python,
    build_flat_peak_arrays,
)
from MassFlow.config import SimilarityConfig
from MassFlow.processing import physical_integrity_reason
from MassFlow.similarity import SimilarityEngine

pytestmark = pytest.mark.core

PrefilterKernel = Callable[..., tuple[np.ndarray, np.ndarray]]

KERNELS = [
    pytest.param(_prefilter_pairs_python, id="python"),
    pytest.param(
        _prefilter_pairs_numba,
        id="numba",
        marks=pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed"),
    ),
]


def _analogue_pair(shifted_peak_count: int) -> tuple[Spectrum, Spectrum]:
    """Construct physically consistent precursors with chosen shifted fragments."""
    precursor_reference = float(mass.calculate_mass(formula="C7H8N4O2", charge=1))
    precursor_query = float(mass.calculate_mass(formula="C8H10N4O2", charge=1))
    # Formula evaluation is over three labels, not a spectral-array hot path.
    reference_peaks = np.array(
        [
            mass.calculate_mass(formula=formula, charge=1)
            for formula in ("C3H2N2", "C4H4N2O", "C5H4N2O")
        ],
        dtype=np.float64,
    )
    query_peaks = reference_peaks.copy()
    if shifted_peak_count:
        query_peaks[-shifted_peak_count:] += precursor_query - precursor_reference
    reference = Spectrum(
        mz=reference_peaks,
        intensities=np.ones(3, dtype=np.float64),
        metadata={
            "id": "synthetic_reference",
            "precursor_mz": precursor_reference,
            "formula": "C7H8N4O2",
            "charge": 1,
            "adduct": "[M+H]+",
        },
    )
    query = Spectrum(
        mz=query_peaks,
        intensities=np.ones(3, dtype=np.float64),
        metadata={
            "id": "synthetic_query",
            "precursor_mz": precursor_query,
            "formula": "C8H10N4O2",
            "charge": 1,
            "adduct": "[M+H]+",
        },
    )
    # Exercise the real formula/adduct/5-ppm validators, without RDKit or mocks.
    assert physical_integrity_reason(reference) is None
    assert physical_integrity_reason(query) is None
    return reference, query


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("shifted_peak_count", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_frames_survive_prefilter(
    kernel: PrefilterKernel, shifted_peak_count: int, reverse: bool
) -> None:
    """Neither frame reaches three alone; their disjoint union does."""
    reference, query = _analogue_pair(shifted_peak_count)
    if reverse:
        reference, query = query, reference
    reference_arrays = build_flat_peak_arrays([reference])
    query_arrays = build_flat_peak_arrays([query])
    direct_count = _count_tolerance_matches_python(
        reference_arrays[0], query_arrays[0], 0.02
    )
    loss_count = _count_tolerance_matches_python(
        reference_arrays[2], query_arrays[2], 0.02
    )
    assert sorted([direct_count, loss_count]) == [1, 2]
    score = ModifiedCosineGreedy(tolerance=0.02).pair(reference, query)
    assert int(score["matches"]) == 3
    assert float(score["score"]) == pytest.approx(1.0, abs=1e-14)
    rows, cols = kernel(
        *reference_arrays[:3],
        *query_arrays[:3],
        0.02,
        3,
        True,
        True,
        reference_arrays[3],
        query_arrays[3],
    )
    assert list(zip(rows.tolist(), cols.tolist())) == [(0, 0)]


@pytest.mark.skipif(not _HAS_NUMBA, reason="engine prefilter requires numba")
@pytest.mark.parametrize("shifted_peak_count", [0, 1, 2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_search_preserves_three_matches(shifted_peak_count: int, reverse: bool) -> None:
    """Default acceleration must return the same score and peaks as full scoring."""
    reference, query = _analogue_pair(shifted_peak_count)
    if reverse:
        reference, query = query, reference
    config = SimilarityConfig(algorithm="modified_cosine", min_score=0.7)
    assert config.min_matched_peaks == 3
    assert config.enable_numba_prefilter
    unfiltered_config = config.model_copy(update={"enable_numba_prefilter": False})
    expected = SimilarityEngine(unfiltered_config).search(
        [query], [reference], include_decoys=False
    )
    assert len(expected) == 1
    assert expected[0]["matched_peaks"] == 3
    assert expected[0]["score"] == pytest.approx(1.0, abs=1e-14)
    assert (
        SimilarityEngine(config).search([query], [reference], include_decoys=False)
        == expected
    )


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "offset,expected", [(0.0, True), (1 / 64, True), (-1 / 64, True), (1 / 32, False)]
)
def test_mixed_frame_tolerance_boundary(
    kernel: PrefilterKernel, offset: float, expected: bool
) -> None:
    """Binary-exact masses isolate inclusive tolerance from rounding error."""
    reference = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensities=np.ones(3, dtype=np.float64),
        metadata={"precursor_mz": 500.0},
    )
    query = Spectrum(
        mz=np.array([100.0, 200.0, 316.0 + offset], dtype=np.float64),
        intensities=np.ones(3, dtype=np.float64),
        metadata={"precursor_mz": 516.0},
    )
    reference_arrays = build_flat_peak_arrays([reference])
    query_arrays = build_flat_peak_arrays([query])
    score = ModifiedCosineGreedy(tolerance=1 / 64).pair(reference, query)
    assert (int(score["matches"]) == 3) == expected
    rows, cols = kernel(
        *reference_arrays[:3],
        *query_arrays[:3],
        1 / 64,
        3,
        True,
        True,
        reference_arrays[3],
        query_arrays[3],
    )
    assert bool(rows.size) == expected
    assert rows.size == cols.size


@pytest.mark.parametrize("enabled", [False, True])
def test_overlapping_frames_do_not_inflate_final_peak_count(enabled: bool) -> None:
    """A permissive prefilter never replaces the exact one-to-one score gate."""
    reference, _ = _analogue_pair(0)
    query_peaks = reference.peaks.mz.copy()
    query_peaks[-1] += 3.0
    query = Spectrum(
        mz=query_peaks,
        intensities=np.ones(3, dtype=np.float64),
        metadata={"id": "overlap", "precursor_mz": reference.get("precursor_mz")},
    )
    score = ModifiedCosineGreedy(tolerance=0.02).pair(reference, query)
    assert int(score["matches"]) == 2
    assert float(score["score"]) == pytest.approx(2 / 3, abs=1e-14)
    config = SimilarityConfig(
        algorithm="modified_cosine",
        min_score=0.5,
        min_matched_peaks=3,
        enable_numba_prefilter=enabled,
    )
    assert (
        SimilarityEngine(config).search([query], [reference], include_decoys=False)
        == []
    )
