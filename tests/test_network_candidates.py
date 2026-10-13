"""
Candidate-pair generation tests for spectral networking (Stage 1).

The candidate stage is pure, deterministic, and depends only on numpy, so these
are fast unit tests with hand-checkable expectations.
"""

from __future__ import annotations

import math

import pytest

from MassFlow.network.candidates import generate_candidate_pairs

pytestmark = pytest.mark.unit


def test_window_selects_only_close_precursors() -> None:
    pairs = generate_candidate_pairs([100.0, 100.01, 200.0], tolerance=0.02)
    assert pairs == [(0, 1)]


def test_pairs_are_canonically_ordered() -> None:
    pairs = generate_candidate_pairs([100.0, 100.005, 100.01], tolerance=0.02)
    assert pairs == [(0, 1), (0, 2), (1, 2)]
    assert pairs == sorted(pairs)


def test_generation_is_deterministic() -> None:
    mzs = [300.0, 100.0, 100.015, 500.0, 100.03]
    assert generate_candidate_pairs(mzs, tolerance=0.02) == generate_candidate_pairs(
        mzs, tolerance=0.02
    )


def test_missing_precursor_is_excluded() -> None:
    pairs = generate_candidate_pairs([100.0, float("nan"), 100.01], tolerance=0.02)
    assert pairs == [(0, 2)]


def test_rt_filter_excludes_far_eluting_pairs() -> None:
    assert (
        generate_candidate_pairs(
            [100.0, 100.01],
            tolerance=0.02,
            rt_seconds=[10.0, 50.0],
            rt_tolerance=5.0,
        )
        == []
    )
    assert generate_candidate_pairs(
        [100.0, 100.01],
        tolerance=0.02,
        rt_seconds=[10.0, 12.0],
        rt_tolerance=5.0,
    ) == [(0, 1)]


def test_rt_filter_fails_closed_on_missing_rt() -> None:
    assert (
        generate_candidate_pairs(
            [100.0, 100.01],
            tolerance=0.02,
            rt_seconds=[float("nan"), 12.0],
            rt_tolerance=5.0,
        )
        == []
    )


def test_missing_rt_can_be_treated_as_compatible() -> None:
    # Feature grouping treats a missing RT as m/z-informative rather than fatal.
    assert generate_candidate_pairs(
        [100.0, 100.01],
        tolerance=0.02,
        rt_seconds=[float("nan"), 12.0],
        rt_tolerance=5.0,
        treat_missing_rt_as_compatible=True,
    ) == [(0, 1)]


def test_empty_and_single_inputs() -> None:
    assert generate_candidate_pairs([], tolerance=0.02) == []
    assert generate_candidate_pairs([100.0], tolerance=0.02) == []


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        generate_candidate_pairs([100.0, 100.01], tolerance=0.0)
    with pytest.raises(ValueError):
        generate_candidate_pairs([100.0, 100.01], tolerance=0.02, rt_tolerance=5.0)
    with pytest.raises(ValueError):
        generate_candidate_pairs(
            [100.0, 100.01, 100.02],
            tolerance=0.02,
            rt_seconds=[1.0, 2.0],
            rt_tolerance=5.0,
        )


def test_does_not_crash_on_non_finite_values() -> None:
    pairs = generate_candidate_pairs([100.0, math.inf, 100.01], tolerance=0.02)
    # inf is not finite and must be excluded from (but not break) candidacy.
    assert pairs == [(0, 2)]
