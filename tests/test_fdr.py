"""Unit tests for calculate_fdr in MassFlow.similarity

These tests check behavior on edge-cases: no decoys, no targets, ties, and general monotonicity.
"""

import numpy as np

from MassFlow.similarity import calculate_fdr


def _is_non_decreasing(arr: np.ndarray) -> bool:
    return bool(np.all(np.diff(arr) >= -1e-8))


def test_calculate_fdr_no_decoys_properties():
    targets = np.array([0.9, 0.8, 0.7], dtype=float)
    decoys = np.array([], dtype=float)

    sorted_scores, q_values, is_target = calculate_fdr(targets, decoys)

    assert sorted_scores.shape == q_values.shape
    assert sorted_scores.size == 3
    assert q_values.size == 3
    assert np.all(q_values >= 0.0) and np.all(q_values <= 1.0)
    # q-values should be non-decreasing as score decreases
    assert _is_non_decreasing(q_values)
    # All entries are targets when no decoys present
    assert np.all(is_target)


def test_calculate_fdr_no_targets_returns_all_ones_and_is_target_false():
    targets = np.array([], dtype=float)
    decoys = np.array([0.5, 0.4], dtype=float)

    sorted_scores, q_values, is_target = calculate_fdr(targets, decoys)

    assert sorted_scores.size == 2
    assert q_values.size == 2
    assert np.allclose(q_values, 1.0)
    assert not np.any(is_target)


def test_calculate_fdr_ties_put_targets_before_decoys():
    targets = np.array([0.8], dtype=float)
    decoys = np.array([0.8], dtype=float)

    sorted_scores, q_values, is_target = calculate_fdr(targets, decoys)

    # When scores tie, targets must be ordered before decoys (is_target True first)
    assert is_target[0]
    assert not is_target[1]


def test_calculate_fdr_mixed_case_monotonicity_and_bounds():
    targets = np.array([0.9, 0.88, 0.5], dtype=float)
    decoys = np.array([0.85, 0.4], dtype=float)

    sorted_scores, q_values, is_target = calculate_fdr(targets, decoys)

    assert sorted_scores.size == 5
    assert q_values.size == 5
    assert np.all(q_values >= 0.0) and np.all(q_values <= 1.0)
    assert _is_non_decreasing(q_values)
