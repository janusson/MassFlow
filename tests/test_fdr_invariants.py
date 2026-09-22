"""
Algebraic invariants of the target-decoy FDR estimator.

``tests/test_fdr_statistics.py`` and ``tests/test_scientific_validation.py``
cover the *statistical contract* (competition unit, traceability, execution-mode
equivalence) and golden known answers. This module covers the *algebraic
properties* of ``MassFlow.similarity.calculate_fdr`` itself — the properties a
reviewer would check by hand against
``docs/user-guide/scoring_logic.md`` §3–§5 — plus an exact agreement check
against an independent transcription of the documented formula.

These invariants previously lived only in ``scripts/benchmark_metrics.py``,
which meant nothing gated them. Every assertion here is a property, not a
snapshot: it constrains how two quantities must relate, so it stays valid if the
implementation is refactored.

Notation (scoring_logic.md §9):
    T_q   best target score of query q, after thresholds
    D_q   best decoy score of query q; -inf when the query has no decoy hit
    FDR(t) = (1 + #{q: D_q >= t}) / #{q: T_q >= t}, clipped to [0, 1]
    q(s)   = min over t <= s of FDR(t)                       (monotone closure)
    p(s)   = (1 + #{D_q >= s}) / (1 + #{q: D_q finite})      (diagnostic only)
Ties rank decoy-first (conservative).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pytest

from MassFlow.similarity import calculate_empirical_p_values, calculate_fdr

pytestmark = pytest.mark.scientific

# Deterministic seeds so a failure is always reproducible.
SEEDS = list(range(200))


def reference_q_p(
    target_scores: List[float], decoy_scores: List[float]
) -> Dict[float, Tuple[float, float]]:
    """Independent transcription of the documented formula (shares no code).

    Returns ``{target score: (q, p)}``. Empty decoy set yields the documented
    ``1/N`` rank bound shared by all N competing queries, with ``p = 1.0``.
    """
    targets = np.asarray(target_scores, dtype=np.float64)
    decoys = np.asarray(decoy_scores, dtype=np.float64)
    if targets.size == 0:
        return {}
    if decoys.size == 0:
        return {float(s): (1.0 / targets.size, 1.0) for s in targets}

    out: Dict[float, Tuple[float, float]] = {}
    thresholds = np.unique(np.concatenate([targets, decoys]))
    for s in targets:
        q = 1.0
        for t in thresholds:
            if t > s:
                continue
            n_targets = int((targets >= t).sum())
            if n_targets == 0:
                continue
            fdr = (1 + int((decoys >= t).sum())) / n_targets
            q = min(q, min(fdr, 1.0))
        p = (1 + int((decoys >= s).sum())) / (1 + decoys.size)
        out[float(s)] = (q, p)
    return out


@pytest.fixture
def random_cases() -> List[Tuple[np.ndarray, np.ndarray]]:
    """300 deterministic random (targets, decoys) score sets."""
    cases: List[Tuple[np.ndarray, np.ndarray]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        n_t = int(rng.integers(1, 40))
        n_d = int(rng.integers(1, 40))
        cases.append((rng.uniform(0, 1, size=n_t), rng.uniform(0, 1, size=n_d)))
    return cases


class TestMonotoneClosure:
    """q is the monotone closure of FDR: weaker scores can never get a better q."""

    def test_q_non_decreasing_as_score_decreases(
        self, random_cases: List[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        for targets, decoys in random_cases:
            _, q_values, _ = calculate_fdr(targets, decoys)
            assert np.all(np.diff(q_values) >= -1e-15), (
                "q decreased as the score decreased; calculate_fdr returns ranks "
                "in descending score order, so q must be non-decreasing there"
            )

    def test_stronger_target_never_has_worse_q(
        self, random_cases: List[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        for targets, decoys in random_cases:
            scores, q_values, is_target = calculate_fdr(targets, decoys)
            pairs = [(s, q) for s, q, t in zip(scores, q_values, is_target) if t]
            for (s_hi, q_hi), (s_lo, q_lo) in zip(pairs, pairs[1:]):
                assert s_hi >= s_lo
                assert q_hi <= q_lo + 1e-15, (
                    "a stronger-scoring query received a worse q than a weaker one"
                )


class TestTieHandling:
    """Equal scores rank decoys before targets, so ties never favour the target."""

    def test_tied_decoy_ranks_before_target(self) -> None:
        scores, _, is_target = calculate_fdr(np.array([0.5]), np.array([0.5]))
        assert not is_target[0], "the tied rank must be the decoy"
        assert is_target[1]

    def test_tied_decoy_is_counted_against_the_target(self) -> None:
        # One tied decoy and one target: FDR at that rank is (1 + 1) / 1 = 2 -> clipped to 1.
        _, q_values, is_target = calculate_fdr(np.array([0.5]), np.array([0.5]))
        target_q = q_values[is_target][0]
        assert target_q == pytest.approx(1.0)

    def test_tie_cannot_lower_a_targets_q_value(self) -> None:
        without_tie = calculate_fdr(np.array([0.5, 0.4]), np.array([0.1]))
        with_tie = calculate_fdr(np.array([0.5, 0.4]), np.array([0.5, 0.1]))
        q_no_tie = without_tie[1][without_tie[2]]
        q_with_tie = with_tie[1][with_tie[2]]
        assert np.all(q_with_tie >= q_no_tie - 1e-15)


class TestPseudoCount:
    """The +1 pseudo-count (Elias & Gygi 2007) forbids an optimistic q == 0."""

    def test_q_is_never_zero(self) -> None:
        _, q_values, _ = calculate_fdr(
            np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.1, 0.05])
        )
        assert np.all(q_values > 0.0)

    def test_perfect_separation_still_has_positive_q(self) -> None:
        # Every target above every decoy: the best attainable FDR is 1/N, not 0.
        targets = np.array([0.9, 0.8])
        decoys = np.array([0.1])
        _, q_values, is_target = calculate_fdr(targets, decoys)
        target_q = q_values[is_target]
        assert np.all(target_q > 0.0)
        assert np.allclose(target_q, 0.5)  # (1 + 0) / 2


class TestEmptyDecoyNull:
    """With no null evidence the q-value is the conservative 1/N rank bound."""

    TARGETS = np.array([0.9, 0.8, 0.7, 0.6])

    def test_q_is_one_over_n_for_every_query(self) -> None:
        _, q_values, _ = calculate_fdr(self.TARGETS, np.array([]))
        assert np.allclose(q_values, 1.0 / len(self.TARGETS))

    def test_p_is_one(self) -> None:
        p_values = calculate_empirical_p_values(self.TARGETS, np.array([]))
        assert np.allclose(p_values, 1.0)

    def test_bound_is_monotone_closed(self) -> None:
        _, q_values, _ = calculate_fdr(self.TARGETS, np.array([]))
        assert np.all(np.diff(q_values) >= -1e-15)


class TestEmptyTargets:
    """Without a target hit there is no discovery to calibrate."""

    def test_decoys_carry_q_one_and_are_not_targets(self) -> None:
        _, q_values, is_target = calculate_fdr(np.array([]), np.array([0.4, 0.5]))
        assert np.allclose(q_values, 1.0)
        assert not is_target.any()

    def test_both_empty_returns_empty(self) -> None:
        scores, q_values, is_target = calculate_fdr(np.array([]), np.array([]))
        assert scores.size == 0 and q_values.size == 0 and not is_target.any()


class TestHandComputedValues:
    """Values computed by hand from the ranking definition."""

    def test_three_targets_one_decoy(self) -> None:
        # ranks: T .9 (FDR 1/1), T .8 (1/2), T .7 (1/3), D .1 (2/3)
        # suffix-min -> 1/3 for each target.
        scores, q_values, is_target = calculate_fdr(
            np.array([0.9, 0.8, 0.7]), np.array([0.1])
        )
        assert np.allclose(q_values[is_target], 1.0 / 3)
        assert scores[0] == pytest.approx(0.9)  # descending order

    def test_two_targets_two_decoys(self) -> None:
        # ranks: T 1.0 (1/1), T 0.9 (1/2), D 0.8 (2/2), D 0.7 (3/2 -> clip 1)
        # target suffix-min -> min(1, 0.5, 1, 1) = 0.5
        _, q_values, is_target = calculate_fdr(
            np.array([1.0, 0.9]), np.array([0.8, 0.7])
        )
        assert np.allclose(q_values[is_target], 0.5)

    def test_p_value_formula(self) -> None:
        # p(s) = (1 + #{D >= s}) / (1 + #D)
        decoys = np.array([0.9, 0.85, 0.1])

        # s above every decoy: numerator is the pseudo-count alone -> 1/4
        p_high = calculate_empirical_p_values(np.array([0.95]), decoys)
        assert p_high[0] == pytest.approx(1 / 4)

        # s below two decoys: 2 decoys match or beat it -> (1 + 2) / 4 = 3/4
        p_low = calculate_empirical_p_values(np.array([0.8]), decoys)
        assert p_low[0] == pytest.approx(3 / 4)


class TestAgainstIndependentFormula:
    """Exact agreement with a from-scratch transcription of the contract."""

    def test_q_and_p_match_on_all_random_cases(
        self, random_cases: List[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        for targets, decoys in random_cases:
            scores, q_values, is_target = calculate_fdr(targets, decoys)
            p_values = calculate_empirical_p_values(targets, decoys)
            expected = reference_q_p(list(targets), list(decoys))

            # q: look up each target rank's score
            for score, q, is_t in zip(scores, q_values, is_target):
                if not is_t:
                    continue
                q_ref, _ = expected[float(score)]
                assert abs(q - q_ref) < 1e-12, (
                    f"q mismatch at score {score}: {q} != {q_ref}"
                )

            # p: aligned with the input target order
            for score, p in zip(targets, p_values):
                _, p_ref = expected[float(score)]
                assert abs(p - p_ref) < 1e-12, (
                    f"p mismatch at score {score}: {p} != {p_ref}"
                )
