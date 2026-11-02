"""Evidence-qualified reporting for the existing query-maxima FDR estimator.

No empirical calibration study is registered for this procedure. Numerical
estimates are therefore never labelled calibrated by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np

from MassFlow.similarity import SearchResult, calibrate_query_level_fdr

ConfidenceState = Literal["estimated_uncalibrated", "unavailable", "invalid"]


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Separate an available numerical estimate from evidence of calibration."""

    state: ConfidenceState
    q_values: dict[str, float | None]
    p_values: dict[str, float | None]
    legacy_q_values: dict[str, float]
    legacy_p_values: dict[str, float]
    summary: dict[str, int]
    limitations: tuple[str, ...]
    invalid_reason: str | None = None

    def metadata(self) -> dict[str, Any]:
        """Return JSON/YAML-compatible methodology and evidence boundaries.

        Returns
        -------
        dict
            Versioned estimation state, competition unit and limitations.

        Examples
        --------
        >>> assess_confidence([]).metadata()['state']
        'unavailable'
        """
        return {
            "schema_version": 1,
            "state": self.state,
            "method": "pooled_query_maxima_plus_one_v1",
            "discovery_unit": "query_best_target",
            "candidate_specific": False,
            "family": "input_file",
            "empirical_calibration": "pending",
            "formula": "min_{t<=s} min(1, (1+D_ge_t)/T_ge_t)",
            "decoy_strategy": "precursor_preserved_intensity_permutation_uniform_mz_jitter",
            "tie_policy": "complete_equal_score_blocks_before_monotone_closure",
            "limitations": list(self.limitations),
            "invalid_reason": self.invalid_reason,
            "counts": dict(self.summary),
            "legacy_fields": "q_value/p_value retain legacy API and empty-null diagnostics; complete-tie counts are corrected; use confidence_state and confidence_q_value for estimation availability",
        }


def assess_confidence(results: Sequence[Mapping[str, Any]]) -> ConfidenceAssessment:
    """Assess availability of the current estimator without claiming calibration.

    Parameters
    ----------
    results : Sequence[Mapping[str, Any]]
        Qualified target/decoy candidate rows with query ID and finite score.
        A nonfinite score or unattributable row invalidates the entire family.

    Returns
    -------
    ConfidenceAssessment
        Nullable statistical estimates, legacy numerical diagnostics and method
        metadata. No surviving decoys means unavailable, never an estimate of zero.

    Examples
    --------
    >>> assess_confidence([]).state
    'unavailable'
    """
    identifiers = {
        str(row["query_id"]) for row in results if row.get("query_id") is not None
    }
    limitations: tuple[str, ...] = (
        "decoy_exchangeability_unvalidated",
        "query_population_calibration_pending",
        "selection_gates_condition_estimate",
        "query_q_inherited_by_candidate_rows",
        "query_dependence_not_assessed",
        "library_redundancy_not_assessed",
    )
    try:
        for row in results:
            if not isinstance(row.get("query_id"), str) or not row["query_id"].strip():
                raise ValueError("Every confidence row requires a nonempty query_id.")
            raw_score = row.get("score")
            if (
                isinstance(raw_score, (bool, np.bool_))
                or not np.isscalar(raw_score)
                or not np.isrealobj(raw_score)
            ):
                raise ValueError("Confidence scores must be finite real scalars.")
            if not np.isfinite(float(cast(Any, raw_score))):
                raise ValueError("Confidence scores must be finite real scalars.")
        legacy_q, legacy_p, summary = calibrate_query_level_fdr(
            cast(Sequence[SearchResult], results)
        )
    except (TypeError, ValueError, KeyError) as error:
        return ConfidenceAssessment(
            "invalid",
            dict.fromkeys(identifiers),
            dict.fromkeys(identifiers),
            {},
            {},
            {
                "n_competing_queries": len(identifiers),
                "n_target_competitions": 0,
                "n_decoy_competitions": 0,
            },
            limitations + ("invalid_input",),
            str(error),
        )
    available = (
        summary["n_target_competitions"] > 0 and summary["n_decoy_competitions"] > 0
    )
    if summary["n_target_competitions"] == 0:
        limitations += ("no_target_observations",)
    if summary["n_decoy_competitions"] == 0:
        limitations += ("empty_decoy_null",)
    elif summary["n_decoy_competitions"] == 1:
        limitations += ("single_decoy_observation",)
    targets = {
        str(row["query_id"]) for row in results if not row.get("is_decoy", False)
    }
    return ConfidenceAssessment(
        "estimated_uncalibrated" if available else "unavailable",
        {
            key: legacy_q[key] if available and key in targets else None
            for key in legacy_q
        },
        {
            key: legacy_p[key] if available and key in targets else None
            for key in legacy_p
        },
        legacy_q,
        legacy_p,
        summary,
        limitations,
    )
