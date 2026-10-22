"""Query-denominated retrieval metrics independent of the matching implementation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

REPORT_SCHEMA_VERSION = 1


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    """Preserve numerator/denominator and distinguish undefined from numerical zero."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    """Summarize observed finite scores; do not imply a probability calibration."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Nonfinite score in benchmark output")
    return {
        "count": len(values),
        "min": float(np.min(array)) if values else None,
        "median": float(np.median(array)) if values else None,
        "max": float(np.max(array)) if values else None,
        "histogram_edges": np.linspace(0, 1, 11).tolist(),
        "histogram_counts": np.histogram(array, bins=np.linspace(0, 1, 11))[0].tolist(),
    }


def _summarize(
    queries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    identity_field: str = "inchikey",
) -> dict[str, Any]:
    """Compute rates once per selected query after structure-level collapse."""
    ids = {q["query_id"] for q in queries}
    if len(ids) != len(queries):
        raise ValueError("Repeated query identity in accounting ledger")
    if any(r.get("query_id") not in ids for r in rows):
        raise ValueError("Result contains unknown query")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("is_decoy", False):
            grouped[row["query_id"]].append(row)
    eligible = [q for q in queries if q["eligible"]]
    counts: Counter[str] = Counter()
    outcomes = []
    for query in eligible:
        candidates = grouped.get(query["query_id"], [])
        structures: dict[str, float] = {}
        for row in candidates:
            key = (
                row["inchikey"]
                if identity_field == "inchikey"
                else row["inchikey"].split("-")[0]
            )
            score = float(row["score"])
            if not np.isfinite(score):
                raise ValueError("Nonfinite candidate score")
            structures[key] = max(score, structures.get(key, float("-inf")))
        ranked = sorted(structures, key=lambda key: (-structures[key], key))
        truth = query[identity_field]
        rank = ranked.index(truth) + 1 if truth in ranked else None
        present = query["present"]
        scored = query["state"] == "scored"
        called = bool(ranked) and scored
        correct = called and present and ranked[0] == truth
        counts["present" if present else "absent"] += 1
        counts["scored"] += scored
        counts["called"] += called
        counts["correct"] += correct
        counts["false_matches"] += called and not correct
        counts["top1"] += present and scored and rank == 1
        counts["top5"] += present and scored and rank is not None and rank <= 5
        counts["retrieved"] += present and scored and rank is not None
        counts["absent_scored"] += not present and scored
        counts["absent_called"] += not present and called
        counts["absent_rejected"] += not present and scored and not called
        outcomes.append(
            {
                "query_id": query["query_id"],
                "present": present,
                "called": called,
                "correct_top1": bool(correct),
                "true_structure_rank": rank,
                "distinct_candidate_structures": len(ranked),
                "top_structure": ranked[0] if ranked else None,
                "best_score": structures[ranked[0]] if ranked else None,
            }
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "identity_endpoint": identity_field,
        "counts": {
            "attempted": len(queries),
            "eligible": len(eligible),
            "excluded": len(queries) - len(eligible),
            "successfully_scored": counts["scored"],
            "calls": counts["called"],
            "correct_top1": counts["correct"],
            "false_matches": counts["false_matches"],
            "missed_matches": counts["present"] - counts["correct"],
            "query_states": dict(Counter(q["state"] for q in queries)),
            "rejection_reasons": dict(
                Counter(reason for q in queries for reason in q.get("reasons", []))
            ),
        },
        "closed_set": {
            "top1": _rate(counts["top1"], counts["present"]),
            "top5": _rate(counts["top5"], counts["present"]),
            "retrieval_coverage": _rate(counts["retrieved"], counts["present"]),
            "recall": _rate(counts["correct"], counts["present"]),
        },
        "open_set": {
            "false_assignment_rate": _rate(counts["absent_called"], counts["absent"]),
            "searched_rejection_rate": _rate(
                counts["absent_rejected"], counts["absent_scored"]
            ),
        },
        "precision": _rate(counts["correct"], counts["called"]),
        "call_coverage": _rate(counts["called"], len(eligible)),
        "observed_fdp": _rate(counts["false_matches"], counts["called"]),
        "outcomes": outcomes,
    }


def evaluate(
    queries: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate selected queries, retaining operational failures in denominators.

    Parameters
    ----------
    queries : list[dict]
        Independent ledger with query ID, full/first-block truth, presence,
        eligibility and terminal pipeline state. Labels never affect scoring.
    rows : list[dict]
        Actual qualified candidate rows enriched with reference labels.

    Returns
    -------
    dict
        Versioned strict/connectivity rates, raw counts and source strata.

    Examples
    --------
    >>> evaluate([],[])['precision']['value'] is None
    True
    """
    result = _summarize(queries, rows)
    result["connectivity_endpoint"] = _summarize(queries, rows, "connectivity")
    strata = sorted({q["stratum"] for q in queries})
    result["by_stratum"] = {
        s: _summarize(
            [q for q in queries if q["stratum"] == s],
            [
                r
                for r in rows
                if any(
                    q["query_id"] == r["query_id"] and q["stratum"] == s
                    for q in queries
                )
            ],
        )
        for s in strata
    }
    result["score_distributions"] = {
        kind: _distribution(
            [float(r["score"]) for r in rows if bool(r.get("is_decoy", False)) == decoy]
        )
        for kind, decoy in [("targets", False), ("decoys", True)]
    }
    return result
