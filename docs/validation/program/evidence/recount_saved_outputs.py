"""Recount historical candidate outputs without importing MassFlow.

This verifies saved-row arithmetic against the supplied label-to-reference map.
It does not authenticate experimental labels, rerun matching, or validate FDR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> dict[str, Any]:
    """Identify one immutable input asset by bytes."""
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _recount(base: Path) -> dict[str, Any]:
    """Recompute query-level counts using the saved evaluator's label mapping."""
    labels: dict[str, Any] = json.loads((base / "labels.json").read_text())
    result: dict[str, Any] = {"runs": {}, "asset_hashes": {}}
    # These are candidate rows/record mappings, not loops over spectral arrays.
    for name in ("narrow", "wide"):
        directory = base / f"real_cosine_{name}"
        with (directory / "queries_results.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row.get("score"):
                by_query[row["query_id"]].append(row)
        correct = sum(
            max(candidates, key=lambda row: float(row["score"]))["reference_id"]
            in labels[query_id]["true_lib_records"]
            for query_id, candidates in by_query.items()
        )
        tied_queries = sum(
            sum(
                float(row["score"])
                == max(float(candidate["score"]) for candidate in candidates)
                for row in candidates
            )
            > 1
            for candidates in by_query.values()
        )
        if tied_queries:
            raise ValueError("Historical recount requires a declared tie policy.")
        result["runs"][name] = {
            "selected_labelled_queries": len(labels),
            "exported_rows_including_unknowns": len(rows),
            "candidate_rows": sum(map(len, by_query.values())),
            "called_queries": len(by_query),
            "correct_top1_connectivity": correct,
            "true_in_candidates": sum(
                any(
                    row["reference_id"] in labels[query_id]["true_lib_records"]
                    for row in candidates
                )
                for query_id, candidates in by_query.items()
            ),
            "top_score_tie_queries": tied_queries,
            "calls_by_source": dict(
                Counter(labels[query_id]["source"] for query_id in by_query)
            ),
            "selected_end_to_end_accuracy": correct / len(labels) if labels else None,
            "accepted_call_accuracy": correct / len(by_query) if by_query else None,
        }
        for filename in (
            "queries_results.csv",
            "queries_results.report.yaml",
            "run_provenance.json",
        ):
            path = directory / filename
            result["asset_hashes"][str(path)] = _sha256(path)
    for filename in (
        "labels.json",
        "queries.mgf",
        "library.msp",
        "report.json",
        "report-narrow.json",
    ):
        path = base / filename
        result["asset_hashes"][str(path)] = _sha256(path)
    return result


def _main() -> None:
    """Print the independent recount; never modify the historical inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_recount(args.base), indent=2))


if __name__ == "__main__":
    _main()
