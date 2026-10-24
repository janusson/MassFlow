"""Reusable, data-only metric and comparison contracts for benchmark scripts."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Optional


REPORT_SCHEMA_VERSION = 2


def _column(rows: Sequence[Mapping[str, Any]], *names: str) -> Optional[str]:
    """Find an output column without depending on capitalization."""
    if not rows:
        return None
    keys = {str(key).lower(): str(key) for key in rows[0]}
    for name in names:
        if name.lower() in keys:
            return keys[name.lower()]
    return None


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """Return a rate, preserving undefined denominators as null values."""
    return numerator / denominator if denominator else None


def _to_float(value: Any, *, field_name: str, query_id: str) -> float:
    """Parse a numeric result field and identify the affected query on error."""
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"query {query_id!r} has a nonnumeric {field_name}: {value!r}"
        ) from error


def evaluate_query_retrieval(
    labels: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    min_query_peaks: int,
    min_score: float = 0.0,
    q_threshold: float = math.inf,
    min_matched_peaks: int = 1,
) -> dict[str, Any]:
    """Compute query-level retrieval metrics from labels and exported rows.

    A query is searched when it passes the configured minimum-peak gate and
    appears in the exported result table. Queries rejected by that gate remain
    in the end-to-end denominator. The CSV contract requires one or more rows
    for every searched query, including queries with no candidate.

    Parameters
    ----------
    labels : Mapping[str, Mapping[str, Any]]
        Per-query truth labels. Each value requires ``connectivity``,
        ``true_lib_records``, ``source``, and ``peak_count``.
    rows : Sequence[Mapping[str, Any]]
        Rows from the result CSV. Candidate rows require ``reference_name``,
        ``score``, and ``matched_peaks``; ``q_value`` is required when a finite
        q-value threshold is applied.
    min_query_peaks : int
        Preprocessing minimum-peak gate used by the benchmark run.
    min_score : float, default=0.0
        Minimum exported-candidate score used for this metric operating point.
    q_threshold : float, default=inf
        Maximum q-value; infinity disables q-value filtering.
    min_matched_peaks : int, default=1
        Minimum matched-fragment count used for this metric operating point.

    Returns
    -------
    dict[str, Any]
        Query accounting, retrieval metrics, source-stratified counts, and
        exported candidate-row count.

    Examples
    --------
    >>> evaluate_query_retrieval({}, [], min_query_peaks=3)["query_counts"]["selected"]
    0
    """
    if min_query_peaks < 0 or min_matched_peaks < 0:
        raise ValueError("peak-count thresholds must be nonnegative")

    query_column = _column(rows, "query_id", "query")
    reference_column = _column(rows, "reference_name", "reference_id")
    score_column = _column(rows, "score")
    matched_peaks_column = _column(rows, "matched_peaks")
    q_value_column = _column(rows, "q_value")
    if rows and query_column is None:
        raise ValueError("results are missing a query_id column")

    selected_ids = {str(query_id) for query_id in labels}
    rejected_ids: set[str] = set()
    for query_id, label in labels.items():
        if "peak_count" not in label:
            raise ValueError(
                "query labels are missing peak_count; rebuild the prepared "
                "benchmark data without --keep-data"
            )
        try:
            peak_count = int(label["peak_count"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"query {query_id!r} has an invalid peak_count") from error
        if peak_count < min_query_peaks:
            rejected_ids.add(str(query_id))

    observed_ids = {
        str(row[query_column])
        for row in rows
        if query_column is not None and row.get(query_column) not in (None, "")
    }
    unknown_ids = observed_ids - selected_ids
    if unknown_ids:
        raise ValueError(f"results contain unknown query ids: {sorted(unknown_ids)}")
    rejected_with_rows = observed_ids & rejected_ids
    if rejected_with_rows:
        raise ValueError(
            "preprocessing-rejected queries unexpectedly appear in results: "
            f"{sorted(rejected_with_rows)}"
        )
    searched_ids = selected_ids - rejected_ids
    missing_ids = searched_ids - observed_ids
    if missing_ids:
        raise ValueError(
            f"searched queries are missing from the results: {sorted(missing_ids)}"
        )

    if rows and reference_column is None:
        raise ValueError("results are missing a reference_name column")
    candidate_rows: list[dict[str, Any]] = []
    if reference_column is not None:
        for row in rows:
            reference_name = row.get(reference_column)
            if reference_name in (None, ""):
                continue
            query_id = str(row[query_column]) if query_column is not None else ""
            if score_column is None or matched_peaks_column is None:
                raise ValueError(
                    "candidate rows require score and matched_peaks columns"
                )
            score = _to_float(
                row.get(score_column), field_name="score", query_id=query_id
            )
            try:
                matched_peaks = int(
                    _to_float(
                        row.get(matched_peaks_column),
                        field_name="matched_peaks",
                        query_id=query_id,
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"query {query_id!r} has a nonnumeric matched_peaks"
                ) from error
            q_value = math.nan
            raw_q_value = (
                row.get(q_value_column) if q_value_column is not None else None
            )
            if raw_q_value not in (None, ""):
                q_value = _to_float(
                    raw_q_value, field_name="q_value", query_id=query_id
                )
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "reference_name": str(reference_name),
                    "score": score,
                    "matched_peaks": matched_peaks,
                    "q_value": q_value,
                }
            )

    hits_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows:
        if candidate["query_id"] not in rejected_ids:
            hits_by_query[candidate["query_id"]].append(candidate)

    for query_hits in hits_by_query.values():
        query_hits.sort(key=lambda candidate: -candidate["score"])

    threshold_hits: dict[str, list[dict[str, Any]]] = {}
    for query_id in searched_ids:
        query_hits = [
            candidate
            for candidate in hits_by_query.get(query_id, [])
            if candidate["score"] >= min_score
            and candidate["matched_peaks"] >= min_matched_peaks
            and (
                math.isinf(q_threshold)
                or (
                    math.isfinite(candidate["q_value"])
                    and candidate["q_value"] <= q_threshold
                )
            )
        ]
        threshold_hits[query_id] = query_hits

    def summarize(group_ids: set[str]) -> dict[str, Any]:
        group_rejected = group_ids & rejected_ids
        group_searched = group_ids - group_rejected
        group_with_candidate = {
            query_id for query_id in group_searched if threshold_hits.get(query_id)
        }
        correct_top1 = 0
        true_compound_in_candidates = 0
        for query_id in group_with_candidate:
            true_references = set(labels[query_id]["true_lib_records"])
            query_hits = threshold_hits[query_id]
            if query_hits[0]["reference_name"] in true_references:
                correct_top1 += 1
            if any(hit["reference_name"] in true_references for hit in query_hits):
                true_compound_in_candidates += 1
        call_count = len(group_with_candidate)
        selected_count = len(group_ids)
        searched_count = len(group_searched)
        reachable_count = true_compound_in_candidates
        return {
            "query_counts": {
                "selected": selected_count,
                "preprocessing_rejected": len(group_rejected),
                "searched": searched_count,
                "searched_without_candidate": searched_count - call_count,
                "with_candidate": call_count,
                "correct_top1": correct_top1,
                "true_compound_in_candidates": reachable_count,
            },
            "metrics": {
                "end_to_end_top1_accuracy": _ratio(correct_top1, selected_count),
                "conditional_top1_accuracy": _ratio(correct_top1, searched_count),
                "call_coverage_selected": _ratio(call_count, selected_count),
                "call_coverage_searched": _ratio(call_count, searched_count),
                "accepted_call_accuracy": _ratio(correct_top1, call_count),
                "true_compound_retrieval_rate_selected": _ratio(
                    reachable_count, selected_count
                ),
                "rank1_of_reachable": _ratio(correct_top1, reachable_count),
            },
            "candidate_rows_at_operating_point": sum(
                len(threshold_hits.get(query_id, [])) for query_id in group_searched
            ),
        }

    summary = summarize(selected_ids)
    source_counts: dict[str, set[str]] = defaultdict(set)
    for query_id, label in labels.items():
        source_counts[str(label.get("source", "unknown"))].add(str(query_id))
    return {
        **summary,
        "candidate_rows_exported": len(candidate_rows),
        "by_source": {
            source: summarize(query_ids)
            for source, query_ids in sorted(source_counts.items())
        },
    }


def compare_benchmark_reports(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two compatible benchmark reports without implying causation.

    Reports must use this module's schema and identify the same exact query and
    library inputs. Configuration differences are returned explicitly; a
    two-tolerance change is labeled descriptive rather than attributable.

    Parameters
    ----------
    baseline : Mapping[str, Any]
        Earlier benchmark report.
    current : Mapping[str, Any]
        Later benchmark report with the same ``dataset_fingerprint``.

    Returns
    -------
    dict[str, Any]
        Per-algorithm candidate-row and wall-time comparisons.

    Examples
    --------
    >>> old = {"schema_version": 2, "dataset_fingerprint": "x", "runs": {}}
    >>> compare_benchmark_reports(old, old)["runs"]
    {}
    """
    if (
        baseline.get("schema_version") != REPORT_SCHEMA_VERSION
        or current.get("schema_version") != REPORT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"both reports must use schema version {REPORT_SCHEMA_VERSION}; "
            "rerun older benchmarks before comparing them"
        )
    baseline_fingerprint = baseline.get("dataset_fingerprint")
    if not baseline_fingerprint or baseline_fingerprint != current.get(
        "dataset_fingerprint"
    ):
        raise ValueError("cannot compare different benchmark datasets")
    baseline_runs = baseline.get("runs")
    current_runs = current.get("runs")
    if not isinstance(baseline_runs, Mapping) or not isinstance(current_runs, Mapping):
        raise ValueError("benchmark reports must contain a runs mapping")
    missing_runs = set(baseline_runs) - set(current_runs)
    if missing_runs:
        raise ValueError(f"current report is missing runs: {sorted(missing_runs)}")

    baseline_runtime = baseline.get("runtime", {})
    current_runtime = current.get("runtime", {})
    if not isinstance(baseline_runtime, Mapping) or not isinstance(
        current_runtime, Mapping
    ):
        raise ValueError("benchmark reports contain invalid runtime metadata")
    runtime_fields = {"platform_system", "platform_machine", "python_version"}
    changed_runtime_fields = sorted(
        field
        for field in runtime_fields
        if baseline_runtime.get(field) != current_runtime.get(field)
    )

    compared: dict[str, Any] = {}
    for run_name, baseline_run in baseline_runs.items():
        current_run = current_runs[run_name]
        if not isinstance(baseline_run, Mapping) or not isinstance(
            current_run, Mapping
        ):
            raise ValueError(f"run {run_name!r} must be a mapping")
        baseline_rows = int(baseline_run["candidate_rows_exported"])
        current_rows = int(current_run["candidate_rows_exported"])
        baseline_time = float(baseline_run["wall_time_seconds"])
        current_time = float(current_run["wall_time_seconds"])
        baseline_configuration = baseline_run.get("configuration", {})
        current_configuration = current_run.get("configuration", {})
        if not isinstance(baseline_configuration, Mapping) or not isinstance(
            current_configuration, Mapping
        ):
            raise ValueError(f"run {run_name!r} has invalid configuration metadata")
        config_keys = set(baseline_configuration) | set(current_configuration)
        changed_fields = sorted(
            field
            for field in config_keys
            if baseline_configuration.get(field) != current_configuration.get(field)
        )
        tolerance_fields = {"ms1_tolerance", "ms2_tolerance"}
        changed_tolerances = [
            field for field in changed_fields if field in tolerance_fields
        ]
        if len(changed_tolerances) == 2:
            interpretation = "descriptive_both_tolerances_changed"
        elif changed_fields == changed_tolerances and len(changed_tolerances) == 1:
            interpretation = "single_tolerance_changed"
        elif changed_fields:
            interpretation = "descriptive_other_configuration_changes"
        else:
            interpretation = "same_configuration"
        compared[str(run_name)] = {
            "baseline_candidate_rows": baseline_rows,
            "current_candidate_rows": current_rows,
            "candidate_row_ratio": (
                current_rows / baseline_rows if baseline_rows else None
            ),
            "baseline_wall_time_seconds": baseline_time,
            "current_wall_time_seconds": current_time,
            "wall_time_change_percent": (
                (current_time - baseline_time) / baseline_time * 100
                if baseline_time
                else None
            ),
            "changed_configuration_fields": changed_fields,
            "changed_tolerances": changed_tolerances,
            "changed_runtime_fields": changed_runtime_fields,
            "wall_time_comparable_runtime": not changed_runtime_fields,
            "interpretation": interpretation,
        }
    return {
        "dataset_fingerprint": baseline_fingerprint,
        "changed_runtime_fields": changed_runtime_fields,
        "runs": compared,
    }


def attribute_truth_candidates(
    labels: Mapping[str, Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    config_path: str,
    prepared_library_path: str,
    run_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attribute benchmark truth-record losses using the configured search gates.

    This benchmark-only audit keeps truth labels outside MassFlow annotation
    output. It replays each query only against its labelled reference records,
    using the effective processing and similarity configurations, then compares
    gate eligibility with the full-library exported ranking.

    Parameters
    ----------
    labels : Mapping[str, Mapping[str, Any]]
        Per-query benchmark truth labels, including ``true_lib_records``.
    result_rows : Sequence[Mapping[str, Any]]
        Full-library result CSV rows, including no-hit query rows.
    config_path : str
        Effective YAML configuration used for the annotation run.
    prepared_library_path : str
        Read-only prepared store used by that run.
    run_diagnostics : Mapping[str, Any] or None
        Production funnel artifact used to determine whether a candidate was
        removed by post-scoring confidence filtering.

    Returns
    -------
    dict[str, Any]
        Per-query truth-record eligibility, full-search rank, and an explicit
        attribution state. Causes not observable from the prepared store or
        search gates remain ``unknown_or_unattributable``.
    """
    from pathlib import Path

    from MassFlow import io, processing
    from MassFlow.config import MassFlowConfig
    from MassFlow.diagnostics import DiagnosticReason
    from MassFlow.library import LibrarySpec, open_library
    from MassFlow.similarity import SimilarityEngine

    config = MassFlowConfig.from_yaml(config_path)
    query_iterator = io.load_spectra(Path(config.input.input_path))
    processed_queries = list(
        processing.process_spectra(query_iterator, config.processing)
    )
    query_by_id = {
        str(query.get("id")): query
        for query in processed_queries
        if query.get("id") is not None
    }

    library_backend = open_library(
        LibrarySpec(path=Path(prepared_library_path), kind="store"),
        config.processing,
    )
    try:
        prepared_references = list(library_backend.iter_spectra())
    finally:
        library_backend.close()
    prepared_by_id = {
        str(reference.get("id")): reference
        for reference in prepared_references
        if reference.get("id") is not None
    }

    query_column = _column(result_rows, "query_id", "query")
    reference_id_column = _column(result_rows, "reference_id")
    score_column = _column(result_rows, "score")
    if result_rows and query_column is None:
        raise ValueError("results are missing a query_id column")
    rows_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in result_rows:
        if query_column is not None:
            rows_by_query[str(row[query_column])].append(row)
    for hits in rows_by_query.values():
        hits.sort(
            key=lambda row: (
                -float(row[score_column])
                if score_column and row.get(score_column)
                else 0.0,
                str(row.get(reference_id_column or "", "")),
            )
        )

    similarity_config = config.similarity.model_copy(
        update={"collect_retrieval_diagnostics": True}
    )
    engine = SimilarityEngine(similarity_config)
    run_search: Mapping[str, Any] = {}
    if run_diagnostics is not None:
        raw_search = run_diagnostics.get("search", {})
        if isinstance(raw_search, Mapping):
            run_search = raw_search.get("queries", {}) or {}

    per_query: dict[str, Any] = {}
    classifications: dict[str, int] = {}
    for query_id, label in sorted(labels.items()):
        truth_reference_ids = [str(item) for item in label.get("true_lib_records", [])]
        missing_reference_ids = [
            reference_id
            for reference_id in truth_reference_ids
            if reference_id not in prepared_by_id
        ]
        prepared_truth_references = [
            prepared_by_id[reference_id]
            for reference_id in truth_reference_ids
            if reference_id in prepared_by_id
        ]
        query = query_by_id.get(str(query_id))
        attribution = "unknown_or_unattributable"
        truth_pair_counts: dict[str, Any] = {}
        if not truth_reference_ids:
            attribution = "truth_record_absent_from_declared_library"
        elif query is None:
            minimum_peaks = config.processing.min_peaks
            if (
                config.processing.filter_min_peaks
                and int(label.get("peak_count", minimum_peaks)) < minimum_peaks
            ):
                attribution = DiagnosticReason.INSUFFICIENT_PEAKS.value
            else:
                attribution = "query_processing_rejected_or_unattributable"
        elif not prepared_truth_references:
            attribution = "truth_records_not_retained_in_prepared_library"
        else:
            engine.search(
                [query],
                prepared_truth_references,
                include_decoys=False,
            )
            truth_diagnostics = engine.retrieval_diagnostics
            query_diagnostics = truth_diagnostics.get("queries", {}).get(
                str(query_id), {}
            )
            truth_pair_counts = query_diagnostics.get("target", {})
            eligible_truth_count = int(
                truth_pair_counts.get("eligible_before_top_n", 0)
            )
            hits = [
                row
                for row in rows_by_query.get(str(query_id), [])
                if row.get(reference_id_column or "") not in (None, "")
            ]
            true_ids = set(truth_reference_ids)
            best_truth_rank = next(
                (
                    rank
                    for rank, row in enumerate(hits, start=1)
                    if str(row.get(reference_id_column or "")) in true_ids
                ),
                None,
            )
            if best_truth_rank is not None:
                if best_truth_rank == 1:
                    attribution = "correct_top1"
                elif best_truth_rank <= 5:
                    attribution = "correct_top5_not_top1"
                else:
                    attribution = "truth_candidate_ranked_below_top5"
            elif eligible_truth_count > 0:
                query_search = run_search.get(str(query_id), {})
                if query_search.get("terminal_outcome") == "confidence_filter_rejected":
                    attribution = DiagnosticReason.CONFIDENCE_FILTER_REJECTED.value
                else:
                    attribution = "truth_pair_eligible_but_not_in_export_unknown"
            else:
                rejection_counts = truth_pair_counts.get("rejection_counts", {})
                observed_reasons = [
                    reason
                    for reason, count in rejection_counts.items()
                    if int(count) > 0
                ]
                if observed_reasons:
                    attribution = observed_reasons[0]
                else:
                    attribution = str(
                        query_diagnostics.get(
                            "terminal_outcome",
                            DiagnosticReason.NO_ELIGIBLE_CANDIDATE.value,
                        )
                    )

        classifications[attribution] = classifications.get(attribution, 0) + 1
        per_query[str(query_id)] = {
            "source": label.get("source"),
            "truth_reference_count": len(truth_reference_ids),
            "prepared_truth_reference_count": len(prepared_truth_references),
            "truth_reference_ids_missing_after_preparation": missing_reference_ids,
            "truth_pair_stages": truth_pair_counts,
            "attribution": attribution,
        }

    return {
        "schema_version": 1,
        "truth_labels_used_only_for_benchmark_attribution": True,
        "query_count": len(labels),
        "classification_counts": dict(sorted(classifications.items())),
        "by_query": per_query,
    }


def compare_result_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare exported query rankings without treating provenance as scores.

    The comparison is query-keyed and ignores metadata payloads such as source
    paths. Missing numeric values remain ``None`` and are never coerced to zero.

    Parameters
    ----------
    baseline_rows : Sequence[Mapping[str, Any]]
        Rows from the earlier exported result table.
    current_rows : Sequence[Mapping[str, Any]]
        Rows from the later exported result table.

    Returns
    -------
    dict[str, Any]
        Per-query candidate/ranking/score/matched-peak/q-value/confidence
        changes and aggregate change counts.
    """

    def parsed_rows(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str, str, str, str | None, str | None]:
        query_column = _column(rows, "query_id", "query")
        if query_column is None:
            raise ValueError("result comparison requires a query_id column")
        reference_column = _column(rows, "reference_id") or _column(
            rows, "reference_name", "reference"
        )
        score_column = _column(rows, "score")
        matched_column = _column(rows, "matched_peaks")
        q_value_column = _column(rows, "q_value")
        confidence_column = _column(rows, "confidence_state")
        if rows and any(
            row.get(reference_column or "") not in (None, "") for row in rows
        ):
            if (
                reference_column is None
                or score_column is None
                or matched_column is None
            ):
                raise ValueError(
                    "candidate rows require reference, score, and matched_peaks"
                )
        return (
            query_column,
            reference_column or "",
            score_column or "",
            matched_column or "",
            q_value_column,
            confidence_column,
        )

    def optional_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def group_rows(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[set[str], dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
        (
            query_column,
            reference_column,
            score_column,
            matched_column,
            q_value_column,
            confidence_column,
        ) = parsed_rows(rows)
        query_ids = {
            str(row[query_column])
            for row in rows
            if row.get(query_column) not in (None, "")
        }
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        confidence_states: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            query_id = str(row.get(query_column, ""))
            state = row.get(confidence_column) if confidence_column else None
            if (
                state not in (None, "")
                and str(state) not in confidence_states[query_id]
            ):
                confidence_states[query_id].append(str(state))
            reference_id = row.get(reference_column) if reference_column else None
            if reference_id in (None, ""):
                continue
            score = optional_float(row.get(score_column))
            if score is None:
                raise ValueError(f"query {query_id!r} has a missing candidate score")
            matched_raw = row.get(matched_column)
            try:
                matched_peaks = (
                    int(float(matched_raw)) if matched_raw is not None else 0
                )
            except (TypeError, ValueError):
                matched_peaks = 0
            candidates[query_id].append(
                {
                    "reference_id": str(reference_id),
                    "score": score,
                    "matched_peaks": matched_peaks,
                    "q_value": optional_float(
                        row.get(q_value_column) if q_value_column else None
                    ),
                }
            )
        for hits in candidates.values():
            hits.sort(key=lambda hit: (-hit["score"], hit["reference_id"]))
        return query_ids, candidates, confidence_states

    baseline_ids, baseline_by_query, baseline_states = group_rows(baseline_rows)
    current_ids, current_by_query, current_states = group_rows(current_rows)
    if baseline_ids != current_ids:
        raise ValueError(
            "result comparison requires the exact same query IDs; "
            f"missing={sorted(baseline_ids - current_ids)}, "
            f"unexpected={sorted(current_ids - baseline_ids)}"
        )

    per_query: dict[str, Any] = {}
    for query_id in sorted(baseline_ids):
        before = baseline_by_query.get(query_id, [])
        after = current_by_query.get(query_id, [])
        before_by_id = {candidate["reference_id"]: candidate for candidate in before}
        after_by_id = {candidate["reference_id"]: candidate for candidate in after}
        shared_ids = set(before_by_id) & set(after_by_id)
        score_changes = [
            {
                "reference_id": reference_id,
                "before": before_by_id[reference_id]["score"],
                "after": after_by_id[reference_id]["score"],
            }
            for reference_id in sorted(shared_ids)
            if before_by_id[reference_id]["score"] != after_by_id[reference_id]["score"]
        ]
        matched_peak_changes = [
            {
                "reference_id": reference_id,
                "before": before_by_id[reference_id]["matched_peaks"],
                "after": after_by_id[reference_id]["matched_peaks"],
            }
            for reference_id in sorted(shared_ids)
            if before_by_id[reference_id]["matched_peaks"]
            != after_by_id[reference_id]["matched_peaks"]
        ]
        q_value_changes = [
            {
                "reference_id": reference_id,
                "before": before_by_id[reference_id]["q_value"],
                "after": after_by_id[reference_id]["q_value"],
            }
            for reference_id in sorted(shared_ids)
            if before_by_id[reference_id]["q_value"]
            != after_by_id[reference_id]["q_value"]
        ]
        before_states = sorted(baseline_states.get(query_id, []))
        after_states = sorted(current_states.get(query_id, []))
        per_query[query_id] = {
            "candidate_count_before": len(before),
            "candidate_count_after": len(after),
            "candidate_ids_added": sorted(set(after_by_id) - set(before_by_id)),
            "candidate_ids_removed": sorted(set(before_by_id) - set(after_by_id)),
            "ranking_before": [candidate["reference_id"] for candidate in before],
            "ranking_after": [candidate["reference_id"] for candidate in after],
            "ranking_changed": [candidate["reference_id"] for candidate in before]
            != [candidate["reference_id"] for candidate in after],
            "score_changes": score_changes,
            "matched_peak_changes": matched_peak_changes,
            "q_value_changes": q_value_changes,
            "confidence_state_before": before_states,
            "confidence_state_after": after_states,
            "confidence_state_changed": before_states != after_states,
        }

    return {
        "query_count": len(per_query),
        "changed_rankings": sum(
            int(query["ranking_changed"]) for query in per_query.values()
        ),
        "queries_with_candidate_set_changes": sum(
            int(bool(query["candidate_ids_added"] or query["candidate_ids_removed"]))
            for query in per_query.values()
        ),
        "queries_with_score_changes": sum(
            int(bool(query["score_changes"])) for query in per_query.values()
        ),
        "queries_with_matched_peak_changes": sum(
            int(bool(query["matched_peak_changes"])) for query in per_query.values()
        ),
        "queries_with_q_value_changes": sum(
            int(bool(query["q_value_changes"])) for query in per_query.values()
        ),
        "queries_with_confidence_state_changes": sum(
            int(query["confidence_state_changed"]) for query in per_query.values()
        ),
        "by_query": per_query,
    }
