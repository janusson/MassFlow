"""Reusable, data-only metric and comparison contracts for benchmark scripts."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from MassFlow.config import MassFlowConfig


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
    identity_column = _column(rows, "inchikey", "connectivity")
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
            raw_identity = row.get(identity_column) if identity_column else None
            identity = (
                str(raw_identity).split("-")[0]
                if raw_identity not in (None, "")
                else str(reference_name)
            )
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "reference_name": str(reference_name),
                    "identity": identity,
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
        query_hits.sort(
            key=lambda candidate: (
                -candidate["score"],
                candidate["identity"],
                candidate["reference_name"],
            )
        )

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
            truth_connectivity = str(labels[query_id].get("connectivity", ""))
            query_hits = threshold_hits[query_id]
            if identity_column is not None and truth_connectivity:
                top_matches_truth = query_hits[0]["identity"] == truth_connectivity
                any_matches_truth = any(
                    hit["identity"] == truth_connectivity for hit in query_hits
                )
            else:
                top_matches_truth = query_hits[0]["reference_name"] in true_references
                any_matches_truth = any(
                    hit["reference_name"] in true_references for hit in query_hits
                )
            correct_top1 += int(top_matches_truth)
            true_compound_in_candidates += int(any_matches_truth)
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

    def optional_seconds(run: Mapping[str, Any], run_name: str) -> float | None:
        value = run.get("wall_time_seconds")
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"run {run_name!r} has an invalid wall_time_seconds"
            ) from error
        if not math.isfinite(parsed):
            raise ValueError(f"run {run_name!r} has a nonfinite wall_time_seconds")
        return parsed

    compared: dict[str, Any] = {}
    for run_name, baseline_run in baseline_runs.items():
        current_run = current_runs[run_name]
        if not isinstance(baseline_run, Mapping) or not isinstance(
            current_run, Mapping
        ):
            raise ValueError(f"run {run_name!r} must be a mapping")
        baseline_rows = int(baseline_run["candidate_rows_exported"])
        current_rows = int(current_run["candidate_rows_exported"])
        baseline_time = optional_seconds(baseline_run, str(run_name))
        current_time = optional_seconds(current_run, str(run_name))
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
                if baseline_time is not None
                and current_time is not None
                and baseline_time
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


def _truth_pair_terminal_reason(
    pair_stages: Mapping[str, Any], candidate_generation: str
) -> str:
    """Return the first observed gate that removes every remaining truth pair."""
    from MassFlow.diagnostics import DiagnosticReason

    if int(pair_stages.get("prefilter_pairs", 0)) == 0:
        if candidate_generation == "ms1_precursor_window":
            return DiagnosticReason.NO_PRECURSOR_CANDIDATES.value
        if candidate_generation == "modified_cosine_pair_prefilter":
            return DiagnosticReason.PREFILTER_REJECTED.value
        return DiagnosticReason.NO_ELIGIBLE_CANDIDATE.value
    if int(pair_stages.get("score_threshold_pass_pairs", 0)) == 0:
        return DiagnosticReason.SCORE_BELOW_THRESHOLD.value
    if int(pair_stages.get("matched_peak_pass_pairs", 0)) == 0:
        return DiagnosticReason.INSUFFICIENT_FRAGMENT_MATCHES.value
    if int(pair_stages.get("polarity_compatible_pairs", 0)) == 0:
        return DiagnosticReason.POLARITY_INCOMPATIBLE.value
    if int(pair_stages.get("rt_eligible_pairs", 0)) == 0:
        return DiagnosticReason.RT_OUTSIDE_TOLERANCE.value
    return DiagnosticReason.NO_ELIGIBLE_CANDIDATE.value


def _distinct_truth_identity_rank(
    hits: Sequence[Mapping[str, Any]],
    *,
    reference_column: str | None,
    identity_column: str | None,
    truth_identity: str,
    truth_reference_ids: set[str],
    identity_endpoint: str,
    score_column: str | None = None,
) -> int | None:
    """Rank a truth identity using the primary metric's structure tie-break.

    Repeated acquisitions of one structure collapse to their best score, and
    equal scores order by the structure identity rather than by accession.
    This is the same deterministic ranking used by
    :func:`scripts.validation_benchmark.metrics._summarize`.
    """
    structure_scores: dict[str, float] = {}
    for row in hits:
        raw_reference_id = (
            row.get(reference_column) if reference_column is not None else None
        )
        if (
            raw_reference_id not in (None, "")
            and str(raw_reference_id) in truth_reference_ids
        ):
            identity = truth_identity
        else:
            raw_identity = row.get(identity_column) if identity_column else None
            if raw_identity in (None, ""):
                raw_identity = raw_reference_id
            if raw_identity in (None, ""):
                continue
            identity = str(raw_identity)
            if identity_endpoint == "connectivity":
                identity = identity.split("-")[0]
        if score_column is None:
            score = 0.0
        else:
            raw_score = row.get(score_column)
            if raw_score is None:
                raise ValueError(f"candidate {identity!r} has a missing score")
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"candidate {identity!r} has a nonnumeric score: {raw_score!r}"
                ) from error
            if not math.isfinite(score):
                raise ValueError(f"candidate {identity!r} has a nonfinite score")
        previous = structure_scores.get(identity)
        if previous is None or score > previous:
            structure_scores[identity] = score
    if truth_identity not in structure_scores:
        return None
    ranked = sorted(structure_scores, key=lambda key: (-structure_scores[key], key))
    return ranked.index(truth_identity) + 1


def attribute_truth_candidates(
    labels: Mapping[str, Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    config_path: str | None = None,
    prepared_library_path: str,
    config: MassFlowConfig | None = None,
    run_diagnostics: Mapping[str, Any] | None = None,
    identity_endpoint: str = "inchikey",
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
    config_path : str or None
        Effective YAML configuration used for the annotation run. Ignored when
        ``config`` is supplied.
    prepared_library_path : str
        Read-only prepared store used by that run.
    config : MassFlowConfig or None
        Already-validated configuration from the run. Supplying it avoids
        reinterpreting relative paths against the configuration file's
        directory.
    identity_endpoint : str, default="inchikey"
        Benchmark truth identity: exact full InChIKey or connectivity block.
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

    if config is None:
        if config_path is None:
            raise ValueError(
                "attribute_truth_candidates requires config or config_path"
            )
        config = MassFlowConfig.from_yaml(str(Path(config_path).resolve()))
    if identity_endpoint not in {"inchikey", "connectivity"}:
        raise ValueError("identity_endpoint must be inchikey or connectivity")
    algorithm = str(config.similarity.algorithm)
    if algorithm not in {"cosine", "modified_cosine"}:
        reason = (
            "truth replay requires classical cosine or modified_cosine pair-stage "
            f"diagnostics; algorithm {algorithm!r} is unsupported"
        )
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reason": reason,
            "truth_labels_used_only_for_benchmark_attribution": True,
            "identity_endpoint": identity_endpoint,
            "query_count": len(labels),
            "classification_counts": {
                "truth_replay_unavailable_for_algorithm": len(labels)
            },
            "by_query": {
                str(query_id): {
                    "source": label.get("source"),
                    "identity_endpoint": identity_endpoint,
                    "truth_identity": str(label.get(identity_endpoint) or "") or None,
                    "truth_reference_count": len(label.get("true_lib_records", [])),
                    "truth_reference_ids_missing_after_preparation": [],
                    "truth_pair_stages": {},
                    "attribution": "truth_replay_unavailable_for_algorithm",
                }
                for query_id, label in sorted(labels.items())
            },
        }
    prepared_library_path = str(Path(prepared_library_path).resolve())
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
    identity_column = _column(result_rows, "inchikey")
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
        truth_identity = str(label.get(identity_endpoint) or "")
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
            if truth_identity:
                best_truth_rank = _distinct_truth_identity_rank(
                    hits,
                    reference_column=reference_id_column,
                    identity_column=identity_column,
                    truth_identity=truth_identity,
                    truth_reference_ids=set(truth_reference_ids),
                    identity_endpoint=identity_endpoint,
                    score_column=score_column,
                )
            else:
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
                attribution = _truth_pair_terminal_reason(
                    truth_pair_counts,
                    str(truth_diagnostics.get("candidate_generation", "unknown")),
                )

        classifications[attribution] = classifications.get(attribution, 0) + 1
        per_query[str(query_id)] = {
            "source": label.get("source"),
            "identity_endpoint": identity_endpoint,
            "truth_identity": truth_identity or None,
            "truth_reference_count": len(truth_reference_ids),
            "prepared_truth_reference_count": len(prepared_truth_references),
            "truth_reference_ids_missing_after_preparation": missing_reference_ids,
            "truth_pair_stages": truth_pair_counts,
            "attribution": attribution,
        }

    return {
        "schema_version": 1,
        "truth_labels_used_only_for_benchmark_attribution": True,
        "identity_endpoint": identity_endpoint,
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
            if matched_raw is None or (
                isinstance(matched_raw, str) and not matched_raw.strip()
            ):
                matched_peaks = None
            else:
                try:
                    parsed_matched = float(matched_raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"query {query_id!r} has a nonnumeric matched_peaks: "
                        f"{matched_raw!r}"
                    ) from error
                if not math.isfinite(parsed_matched) or parsed_matched != int(
                    parsed_matched
                ):
                    raise ValueError(
                        f"query {query_id!r} has an invalid matched_peaks: "
                        f"{matched_raw!r}"
                    )
                matched_peaks = int(parsed_matched)
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


def summarize_diagnostics_by_source(
    labels: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    *,
    min_query_peaks: int,
) -> dict[str, Any]:
    """Aggregate production funnel evidence by benchmark query source.

    Query selection and truth labels are used only to stratify benchmark
    accounting. Reference-pair counts come from MassFlow's production funnel.

    Parameters
    ----------
    labels : Mapping[str, Mapping[str, Any]]
        Benchmark labels with ``source`` and ``peak_count`` fields.
    diagnostics : Mapping[str, Any]
        Versioned production retrieval-diagnostics artifact.
    min_query_peaks : int
        Effective preprocessing minimum-peak threshold.

    Returns
    -------
    dict[str, Any]
        Source-stratified terminal outcomes, pair-stage counts, and score
        histograms. Missing query-stage observations remain unattributed.
    """
    search = diagnostics.get("search", {})
    search_queries = search.get("queries", {}) if isinstance(search, Mapping) else {}
    if not isinstance(search_queries, Mapping):
        return {"status": "unavailable", "reason": "search queries were not reported"}

    count_fields = (
        "reference_pairs_total",
        "prefilter_pairs",
        "scored_pairs",
        "score_threshold_pass_pairs",
        "matched_peak_pass_pairs",
        "polarity_compatible_pairs",
        "rt_eligible_pairs",
        "eligible_before_top_n",
        "returned_by_search",
        "unattributable_pairs",
    )
    source_ids: dict[str, set[str]] = defaultdict(set)
    for query_id, label in labels.items():
        source_ids[str(label.get("source", "unknown"))].add(str(query_id))

    by_source: dict[str, Any] = {}
    for source, selected_ids in sorted(source_ids.items()):
        queries = {
            query_id: search_queries[query_id]
            for query_id in selected_ids
            if query_id in search_queries
        }
        expected_preprocessing_rejections = {
            query_id
            for query_id in selected_ids
            if int(labels[query_id].get("peak_count", min_query_peaks))
            < min_query_peaks
        }
        unobserved_ids = selected_ids - set(queries)
        unknown_ids = unobserved_ids - expected_preprocessing_rejections
        terminal_counts: dict[str, int] = defaultdict(int)
        terminal_counts["preprocessing_rejected"] = len(unobserved_ids - unknown_ids)
        terminal_counts["unknown"] = len(unknown_ids)
        confidence_counts: dict[str, int] = defaultdict(int)
        pair_totals: dict[str, dict[str, Any]] = {}
        for kind in ("target", "decoy"):
            pair_totals[kind] = {
                **{field: 0 for field in count_fields},
                "rejection_counts": {},
                "score_distribution": _empty_distribution_payload(),
            }
        for query in queries.values():
            outcome = str(query.get("terminal_outcome", "unknown"))
            terminal_counts[outcome] += 1
            confidence_state = query.get("confidence_state")
            if confidence_state not in (None, ""):
                confidence_counts[str(confidence_state)] += 1
            for kind in ("target", "decoy"):
                source_counts = query.get(kind, {})
                destination = pair_totals[kind]
                for field in count_fields:
                    destination[field] += int(source_counts.get(field, 0))
                for reason, count in source_counts.get("rejection_counts", {}).items():
                    destination["rejection_counts"][reason] = destination[
                        "rejection_counts"
                    ].get(reason, 0) + int(count)
                _merge_distribution_payload(
                    destination["score_distribution"],
                    source_counts.get(
                        "score_distribution", _empty_distribution_payload()
                    ),
                )
        searched = len(queries)
        called = terminal_counts.get("candidate_returned", 0)
        by_source[source] = {
            "selected": len(selected_ids),
            "preprocessing_rejected": terminal_counts["preprocessing_rejected"],
            "searched": searched,
            "searched_without_candidate": searched - called,
            "with_candidate": called,
            "unknown_or_unattributable": terminal_counts["unknown"],
            "terminal_query_outcomes": dict(sorted(terminal_counts.items())),
            "confidence_state_counts": dict(sorted(confidence_counts.items())),
            "pair_stages": pair_totals,
        }
    return {"status": "collected", "by_source": by_source}


def _empty_distribution_payload() -> dict[str, Any]:
    """Create a zeroed score histogram compatible with core diagnostics."""
    return {
        "bin_edges": [index / 10 for index in range(11)],
        "counts": [0] * 10,
        "below_range": 0,
        "above_range": 0,
        "nonfinite": 0,
    }


def _merge_distribution_payload(
    destination: dict[str, Any], source: Mapping[str, Any]
) -> None:
    """Add one fixed-bin score histogram into a source-level histogram."""
    source_edges = source.get("bin_edges", [])
    if len(destination["bin_edges"]) != len(source_edges) or any(
        abs(float(left) - float(right)) > 1e-15
        for left, right in zip(destination["bin_edges"], source_edges)
    ):
        raise ValueError("cannot merge score histograms with different bin edges")
    destination["counts"] = [
        int(left) + int(right)
        for left, right in zip(destination["counts"], source.get("counts", []))
    ]
    for field in ("below_range", "above_range", "nonfinite"):
        destination[field] += int(source.get(field, 0))
