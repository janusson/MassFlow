from __future__ import annotations

import pytest

from scripts.benchmark_reporting import (
    compare_benchmark_reports,
    compare_result_rows,
    evaluate_query_retrieval,
    summarize_diagnostics_by_source,
)
from scripts.benchmark_real_corpus import (
    audit_truth_precursor_eligibility,
    evaluate_real,
)


@pytest.fixture
def labels() -> dict[str, dict[str, object]]:
    return {
        "q_ranked_wrong": {
            "connectivity": "A",
            "true_lib_records": ["A_ref"],
            "source": "QTOF",
            "peak_count": 4,
        },
        "q_gate_rejected": {
            "connectivity": "B",
            "true_lib_records": ["B_ref"],
            "source": "LTQ",
            "peak_count": 2,
        },
        "q_correct": {
            "connectivity": "C",
            "true_lib_records": ["C_ref"],
            "source": "LTQ",
            "peak_count": 5,
        },
        "q_ranked_wrong_2": {
            "connectivity": "D",
            "true_lib_records": ["D_ref"],
            "source": "QTOF",
            "peak_count": 3,
        },
    }


@pytest.fixture
def rows() -> list[dict[str, str]]:
    return [
        {
            "query_id": "q_ranked_wrong",
            "reference_name": "X_ref",
            "score": "0.95",
            "matched_peaks": "4",
            "q_value": "1.0",
        },
        {
            "query_id": "q_ranked_wrong",
            "reference_name": "A_ref",
            "score": "0.80",
            "matched_peaks": "3",
            "q_value": "1.0",
        },
        {
            "query_id": "q_correct",
            "reference_name": "C_ref",
            "score": "0.90",
            "matched_peaks": "3",
            "q_value": "1.0",
        },
        {
            "query_id": "q_ranked_wrong_2",
            "reference_name": "Y_ref",
            "score": "0.70",
            "matched_peaks": "3",
            "q_value": "1.0",
        },
    ]


def test_retrieval_metrics_use_explicit_query_denominators(
    labels: dict[str, dict[str, object]], rows: list[dict[str, str]]
) -> None:
    result = evaluate_query_retrieval(labels, rows, min_query_peaks=3)

    assert result["query_counts"] == {
        "selected": 4,
        "preprocessing_rejected": 1,
        "searched": 3,
        "searched_without_candidate": 0,
        "with_candidate": 3,
        "correct_top1": 1,
        "true_compound_in_candidates": 2,
    }
    assert result["candidate_rows_exported"] == 4
    assert result["metrics"] == {
        "end_to_end_top1_accuracy": 0.25,
        "conditional_top1_accuracy": pytest.approx(1 / 3),
        "call_coverage_selected": 0.75,
        "call_coverage_searched": 1.0,
        "accepted_call_accuracy": pytest.approx(1 / 3),
        "true_compound_retrieval_rate_selected": 0.5,
        "rank1_of_reachable": 0.5,
    }
    assert result["by_source"]["QTOF"]["query_counts"]["selected"] == 2
    assert result["by_source"]["LTQ"]["query_counts"]["preprocessing_rejected"] == 1


def test_retrieval_metrics_keep_undefined_rates_null_when_denominator_is_zero() -> None:
    labels = {
        "q1": {
            "connectivity": "A",
            "true_lib_records": ["A_ref"],
            "source": "LTQ",
            "peak_count": 1,
        }
    }

    result = evaluate_query_retrieval(labels, [], min_query_peaks=3)

    assert result["query_counts"]["preprocessing_rejected"] == 1
    assert result["metrics"]["end_to_end_top1_accuracy"] == 0.0
    assert result["metrics"]["conditional_top1_accuracy"] is None
    assert result["metrics"]["call_coverage_searched"] is None
    assert result["metrics"]["accepted_call_accuracy"] is None


def test_retrieval_metrics_reject_unaccounted_queries(
    labels: dict[str, dict[str, object]], rows: list[dict[str, str]]
) -> None:
    incomplete_rows = [row for row in rows if row["query_id"] != "q_correct"]

    with pytest.raises(ValueError, match="missing from the results"):
        evaluate_query_retrieval(labels, incomplete_rows, min_query_peaks=3)


def test_real_corpus_evaluator_reports_retrieval_metrics_without_weighted_labels(
    labels: dict[str, dict[str, object]], rows: list[dict[str, str]]
) -> None:
    result = evaluate_real(
        rows,
        labels,
        score_thresholds=(0.0,),
        q_thresholds=(float("inf"),),
        min_matched_peaks=1,
    )

    assert result["metrics"]["end_to_end_top1_accuracy"] == 0.25
    assert result["query_counts"]["preprocessing_rejected"] == 1
    assert result["sweep"][0]["accepted_call_accuracy"] == pytest.approx(1 / 3)
    assert "precision_weighted" not in result["sweep"][0]


def test_comparison_computes_row_ratio_and_flags_two_tolerance_changes() -> None:
    baseline = {
        "schema_version": 2,
        "dataset_fingerprint": "same-data",
        "runs": {
            "cosine": {
                "candidate_rows_exported": 5584,
                "wall_time_seconds": 75.65,
                "configuration": {"ms1_tolerance": 0.01, "ms2_tolerance": 0.01},
            }
        },
    }
    current = {
        "schema_version": 2,
        "dataset_fingerprint": "same-data",
        "runs": {
            "cosine": {
                "candidate_rows_exported": 30401,
                "wall_time_seconds": 78.92,
                "configuration": {"ms1_tolerance": 0.02, "ms2_tolerance": 0.02},
            }
        },
    }

    comparison = compare_benchmark_reports(baseline, current)
    result = comparison["runs"]["cosine"]

    assert result["candidate_row_ratio"] == pytest.approx(30401 / 5584)
    assert result["wall_time_change_percent"] == pytest.approx(
        (78.92 - 75.65) / 75.65 * 100
    )
    assert result["changed_tolerances"] == ["ms1_tolerance", "ms2_tolerance"]
    assert result["wall_time_comparable_runtime"] is True
    assert result["interpretation"] == "descriptive_both_tolerances_changed"


def test_comparison_rejects_different_dataset_fingerprints() -> None:
    baseline = {"schema_version": 2, "dataset_fingerprint": "dataset-a", "runs": {}}
    current = {"schema_version": 2, "dataset_fingerprint": "dataset-b", "runs": {}}

    with pytest.raises(ValueError, match="different benchmark datasets"):
        compare_benchmark_reports(baseline, current)


def test_comparison_flags_different_runtime_for_wall_time_context() -> None:
    baseline = {
        "schema_version": 2,
        "dataset_fingerprint": "same-data",
        "runtime": {"platform_system": "Linux", "platform_machine": "x86_64"},
        "runs": {
            "cosine": {
                "candidate_rows_exported": 1,
                "wall_time_seconds": 1.0,
                "configuration": {},
            }
        },
    }
    current = {
        **baseline,
        "runtime": {"platform_system": "Linux", "platform_machine": "aarch64"},
    }

    comparison = compare_benchmark_reports(baseline, current)

    assert comparison["changed_runtime_fields"] == ["platform_machine"]
    assert comparison["runs"]["cosine"]["wall_time_comparable_runtime"] is False


def test_result_comparison_reports_query_changes_and_keeps_null_distinct_from_zero() -> (
    None
):
    baseline = [
        {
            "query_id": "q1",
            "reference_id": "ref-a",
            "score": "0.0",
            "matched_peaks": "0",
            "q_value": "",
            "confidence_state": "unavailable",
        },
        {
            "query_id": "q2",
            "reference_id": "ref-old",
            "score": "0.7",
            "matched_peaks": "3",
            "q_value": "1.0",
            "confidence_state": "estimated_uncalibrated",
        },
    ]
    current = [
        {
            "query_id": "q1",
            "reference_id": "ref-a",
            "score": "0.0",
            "matched_peaks": "0",
            "q_value": "0.0",
            "confidence_state": "estimated_uncalibrated",
        },
        {
            "query_id": "q2",
            "reference_id": "ref-new",
            "score": "0.7",
            "matched_peaks": "3",
            "q_value": "1.0",
            "confidence_state": "estimated_uncalibrated",
        },
    ]

    comparison = compare_result_rows(baseline, current)

    assert comparison["changed_rankings"] == 1
    assert comparison["queries_with_candidate_set_changes"] == 1
    assert comparison["queries_with_q_value_changes"] == 1
    assert comparison["queries_with_confidence_state_changes"] == 1
    assert comparison["by_query"]["q1"]["q_value_changes"] == [
        {"reference_id": "ref-a", "before": None, "after": 0.0}
    ]
    assert comparison["by_query"]["q2"]["candidate_ids_added"] == ["ref-new"]
    assert comparison["by_query"]["q2"]["candidate_ids_removed"] == ["ref-old"]


def test_result_comparison_requires_identical_query_denominators() -> None:
    with pytest.raises(ValueError, match="exact same query IDs"):
        compare_result_rows(
            [{"query_id": "q1"}],
            [{"query_id": "q2"}],
        )


def test_diagnostic_summary_by_source_keeps_preprocessing_and_search_denominators_separate() -> (
    None
):
    labels = {
        "ltq_rejected": {"source": "LTQ", "peak_count": 2},
        "ltq_searched": {"source": "LTQ", "peak_count": 4},
        "qtof_called": {"source": "QTOF", "peak_count": 4},
    }
    empty_distribution = {
        "bin_edges": [index / 10 for index in range(11)],
        "counts": [0] * 10,
        "below_range": 0,
        "above_range": 0,
        "nonfinite": 0,
    }
    one_pair = {
        "reference_pairs_total": 1,
        "prefilter_pairs": 1,
        "scored_pairs": 1,
        "score_threshold_pass_pairs": 0,
        "matched_peak_pass_pairs": 0,
        "polarity_compatible_pairs": 0,
        "rt_eligible_pairs": 0,
        "eligible_before_top_n": 0,
        "returned_by_search": 0,
        "unattributable_pairs": 0,
        "rejection_counts": {"score_below_threshold": 1},
        "score_distribution": empty_distribution,
    }
    called_pair = {
        **one_pair,
        "score_threshold_pass_pairs": 1,
        "matched_peak_pass_pairs": 1,
        "polarity_compatible_pairs": 1,
        "rt_eligible_pairs": 1,
        "eligible_before_top_n": 1,
        "returned_by_search": 1,
        "rejection_counts": {},
    }
    diagnostics = {
        "search": {
            "queries": {
                "ltq_searched": {
                    "terminal_outcome": "score_below_threshold",
                    "target": one_pair,
                    "decoy": {**one_pair, "reference_pairs_total": 0},
                },
                "qtof_called": {
                    "terminal_outcome": "candidate_returned",
                    "confidence_state": "unavailable",
                    "target": called_pair,
                    "decoy": {**one_pair, "reference_pairs_total": 0},
                },
            }
        }
    }

    result = summarize_diagnostics_by_source(labels, diagnostics, min_query_peaks=3)

    assert result["by_source"]["LTQ"]["selected"] == 2
    assert result["by_source"]["LTQ"]["preprocessing_rejected"] == 1
    assert result["by_source"]["LTQ"]["searched"] == 1
    assert result["by_source"]["LTQ"]["searched_without_candidate"] == 1
    assert result["by_source"]["QTOF"]["with_candidate"] == 1
    assert result["by_source"]["QTOF"]["confidence_state_counts"] == {"unavailable": 1}


def test_truth_precursor_audit_uses_serialized_values_and_explicit_denominators() -> (
    None
):
    labels = {
        "q_in_window": {
            "source": "LTQ",
            "peak_count": 4,
            "true_lib_records": ["ref-a", "ref-b"],
        },
        "q_preprocessing_rejected": {
            "source": "LTQ",
            "peak_count": 2,
            "true_lib_records": ["ref-a"],
        },
    }
    stats = {
        "selected_source_metadata": {
            "q_in_window": {"search_precursor_mz": 200.0},
            "q_preprocessing_rejected": {"search_precursor_mz": 300.0},
        },
        "selected_truth_reference_precursor_mz": {
            "ref-a": 200.005,
            "ref-b": 200.03,
        },
    }

    result = audit_truth_precursor_eligibility(
        labels,
        stats,
        min_query_peaks=3,
    )

    source = result["by_source"]["LTQ"]
    assert source["selected_queries"] == 2
    assert source["searched_queries"] == 1
    assert source["selected_truth_pairs"]["pair_count"] == 3
    assert source["searched_truth_pairs"]["pair_count"] == 2
    assert (
        source["tolerances"]["0.01_Da"][
            "searched_queries_with_any_truth_pair_in_window"
        ]
        == 1
    )
    assert (
        source["tolerances"]["0.02_Da"]["searched_true_reference_pairs_in_window"] == 1
    )
