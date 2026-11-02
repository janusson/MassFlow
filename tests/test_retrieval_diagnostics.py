from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from matchms import Spectrum

from MassFlow import io, processing
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.diagnostics import (
    DiagnosticReason,
    DiagnosticStage,
    RejectionEvent,
    RejectionSummary,
)
from MassFlow.similarity import SimilarityEngine
from MassFlow.workflow import FileExecutionResult, _write_analysis_report


def _spectrum(
    spectrum_id: str,
    precursor_mz: float,
    fragment_mz: tuple[float, ...] = (100.0, 150.0, 175.0),
    *,
    adduct: str | None = None,
    retention_time: float | None = None,
) -> Spectrum:
    metadata: dict[str, Any] = {
        "id": spectrum_id,
        "precursor_mz": precursor_mz,
        "charge": 1,
    }
    if adduct is not None:
        metadata["adduct"] = adduct
    if retention_time is not None:
        metadata["retention_time"] = retention_time
    return Spectrum(
        mz=np.asarray(fragment_mz, dtype=np.float64),
        intensities=np.arange(1, len(fragment_mz) + 1, dtype=np.float64),
        metadata=metadata,
    )


def _cosine_config(**overrides: Any) -> SimilarityConfig:
    values: dict[str, Any] = {
        "algorithm": "cosine",
        "ms1_tolerance": 0.02,
        "ms2_tolerance": 0.01,
        "min_score": 0.4,
        "min_matched_peaks": 3,
    }
    values.update(overrides)
    return SimilarityConfig(**values)


def test_diagnostics_are_opt_in_and_do_not_change_scores_or_results() -> None:
    query = _spectrum("query", 200.0)
    references = [
        _spectrum("ref-a", 200.0),
        _spectrum("ref-b", 200.01),
        _spectrum("ref-c", 300.0),
    ]
    plain_engine = SimilarityEngine(_cosine_config())
    diagnostic_engine = SimilarityEngine(
        _cosine_config(collect_retrieval_diagnostics=True)
    )

    plain_results = plain_engine.search([query], references, include_decoys=False)
    diagnostic_results = diagnostic_engine.search(
        [query], references, include_decoys=False
    )

    assert diagnostic_results == plain_results
    assert plain_engine.retrieval_diagnostics == {}
    funnel = diagnostic_engine.retrieval_diagnostics
    query_funnel = funnel["queries"]["query"]
    target = query_funnel["target"]
    assert target["reference_pairs_total"] == 3
    assert target["prefilter_pairs"] == target["scored_pairs"] == 2
    assert target["score_threshold_pass_pairs"] == 2
    assert target["matched_peak_pass_pairs"] == 2
    assert target["polarity_compatible_pairs"] == 2
    assert target["rt_eligible_pairs"] == 2
    assert target["eligible_before_top_n"] == 2
    assert target["returned_by_search"] == 2
    score_distribution = target["score_distribution"]
    assert sum(score_distribution["counts"]) == target["scored_pairs"]
    assert score_distribution["counts"][-1] == 2
    assert score_distribution["below_range"] == 0
    assert score_distribution["above_range"] == 0
    assert score_distribution["nonfinite"] == 0
    assert target["rejection_counts"] == {"prefilter_rejected": 1}
    assert query_funnel["terminal_outcome"] == DiagnosticReason.CANDIDATE_RETURNED.value


def test_pre_generated_decoys_use_effective_flags_in_pair_diagnostics() -> None:
    query = _spectrum("query", 200.0)
    target = _spectrum("target", 200.0)
    decoy = _spectrum("decoy", 200.0)
    engine = SimilarityEngine(_cosine_config(collect_retrieval_diagnostics=True))

    results = engine.search(
        [query],
        [target, decoy],
        include_decoys=False,
        ref_is_decoy=np.asarray([False, True], dtype=bool),
    )

    assert {row["reference_id"]: row["is_decoy"] for row in results} == {
        "target": False,
        "decoy": True,
    }
    funnel = engine.retrieval_diagnostics
    query_counts = funnel["queries"]["query"]
    assert query_counts["target"]["reference_pairs_total"] == 1
    assert query_counts["decoy"]["reference_pairs_total"] == 1
    assert query_counts["target"]["scored_pairs"] == 1
    assert query_counts["decoy"]["scored_pairs"] == 1
    assert query_counts["target"]["returned_by_search"] == 1
    assert query_counts["decoy"]["returned_by_search"] == 1
    assert funnel["pair_stages"]["target"]["reference_pairs_total"] == 1
    assert funnel["pair_stages"]["decoy"]["reference_pairs_total"] == 1


def test_no_precursor_candidate_is_a_distinct_terminal_outcome() -> None:
    engine = SimilarityEngine(_cosine_config(collect_retrieval_diagnostics=True))
    engine.search(
        [_spectrum("far-query", 500.0)],
        [_spectrum("ref", 200.0)],
        include_decoys=False,
    )

    query_funnel = engine.retrieval_diagnostics["queries"]["far-query"]
    assert query_funnel["target"]["prefilter_pairs"] == 0
    assert query_funnel["target"]["rejection_counts"] == {"prefilter_rejected": 1}
    assert (
        query_funnel["terminal_outcome"]
        == DiagnosticReason.NO_PRECURSOR_CANDIDATES.value
    )


def test_score_and_fragment_gates_are_reported_sequentially() -> None:
    query = _spectrum("query", 200.0)
    unrelated = _spectrum("unrelated", 200.0, (50.0, 60.0, 70.0))
    score_engine = SimilarityEngine(_cosine_config(collect_retrieval_diagnostics=True))
    score_engine.search([query], [unrelated], include_decoys=False)
    score_query = score_engine.retrieval_diagnostics["queries"]["query"]
    assert score_query["target"]["score_threshold_pass_pairs"] == 0
    assert score_query["target"]["rejection_counts"] == {"score_below_threshold": 1}
    assert (
        score_query["terminal_outcome"] == DiagnosticReason.SCORE_BELOW_THRESHOLD.value
    )

    one_match = _spectrum("one-match", 200.0, (60.0, 70.0, 100.0))
    fragment_engine = SimilarityEngine(
        _cosine_config(
            min_score=0.0,
            min_matched_peaks=2,
            collect_retrieval_diagnostics=True,
        )
    )
    fragment_engine.search([query], [one_match], include_decoys=False)
    fragment_query = fragment_engine.retrieval_diagnostics["queries"]["query"]
    assert fragment_query["target"]["score_threshold_pass_pairs"] == 1
    assert fragment_query["target"]["matched_peak_pass_pairs"] == 0
    assert fragment_query["target"]["rejection_counts"] == {
        "insufficient_fragment_matches": 1
    }
    assert (
        fragment_query["terminal_outcome"]
        == DiagnosticReason.INSUFFICIENT_FRAGMENT_MATCHES.value
    )


def test_polarity_rt_and_top_n_losses_have_distinct_pair_reasons() -> None:
    query = _spectrum("query", 200.0, adduct="[M+H]+", retention_time=10.0)
    opposite_polarity = _spectrum(
        "negative", 200.0, adduct="[M-H]-", retention_time=10.0
    )
    polarity_engine = SimilarityEngine(
        _cosine_config(collect_retrieval_diagnostics=True)
    )
    polarity_engine.search([query], [opposite_polarity], include_decoys=False)
    polarity_query = polarity_engine.retrieval_diagnostics["queries"]["query"]
    assert polarity_query["target"]["matched_peak_pass_pairs"] == 1
    assert polarity_query["target"]["polarity_compatible_pairs"] == 0
    assert polarity_query["target"]["rejection_counts"] == {"polarity_incompatible": 1}
    assert (
        polarity_query["terminal_outcome"]
        == DiagnosticReason.POLARITY_INCOMPATIBLE.value
    )

    late_reference = _spectrum("late", 200.0, adduct="[M+H]+", retention_time=20.0)
    rt_engine = SimilarityEngine(
        _cosine_config(
            rt_tolerance=5.0,
            collect_retrieval_diagnostics=True,
        )
    )
    rt_engine.search([query], [late_reference], include_decoys=False)
    rt_query = rt_engine.retrieval_diagnostics["queries"]["query"]
    assert rt_query["target"]["polarity_compatible_pairs"] == 1
    assert rt_query["target"]["rt_eligible_pairs"] == 0
    assert rt_query["target"]["rejection_counts"] == {"rt_outside_tolerance": 1}
    assert rt_query["terminal_outcome"] == DiagnosticReason.RT_OUTSIDE_TOLERANCE.value

    top_n_engine = SimilarityEngine(_cosine_config(collect_retrieval_diagnostics=True))
    ranked = top_n_engine.search(
        [query],
        [_spectrum("ref-b", 200.0), _spectrum("ref-a", 200.0)],
        include_decoys=False,
        top_n=1,
    )
    top_n_query = top_n_engine.retrieval_diagnostics["queries"]["query"]
    assert [row["reference_id"] for row in ranked] == ["ref-a"]
    assert top_n_query["target"]["eligible_before_top_n"] == 2
    assert top_n_query["target"]["returned_by_search"] == 1
    assert top_n_query["target"]["rejection_counts"] == {
        "candidate_ranked_below_limit": 1
    }


def test_lazy_library_diagnostics_merge_chunk_counts(monkeypatch) -> None:
    import MassFlow.similarity as similarity_module

    def one_reference_per_chunk(references, chunk_size):
        iterator = iter(references)
        while True:
            chunk = list(__import__("itertools").islice(iterator, 1))
            if not chunk:
                break
            yield chunk

    monkeypatch.setattr(
        similarity_module, "yield_fixed_chunks", one_reference_per_chunk
    )
    engine = SimilarityEngine(_cosine_config(collect_retrieval_diagnostics=True))
    results = engine.search(
        [_spectrum("query", 200.0)],
        iter([_spectrum("ref-a", 200.0), _spectrum("ref-b", 300.0)]),
        include_decoys=False,
    )

    assert len(results) == 1
    query_funnel = engine.retrieval_diagnostics["queries"]["query"]
    assert query_funnel["target"]["reference_pairs_total"] == 2
    assert query_funnel["target"]["prefilter_pairs"] == 1
    assert query_funnel["target"]["eligible_before_top_n"] == 1
    assert sum(query_funnel["target"]["score_distribution"]["counts"]) == 1
    aggregate_target = engine.retrieval_diagnostics["pair_stages"]["target"]
    assert aggregate_target["reference_pairs_total"] == 2
    assert sum(aggregate_target["score_distribution"]["counts"]) == 1


def test_typed_rejection_events_are_bounded_and_preserve_details(
    tmp_path: Path,
) -> None:
    missing_precursor = Spectrum(
        mz=np.asarray([100.0, 150.0], dtype=np.float64),
        intensities=np.asarray([1.0, 2.0], dtype=np.float64),
        metadata={"id": "missing-precursor"},
    )
    events: list[RejectionEvent] = []
    loaded = list(
        io._validate_spectra_iterator(
            [missing_precursor],
            tmp_path / "input.msp",
            diagnostic_reporter=events.append,
        )
    )
    assert loaded == []
    assert events[0].reason is DiagnosticReason.PRECURSOR_MISSING
    assert events[0].stage is DiagnosticStage.INPUT_VALIDATION
    assert events[0].detail == "Missing precursor_mz"

    too_few_peaks = _spectrum("short", 200.0, (100.0, 150.0))
    processing_events: list[RejectionEvent] = []
    processed = list(
        processing.process_spectra(
            [too_few_peaks],
            ProcessingConfig(filter_min_peaks=True, min_peaks=3),
            diagnostic_reporter=processing_events.append,
        )
    )
    assert processed == []
    assert processing_events[0].reason is DiagnosticReason.INSUFFICIENT_PEAKS
    assert processing_events[0].spectrum_id == "short"
    assert "peak_count=2" in processing_events[0].detail

    bounded = RejectionSummary(sample_limit=1)
    for number in range(3):
        bounded.record(
            RejectionEvent(
                stage=DiagnosticStage.PEAK_PROCESSING,
                reason=DiagnosticReason.INSUFFICIENT_PEAKS,
                detail=f"item={number}",
            )
        )
    assert bounded.as_dict()["reason_counts"] == {"insufficient_peaks": 3}
    assert bounded.as_dict()["samples"] == [
        {
            "stage": "peak_processing",
            "reason": "insufficient_peaks",
            "detail": "item=0",
            "spectrum_id": None,
        }
    ]


def test_json_diagnostic_sidecar_is_deterministic(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "terminal_query_outcomes": {"no_precursor_candidates": 2},
        "missing_precursor_mz": None,
    }
    path = tmp_path / "query.diagnostics.json"
    io.save_retrieval_diagnostics_json(path, payload)
    first_write = path.read_text()
    io.save_retrieval_diagnostics_json(path, payload)
    assert path.read_text() == first_write
    assert json.loads(first_write) == payload


def test_workflow_writes_diagnostic_json_and_links_it_from_yaml(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.mgf"
    library_path = tmp_path / "library.msp"
    query_path.write_text("query bytes")
    library_path.write_text("library bytes")
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path),
        input=InputConfig(input_path=query_path, library_path=library_path),
        similarity=SimilarityConfig(collect_retrieval_diagnostics=True),
    )
    result = FileExecutionResult(
        status="success",
        input_path=query_path,
        retrieval_diagnostics={
            "schema_version": 1,
            "terminal_query_outcomes": {"no_precursor_candidates": 1},
        },
    )
    report_path = tmp_path / "queries_results.report.yaml"

    _write_analysis_report(report_path, config, result)

    diagnostic_path = tmp_path / "queries_results.diagnostics.json"
    diagnostic_payload = json.loads(diagnostic_path.read_text())
    report_payload = yaml.safe_load(report_path.read_text())
    assert diagnostic_payload["terminal_query_outcomes"] == {
        "no_precursor_candidates": 1
    }
    assert diagnostic_payload["fingerprints"] == {
        "query_file_sha256": hashlib.sha256(b"query bytes").hexdigest(),
        "library_file_sha256": hashlib.sha256(b"library bytes").hexdigest(),
    }
    assert report_payload["retrieval_diagnostics_json"] == str(diagnostic_path)


def test_json_diagnostic_sidecar_rejects_nonfinite_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        io.save_retrieval_diagnostics_json(
            tmp_path / "invalid.diagnostics.json", {"score": float("nan")}
        )


def test_zero_score_and_zero_match_threshold_still_respects_ms1_prefilter() -> None:
    engine = SimilarityEngine(
        _cosine_config(
            min_score=0.0,
            min_matched_peaks=0,
            collect_retrieval_diagnostics=True,
        )
    )
    results = engine.search(
        [_spectrum("query", 500.0)],
        [_spectrum("outside-window", 200.0)],
        include_decoys=False,
    )

    assert len(results) == 1
    assert results[0]["score"] == 0.0
    query_funnel = engine.retrieval_diagnostics["queries"]["query"]
    assert query_funnel["target"]["prefilter_pairs"] == 0
    assert query_funnel["target"]["eligible_before_top_n"] == 1
    assert query_funnel["target"]["unattributable_pairs"] == 1
    assert query_funnel["target"]["stage_accounting_consistent"] is False
    assert query_funnel["target"]["rejection_counts"] == {}
    assert query_funnel["terminal_outcome"] == DiagnosticReason.CANDIDATE_RETURNED.value


def test_annotation_pipeline_exports_query_funnel_and_confidence_state(
    tmp_path: Path,
) -> None:
    from MassFlow.workflow import run_annotation_pipeline

    library_path = tmp_path / "library.msp"
    library_path.write_text(
        "NAME: ref\n"
        "ID: ref\n"
        "PRECURSOR_MZ: 200\n"
        "CHARGE: 1\n"
        "NUM PEAKS: 3\n"
        "100 1\n"
        "150 2\n"
        "175 3\n\n"
    )
    query_path = tmp_path / "query.mgf"
    query_path.write_text(
        "BEGIN IONS\n"
        "TITLE=query\n"
        "ID=query\n"
        "PEPMASS=200\n"
        "CHARGE=1+\n"
        "100 1\n"
        "150 2\n"
        "175 3\n"
        "END IONS\n"
    )
    output_directory = tmp_path / "output"
    config = MassFlowConfig(
        project=ProjectConfig(
            name="diagnostics-test", output_directory=output_directory
        ),
        input=InputConfig(
            input_path=query_path,
            library_path=library_path,
            format="mgf",
        ),
        processing=ProcessingConfig(filter_min_peaks=True, min_peaks=3),
        similarity=_cosine_config(
            collect_retrieval_diagnostics=True,
            fdr_threshold=1.0,
        ),
    )

    result = run_annotation_pipeline(config)[0]

    assert result.status in {"success", "degraded"}
    assert result.retrieval_diagnostics is not None
    assert result.retrieval_diagnostics["terminal_query_outcomes"] == {
        "candidate_returned": 1,
        "preprocessing_rejected": 0,
    }
    assert result.retrieval_diagnostics["confidence"]["state"] in {
        "unavailable",
        "estimated_uncalibrated",
    }
    json_sidecar = output_directory / "query_results.diagnostics.json"
    payload = json.loads(json_sidecar.read_text())
    assert payload["search"]["queries"]["query"]["target"]["returned_by_search"] == 1
    yaml_sidecar = output_directory / "query_results.report.yaml"
    assert str(json_sidecar) in yaml_sidecar.read_text()


def test_workflow_accounts_for_searches_without_engine_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from MassFlow.workflow import run_annotation_pipeline

    library_path = tmp_path / "library.msp"
    library_path.write_text(
        "NAME: ref\n"
        "ID: ref\n"
        "PRECURSOR_MZ: 200\n"
        "CHARGE: 1\n"
        "NUM PEAKS: 3\n"
        "100 1\n"
        "150 2\n"
        "175 3\n\n"
    )
    query_path = tmp_path / "query.mgf"
    query_path.write_text(
        "BEGIN IONS\n"
        "TITLE=query\n"
        "ID=query\n"
        "PEPMASS=200\n"
        "CHARGE=1+\n"
        "100 1\n"
        "150 2\n"
        "175 3\n"
        "END IONS\n"
    )
    config = MassFlowConfig(
        project=ProjectConfig(
            name="missing-diagnostics-test",
            output_directory=tmp_path / "output",
        ),
        input=InputConfig(
            input_path=query_path,
            library_path=library_path,
            format="mgf",
        ),
        processing=ProcessingConfig(filter_min_peaks=True, min_peaks=3),
        similarity=_cosine_config(
            collect_retrieval_diagnostics=True,
            fdr_threshold=1.0,
        ),
    )
    original_search = SimilarityEngine.search

    def search_without_diagnostics(
        engine: SimilarityEngine, *args: Any, **kwargs: Any
    ) -> Any:
        results = original_search(engine, *args, **kwargs)
        engine._retrieval_diagnostics = {}
        return results

    monkeypatch.setattr(SimilarityEngine, "search", search_without_diagnostics)
    result = run_annotation_pipeline(config)[0]

    assert result.retrieval_diagnostics is not None
    assert result.retrieval_diagnostics["search"]["status"] == "unavailable"
    assert result.retrieval_diagnostics["terminal_query_outcomes"] == {
        "preprocessing_rejected": 0,
        "unknown": 1,
    }
    confidence_state = result.retrieval_diagnostics["confidence"]["state"]
    assert result.retrieval_diagnostics["confidence_state_counts"] == {
        confidence_state: 1
    }
    assert result.retrieval_diagnostics["unknown_or_unattributable"] == 1


def test_modified_cosine_diagnostics_preserve_full_candidate_domain() -> None:
    query = _spectrum("query", 200.0)
    references = [
        _spectrum("same-precursor", 200.0),
        _spectrum("shifted-precursor", 300.0),
    ]
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm="modified_cosine",
            ms1_tolerance=0.02,
            ms2_tolerance=0.01,
            min_score=0.4,
            min_matched_peaks=3,
            collect_retrieval_diagnostics=True,
        )
    )

    results = engine.search([query], references, include_decoys=False)

    query_funnel = engine.retrieval_diagnostics["queries"]["query"]
    assert {row["reference_id"] for row in results} == {
        "same-precursor",
        "shifted-precursor",
    }
    assert engine.retrieval_diagnostics["candidate_generation"] == "full_scoring"
    assert query_funnel["target"]["prefilter_pairs"] == 2
    assert query_funnel["target"]["score_distribution"]["counts"][-1] == 2
