"""Adversarial confidence-state checks; these do not establish empirical FDR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from matchms import Spectrum

from MassFlow import io, workflow
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.similarity import calculate_empirical_p_values, calculate_fdr

pytestmark = pytest.mark.core


def _assess(rows: list[dict[str, Any]]) -> Any:
    """Import the public assessment API lazily so missing functionality fails explicitly."""
    from MassFlow import confidence

    return confidence.assess_confidence(rows)


def _hit(query: str, score: Any, decoy: bool = False) -> dict[str, Any]:
    """Make a numerical ranking fixture, not a purported experimental spectrum."""
    return {
        "query_id": query,
        "reference_id": "decoy" if decoy else "target",
        "score": score,
        "is_decoy": decoy,
    }


def _reference(targets: list[float], decoys: list[float], score: float) -> float:
    """Enumerate complete score thresholds independently of the ranking code."""
    cutoffs = {value for value in targets + decoys if value <= score}
    return min(
        min(1.0, (1 + sum(x >= t for x in decoys)) / sum(x >= t for x in targets))
        for t in cutoffs
        if any(x >= t for x in targets)
    )


def test_empty_null_has_no_statistical_estimate() -> None:
    assessment = _assess([_hit("q1", 0.9), _hit("q2", 0.8)])
    assert assessment.state == "unavailable"
    assert assessment.q_values == {"q1": None, "q2": None}
    assert assessment.legacy_q_values == {"q1": 0.5, "q2": 0.5}
    assert "empty_decoy_null" in assessment.limitations
    assert assessment.metadata()["empirical_calibration"] == "pending"


@pytest.mark.parametrize("rows", [[], [_hit("q", 0.8, True)]])
def test_no_targets_is_unavailable(rows: list[dict[str, Any]]) -> None:
    assessment = _assess(rows)
    assert assessment.state == "unavailable"
    assert all(value is None for value in assessment.q_values.values())
    assert "no_target_observations" in assessment.limitations


@pytest.mark.parametrize(
    "targets,decoys",
    [
        ([0.9, 0.8, 0.7], [0.85]),
        ([0.9, 0.9, 0.6], [0.9, 0.6]),
        ([0.95, 0.8, 0.7, 0.5], [0.9, 0.6]),
        ([0.0], [0.0]),
        ([0.0, 0.5, 0.5, 0.5], [0.0, 0.0]),
        ([0.6, 0.9, 0.9, 0.9], [0.6, 0.6]),
    ],
)
def test_estimate_matches_independent_threshold_enumeration(
    targets: list[float], decoys: list[float]
) -> None:
    assessment = _assess(
        [_hit(f"q{i}", value) for i, value in enumerate(targets)]
        + [_hit(f"d{i}", value, True) for i, value in enumerate(decoys)]
    )
    assert assessment.state == "estimated_uncalibrated"
    assert assessment.metadata()["discovery_unit"] == "query_best_target"
    assert "decoy_exchangeability_unvalidated" in assessment.limitations
    for i, value in enumerate(targets):
        assert assessment.q_values[f"q{i}"] == pytest.approx(
            _reference(targets, decoys, value)
        )
    ordered = [
        assessment.q_values[f"q{i}"]
        for i in sorted(range(len(targets)), key=lambda i: -targets[i])
    ]
    assert all(left <= right for left, right in zip(ordered, ordered[1:]))
    assert all(value > 0 for value in ordered)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "not-a-score", None])
def test_invalid_input_remains_invalid(bad: Any) -> None:
    assessment = _assess([_hit("q", bad), _hit("d", 0.5, True)])
    assert assessment.state == "invalid"
    assert assessment.q_values.get("q") is None
    assert assessment.metadata()["invalid_reason"]


@pytest.mark.parametrize(
    "bad",
    [
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-np.inf]),
        np.ones((1, 1)),
        np.array([1j]),
        np.array([True]),
    ],
)
@pytest.mark.parametrize("function", [calculate_fdr, calculate_empirical_p_values])
def test_nonfinite_or_nonnumeric_arrays_are_rejected(
    bad: np.ndarray, function: Any
) -> None:
    with pytest.raises(ValueError):
        function(bad, np.array([0.5]))
    with pytest.raises(ValueError):
        function(np.array([0.5]), bad)


def test_redundant_rows_do_not_add_query_competitions() -> None:
    original = [_hit("q", 0.9), _hit("d", 0.8, True)]
    base = _assess(original)
    redundant = _assess(original * 7 + [_hit("q", 0.7)])
    assert redundant.q_values == base.q_values
    assert redundant.summary == base.summary
    assert "library_redundancy_not_assessed" in redundant.limitations


def test_target_absence_cannot_be_known_from_scores() -> None:
    assessment = _assess([_hit("known_absent_query", 0.99), _hit("d", 0.4, True)])
    assert assessment.state == "estimated_uncalibrated"
    assert assessment.metadata()["empirical_calibration"] == "pending"
    assert assessment.metadata()["candidate_specific"] is False


def test_missing_query_identifier_invalidates_the_family() -> None:
    assessment = _assess([{"score": 0.9, "is_decoy": False}])
    assert assessment.state == "invalid"


@pytest.mark.parametrize("minimum_matches", [3, 4])
def test_pipeline_serializes_unavailable_confidence_separately(
    tmp_path: Path, minimum_matches: int
) -> None:
    # Constructed caffeine ions exercise real file ingestion/chemistry, not identity accuracy.
    query = Spectrum(
        np.array([50.0, 80.0, 110.0], dtype=np.float64),
        np.array([1.0, 0.5, 0.2], dtype=np.float64),
        metadata={
            "id": "q",
            "compound_name": "blind query",
            "precursor_mz": 195.08765,
            "adduct": "[M+H]+",
            "charge": 1,
            "ionmode": "positive",
        },
        metadata_harmonization=False,
    )
    reference = query.clone()
    reference.set("id", "ref")
    reference.set("formula", "C8H10N4O2")
    io.save_spectra_to_msp([reference], tmp_path / "library.msp")
    io.save_spectra_to_mgf([query], tmp_path / "query.mgf")
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "out"),
        input=InputConfig(
            input_path=tmp_path / "query.mgf",
            library_path=tmp_path / "library.msp",
            format="mgf",
        ),
        processing=ProcessingConfig(repair_inchi_inchikey_smiles=False),
        similarity=SimilarityConfig(
            algorithm="cosine",
            min_score=0.8,
            min_matched_peaks=minimum_matches,
            fdr_threshold=1.0,
        ),
    )
    result = workflow.run_annotation_pipeline(config)[0]
    if minimum_matches == 3:
        assert result.results[0]["score"] == pytest.approx(1.0)
        assert result.results[0]["confidence_state"] == "unavailable"
        assert result.results[0]["confidence_q_value"] is None
    else:
        assert not result.results
        frame = io._build_results_dataframe([], result.query_spectra)
        assert frame is not None
        assert frame["confidence_state"].to_list() == ["unavailable"]
        assert frame["confidence_q_value"].to_list() == [None]
    sidecar = yaml.safe_load((tmp_path / "out/query_results.report.yaml").read_text())
    assert sidecar["confidence"]["state"] == "unavailable"
    assert sidecar["confidence"]["empirical_calibration"] == "pending"
    assert "confidence_q_value" in (tmp_path / "out/query_results.csv").read_text()


@pytest.mark.parametrize("value", [0.0, 0.25, None])
def test_streaming_serialization_distinguishes_zero_from_missing(
    value: float | None,
) -> None:
    from types import SimpleNamespace
    from MassFlow.streaming.engine import _packet_to_search_result

    row: Any = {"reference_id": "ref", "score": 0.8, "q_value": value}
    serialized = _packet_to_search_result(SimpleNamespace(spectrum_id="query"), row)
    if value is None:
        assert np.isnan(serialized["fdr_q_value"])
    else:
        assert serialized["fdr_q_value"] == value


def test_streaming_response_does_not_fabricate_zero_confidence() -> None:
    from MassFlow.streaming.server import MassFlowStreamingServicer

    response = MassFlowStreamingServicer._build_response(
        object.__new__(MassFlowStreamingServicer),
        {"spectrum_id": "q", "status": "unmatched", "top_hits": []},
    )
    assert np.isnan(response.fdr_q_value)


def test_exhaustive_small_tie_blocks_match_complete_thresholds() -> None:
    from itertools import combinations_with_replacement

    vectors = [
        values
        for size in range(1, 5)
        for values in combinations_with_replacement([0.6, 0.8, 1.0], size)
    ]
    for targets in vectors:
        for decoys in vectors:
            assessment = _assess(
                [_hit(f"q{i}", score) for i, score in enumerate(targets)]
                + [_hit(f"d{i}", score, True) for i, score in enumerate(decoys)]
            )
            for i, score in enumerate(targets):
                assert assessment.q_values[f"q{i}"] == pytest.approx(
                    _reference(list(targets), list(decoys), score)
                ), (targets, decoys, score)


def _spectrum(spec_id: str, precursor_mz: float) -> Spectrum:
    return Spectrum(
        np.array([50.0, 80.0, 110.0], dtype=np.float64),
        np.array([1.0, 0.5, 0.2], dtype=np.float64),
        metadata={
            "id": spec_id,
            "precursor_mz": precursor_mz,
            "adduct": "[M+H]+",
            "charge": 1,
            "ionmode": "positive",
        },
        metadata_harmonization=False,
    )


def test_unavailable_confidence_does_not_discard_candidates_with_legacy_rank(
    tmp_path: Path,
) -> None:
    """Empty decoy null + strict fdr_threshold preserves eligible uncalibrated hits."""
    queries = [_spectrum(f"q{i}", 200.0) for i in range(3)]
    ref = _spectrum("ref1", 200.0)
    fake_hits = [
        {
            "query_id": f"q{i}",
            "query_precursor_mz": 200.0,
            "reference_id": "ref1",
            "reference_name": "ref1",
            "reference_precursor_mz": 200.0,
            "score": 0.9 - 0.05 * i,
            "matched_peaks": 3,
            "smiles": None,
            "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
            "is_decoy": False,
            "q_value": 1.0,
            "p_value": None,
        }
        for i in range(3)
    ]

    class _StubEngine:
        degraded_mode_flags = ["decoy_null_empty"]
        decoy_diagnostics = {
            "n_decoy_pairs_scored": 9,
            "best_decoy_score": 0.0,
            "n_decoy_hits": 0,
            "decoy_null_empty": True,
        }
        retrieval_diagnostics = {
            "algorithm": "cosine",
            "candidate_generation": "ms1_precursor_window",
            "queries": {
                f"q{i}": {
                    "target": {"returned_by_search": 1},
                    "decoy": {"returned_by_search": 0},
                    "terminal_outcome": "candidate_returned",
                }
                for i in range(3)
            },
        }

        def search(
            self, query_spectra: Any, reference_spectra: Any, **kwargs: Any
        ) -> Any:
            list(reference_spectra)
            return [dict(h) for h in fake_hits]

    from unittest.mock import patch

    for threshold in (0.01, 0.0):
        out_dir = tmp_path / f"empty_{threshold}"
        q_path = tmp_path / "q.mgf"
        r_path = tmp_path / "r.msp"
        io.save_spectra_to_mgf(queries, q_path)
        io.save_spectra_to_msp([ref], r_path)
        config = MassFlowConfig(
            project=ProjectConfig(output_directory=out_dir),
            input=InputConfig(input_path=q_path, library_path=r_path, format="mgf"),
            similarity=SimilarityConfig(
                algorithm="cosine",
                fdr_threshold=threshold,
                collect_retrieval_diagnostics=True,
            ),
        )
        with patch("MassFlow.workflow._worker_engine", _StubEngine()):
            result = workflow.run_annotation_pipeline(config)[0]
        assert len(result.results) == 3
        assert all(r["confidence_state"] == "unavailable" for r in result.results)
        assert all(r["confidence_q_value"] is None for r in result.results)
        assert all(r["q_value"] == pytest.approx(1.0 / 3.0) for r in result.results)
        assert "fdr_uncalibrated" in result.degraded_mode_flags
        assert any(
            "was not applied because confidence is unavailable" in w
            for w in result.warnings
        )
        assert result.confidence_summary is not None
        assert result.confidence_summary["filtering"] == {
            "applied": False,
            "fdr_threshold": threshold,
            "reason": "bypassed_unavailable_empty_decoy_null",
        }
        assert result.retrieval_diagnostics is not None
        assert result.retrieval_diagnostics["confidence"]["filtering_applied"] is False
        assert (
            result.retrieval_diagnostics["terminal_query_outcomes"][
                "candidate_returned"
            ]
            == 3
        )


@pytest.mark.parametrize(
    ("threshold", "expected_ids"),
    [
        (0.05, set()),
        (0.40, {"q0", "q1", "q2"}),
        (1.00, {"q0", "q1", "q2", "q3"}),
    ],
)
def test_populated_decoy_and_small_batch_filtering_respects_confidence_q(
    tmp_path: Path, threshold: float, expected_ids: set[str]
) -> None:
    """When decoys survive, fdr_threshold filters strictly via confidence_q_value."""
    # Small 4-query batch: 3 clean targets (q0=0.95, q1=0.90, q2=0.85 -> q=1/3)
    # and 1 decoy-outscored target (q3=0.40 vs decoy=0.70 -> q=2/4=0.5).
    queries = [_spectrum(f"q{i}", 200.0) for i in range(4)]
    ref = _spectrum("ref1", 200.0)
    fake_hits = [
        {
            "query_id": "q0",
            "reference_id": "r0",
            "reference_name": "r0",
            "score": 0.95,
            "is_decoy": False,
        },
        {
            "query_id": "q1",
            "reference_id": "r1",
            "reference_name": "r1",
            "score": 0.90,
            "is_decoy": False,
        },
        {
            "query_id": "q2",
            "reference_id": "r2",
            "reference_name": "r2",
            "score": 0.85,
            "is_decoy": False,
        },
        {
            "query_id": "q3",
            "reference_id": "r3",
            "reference_name": "r3",
            "score": 0.40,
            "is_decoy": False,
        },
        {
            "query_id": "q3",
            "reference_id": "d3",
            "reference_name": "d3",
            "score": 0.70,
            "is_decoy": True,
        },
    ]

    class _StubEngine:
        degraded_mode_flags: list[str] = []
        decoy_diagnostics = {
            "n_decoy_pairs_scored": 4,
            "best_decoy_score": 0.70,
            "n_decoy_hits": 1,
            "decoy_null_empty": False,
        }
        retrieval_diagnostics = {
            "algorithm": "cosine",
            "candidate_generation": "ms1_precursor_window",
            "queries": {
                f"q{i}": {
                    "target": {"returned_by_search": 1},
                    "decoy": {"returned_by_search": 1 if i == 3 else 0},
                    "terminal_outcome": "candidate_returned",
                }
                for i in range(4)
            },
        }

        def search(
            self, query_spectra: Any, reference_spectra: Any, **kwargs: Any
        ) -> Any:
            list(reference_spectra)
            return [dict(h) for h in fake_hits]

    from unittest.mock import patch

    out_dir = tmp_path / f"pop_{threshold}"
    q_path = tmp_path / "q.mgf"
    r_path = tmp_path / "r.msp"
    io.save_spectra_to_mgf(queries, q_path)
    io.save_spectra_to_msp([ref], r_path)
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=out_dir),
        input=InputConfig(input_path=q_path, library_path=r_path, format="mgf"),
        similarity=SimilarityConfig(
            algorithm="cosine",
            fdr_threshold=threshold,
            collect_retrieval_diagnostics=True,
        ),
    )
    with patch("MassFlow.workflow._worker_engine", _StubEngine()):
        result = workflow.run_annotation_pipeline(config)[0]

    assert {r["query_id"] for r in result.results} == expected_ids
    assert result.confidence_summary is not None
    assert result.confidence_summary["state"] == "estimated_uncalibrated"
    assert result.confidence_summary["filtering"] == {
        "applied": True,
        "fdr_threshold": threshold,
        "reason": "query_best_target_tdc_q_value",
    }
    assert result.retrieval_diagnostics is not None
    outcomes = result.retrieval_diagnostics["terminal_query_outcomes"]
    assert outcomes.get("confidence_filter_rejected", 0) == 4 - len(expected_ids)
    assert outcomes.get("candidate_returned", 0) == len(expected_ids)


def test_small_library_warning_makes_no_error_bound_claim(caplog: Any) -> None:
    workflow._emit_small_library_warning(12, 0.01)
    text = caplog.text
    assert "calibration" in text
    assert "fundamentally" not in text
    assert "relax" not in text
