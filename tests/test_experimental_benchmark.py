"""Metric arithmetic and real, redistributable MassBank pipeline integration."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.core
FIXTURES = Path(__file__).parent / "fixtures/massbank"


def _api() -> tuple[Any, Any, Any]:
    from scripts.validation_benchmark import dataset, metrics, runner

    return dataset, metrics, runner


def test_metrics_count_every_query_and_distinct_structures() -> None:
    _, metrics, _ = _api()
    queries = [
        {
            "query_id": "a",
            "inchikey": "A",
            "connectivity": "A",
            "eligible": True,
            "present": True,
            "state": "scored",
            "stratum": "one",
        },
        {
            "query_id": "b",
            "inchikey": "B",
            "connectivity": "B",
            "eligible": True,
            "present": True,
            "state": "processing_rejected",
            "stratum": "one",
        },
        {
            "query_id": "c",
            "inchikey": "C",
            "connectivity": "C",
            "eligible": True,
            "present": False,
            "state": "scored",
            "stratum": "two",
        },
        {
            "query_id": "d",
            "inchikey": "D",
            "connectivity": "D",
            "eligible": False,
            "present": True,
            "state": "scored",
            "stratum": "two",
        },
    ]
    rows = [
        {
            "query_id": "a",
            "reference_id": str(i),
            "inchikey": key,
            "score": score,
            "q_value": 0.1,
            "is_decoy": False,
        }
        for i, (key, score) in enumerate([("X", 0.99), ("X", 0.98), ("A", 0.8)])
    ] + [
        {
            "query_id": "c",
            "reference_id": "bad",
            "inchikey": "Z",
            "score": 0.9,
            "q_value": 0.1,
            "is_decoy": False,
        }
    ]
    r = metrics.evaluate(queries, rows)
    assert r["counts"]["attempted"] == 4
    assert r["counts"]["eligible"] == 3
    assert r["counts"]["successfully_scored"] == 2
    assert r["closed_set"]["top1"] == {"numerator": 0, "denominator": 2, "value": 0.0}
    assert r["closed_set"]["top5"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert r["closed_set"]["retrieval_coverage"]["value"] == 0.5
    assert r["counts"]["false_matches"] == 2
    assert r["counts"]["missed_matches"] == 2
    assert r["open_set"]["false_assignment_rate"]["value"] == 1.0
    assert r["schema_version"] == 1


def test_metrics_do_not_invent_zero_precision() -> None:
    _, metrics, _ = _api()
    r = metrics.evaluate([], [])
    assert r["precision"]["value"] is None
    assert r["closed_set"]["top1"]["value"] is None


def test_metrics_reject_unattributable_results() -> None:
    _, metrics, _ = _api()
    with pytest.raises(ValueError, match="unknown query"):
        metrics.evaluate([], [{"query_id": "unseen", "score": 1.0, "is_decoy": False}])


def test_massbank_parser_preserves_real_source_metadata() -> None:
    dataset, _, _ = _api()
    r = dataset.read_record(FIXTURES / "MSBNK-Eawag-EA262601.txt")
    assert r.inchikey == "AAOVKJBEBIDNHE-UHFFFAOYSA-N"
    assert r.metadata["license"] == "CC BY"
    assert r.metadata["confidence"] == "standard compound"
    assert r.metadata["collision_energy"]
    assert r.mz.dtype.name == "float64"
    assert (
        r.source_sha256
        == "19e82b7350ee53ad2d482df6deb8bb380290dfcc58cd47bad6023cb0f6d6ec1a"
    )


def test_exact_duplicate_detection_is_intensity_scale_invariant() -> None:
    dataset, _, _ = _api()
    r = dataset.read_record(FIXTURES / "MSBNK-Eawag-EA262601.txt")
    changed = copy.deepcopy(r)
    changed.intensities = changed.intensities * 17
    assert dataset.peak_hash(r) == dataset.peak_hash(changed)
    changed.mz = changed.mz + 0.1
    assert dataset.peak_hash(r) != dataset.peak_hash(changed)


def test_manifest_hash_and_schema_changes_are_detected() -> None:
    dataset, _, _ = _api()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": {"sha256": "a" * 64},
        "selection": {"seed": 42},
        "queries": [],
    }
    digest = dataset.manifest_digest(manifest)
    dataset.verify_manifest(manifest, digest)
    manifest["selection"]["seed"] = 43
    with pytest.raises(ValueError, match="digest"):
        dataset.verify_manifest(manifest, digest)
    manifest["schema_version"] = 99
    with pytest.raises(ValueError, match="schema"):
        dataset.verify_manifest(manifest, dataset.manifest_digest(manifest))


def test_real_record_pipeline_present_and_absent(tmp_path: Path) -> None:
    dataset, _, runner = _api()
    query = dataset.read_record(FIXTURES / "MSBNK-Athens_Univ-AU111202.txt")
    reference = dataset.read_record(FIXTURES / "MSBNK-Eawag-EA258301.txt")
    other = dataset.read_record(FIXTURES / "MSBNK-Eawag-EA262601.txt")
    settings = {
        "algorithm": "cosine",
        "ms1_tolerance": 0.02,
        "ms2_tolerance": 0.02,
        "min_score": 0.6,
        "min_matched_peaks": 3,
        "fdr_threshold": 1.0,
        "enable_numba_prefilter": False,
    }
    present = runner.run_arm(
        [query], [reference, other], settings, tmp_path / "present", absent=set()
    )
    absent = runner.run_arm(
        [query], [other], settings, tmp_path / "absent", absent={query.connectivity}
    )
    assert present["metrics"]["counts"]["attempted"] == 1
    assert present["queries"][0]["state"] in {"scored", "processing_rejected"}
    assert (
        present["provenance"]["pipeline"] == "MassFlow.workflow.run_annotation_pipeline"
    )
    assert present["provenance"]["dependencies"]["matchms"]
    assert absent["queries"][0]["present"] is False
    assert (tmp_path / "present/queries.jsonl").exists()
    assert (tmp_path / "present/native/query_results.report.yaml").exists()
    assert present["calibration"]["empirical_calibration"] == "pending"
    assert (
        "no_independent_deployment_validation" in present["calibration"]["limitations"]
    )
    assert "unmerged_correctness_fixes" not in present["calibration"]["limitations"]
    assert (
        hashlib.sha256((tmp_path / "present/query.mgf").read_bytes()).hexdigest()
        == present["provenance"]["query_sha256"]
    )


def test_relative_output_path_is_resolved_before_truth_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    dataset, _, runner = _api()
    query = dataset.read_record(FIXTURES / "MSBNK-Athens_Univ-AU111202.txt")
    reference = dataset.read_record(FIXTURES / "MSBNK-Eawag-EA258301.txt")
    settings = {
        "algorithm": "cosine",
        "ms1_tolerance": 0.02,
        "ms2_tolerance": 0.02,
        "min_score": 0.6,
        "min_matched_peaks": 3,
        "fdr_threshold": 1.0,
        "enable_numba_prefilter": False,
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "_environment", lambda: {"pipeline": "test"})

    result = runner.run_arm(
        [query], [reference], settings, Path("relative-run"), absent=set()
    )

    effective_config = json.loads(
        (tmp_path / "relative-run" / "effective_config.json").read_text()
    )
    assert Path(effective_config["input"]["input_path"]).is_absolute()
    assert Path(effective_config["project"]["output_directory"]).is_absolute()
    assert result["truth_attribution"]["query_count"] == 1


def test_production_numpy_scalars_serialize_without_silent_nan(tmp_path: Path) -> None:
    import json
    import numpy as np

    _, _, runner = _api()
    runner._dump(
        tmp_path / "result.json",
        {"is_decoy": np.bool_(False), "score": np.float64(0.8)},
    )
    assert json.loads((tmp_path / "result.json").read_text()) == {
        "is_decoy": False,
        "score": 0.8,
    }
    with pytest.raises(ValueError):
        runner._dump(tmp_path / "invalid.json", {"score": float("nan")})


def test_leakage_audit_flags_copies_and_repeated_chemical_identity() -> None:
    dataset, _, _ = _api()
    record = dataset.read_record(FIXTURES / "MSBNK-Eawag-EA262601.txt")
    repeated = copy.deepcopy(record)
    repeated.accession = "second_real_record_derivative_for_unit_test"
    audit = dataset.audit_leakage([record], [record, repeated])
    assert len(audit["exact_cross_pairs"]) == 2
    assert audit["repeated_library_identities"][record.inchikey] == 2
    assert audit["within_library_exact_groups"]


def test_low_peak_real_query_is_not_hidden(tmp_path: Path) -> None:
    dataset, _, runner = _api()
    query = dataset.read_record(FIXTURES / "MSBNK-Athens_Univ-AU160801.txt")
    reference = dataset.read_record(FIXTURES / "MSBNK-Eawag-EA262601.txt")
    result = runner.run_arm(
        [query],
        [reference],
        {"algorithm": "cosine", "fdr_threshold": 1.0},
        tmp_path / "low-peaks",
        absent=set(),
    )
    assert result["metrics"]["counts"]["attempted"] == 1
    assert result["metrics"]["counts"]["eligible"] == 1
    assert result["queries"][0]["state"] == "processing_rejected"
    assert "minimum_peak_count" in result["queries"][0]["reasons"]
    assert result["metrics"]["counts"]["missed_matches"] == 1
    assert result["reference_accounting"]["attempted"] == 1
    assert (tmp_path / "low-peaks/references.jsonl").exists()
    assert result["provenance"]["harness_source_sha256"]


def test_frozen_settings_reject_tampering() -> None:
    _, _, runner = _api()
    parameters: dict[str, Any] = {"schema_version": 1, "similarity": {"min_score": 0.6}}
    digest = hashlib.sha256(
        __import__("json")
        .dumps(parameters, sort_keys=True, separators=(",", ":"))
        .encode()
    ).hexdigest()
    runner.verify_parameters(parameters, digest)
    parameters["similarity"]["min_score"] = 0.7
    with pytest.raises(ValueError, match="parameters"):
        runner.verify_parameters(parameters, digest)


def test_output_metric_schema_contract_is_versioned() -> None:
    import json

    _, metrics, _ = _api()
    schema = json.loads(
        (
            Path(__file__).parents[1] / "benchmarks/massbank-v1/metrics.schema.json"
        ).read_text()
    )
    result = metrics.evaluate([], [])
    assert (
        result["schema_version"] == schema["properties"]["schema_version"]["const"] == 1
    )
    assert set(result) == set(schema["required"])


def test_failed_family_cannot_be_counted_as_successful_retrieval() -> None:
    _, metrics, _ = _api()
    query = {
        "query_id": "q",
        "inchikey": "A",
        "connectivity": "A",
        "eligible": True,
        "present": True,
        "state": "pipeline_failed",
        "stratum": "one",
    }
    row = {"query_id": "q", "inchikey": "A", "score": 1.0, "is_decoy": False}
    result = metrics.evaluate([query], [row])
    assert result["closed_set"]["top1"]["value"] == 0.0
    assert result["closed_set"]["retrieval_coverage"]["value"] == 0.0
    assert result["counts"]["missed_matches"] == 1
