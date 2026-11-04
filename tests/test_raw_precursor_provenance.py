"""Raw precursor input-policy provenance and source identity regressions.

These are deterministic synthetic software fixtures, not identification-accuracy
or FDR calibration evidence. The existing finite guard must not be removed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from matchms import Spectrum

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
    raw_rejection_category,
)
from MassFlow.io import RawRejectionLedger, _validate_spectra_iterator
from MassFlow.workflow import run_annotation_pipeline

pytestmark = pytest.mark.core


def _spectrum(identifier: str, precursor_mz: object) -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 150.0, 200.0], dtype=np.float64),
        intensities=np.array([1.0, 2.0, 1.0], dtype=np.float64),
        metadata={"id": identifier, "precursor_mz": precursor_mz},
        metadata_harmonization=False,
    )


@pytest.mark.parametrize(
    ("value", "category"),
    [
        pytest.param(None, "precursor_missing", id="missing"),
        pytest.param(float("nan"), "precursor_nan", id="nan"),
        pytest.param(float("inf"), "precursor_positive_infinity", id="positive-inf"),
        pytest.param(float("-inf"), "precursor_negative_infinity", id="negative-inf"),
        pytest.param(0.0, "precursor_non_positive", id="zero"),
        pytest.param(-1.0, "precursor_non_positive", id="negative"),
        pytest.param("not-a-mass", "precursor_non_numeric", id="non-numeric"),
    ],
)
def test_invalid_precursors_have_distinct_raw_categories_and_source_ordinal(
    value: object, category: str
) -> None:
    source = Path("hypothetical.mgf")
    events: list[RejectionEvent] = []
    reasons: list[str] = []
    accepted = list(
        _validate_spectra_iterator(
            [_spectrum("accepted", 195.0), _spectrum("bad", value)],
            source,
            rejection_reporter=reasons.append,
            diagnostic_reporter=events.append,
        )
    )
    assert len(accepted) == 1
    assert len(events) == 1
    assert len(reasons) == 1
    event = events[0]
    assert event.stage is DiagnosticStage.INPUT_VALIDATION
    assert event.reason.value == category
    assert event.source_path == str(source)
    assert event.source_ordinal == 1
    assert event.spectrum_id == "bad"
    assert event.detail == reasons[0]


def test_finite_positive_precursor_is_retained() -> None:
    events: list[RejectionEvent] = []
    accepted = list(
        _validate_spectra_iterator(
            [_spectrum("positive", 195.088)],
            Path("positive.mgf"),
            diagnostic_reporter=events.append,
        )
    )
    assert len(accepted) == 1
    assert events == []


def test_rejection_ledger_retains_every_source_ordinal(tmp_path: Path) -> None:
    path = tmp_path / "raw-rejections.jsonl"
    specs = [_spectrum("good", 195.0)] + [
        _spectrum(f"invalid-{i}", float("nan")) for i in range(25)
    ]
    with RawRejectionLedger(path) as ledger:
        accepted = list(
            _validate_spectra_iterator(
                specs, Path("source.msp"), diagnostic_reporter=ledger.record
            )
        )
    assert len(accepted) == 1
    assert ledger.count == 25
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [item["source_ordinal"] for item in events] == list(range(1, 26))
    assert [item["spectrum_id"] for item in events] == [
        f"invalid-{i}" for i in range(25)
    ]
    assert all(item["reason"] == "precursor_nan" for item in events)
    assert all(item["source_path"] == "source.msp" for item in events)


def test_reference_reason_grouping_avoids_source_value_cardinality() -> None:
    for value in ("bad-1", "bad-2", "bad-3"):
        assert raw_rejection_category(
            RejectionEvent(
                stage=DiagnosticStage.INPUT_VALIDATION,
                reason=DiagnosticReason.PRECURSOR_NON_NUMERIC,
                detail=f"Non-numeric precursor_mz: {value}",
            )
        ) == "precursor_non_numeric"
    assert raw_rejection_category(
        RejectionEvent(
            stage=DiagnosticStage.INPUT_VALIDATION,
            reason=DiagnosticReason.INPUT_INVALID,
            detail="Duplicate fragment m/z values",
        )
    ) == "Duplicate fragment m/z values"


def _write_mgf(path: Path, precursor_values: list[tuple[str, str | None]]) -> None:
    lines: list[str] = []
    for name, precursor in precursor_values:
        lines.extend(["BEGIN IONS", f"TITLE={name}"])
        if precursor is not None:
            lines.append(f"PEPMASS={precursor}")
        lines.extend(["CHARGE=1+", "100.0 1.0", "150.0 2.0", "200.0 1.0", "END IONS"])
    path.write_text("\n".join(lines) + "\n")


def _config(tmp_path: Path, query: Path, library: Path) -> MassFlowConfig:
    return MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(input_path=query, library_path=library, format="mgf"),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(
            algorithm="cosine", min_matched_peaks=1, min_score=0.0, fdr_threshold=1.0
        ),
    )


def _read_report_and_ledger(report_path: Path) -> tuple[dict, list[dict]]:
    report = yaml.safe_load(report_path.read_text())
    provenance = report["raw_input_rejections"]
    ledger = Path(provenance["ledger_path"])
    assert ledger.is_file()
    assert provenance["source_sha256"] == hashlib.sha256(
        Path(provenance["source_path"]).read_bytes()
    ).hexdigest()
    assert provenance["ledger_sha256"] == hashlib.sha256(ledger.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == provenance["total"]
    return report, rows


def test_mixed_query_exports_rejection_reasons_and_record_ledger(
    tmp_path: Path,
) -> None:
    query = tmp_path / "mixed.mgf"
    library = tmp_path / "reference.mgf"
    _write_mgf(
        query,
        [("good", "195.088"), ("missing", None), ("zero", "0")],
    )
    _write_mgf(library, [("reference", "195.088")])
    result = run_annotation_pipeline(_config(tmp_path, query, library))[0]
    assert result.status in {"success", "degraded"}
    assert result.spectra_loaded == 1
    assert result.spectra_rejected == 2
    report, ledger = _read_report_and_ledger(
        tmp_path / "results" / "mixed_results.report.yaml"
    )
    assert report["raw_input_rejections"]["input_scan_complete"] is True
    assert report["raw_input_rejections"]["reason_counts"] == {
        "precursor_missing": 1,
        "precursor_non_positive": 1,
    }
    assert report["spectra_rejected"] == 2
    assert [row["source_ordinal"] for row in ledger] == [1, 2]
    assert (tmp_path / "results" / "mixed_results.csv").exists()


def test_all_invalid_query_exports_failure_and_ledger_without_csv(
    tmp_path: Path,
) -> None:
    query = tmp_path / "invalid.mgf"
    library = tmp_path / "reference.mgf"
    _write_mgf(query, [("missing", None), ("zero", "0")])
    _write_mgf(library, [("reference", "195.088")])
    result = run_annotation_pipeline(_config(tmp_path, query, library))[0]
    assert result.status == "failed"
    assert result.spectra_rejected == 2
    report, ledger = _read_report_and_ledger(
        tmp_path / "results" / "invalid_failed.report.yaml"
    )
    assert report["raw_input_rejections"]["input_scan_complete"] is True
    assert report["raw_input_rejections"]["reason_counts"] == {
        "precursor_missing": 1,
        "precursor_non_positive": 1,
    }
    assert len(ledger) == 2
    assert report["results_csv"] is None
    assert not (tmp_path / "results" / "invalid_results.csv").exists()
