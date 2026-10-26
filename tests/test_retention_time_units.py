"""Seconds-based retention-time contracts at real input and search boundaries."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import logging
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matchms import Spectrum
from pyteomics.auxiliary import unitfloat

from MassFlow import io, processing
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    SimilarityConfig,
)
from MassFlow.library import prepare_library
from MassFlow.similarity import SimilarityEngine, _sub_engine_config
from MassFlow.storage import create_spectral_store
from MassFlow.workflow import _build_classical_fallback_engine

pytestmark = pytest.mark.core


def _spectrum(metadata: dict[str, Any]) -> Spectrum:
    """Provide deterministic identical fragments without structural assumptions."""
    return Spectrum(
        mz=np.array([50.0, 70.0, 90.0], dtype=np.float64),
        intensities=np.ones(3, dtype=np.float64),
        metadata={"id": "query", "precursor_mz": 200.0, **metadata},
        metadata_harmonization=False,
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"rtinminutes": "2"},
        {"rtinseconds": "120"},
        {"retentiontime": "2 min"},
        {"rt": "120 sec"},
        {"scan_start_time": [unitfloat(2, "minute")]},
        {"retention_time": [unitfloat(2, "minute")]},
        {"retention_time": 2, "retention_time_unit": "minutes"},
        {"retention_time": [120]},
    ],
)
def test_processing_normalizes_explicit_time_without_mutating_source(
    metadata: dict[str, Any],
) -> None:
    spectrum = _spectrum(metadata)
    before = spectrum.metadata
    normalized = processing.metadata_processing(spectrum)
    assert normalized is not None
    assert normalized.get("retention_time") == 120.0
    assert normalized.get("retention_time_unit") == "seconds"
    assert normalized.get("retention_time_source")
    assert spectrum.metadata == before
    repeated = processing.metadata_processing(normalized)
    assert repeated is not None
    assert repeated.metadata == normalized.metadata


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"retention_time": "CCS: 123"},
        {"retention_time": float("nan")},
        {"retention_time": float("inf")},
        {"retention_time": -1},
        {"retention_time": [1, 2]},
        {"retention_time": True},
        {"retention_time": 2, "retention_time_unit": "unknown"},
        {"scan_start_time": [2]},
        {"retention_time": "2 min", "retention_time_unit": "seconds", "rtinminutes": 2},
        {"rtinminutes": "2 sec", "rtinseconds": 2},
        {"rtinseconds": 120, "rtinminutes": 3},
        {"retention_time": 120, "retention_time_unit": "seconds", "rtinminutes": 3},
    ],
)
def test_unknown_or_ambiguous_time_stays_missing(metadata: dict[str, Any]) -> None:
    normalized = processing.metadata_processing(_spectrum(metadata))
    assert normalized is not None
    assert normalized.get("retention_time") is None


def test_zero_seconds_is_observed_time() -> None:
    normalized = processing.metadata_processing(_spectrum({"rtinminutes": "0"}))
    assert normalized is not None
    assert normalized.get("retention_time") == 0.0


@pytest.mark.parametrize(
    "extension,field", [("mgf", "RTINMINUTES=2"), ("msp", "RetentionTime: 2 min")]
)
def test_real_text_import_normalizes_minutes(
    tmp_path: Path, extension: str, field: str
) -> None:
    path = tmp_path / f"query.{extension}"
    if extension == "mgf":
        text = f"BEGIN IONS\nTITLE=query\nPEPMASS=200\n{field}\n50 1\n70 1\n90 1\nEND IONS\n"
    else:
        text = f"Name: query\nPrecursorMZ: 200\n{field}\nNum Peaks: 3\n50 1\n70 1\n90 1\n\n"
    path.write_text(text)
    normalized = processing.metadata_processing(next(io.load_spectra(path)))
    assert normalized is not None
    assert normalized.get("retention_time") == 120.0
    assert normalized.get("retention_time_unit") == "seconds"


def _write_xml(path: Path, kind: str) -> None:
    """Write real, minimal MS2 records with independently specified units."""
    if kind == "mzml":
        arrays = []
        for values, accession, name in [
            ([50.0, 70.0, 90.0], "MS:1000514", "m/z array"),
            ([1.0, 1.0, 1.0], "MS:1000515", "intensity array"),
        ]:
            encoded = base64.b64encode(struct.pack("<3d", *values)).decode()
            arrays.append(
                f'<binaryDataArray encodedLength="{len(encoded)}"><cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/><cvParam cvRef="MS" accession="MS:1000576" name="no compression"/><cvParam cvRef="MS" accession="{accession}" name="{name}"/><binary>{encoded}</binary></binaryDataArray>'
            )
        text = f"""<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0"><run id="run"><spectrumList count="1"><spectrum id="scan=1" index="0" defaultArrayLength="3"><cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2"/><scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="2" unitName="minute" unitAccession="UO:0000031" unitCvRef="UO"/></scan></scanList><precursorList count="1"><precursor><selectedIonList count="1"><selectedIon><cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="200"/></selectedIon></selectedIonList></precursor></precursorList><binaryDataArrayList count="2">{"".join(arrays)}</binaryDataArrayList></spectrum></spectrumList></run></mzML>"""
    else:
        encoded = base64.b64encode(struct.pack(">6d", 50, 1, 70, 1, 90, 1)).decode()
        text = f"""<mzXML><msRun scanCount="1"><scan num="1" msLevel="2" peaksCount="3" retentionTime="PT120S" polarity="+"><precursorMz precursorCharge="1">200</precursorMz><peaks precision="64" byteOrder="network" pairOrder="m/z-int">{encoded}</peaks></scan></msRun></mzXML>"""
    path.write_text(text)


@pytest.mark.parametrize("kind", ["mzml", "mzxml"])
def test_real_xml_import_preserves_time_units(tmp_path: Path, kind: str) -> None:
    path = tmp_path / f"query.{kind}"
    _write_xml(path, kind)
    normalized = processing.metadata_processing(next(io.load_spectra(path)))
    assert normalized is not None
    assert normalized.get("retention_time") == 120.0
    assert normalized.get("retention_time_unit") == "seconds"
    source = normalized.get("retention_time_source")
    assert source["inputs"][0]["unit"] == "minute"
    if kind == "mzxml":
        assert source["original"]["value"] == "PT120S"
        assert source["original"]["field"] == "retentionTime"
        assert source["input_origin"] == "importer_metadata"


@pytest.mark.parametrize(
    "difference_seconds,expected_hits", [(10.0, 1), (12.0, 1), (13.0, 0)]
)
def test_minute_tolerance_has_hand_computed_second_boundary(
    difference_seconds: float,
    expected_hits: int,
) -> None:
    config = SimilarityConfig(
        algorithm="cosine",
        min_score=0.0,
        min_matched_peaks=3,
        rt_tolerance=0.2,
        rt_tolerance_unit="minutes",
    )
    results = SimilarityEngine(config).search(
        query_spectra=[_spectrum({"retention_time": 120.0})],
        reference_spectra=[
            _spectrum({"id": "reference", "retention_time": 120.0 + difference_seconds})
        ],
        include_decoys=False,
    )
    assert len(results) == expected_hits
    if results:
        assert results[0]["score"] == pytest.approx(1.0)
        assert results[0]["matched_peaks"] == 3


def test_legacy_tolerance_keeps_seconds_with_migration_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="MassFlow.config"):
        config = SimilarityConfig(rt_tolerance=0.2)
    assert config.rt_tolerance_seconds == 0.2
    assert "seconds" in caplog.text and "rt_tolerance_unit" in caplog.text
    assert config.model_dump()["rt_tolerance_unit"] == "seconds"


def test_explicit_tolerance_unit_is_forwarded_to_child_and_fallback() -> None:
    config = SimilarityConfig(rt_tolerance=0.2, rt_tolerance_unit="minutes")
    assert _sub_engine_config(config, "cosine").rt_tolerance_seconds == 12.0
    assert _build_classical_fallback_engine(config).config.rt_tolerance_seconds == 12.0


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf")])
def test_nonfinite_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError):
        SimilarityConfig(rt_tolerance=tolerance, rt_tolerance_unit="seconds")


@pytest.mark.parametrize("backend", ["sqlite", "zarr", "hybrid"])
def test_normalized_time_provenance_survives_store_roundtrip(
    tmp_path: Path, backend: str
) -> None:
    spectrum = processing.metadata_processing(
        _spectrum({"scan_start_time": [unitfloat(2, "minute")]})
    )
    assert spectrum is not None
    path = tmp_path / ("library.zarr" if backend == "zarr" else "library.db")
    store = create_spectral_store(path, backend=backend)
    try:
        store.add_spectra(iter([spectrum]))
        restored = next(store.get_spectra())
        normalized = processing.metadata_processing(restored)
        assert normalized is not None
        assert normalized.get("retention_time") == 120.0
        assert normalized.get("retention_time_source") == spectrum.get(
            "retention_time_source"
        )
    finally:
        store.close()


def test_exports_include_seconds_and_original_time_metadata(tmp_path: Path) -> None:
    spectrum = processing.metadata_processing(_spectrum({"rtinminutes": "2"}))
    assert spectrum is not None
    path = tmp_path / "results.csv"
    io.save_match_results([], path, query_spectra=[spectrum])
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["query_retention_time"] == "120.0"  # legacy column retained
    assert row["query_retention_time_seconds"] == "120.0"
    assert row["query_retention_time_unit"] == "seconds"
    assert json.loads(row["query_retention_time_source"])["inputs"][0]["value"] == "2"


def test_old_managed_time_cache_rebuilds_once(tmp_path: Path) -> None:
    library = tmp_path / "library.msp"
    library.write_text(
        "Name: reference\nPrecursorMZ: 200\nRetentionTime: 2 min\nNum Peaks: 3\n50 1\n70 1\n90 1\n\n"
    )
    config = MassFlowConfig(
        input=InputConfig(input_path=library, library_path=library),
        processing=ProcessingConfig(min_peaks=1),
    )
    output = tmp_path / "results"
    spec, _ = prepare_library(config, output)
    cache_metadata_path = spec.path.with_suffix(spec.path.suffix + ".meta.json")
    metadata = json.loads(cache_metadata_path.read_text())
    metadata["fingerprint"] = hashlib.sha256(
        config.processing.model_dump_json().encode()
    ).hexdigest()
    cache_metadata_path.write_text(json.dumps(metadata))
    first_mtime = spec.path.stat().st_mtime_ns
    rebuilt, _ = prepare_library(config, output)
    assert rebuilt.path.stat().st_mtime_ns > first_mtime
    store = create_spectral_store(rebuilt.path, backend="sqlite")
    try:
        assert next(store.get_spectra()).get("retention_time") == 120.0
    finally:
        store.close()
    rebuilt_mtime = rebuilt.path.stat().st_mtime_ns
    reused, _ = prepare_library(config, output)
    assert reused.path.stat().st_mtime_ns == rebuilt_mtime


def test_explicit_seconds_and_disabled_gate_do_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="MassFlow.config"):
        explicit = SimilarityConfig(rt_tolerance=12.0, rt_tolerance_unit="seconds")
        disabled = SimilarityConfig()
    assert explicit.rt_tolerance_seconds == 12.0
    assert disabled.rt_tolerance_seconds is None
    assert "rt_tolerance_unit" not in caplog.text


def test_missing_time_bypasses_gate_without_becoming_zero() -> None:
    config = SimilarityConfig(
        algorithm="cosine",
        min_score=0.0,
        min_matched_peaks=3,
        rt_tolerance=0.0,
        rt_tolerance_unit="seconds",
    )
    results = SimilarityEngine(config).search(
        query_spectra=[_spectrum({"retention_time": "invalid"})],
        reference_spectra=[_spectrum({"id": "reference", "retention_time": 120.0})],
        include_decoys=False,
    )
    assert len(results) == 1


def test_time_normalization_preserves_real_precursor_physics_gate() -> None:
    valid = _spectrum(
        {
            "formula": "C8H10N4O2",
            "precursor_mz": 195.087652,
            "charge": 1,
            "adduct": "[M+H]+",
            "rtinminutes": "2",
        }
    )
    invalid = _spectrum(
        {
            "formula": "C8H10N4O2",
            "precursor_mz": 195.097652,
            "charge": 1,
            "adduct": "[M+H]+",
            "rtinminutes": "2",
        }
    )
    normalized_valid = processing.metadata_processing(valid)
    normalized_invalid = processing.metadata_processing(invalid)
    assert normalized_valid is not None and normalized_invalid is not None
    assert normalized_valid.get("retention_time") == 120.0
    assert processing.physical_integrity_reason(normalized_valid) is None
    assert processing.physical_integrity_reason(normalized_invalid) is not None


def test_streaming_inherits_explicit_tolerance_unit(tmp_path: Path) -> None:
    from MassFlow.streaming.engine import StreamingEngine

    config = MassFlowConfig(
        input=InputConfig(input_path=tmp_path / "query.mgf"),
        similarity=SimilarityConfig(rt_tolerance=0.2, rt_tolerance_unit="minutes"),
    )
    engine = StreamingEngine(config, [_spectrum({"id": "reference"})])
    assert engine._engine.config.rt_tolerance_seconds == 12.0


def test_annotation_reports_effective_time_unit(tmp_path: Path) -> None:
    import yaml
    from MassFlow.workflow import run_annotation_pipeline

    query = tmp_path / "query.mgf"
    query.write_text(
        "BEGIN IONS\nTITLE=query\nPEPMASS=200\nRTINMINUTES=2\n50 1\n70 1\n90 1\nEND IONS\n"
    )
    library = tmp_path / "library.msp"
    library.write_text(
        "Name: reference\nPrecursorMZ: 200\nRetentionTime: 130 sec\nNum Peaks: 3\n50 1\n70 1\n90 1\n\n"
    )
    config = MassFlowConfig(
        project={"output_directory": tmp_path / "results"},
        input=InputConfig(input_path=query, library_path=library),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(
            algorithm="cosine",
            min_matched_peaks=3,
            min_score=0.0,
            fdr_threshold=1.0,
            rt_tolerance=0.2,
            rt_tolerance_unit="minutes",
        ),
    )
    result = run_annotation_pipeline(config)[0]
    assert result.status != "failed", result.fatal_errors
    assert result.hits_produced == 1
    assert result.output_path is not None
    with result.output_path.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["query_retention_time_seconds"] == "120.0"
    assert row["reference_name"] == "reference"
    report_path = result.output_path.with_suffix(".report.yaml")
    report = yaml.safe_load(report_path.read_text())
    assert report["retention_time"] == {
        "unit": "seconds",
        "tolerance_value": 0.2,
        "tolerance_unit": "minutes",
        "tolerance_seconds": 12.0,
    }


def test_conflicting_units_preserve_both_claims() -> None:
    normalized = processing.metadata_processing(
        _spectrum(
            {
                "retention_time": "2 min",
                "retention_time_unit": "seconds",
                "rtinminutes": 2,
            }
        )
    )
    assert normalized is not None
    assert normalized.get("retention_time") is None
    source = normalized.get("retention_time_source")
    assert source["status"] == "conflicting"
    assert source["inputs"][0]["declared_unit"] == "seconds"
    assert source["inputs"][0]["intrinsic_unit"] == "min"
    repeated = processing.metadata_processing(normalized)
    assert repeated is not None
    assert repeated.get("retention_time_source") == source


def test_mzxml_original_duration_joins_by_scan_id(tmp_path: Path) -> None:
    path = tmp_path / "query.mzxml"
    _write_xml(path, "mzxml")
    text = path.read_text()
    start = text.index("<scan ")
    end = text.index("</scan>") + len("</scan>")
    other_scan = (
        text[start:end].replace('num="1"', 'num="2"').replace("PT120S", "PT180S")
    )
    path.write_text(text[:end] + other_scan + text[end:])
    spectra = [
        processing.metadata_processing(spectrum) for spectrum in io.load_spectra(path)
    ]
    assert all(spectrum is not None for spectrum in spectra)
    assert [
        (
            spectrum.get("scan_number"),
            spectrum.get("retention_time"),
            spectrum.get("retention_time_source")["original"]["value"],
        )
        for spectrum in spectra
        if spectrum is not None
    ] == [("1", 120.0, "PT120S"), ("2", 180.0, "PT180S")]
