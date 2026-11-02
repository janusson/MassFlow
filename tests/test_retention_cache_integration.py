"""Prepared caches from either predecessor must rebuild under combined semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    SimilarityConfig,
)
from MassFlow.library import open_library, prepare_library
from MassFlow.processing import metadata_processing
from MassFlow.similarity import SimilarityEngine
from MassFlow.storage import create_spectral_store

pytestmark = pytest.mark.core


@pytest.mark.parametrize("backend", ["sqlite", "zarr", "hybrid"])
@pytest.mark.parametrize("predecessor", ["main", "stabilization-v4", "retention-v1"])
def test_predecessor_cache_rebuilds_with_record_and_time_evidence_then_reuses(
    tmp_path: Path, backend: str, predecessor: str
) -> None:
    """Reusing either old contract would retain the stale record and lose evidence."""
    library_path = tmp_path / "library.msp"
    library_path.write_text(
        "Name: reference\nPrecursorMZ: 200\nRetentionTime: 2 min\n"
        "Num Peaks: 3\n50 1\n70 1\n90 1\n\n"
    )
    config = MassFlowConfig(
        input=InputConfig(
            input_path=library_path,
            library_path=library_path,
            storage_backend=backend,
        ),
        processing=ProcessingConfig(min_peaks=1),
    )
    # These payloads describe the released-main and two unmerged predecessor
    # contracts, independently of the combined fingerprint implementation.
    if predecessor == "main":
        payload = config.processing.model_dump_json()
    else:
        semantic_field = (
            {"processing_semantics_revision": 4}
            if predecessor == "stabilization-v4"
            else {"retention_time_semantics": 1}
        )
        payload = json.dumps(
            {"processing": config.processing.model_dump(mode="json"), **semantic_field},
            sort_keys=True,
            separators=(",", ":"),
        )
    old_fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    output = tmp_path / "results"
    output.mkdir()
    path = output / (
        "library_library.zarr" if backend == "zarr" else "library_library.db"
    )
    store = create_spectral_store(path, backend=backend)
    try:
        store.add_spectra(
            iter(
                [
                    Spectrum(
                        mz=np.array([50.0, 70.0, 90.0], dtype=np.float64),
                        intensities=np.ones(3, dtype=np.float64),
                        metadata={"id": "stale-record", "precursor_mz": 200.0},
                        metadata_harmonization=False,
                    )
                ]
            )
        )
    finally:
        store.close()
    source_stat = library_path.stat()
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source": str(library_path),
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_size": source_stat.st_size,
                "fingerprint": old_fingerprint,
                "backend": backend,
            }
        )
    )

    spec, count = prepare_library(config, output)
    assert count == 1
    assert json.loads(metadata_path.read_text())["fingerprint"] != old_fingerprint
    store = open_library(spec, config.processing)
    try:
        records = list(store.get_spectra())
    finally:
        store.close()
    assert len(records) == 1
    restored = records[0]
    assert restored.get("id") != "stale-record"
    assert restored.get("massflow_source_index") == 0
    assert (
        restored.get("massflow_source_sha256")
        == hashlib.sha256(library_path.read_bytes()).hexdigest()
    )
    assert restored.get("retention_time") == 120.0
    assert restored.get("retention_time_unit") == "seconds"
    assert restored.get("retention_time_source")["status"] == "normalized"

    metadata_before_reuse = metadata_path.read_bytes()
    store_mtime_before_reuse = spec.path.stat().st_mtime_ns
    reused, reused_count = prepare_library(config, output)
    assert reused == spec
    assert reused_count == 1
    assert metadata_path.read_bytes() == metadata_before_reuse
    assert reused.path.stat().st_mtime_ns == store_mtime_before_reuse


def test_minutes_gate_keeps_diagnostic_counts_and_ranks_only_eligible_records() -> None:
    """An early tied record outside 12 seconds must not consume the result slot."""
    spectra = []
    for record_id, seconds in [
        ("query", 120.0),
        ("a-late", 133.0),
        ("b", 130.0),
        ("c", 132.0),
    ]:
        spectrum = metadata_processing(
            Spectrum(
                mz=np.array([50.0, 70.0, 90.0], dtype=np.float64),
                intensities=np.ones(3, dtype=np.float64),
                metadata={
                    "id": record_id,
                    "precursor_mz": 200.0,
                    "retention_time": seconds,
                    "massflow_source_index": 0,
                },
                metadata_harmonization=False,
            )
        )
        assert spectrum is not None
        spectra.append(spectrum)
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm="cosine",
            min_score=0.0,
            min_matched_peaks=3,
            rt_tolerance=0.2,
            rt_tolerance_unit="minutes",
            collect_retrieval_diagnostics=True,
        )
    )
    rows = engine.search(spectra[:1], spectra[1:], include_decoys=False, top_n=1)
    assert [row["reference_id"] for row in rows] == ["b"]
    assert rows[0]["score"] == pytest.approx(1.0)
    metadata = rows[0]["reference_metadata"]
    assert metadata["massflow_source_index"] == 0
    assert metadata["retention_time_source"]["retention_time_seconds"] == 130.0
    counts = engine.retrieval_diagnostics["queries"]["query"]["target"]
    assert counts["rt_eligible_pairs"] == 2
    assert counts["eligible_before_top_n"] == 2
    assert counts["returned_by_search"] == 1
    assert counts["rejection_counts"] == {
        "rt_outside_tolerance": 1,
        "candidate_ranked_below_limit": 1,
    }
