"""Regression tests for Issue #124: benchmark polarity, adduct, and charge fidelity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pyteomics import mass

from MassFlow.cheminformatics import compute_adduct_offset
from MassFlow.config import ProcessingConfig, SimilarityConfig
from MassFlow.io import load_spectra
from MassFlow.processing import metadata_processing
from MassFlow.similarity import SimilarityEngine
from scripts.benchmark_real_corpus import (
    _assess_source_ion_metadata,
    _resolve_source_ion_metadata,
    build_real_benchmark,
    fmt_mgf,
    fmt_msp,
    parse_mgf,
    parse_msp,
    verify_written_spectra,
)


def _caffeine_mz(adduct: str) -> float:
    neutral = float(mass.calculate_mass(formula="C8H10N4O2"))
    offset = compute_adduct_offset(adduct)
    assert offset is not None
    return neutral + offset


def _msp_block(
    *,
    name: str,
    inchikey: str,
    precursor_mz: float,
    peaks: list[tuple[float, float]],
    ion_mode: str | None = None,
    precursor_type: str | None = None,
    charge: str | None = None,
) -> str:
    lines = [
        f"Name: {name}",
        f"InChIKey: {inchikey}",
        f"PrecursorMZ: {precursor_mz:.6f}",
    ]
    if ion_mode is not None:
        lines.append(f"Ion_mode: {ion_mode}")
    if precursor_type is not None:
        lines.append(f"Precursor_type: {precursor_type}")
    if charge is not None:
        lines.append(f"Charge: {charge}")
    lines.append(f"Num Peaks: {len(peaks)}")
    lines.extend(f"{mz:.5f} {intensity:.6g}" for mz, intensity in peaks)
    return "\n".join(lines) + "\n\n"


def test_fmt_msp_and_mgf_preserve_positive_negative_multicharge_and_missing() -> None:
    """Writer must preserve source polarity/adduct/charge and never fabricate Positive/1."""
    cases: list[tuple[dict[str, str], dict[str, Any]]] = [
        (
            {"ion_mode": "P", "precursor_type": "[M+H]+", "precursormz": "195.08765"},
            {"ionmode": "Positive", "adduct": "[M+H]+", "charge": None},
        ),
        (
            {"ion_mode": "N", "precursor_type": "[M-H]-", "precursormz": "193.07310"},
            {"ionmode": "Negative", "adduct": "[M-H]-", "charge": None},
        ),
        (
            {
                "ionmode": "positive",
                "adduct": "[M+2H]2+",
                "charge": "2+",
                "precursormz": "98.04746",
            },
            {"ionmode": "Positive", "adduct": "[M+2H]2+", "charge": "2+"},
        ),
        (
            {"precursormz": "300.12345"},
            {"ionmode": None, "adduct": None, "charge": None},
        ),
    ]
    peaks = [(100.0, 100.0), (150.0, 50.0)]

    for idx, (meta, expected) in enumerate(cases):
        record = {"meta": meta, "peaks": peaks}
        msp_text = fmt_msp(record, f"LIB_{idx:03d}")
        mgf_text = fmt_mgf(record, f"Q_{idx:03d}")

        assert _resolve_source_ion_metadata(meta) == expected
        if expected["ionmode"] != "Positive":
            assert "IONMODE: Positive" not in msp_text
            assert "IONMODE=Positive" not in mgf_text
        if expected["charge"] is None:
            assert "CHARGE:" not in msp_text
            assert "CHARGE=" not in mgf_text


def test_verify_written_spectra_detects_ion_metadata_mismatch(tmp_path: Path) -> None:
    """Round-trip check rejects files whose ionmode/adduct/charge diverge from source."""
    peaks = [(50.0 + i, float(10 * (i + 1))) for i in range(10)]
    source = [
        {
            "meta": {
                "ion_mode": "N",
                "precursor_type": "[M-H]-",
                "precursormz": "193.07310",
            },
            "peaks": peaks,
        }
    ]
    corrupted = tmp_path / "corrupted.msp"
    corrupted.write_text(
        fmt_msp(
            {
                "meta": {
                    "ion_mode": "P",
                    "precursor_type": "[M+H]+",
                    "precursormz": "195.08765",
                },
                "peaks": peaks,
            },
            "LIB_000",
        )
    )

    with pytest.raises(AssertionError, match="round-trip fidelity"):
        verify_written_spectra(corrupted, "msp", source)


def test_build_real_benchmark_preserves_polarity_and_supports_stratification(
    tmp_path: Path,
) -> None:
    """build_real_benchmark preserves mixed polarities and supports --polarity filters."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    lib_file = corpus_dir / "lib.msp"
    query_file = corpus_dir / "queries.msp"

    pos_mz = _caffeine_mz("[M+H]+")
    neg_mz = _caffeine_mz("[M-H]-")
    caffeine_ik = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
    neg_only_ik = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    # Provide >= 10 distinct intensity values per spectrum for verify_written_spectra.
    pos_lib_peaks = [(100.0 + i, float(10 + i)) for i in range(10)]
    neg_lib_peaks = [(100.0 + i, float(30 + i)) for i in range(10)]
    pos_q_peaks = [(100.0 + i, float(100 + i)) for i in range(10)]
    neg_q_peaks = [(100.0 + i, float(300 + i)) for i in range(10)]

    lib_file.write_text(
        _msp_block(
            name="CaffeinePos",
            inchikey=caffeine_ik,
            precursor_mz=pos_mz,
            ion_mode="P",
            precursor_type="[M+H]+",
            peaks=pos_lib_peaks,
        )
        + _msp_block(
            name="AspirinNeg",
            inchikey=neg_only_ik,
            precursor_mz=neg_mz,
            ion_mode="N",
            precursor_type="[M-H]-",
            peaks=neg_lib_peaks,
        )
    )
    query_file.write_text(
        _msp_block(
            name="CaffeinePosQ",
            inchikey=caffeine_ik,
            precursor_mz=pos_mz,
            ion_mode="P",
            precursor_type="[M+H]+",
            peaks=pos_q_peaks,
        )
        + _msp_block(
            name="AspirinNegQ",
            inchikey=neg_only_ik,
            precursor_mz=neg_mz,
            ion_mode="N",
            precursor_type="[M-H]-",
            peaks=neg_q_peaks,
        )
    )

    workdir_all = tmp_path / "work_all"
    built = build_real_benchmark(
        corpus=corpus_dir,
        workdir=workdir_all,
        library_file="lib.msp",
        query_files=["queries.msp"],
        max_queries=10,
        seed=20260921,
        polarity="all",
    )

    assert built["stats"]["library_write_check"]["ionmode_counts"] == {
        "Negative": 1,
        "Positive": 1,
    }
    assert built["stats"]["queries_write_check"]["ionmode_counts"] == {
        "Negative": 1,
        "Positive": 1,
    }
    assert built["stats"]["library_write_check"]["adduct_present_count"] == 2
    assert built["stats"]["queries_write_check"]["adduct_present_count"] == 2
    assert set(built["stats"]["source_files"]) == {"lib.msp", "queries.msp"}

    reparsed_lib = parse_msp(workdir_all / "library.msp")
    reparsed_queries = parse_mgf(workdir_all / "queries.mgf")
    assert {r["meta"]["ionmode"] for r in reparsed_lib} == {"Positive", "Negative"}
    assert {r["meta"]["ionmode"] for r in reparsed_queries} == {"Positive", "Negative"}
    assert {r["meta"]["adduct"] for r in reparsed_queries} == {"[M+H]+", "[M-H]-"}

    # Verify MassFlow loads the rewritten files with intact polarity/adduct metadata
    # and keeps positive and negative spectra separated during similarity search.
    proc_cfg = ProcessingConfig(
        min_peaks=1,
        noise_threshold=0.0,
        filter_by_mz=False,
        filter_by_intensity=False,
    )
    loaded_lib = [
        s
        for raw in load_spectra(workdir_all / "library.msp")
        if (s := metadata_processing(raw, proc_cfg)) is not None
    ]
    loaded_queries = [
        s
        for raw in load_spectra(workdir_all / "queries.mgf")
        if (s := metadata_processing(raw, proc_cfg)) is not None
    ]
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=5.0,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
        )
    )
    hits = engine.search(loaded_queries, loaded_lib, include_decoys=False)
    assert len(hits) == 2
    by_query = {hit["query_id"]: hit["reference_id"] for hit in hits}
    pos_q = next(q for q in loaded_queries if q.get("ionmode") == "positive")
    neg_q = next(q for q in loaded_queries if q.get("ionmode") == "negative")
    pos_ref = next(r for r in loaded_lib if r.get("ionmode") == "positive")
    neg_ref = next(r for r in loaded_lib if r.get("ionmode") == "negative")
    assert by_query[pos_q.get("id")] == pos_ref.get("id")
    assert by_query[neg_q.get("id")] == neg_ref.get("id")

    workdir_neg = tmp_path / "work_neg"
    built_neg = build_real_benchmark(
        corpus=corpus_dir,
        workdir=workdir_neg,
        library_file="lib.msp",
        query_files=["queries.msp"],
        max_queries=10,
        seed=20260921,
        polarity="negative",
    )
    assert built_neg["stats"]["library_records"] == 1
    assert built_neg["stats"]["library_write_check"]["ionmode_counts"] == {
        "Negative": 1
    }
    assert built_neg["stats"]["queries_write_check"]["ionmode_counts"] == {
        "Negative": 1
    }


def test_assess_source_ion_metadata_flags_conflicts() -> None:
    ltq_conflict = _assess_source_ion_metadata(
        {"ion_mode": "P", "precursor_type": "M-H", "charge": "1+"}
    )
    assert ltq_conflict["status"] == "conflicting"
    assert ltq_conflict["conflict_reason"] == "ionmode Positive contradicts adduct M-H"

    qtof_conflict = _assess_source_ion_metadata(
        {"ion_mode": "N", "precursor_type": "[M+Na]+"}
    )
    assert qtof_conflict["status"] == "conflicting"
    assert (
        qtof_conflict["conflict_reason"]
        == "ionmode Negative contradicts adduct [M+Na]+"
    )

    charge_conflict = _assess_source_ion_metadata(
        {"ion_mode": "P", "precursor_type": "[M+H]+", "charge": "1-"}
    )
    assert charge_conflict["status"] == "conflicting"
    assert (
        charge_conflict["conflict_reason"] == "ionmode Positive contradicts charge 1-"
    )

    clean = _assess_source_ion_metadata(
        {"ion_mode": "P", "precursor_type": "[M+H]+", "charge": "1+"}
    )
    assert clean["status"] == "consistent"
    assert clean["conflict_reason"] is None


def test_build_real_benchmark_ion_conflict_policy_mark_vs_exclude(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    lib_ik = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
    caff_pos_mz = _caffeine_mz("[M+H]+")
    lib_peaks = [(100.0 + i, 100.0 + i * 37.0) for i in range(12)]
    valid_q_peaks = [(100.0 + i, 200.0 + i * 19.0) for i in range(12)]
    corrupt_q_peaks = [(100.0 + i, 300.0 + i * 23.0) for i in range(12)]

    (corpus_dir / "lib.msp").write_text(
        _msp_block(
            name="CaffeineRefPos",
            inchikey=lib_ik,
            precursor_mz=caff_pos_mz,
            ion_mode="P",
            precursor_type="[M+H]+",
            peaks=lib_peaks,
        )
    )
    (corpus_dir / "queries.msp").write_text(
        _msp_block(
            name="CaffeineValidPos",
            inchikey=lib_ik,
            precursor_mz=caff_pos_mz,
            ion_mode="P",
            precursor_type="[M+H]+",
            peaks=valid_q_peaks,
        )
        + _msp_block(
            name="CaffeineCorruptLTQ",
            inchikey=lib_ik,
            precursor_mz=caff_pos_mz,
            ion_mode="P",
            precursor_type="M-H",
            peaks=corrupt_q_peaks,
        )
    )

    marked = build_real_benchmark(
        corpus=corpus_dir,
        workdir=tmp_path / "mark",
        library_file="lib.msp",
        query_files=["queries.msp"],
        max_queries=10,
        seed=20260921,
        polarity="positive",
        ion_conflict_policy="mark",
    )
    assert marked["stats"]["ion_conflict_policy"] == "mark"
    assert marked["stats"]["query_sources"]["queries.msp"]["selected"] == 2
    assert (
        marked["stats"]["query_sources"]["queries.msp"][
            "rejected_conflicting_ion_metadata"
        ]
        == 0
    )
    audit_marked = marked["stats"]["source_ion_conflict_audit"]["query_sources"][
        "queries.msp"
    ]
    assert audit_marked["status_counts"] == {"conflicting": 1, "consistent": 1}
    assert audit_marked["excluded_conflicting"] == 0
    statuses = {
        meta["ion_metadata_status"]
        for meta in marked["stats"]["selected_source_metadata"].values()
    }
    assert statuses == {"consistent", "conflicting"}

    excluded = build_real_benchmark(
        corpus=corpus_dir,
        workdir=tmp_path / "exclude",
        library_file="lib.msp",
        query_files=["queries.msp"],
        max_queries=10,
        seed=20260921,
        polarity="positive",
        ion_conflict_policy="exclude",
    )
    assert excluded["stats"]["ion_conflict_policy"] == "exclude"
    assert excluded["stats"]["query_sources"]["queries.msp"]["selected"] == 1
    assert (
        excluded["stats"]["query_sources"]["queries.msp"][
            "rejected_conflicting_ion_metadata"
        ]
        == 1
    )
    audit_excluded = excluded["stats"]["source_ion_conflict_audit"]["query_sources"][
        "queries.msp"
    ]
    assert audit_excluded["status_counts"] == {"conflicting": 1, "consistent": 1}
    assert audit_excluded["excluded_conflicting"] == 1

    with pytest.raises(ValueError, match="Unsupported ion_conflict_policy"):
        build_real_benchmark(
            corpus=corpus_dir,
            workdir=tmp_path / "bad_policy",
            library_file="lib.msp",
            query_files=["queries.msp"],
            max_queries=10,
            seed=20260921,
            ion_conflict_policy="drop",
        )
