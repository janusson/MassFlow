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
