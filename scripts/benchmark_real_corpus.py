#!/usr/bin/env python3
"""Real-data, objectively-labelled reference-library retrieval benchmark.

Replaces the synthetic benchmark's biggest weakness (made-up spectra) with real
acquired spectra and labels that come from the data itself rather than from
construction.

**Design: cross-instrument reference-library retrieval.**
The corpus contains the same Vaniya-Fiehn natural-products library acquired on
three different instruments (LTQ, QTOF, QExactive). Using one instrument as the
reference library and another as the queries is a genuine cross-instrument
library-matching task — the same design used in the Spec2Vec evaluation — and
the label is objective: a hit is correct iff the query's and the hit's InChIKey
connectivity block agree. This evaluates retrieval from a known reference
library; it is not identification of compounds in an unknown sample.

Why InChIKey rather than "did it return the same record": the same compound
appears as several records per library, and MS/MS generally cannot resolve
stereochemistry, so the connectivity block (first 14 characters) is the
defensible identity criterion. The full key is also reported.

Identical spectra are excluded from the query set (a query whose peak list
already exists in the library would be a trivial self-match). The excluded
fraction is reported — it quantifies how much of the corpus is duplicated.

Usage
-----
::

    uv run python scripts/benchmark_real_corpus.py \
        --corpus ~/Projects/MassFlow/assets/spectral-libraries/msp

Nothing is written into the repository; all artefacts go to ``--workdir``.
"""

from __future__ import annotations

import argparse
import csv as _csv
from collections import Counter
import hashlib
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_reporting import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    attribute_truth_candidates,
    compare_benchmark_reports,
    compare_result_rows,
    evaluate_query_retrieval,
    summarize_diagnostics_by_source,
)

DEFAULT_CORPUS = Path.home() / "Projects/MassFlow/assets/spectral-libraries/msp"

# Files with (near-)complete InChIKey coverage, all real acquisitions.
DEFAULT_LIBRARY = "MoNA-export-VF-NPL_QExactive.msp"
DEFAULT_QUERIES = ["MoNA-export-VF-NPL_QTOF.msp", "MoNA-export-VF-NPL_LTQ.msp"]


# ---------------------------------------------------------------------------
# MSP parsing (deliberately independent of matchms, so a matchms bug cannot
# silently shape the benchmark data)
# ---------------------------------------------------------------------------

_META_KEYS = {
    "name",
    "id",
    "precursormz",
    "precursor_mz",
    "precursor",
    "charge",
    "ionmode",
    "ion_mode",
    "adduct",
    "adductionname",
    "rt",
    "retentiontime",
    "retention_time",
    "num peaks",
    "num_peaks",
    "comment",
    "smiles",
    "inchi",
    "inchikey",
    "formula",
    "links",
    "instrument",
    "instrumenttype",
    "db#",
    "synon",
    "cas",
    "mw",
    "exactmass",
    "precursortype",
    "collisionenergy",
}


def parse_msp(path: Path) -> list[dict[str, Any]]:
    """Stream an MSP file into records: metadata + peaks.

    Tolerant of the three key spellings present in this corpus
    (``NAME:``/``Name:``/``NAME :``) because that variability is real.
    """
    records: list[dict[str, Any]] = []
    meta: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []

    def flush() -> None:
        if peaks and meta:
            # NB: copy. Appending the live dict/list and then clearing them
            # would leave every record aliasing one emptied structure.
            records.append({"meta": dict(meta), "peaks": list(peaks)})
        meta.clear()
        peaks.clear()

    with path.open(errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if ":" in line or "=" in line:
                sep = ":" if ":" in line else "="
                key, _, value = line.partition(sep)
                k = key.strip().lower()
                if k in _META_KEYS or (
                    not k.replace(" ", "").replace("#", "").isalpha()
                ):
                    # metadata line
                    if k in {"name"} and peaks:
                        flush()
                    meta[k] = value.strip()
                    continue
            # peak line: "mz intensity" or "mz:intensity" or "mz\tintensity"
            parts = line.replace(":", " ").replace("\t", " ").split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    flush()
    return records


def parse_mgf(path: Path) -> list[dict[str, Any]]:
    """Stream an MGF file into records: metadata + peaks.

    Mirrors :func:`parse_msp` so the harness can round-trip its own output.
    """
    records: list[dict[str, Any]] = []
    meta: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    in_ions = False

    def flush() -> None:
        if peaks and meta:
            records.append({"meta": dict(meta), "peaks": list(peaks)})
        meta.clear()
        peaks.clear()

    with path.open(errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("BEGIN IONS"):
                in_ions = True
                continue
            if line.upper().startswith("END IONS"):
                in_ions = False
                flush()
                continue
            if not in_ions:
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key.strip().lower()] = value.strip()
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    flush()
    return records


def inchikey_of(meta: dict[str, str]) -> Optional[str]:
    """Return a usable InChIKey or None (rejects N/A and malformed values)."""
    raw = meta.get("inchikey", "").strip()
    if not raw or raw.upper() in {"N/A", "NA", "NONE", "NULL", "-"}:
        return None
    if raw.count("-") != 2 or len(raw) < 27:
        return None
    return raw


def precursor_of(meta: dict[str, str]) -> Optional[float]:
    for key in ("precursormz", "precursor_mz", "precursor"):
        if key in meta:
            try:
                value = float(meta[key].split()[0])
            except (ValueError, IndexError):
                continue
            if value > 0:
                return value
    return None


def peak_hash(peaks: Iterable[tuple[float, float]]) -> str:
    """Hash the peak list at 0.01 Da / 1 % resolution (acquisition-noise robust)."""
    norm = sorted((round(mz, 2), round(i)) for mz, i in peaks)
    return hashlib.sha256(repr(norm).encode()).hexdigest()


def fmt_msp(rec: dict[str, Any], name: str) -> str:
    meta, peaks = rec["meta"], rec["peaks"]
    lines = [f"NAME: {name}", f"ID: {name}"]
    pmz = precursor_of(meta)
    if pmz:
        lines.append(f"PRECURSOR_MZ: {pmz:.6f}")
    lines.append("CHARGE: 1")
    lines.append("IONMODE: Positive")
    adduct = meta.get("adduct") or meta.get("adductionname")
    if adduct:
        lines.append(f"ADDUCT: {adduct}")
    lines.append(f"NUM PEAKS: {len(peaks)}")
    lines += [f"{mz:.5f}\t{i:.6g}" for mz, i in peaks]
    return "\n".join(lines) + "\n"


def fmt_mgf(rec: dict[str, Any], name: str) -> str:
    meta, peaks = rec["meta"], rec["peaks"]
    pmz = precursor_of(meta)
    lines = ["BEGIN IONS", f"TITLE={name}", f"ID={name}"]
    if pmz:
        lines.append(f"PEPMASS={pmz:.6f}")
    lines += ["CHARGE=1", "MSLEVEL=2", "IONMODE=Positive"]
    lines += [f"{mz:.5f} {i:.6g}" for mz, i in peaks]
    lines.append("END IONS")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------


def verify_written_spectra(
    path: Path, kind: str, source: list[dict[str, Any]], sample: int = 40
) -> dict[str, Any]:
    """Confirm the written file preserves the source intensities and peak counts.

    This exists because a formatting mistake silently destroyed every intensity
    (``%.1f`` turned 0.033 into ``0.0``), which made the whole benchmark measure
    nothing. A round-trip assertion per record is cheap next to the pipeline run
    it guards.
    """
    reparsed = parse_msp(path) if kind == "msp" else parse_mgf(path)
    if len(reparsed) != len(source):
        raise AssertionError(
            f"{path.name}: wrote {len(reparsed)} records but expected {len(source)}"
        )
    mismatched_counts = 0
    intensity_values: set[float] = set()
    for src, out in zip(source[:sample], reparsed[:sample]):
        if len(src["peaks"]) != len(out["peaks"]):
            mismatched_counts += 1
        intensity_values.update(i for _, i in out["peaks"])
    if mismatched_counts:
        raise AssertionError(
            f"{path.name}: {mismatched_counts}/{min(sample, len(source))} sampled "
            "records lost peaks on write"
        )
    max_intensity = max(intensity_values) if intensity_values else 0.0
    if max_intensity <= 0.0:
        raise AssertionError(
            f"{path.name}: every written intensity is zero — the writer is "
            "destroying intensity information"
        )
    if len(intensity_values) < 10:
        raise AssertionError(
            f"{path.name}: only {len(intensity_values)} distinct intensity values "
            "in the sample; intensities were almost certainly rounded away"
        )
    return {
        "records": len(reparsed),
        "sampled_distinct_intensities": len(intensity_values),
        "sampled_max_intensity": max_intensity,
    }


def build_real_benchmark(
    corpus: Path,
    workdir: Path,
    library_file: str,
    query_files: list[str],
    max_queries: int,
    seed: int,
) -> dict[str, Any]:
    lib_path = corpus / library_file
    if not lib_path.exists():
        raise FileNotFoundError(f"library not found: {lib_path}")

    lib_records = parse_msp(lib_path)
    # InChIKey -> library record indices
    lib_by_key: dict[str, list[int]] = {}
    lib_hashes: set[str] = set()
    for idx, rec in enumerate(lib_records):
        ik = inchikey_of(rec["meta"])
        if ik:
            lib_by_key.setdefault(ik[:14], []).append(idx)
        lib_hashes.add(peak_hash(rec["peaks"]))

    workdir.mkdir(parents=True, exist_ok=True)
    library_mode_counts: Counter[str] = Counter()
    library_type_counts: Counter[str] = Counter()
    library_adduct_counts: Counter[str] = Counter()
    library_charge_counts: Counter[str] = Counter()
    for library_record in lib_records:
        library_metadata = library_record["meta"]
        library_mode_counts[
            (
                library_metadata.get("ionmode")
                or library_metadata.get("ion_mode")
                or "unknown"
            )
            .strip()
            .upper()
        ] += 1
        library_type_counts[
            (
                library_metadata.get("precursor_type")
                or library_metadata.get("precursortype")
                or "unknown"
            )
            .strip()
            .upper()
        ] += 1
        library_charge_counts[
            (library_metadata.get("charge") or "unknown").strip()
        ] += 1
    stats: dict[str, Any] = {
        "library_file": library_file,
        "library_records": len(lib_records),
        "library_distinct_connectivity": len(lib_by_key),
        "library_duplicate_spectra": len(lib_records) - len(lib_hashes),
        "library_source_metadata_counts": {
            "ion_mode": dict(sorted(library_mode_counts.items())),
            "precursor_type": dict(sorted(library_type_counts.items())),
            "adduct": dict(sorted(library_adduct_counts.items())),
            "charge": dict(sorted(library_charge_counts.items())),
        },
        "query_sources": {},
        "selected_source_metadata": {},
        "selected_truth_reference_precursor_mz": {},
        "truth_pair_source_ion_mode_cross_tab": {},
        "truth_pair_source_precursor_type_cross_tab": {},
    }
    (workdir / "library.msp").write_text(
        "".join(fmt_msp(r, f"LIB_{i:06d}") for i, r in enumerate(lib_records))
    )
    stats["library_write_check"] = verify_written_spectra(
        workdir / "library.msp", "msp", lib_records
    )

    rng = random.Random(seed)
    labels: dict[str, Any] = {}
    blocks: list[str] = []
    selected_records: list[dict[str, Any]] = []
    truth_pair_mode_counts: Counter[str] = Counter()
    truth_pair_type_counts: Counter[str] = Counter()

    for qf in query_files:
        qpath = corpus / qf
        if not qpath.exists():
            stats["query_sources"][qf] = {"error": "not found"}
            continue
        q_records = parse_msp(qpath)
        candidates: list[tuple[dict[str, Any], str, str]] = []
        n_no_key = n_not_in_lib = n_identical = 0
        for rec in q_records:
            ik = inchikey_of(rec["meta"])
            if not ik:
                n_no_key += 1
                continue
            if ik[:14] not in lib_by_key:
                n_not_in_lib += 1
                continue
            if peak_hash(rec["peaks"]) in lib_hashes:
                n_identical += 1
                continue
            candidates.append((rec, ik, peak_hash(rec["peaks"])))

        # one query per (connectivity, distinct spectrum) to avoid weighting a
        # compound by how many replicate acquisitions it happens to have
        seen: set[tuple[str, str]] = set()
        deduped = []
        for rec, ik, h in candidates:
            token = (ik[:14], h)
            if token in seen:
                continue
            seen.add(token)
            deduped.append((rec, ik, h))
        rng.shuffle(deduped)
        selected = deduped[:max_queries]

        for n, (rec, ik, _h) in enumerate(selected):
            qid = f"{qf.split('-')[-1].replace('.msp', '')}_Q{n:05d}"
            blocks.append(fmt_mgf(rec, qid))
            selected_records.append(rec)
            source_ion_mode = (
                (rec["meta"].get("ionmode") or rec["meta"].get("ion_mode") or "unknown")
                .strip()
                .upper()
            )
            source_precursor_type = rec["meta"].get("precursor_type") or rec[
                "meta"
            ].get("precursortype")
            source_adduct = rec["meta"].get("adduct") or rec["meta"].get(
                "adductionname"
            )
            stats["selected_source_metadata"][qid] = {
                "source": qf,
                "ion_mode": source_ion_mode,
                "precursor_type": source_precursor_type,
                "adduct": source_adduct,
                "charge": rec["meta"].get("charge"),
                "precursor_mz": precursor_of(rec["meta"]),
                "search_precursor_mz": (
                    float(f"{precursor_of(rec['meta']):.6f}")
                    if precursor_of(rec["meta"]) is not None
                    else None
                ),
                "instrument": rec["meta"].get("instrument")
                or rec["meta"].get("instrumenttype"),
            }
            for library_index in lib_by_key[ik[:14]]:
                library_record_id = f"LIB_{library_index:06d}"
                library_precursor_mz = precursor_of(lib_records[library_index]["meta"])
                stats["selected_truth_reference_precursor_mz"][library_record_id] = (
                    float(f"{library_precursor_mz:.6f}")
                    if library_precursor_mz is not None
                    else None
                )
                library_meta = lib_records[library_index]["meta"]
                library_ion_mode = (
                    (
                        library_meta.get("ionmode")
                        or library_meta.get("ion_mode")
                        or "unknown"
                    )
                    .strip()
                    .upper()
                )
                library_precursor_type = (
                    (
                        library_meta.get("precursor_type")
                        or library_meta.get("precursortype")
                        or "unknown"
                    )
                    .strip()
                    .upper()
                )
                query_precursor_type = (
                    (source_precursor_type or "unknown").strip().upper()
                )
                truth_pair_mode_counts[f"{source_ion_mode}/{library_ion_mode}"] += 1
                truth_pair_type_counts[
                    f"{query_precursor_type}/{library_precursor_type}"
                ] += 1
            labels[qid] = {
                "inchikey": ik,
                "connectivity": ik[:14],
                "true_lib_records": [f"LIB_{i:06d}" for i in lib_by_key[ik[:14]]],
                "precursor_mz": precursor_of(rec["meta"]),
                "source": qf,
                "peak_count": len(rec["peaks"]),
            }
        stats["query_sources"][qf] = {
            "records": len(q_records),
            "rejected_no_inchikey": n_no_key,
            "rejected_not_in_library": n_not_in_lib,
            "rejected_identical_spectrum": n_identical,
            "candidates": len(candidates),
            "after_compound_dedup": len(deduped),
            "selected": len(selected),
        }

    stats["truth_pair_source_ion_mode_cross_tab"] = dict(
        sorted(truth_pair_mode_counts.items())
    )
    stats["truth_pair_source_precursor_type_cross_tab"] = dict(
        sorted(truth_pair_type_counts.items())
    )
    (workdir / "queries.mgf").write_text("\n".join(blocks) + "\n")
    stats["queries_write_check"] = verify_written_spectra(
        workdir / "queries.mgf", "mgf", selected_records
    )
    (workdir / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    return {"stats": stats, "labels": labels}


def audit_truth_precursor_eligibility(
    labels: dict[str, Any],
    stats: dict[str, Any],
    *,
    min_query_peaks: int,
) -> dict[str, Any]:
    """Quantify truth-record precursor differences at the configured MS1 windows.

    This benchmark-only analysis uses only the selected truth labels and the
    exact six-decimal query/library precursor values serialized for search. It
    does not pass truth labels into MassFlow or change the candidate cohort.

    Parameters
    ----------
    labels : dict[str, Any]
        Selected benchmark query labels.
    stats : dict[str, Any]
        Dataset provenance with source and truth-reference precursor metadata.
    min_query_peaks : int
        Preprocessing threshold that defines the searched-query denominator.

    Returns
    -------
    dict[str, Any]
        Per-source true-record MS1-window coverage and absolute-mass-difference
        summaries. Quantiles use NumPy's linear interpolation method.
    """
    source_metadata = stats.get("selected_source_metadata", {})
    reference_precursors = stats.get("selected_truth_reference_precursor_mz", {})
    by_source: dict[str, list[tuple[str, float, list[float]]]] = {}
    missing_query_precursors: dict[str, int] = {}
    missing_reference_precursors: dict[str, int] = {}
    for query_id, label in labels.items():
        source = str(label.get("source", "unknown"))
        metadata = source_metadata.get(query_id, {})
        query_precursor_mz = metadata.get("search_precursor_mz")
        if query_precursor_mz is None:
            missing_query_precursors[source] = (
                missing_query_precursors.get(source, 0) + 1
            )
            continue
        differences: list[float] = []
        for reference_id in label.get("true_lib_records", []):
            reference_precursor_mz = reference_precursors.get(str(reference_id))
            if reference_precursor_mz is None:
                missing_reference_precursors[source] = (
                    missing_reference_precursors.get(source, 0) + 1
                )
                continue
            differences.append(
                abs(float(query_precursor_mz) - float(reference_precursor_mz))
            )
        by_source.setdefault(source, []).append(
            (str(query_id), float(query_precursor_mz), differences)
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "mass_error_summary": "absolute query minus truth-reference precursor m/z in Da",
        "quantile_method": "numpy.quantile(method='linear')",
        "tolerance_denominator": "configured cosine MS1 half-window in Da",
        "by_source": {},
    }
    for source, query_records in sorted(by_source.items()):
        all_differences = [
            difference
            for _query_id, _precursor_mz, differences in query_records
            for difference in differences
        ]
        searched_records = [
            record
            for record in query_records
            if int(labels[record[0]].get("peak_count", min_query_peaks))
            >= min_query_peaks
        ]
        searched_differences = [
            difference
            for _query_id, _precursor_mz, differences in searched_records
            for difference in differences
        ]

        def summarize(differences: list[float]) -> dict[str, Any]:
            if not differences:
                return {"pair_count": 0, "absolute_difference_da": None}
            values = np.asarray(differences, dtype=np.float64)
            quantiles = np.quantile(values, [0.0, 0.5, 0.9, 0.95, 1.0])
            return {
                "pair_count": int(values.size),
                "absolute_difference_da": {
                    "min": float(quantiles[0]),
                    "median": float(quantiles[1]),
                    "p90": float(quantiles[2]),
                    "p95": float(quantiles[3]),
                    "max": float(quantiles[4]),
                },
            }

        tolerances: dict[str, Any] = {}
        for tolerance in (0.01, 0.02):
            selected_pairs = [
                difference for difference in all_differences if difference <= tolerance
            ]
            searched_pairs = [
                difference
                for difference in searched_differences
                if difference <= tolerance
            ]
            selected_queries = sum(
                any(difference <= tolerance for difference in differences)
                for _query_id, _precursor_mz, differences in query_records
            )
            searched_queries = sum(
                any(difference <= tolerance for difference in differences)
                for _query_id, _precursor_mz, differences in searched_records
            )
            tolerances[f"{tolerance:.2f}_Da"] = {
                "selected_true_reference_pairs_in_window": len(selected_pairs),
                "selected_queries_with_any_truth_pair_in_window": selected_queries,
                "searched_true_reference_pairs_in_window": len(searched_pairs),
                "searched_queries_with_any_truth_pair_in_window": searched_queries,
            }
        report["by_source"][source] = {
            "selected_queries": len(query_records),
            "searched_queries": len(searched_records),
            "selected_truth_pairs": summarize(all_differences),
            "searched_truth_pairs": summarize(searched_differences),
            "tolerances": tolerances,
            "missing_query_precursor_count": missing_query_precursors.get(source, 0),
            "missing_truth_reference_precursor_count": missing_reference_precursors.get(
                source, 0
            ),
        }
    return report


# ---------------------------------------------------------------------------
# Running + metrics
# ---------------------------------------------------------------------------


def write_config(
    workdir: Path,
    algorithm: str,
    min_score: float,
    min_matched_peaks: int,
    fdr: float,
    name: str,
    ms1_tolerance: float = 0.02,
    ms2_tolerance: float = 0.02,
    min_query_peaks: int = 3,
) -> Path:
    import yaml  # noqa: PLC0415

    cfg = {
        "project": {"name": name, "output_directory": str(workdir / name)},
        "input": {
            "input_path": str(workdir / "queries.mgf"),
            "library_path": str(workdir / "library.msp"),
            "format": "mgf",
        },
        "processing": {
            "clean_metadata": True,
            "normalize_intensity": True,
            "filter_min_peaks": True,
            "min_peaks": min_query_peaks,
            "decoy_min_relative_intensity": 0.01,
            "decoy_mz_shift_da": 1.0,
        },
        "similarity": {
            "algorithm": algorithm,
            "ms1_tolerance": ms1_tolerance,
            "ms2_tolerance": ms2_tolerance,
            "min_score": min_score,
            "min_matched_peaks": min_matched_peaks,
            "fdr_threshold": fdr,
            "collect_retrieval_diagnostics": True,
        },
        "export": {"format": "csv"},
    }
    path = workdir / f"config_{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def run_cli(config: Path, run_dir: Path) -> dict[str, Any]:
    cli = REPO_ROOT / ".venv" / "bin" / "massflow"
    cmd = (
        [str(cli)]
        if cli.exists()
        else [
            sys.executable,
            "-c",
            "import sys; from MassFlow.cli import main; sys.exit(main())",
        ]
    )
    started = time.perf_counter()
    proc = subprocess.run(
        cmd + ["annotate", "--config", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=14400,
    )
    wall_time_seconds = time.perf_counter() - started
    csvs = sorted(run_dir.rglob("*_results.csv")) if run_dir.exists() else []
    return {
        "exit_code": proc.returncode,
        "results_csv": str(csvs[0]) if csvs else None,
        "stdout_tail": proc.stdout[-1500:],
        "wall_time_seconds": wall_time_seconds,
    }


def _col(rows: list[dict[str, Any]], *names: str) -> Optional[str]:
    if not rows:
        return None
    keys = {k.lower(): k for k in rows[0]}
    for n in names:
        if n.lower() in keys:
            return keys[n.lower()]
    return None


def evaluate_real(
    rows: list[dict[str, Any]],
    labels: dict[str, Any],
    score_thresholds: Iterable[float] = (0.0,),
    q_thresholds: Iterable[float] = (float("inf"),),
    min_matched_peaks: int = 1,
    min_query_peaks: int = 3,
) -> dict[str, Any]:
    """Evaluate retrieval and ranking from the complete exported query table."""
    ref_col = _col(rows, "reference_name")
    query_col = _col(rows, "query_id", "query")
    score_col = _col(rows, "score")
    if not query_col:
        raise RuntimeError("results are missing the query_id column")
    if not rows:
        raise RuntimeError("results CSV contains no query accounting rows")
    accounting_rows = rows
    base_metrics = evaluate_query_retrieval(
        labels,
        accounting_rows,
        min_query_peaks=min_query_peaks,
        min_score=0.0,
        min_matched_peaks=min_matched_peaks,
    )
    candidate_rows = [row for row in rows if ref_col and row.get(ref_col)]
    if candidate_rows and (ref_col is None or score_col is None):
        raise RuntimeError("candidate rows require reference_name and score columns")
    if candidate_rows:
        assert ref_col is not None and score_col is not None
    hits_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        if ref_col is None or score_col is None:
            raise RuntimeError(
                "candidate rows require reference_name and score columns"
            )
        query_id = str(row[query_col])
        hits_by_query.setdefault(query_id, []).append(
            {
                "ref": str(row[ref_col]),
                "score": float(row[score_col]) if score_col else 0.0,
            }
        )
    for hit_list in hits_by_query.values():
        hit_list.sort(key=lambda hit: -hit["score"])

    out: dict[str, Any] = {
        **base_metrics,
        "n_rows": base_metrics["candidate_rows_exported"],
        "n_queries": len(labels),
        "sweep": [],
    }
    for score_threshold in score_thresholds:
        for q_threshold in q_thresholds:
            point = evaluate_query_retrieval(
                labels,
                accounting_rows,
                min_query_peaks=min_query_peaks,
                min_score=score_threshold,
                q_threshold=q_threshold,
                min_matched_peaks=min_matched_peaks,
            )
            out["sweep"].append(
                {
                    "min_score": score_threshold,
                    "q_threshold": None if q_threshold == float("inf") else q_threshold,
                    **point["query_counts"],
                    **point["metrics"],
                }
            )

    # Ranking funnel: separates the two ways a query can fail. "The true
    # compound was never a candidate" is a gating/prefilter problem; "it was a
    # candidate but ranked below something else" is a scoring/ranking problem.
    # Without this split, a low top-1 number says nothing about which to fix.
    lib_of = {
        query_id: set(value["true_lib_records"]) for query_id, value in labels.items()
    }
    funnel: dict[str, Any] = {
        "queries": len(labels),
        "with_any_candidate": 0,
        "true_compound_among_candidates": 0,
        "true_compound_ranked_first": 0,
        "median_rank_of_true_compound": None,
        "true_compound_last_rank": None,
    }
    true_ranks: list[int] = []
    for qid in sorted(labels):
        hits = hits_by_query.get(qid, [])
        if not hits:
            continue
        funnel["with_any_candidate"] += 1
        for rank, h in enumerate(hits, start=1):
            if h["ref"] in lib_of[qid]:
                true_ranks.append(rank)
                funnel["true_compound_among_candidates"] += 1
                if rank == 1:
                    funnel["true_compound_ranked_first"] += 1
                break
    if true_ranks:
        funnel["median_rank_of_true_compound"] = float(statistics.median(true_ranks))
        funnel["true_compound_last_rank"] = int(max(true_ranks))
    funnel["ranked_first_of_those_reachable"] = (
        funnel["true_compound_ranked_first"] / funnel["true_compound_among_candidates"]
        if funnel["true_compound_among_candidates"]
        else None
    )

    out["by_source"] = base_metrics["by_source"]
    out["ranking_funnel"] = funnel
    return out


def _sha256_file(path: Path) -> str:
    """Hash a prepared input file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    """Return the checkout revision and whether tracked files are modified."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(status.stdout.strip())
        if status.returncode == 0
        else None,
    }


def _dataset_fingerprint(workdir: Path, labels: dict[str, Any]) -> str:
    """Identify the exact normalized library, query file, and selected labels."""
    identity = {
        "library_sha256": _sha256_file(workdir / "library.msp"),
        "queries_sha256": _sha256_file(workdir / "queries.mgf"),
        "labels": labels,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _metadata_counts_by_source(
    selected_metadata: dict[str, dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    """Summarize original source metadata without adding it to truth labels."""
    counts_by_source: dict[str, Counter[str]] = {}
    for metadata in selected_metadata.values():
        source = str(metadata.get("source", "unknown"))
        counts_by_source.setdefault(source, Counter())[
            str(metadata.get(field) or "unknown")
        ] += 1
    return {
        source: dict(sorted(counts.items()))
        for source, counts in sorted(counts_by_source.items())
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/massflow-real"))
    ap.add_argument("--library-file", default=DEFAULT_LIBRARY)
    ap.add_argument("--query-files", nargs="*", default=DEFAULT_QUERIES)
    ap.add_argument("--max-queries", type=int, default=400)
    ap.add_argument(
        "--min-query-peaks",
        type=int,
        default=3,
        help="preprocessing gate; rejected selected queries remain in end-to-end denominators",
    )
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument(
        "--keep-data",
        action="store_true",
        help="reuse an existing prepared dataset in --workdir",
    )
    # Production gates by default: realistic operating point, and it keeps the
    # modified_cosine path tractable (that engine applies no MS1 prefilter by
    # design, so permissive gates make it score the full library x query matrix).
    ap.add_argument("--min-score", type=float, default=0.7)
    ap.add_argument("--min-matched-peaks", type=int, default=3)
    ap.add_argument("--fdr-threshold", type=float, default=0.05)
    ap.add_argument(
        "--sweep-scores",
        type=float,
        nargs="*",
        default=None,
        help="score thresholds to evaluate from the exported candidates "
        "(default: the single --min-score value)",
    )
    ap.add_argument(
        "--sweep-q",
        type=float,
        nargs="*",
        default=None,
        help="q-value thresholds to evaluate (default: ignore q)",
    )
    ap.add_argument(
        "--algorithms",
        nargs="*",
        default=["cosine", "modified_cosine"],
        help="which similarity engines to run",
    )
    ap.add_argument(
        "--ms1-tolerance",
        type=float,
        default=0.02,
        help="precursor window in Da; for cosine this is a hard gate "
        "applied before any scoring",
    )
    ap.add_argument(
        "--ms2-tolerance",
        type=float,
        default=0.02,
        help="fragment matching window in Da",
    )
    ap.add_argument(
        "--run-tag",
        default="",
        help="suffix for the run directory, to keep runs side by side",
    )
    ap.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="compare this report.json to a prior schema-v2 report on the same dataset",
    )
    ap.add_argument(
        "--compare-results-to",
        type=Path,
        default=None,
        help="compare exported query rankings to a previous results CSV",
    )
    args = ap.parse_args(argv)

    print("=" * 80)
    print("MassFlow cross-instrument reference-library retrieval benchmark")
    print("=" * 80)

    baseline_report: Optional[dict[str, Any]] = None
    if args.compare_to is not None:
        baseline_report = json.loads(args.compare_to.read_text())
    if args.workdir.exists() and not args.keep_data:
        shutil.rmtree(args.workdir)
    labels_file = args.workdir / "labels.json"
    if args.keep_data and labels_file.exists():
        # Reuse the prepared dataset: parsing ~400 MB of MSP is the slow part.
        labels = json.loads(labels_file.read_text())
        stats_file = args.workdir / "stats.json"
        s = json.loads(stats_file.read_text()) if stats_file.exists() else {}
        if (
            s.get("report_schema_version") != REPORT_SCHEMA_VERSION
            or "dataset_fingerprint" not in s
            or any("peak_count" not in label for label in labels.values())
        ):
            raise RuntimeError(
                "prepared benchmark data predates reporting schema v2; "
                "rerun without --keep-data"
            )
        if s["dataset_fingerprint"] != _dataset_fingerprint(args.workdir, labels):
            raise RuntimeError(
                "prepared benchmark inputs no longer match stats.json; "
                "rerun without --keep-data"
            )
        print(f"\n[corpus] reusing prepared dataset in {args.workdir}")
        print(
            f"[library] {s.get('library_records', '?')} real spectra, "
            f"{s.get('library_distinct_connectivity', '?')} distinct compounds"
        )
        print(f"[queries] {len(labels)} selected")
        built = {"stats": s, "labels": labels}
    else:
        built = build_real_benchmark(
            corpus=args.corpus,
            workdir=args.workdir,
            library_file=args.library_file,
            query_files=args.query_files,
            max_queries=args.max_queries,
            seed=args.seed,
        )
        if any("peak_count" not in label for label in built["labels"].values()):
            raise RuntimeError("prepared query labels are missing peak_count")
        built["stats"]["report_schema_version"] = REPORT_SCHEMA_VERSION
        built["stats"]["library_sha256"] = _sha256_file(args.workdir / "library.msp")
        built["stats"]["queries_sha256"] = _sha256_file(args.workdir / "queries.mgf")
        built["stats"]["dataset_fingerprint"] = _dataset_fingerprint(
            args.workdir, built["labels"]
        )
        (args.workdir / "stats.json").write_text(
            json.dumps(built["stats"], indent=2, default=str)
        )
        s = built["stats"]
        labels = built["labels"]
        print(f"\n[corpus] {args.corpus}")
        print(
            f"[library] {s['library_file']}: {s['library_records']} real spectra, "
            f"{s['library_distinct_connectivity']} distinct compounds (InChIKey connectivity)"
        )
        if s["library_duplicate_spectra"]:
            print(
                f"          {s['library_duplicate_spectra']} records are exact/near duplicates"
            )
        print(
            f"[queries] {len(labels)} selected across {len(args.query_files)} instruments"
        )
        for qf, st in s["query_sources"].items():
            if "error" in st:
                print(f"    {qf}: {st['error']}")
                continue
            print(
                f"    {qf}: {st['records']} records -> "
                f"{st['rejected_not_in_library']} compound not in library, "
                f"{st['rejected_identical_spectrum']} identical spectrum, "
                f"{st['rejected_no_inchikey']} no InChIKey -> {st['selected']} used"
            )

    results: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset": s,
        "dataset_fingerprint": s["dataset_fingerprint"],
        "runtime": {
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "python_version": sys.version.split()[0],
            "git": _git_provenance(),
        },
        "truth_precursor_eligibility_audit": audit_truth_precursor_eligibility(
            labels, s, min_query_peaks=args.min_query_peaks
        ),
        "source_metadata_audit": {
            "normalized_query_writer": {
                "ionmode": "Positive",
                "charge": 1,
                "adduct_emitted": False,
            },
            "raw_query_ion_mode_counts_by_source": _metadata_counts_by_source(
                s.get("selected_source_metadata", {}), "ion_mode"
            ),
            "raw_query_precursor_type_counts_by_source": _metadata_counts_by_source(
                s.get("selected_source_metadata", {}), "precursor_type"
            ),
            "raw_query_adduct_counts_by_source": _metadata_counts_by_source(
                s.get("selected_source_metadata", {}), "adduct"
            ),
            "raw_query_charge_counts_by_source": _metadata_counts_by_source(
                s.get("selected_source_metadata", {}), "charge"
            ),
            "reference_source_metadata_counts": s.get(
                "library_source_metadata_counts", {}
            ),
            "truth_pair_raw_ion_mode_cross_tab": s.get(
                "truth_pair_source_ion_mode_cross_tab", {}
            ),
            "truth_pair_raw_precursor_type_cross_tab": s.get(
                "truth_pair_source_precursor_type_cross_tab", {}
            ),
            "selected_query_metadata": s.get("selected_source_metadata", {}),
            "interpretation": (
                "The normalized benchmark inputs do not preserve source ion mode, "
                "charge, or precursor-type metadata; polarity/adduct eligibility "
                "in the MassFlow search therefore cannot explain these benchmark calls."
            ),
        },
        "runs": {},
    }
    for algorithm in args.algorithms:
        name = f"real_{algorithm}{args.run_tag}"
        run_config_path = args.workdir / f"config_{name}.yaml"
        if args.skip_run:
            run: dict[str, Any] = {"exit_code": 0, "results_csv": None}
        else:
            cfg = write_config(
                args.workdir,
                algorithm,
                args.min_score,
                args.min_matched_peaks,
                args.fdr_threshold,
                name,
                args.ms1_tolerance,
                args.ms2_tolerance,
                args.min_query_peaks,
            )
            print(
                f"\n[run] {algorithm} over {s.get('library_records', '?')} references "
                f"(min_score={args.min_score}, min_matched_peaks={args.min_matched_peaks}, "
                f"q<={args.fdr_threshold}) …",
                flush=True,
            )
            run_config_path = cfg
            run = run_cli(cfg, args.workdir / name)
            print(f"      exit_code={run['exit_code']}")
        csvs = [Path(run["results_csv"])] if run.get("results_csv") else []
        rows = []
        if csvs:
            with csvs[0].open(newline="") as fh:
                rows = [dict(r) for r in _csv.DictReader(fh)]
        if not rows:
            print("      no rows produced")
            results["runs"][algorithm] = {"error": "no rows"}
            continue
        metrics = evaluate_real(
            rows,
            labels,
            score_thresholds=args.sweep_scores or (args.min_score,),
            q_thresholds=args.sweep_q or (float("inf"),),
            min_matched_peaks=args.min_matched_peaks,
            min_query_peaks=args.min_query_peaks,
        )
        metrics.update(
            {
                "algorithm": algorithm,
                "exit_code": run.get("exit_code"),
                "wall_time_seconds": run.get("wall_time_seconds"),
                "configuration": {
                    "min_score": args.min_score,
                    "min_matched_peaks": args.min_matched_peaks,
                    "fdr_threshold": args.fdr_threshold,
                    "ms1_tolerance": args.ms1_tolerance,
                    "ms2_tolerance": args.ms2_tolerance,
                    "min_query_peaks": args.min_query_peaks,
                },
            }
        )
        diagnostics_payload: dict[str, Any] = {
            "status": "unavailable",
            "reason": "diagnostic JSON was not produced by the annotation run",
        }
        truth_attribution: dict[str, Any] = {
            "status": "unavailable",
            "reason": "run output or prepared library store was unavailable",
        }
        if csvs:
            diagnostic_path = csvs[0].with_name(f"{csvs[0].stem}.diagnostics.json")
            if diagnostic_path.is_file():
                diagnostics_payload = json.loads(diagnostic_path.read_text())
            prepared_library_path = csvs[0].parent / "library_library.db"
            if prepared_library_path.is_file() and run_config_path.is_file():
                truth_attribution = attribute_truth_candidates(
                    labels,
                    rows,
                    config_path=str(run_config_path),
                    prepared_library_path=str(prepared_library_path),
                    run_diagnostics=diagnostics_payload,
                )
                for query_id, query_attribution in truth_attribution.get(
                    "by_query", {}
                ).items():
                    query_attribution["raw_source_metadata"] = s.get(
                        "selected_source_metadata", {}
                    ).get(query_id)
        metrics["retrieval_diagnostics"] = diagnostics_payload
        metrics["retrieval_diagnostics_by_source"] = summarize_diagnostics_by_source(
            labels,
            diagnostics_payload,
            min_query_peaks=args.min_query_peaks,
        )
        metrics["truth_attribution"] = truth_attribution
        if args.compare_results_to is not None:
            with args.compare_results_to.open(newline="") as baseline_file:
                baseline_rows = [dict(row) for row in _csv.DictReader(baseline_file)]
            metrics["result_comparison_to_baseline"] = {
                "baseline_csv_name": args.compare_results_to.name,
                "baseline_csv_sha256": _sha256_file(args.compare_results_to),
                **compare_result_rows(baseline_rows, rows),
            }
        results["runs"][algorithm] = metrics
        print(
            f"      candidate_rows_exported={metrics['candidate_rows_exported']} "
            f"selected={metrics['query_counts']['selected']} "
            f"searched={metrics['query_counts']['searched']}"
        )
        print(
            f"      {'min_score':>9} {'calls':>7} {'coverage':>9} "
            f"{'top1/all':>9} {'top1/searched':>14} {'accepted_acc':>13}"
        )
        for row in metrics["sweep"]:
            conditional_accuracy = row["conditional_top1_accuracy"]
            accepted_accuracy = row["accepted_call_accuracy"]
            print(
                f"      {row['min_score']:>9.2f} {row['with_candidate']:>7} "
                f"{row['call_coverage_selected']:>9.4f} "
                f"{row['end_to_end_top1_accuracy']:>9.4f} "
                f"{(f'{conditional_accuracy:.4f}' if conditional_accuracy is not None else 'n/a'):>14} "
                f"{(f'{accepted_accuracy:.4f}' if accepted_accuracy is not None else 'n/a'):>13}"
            )
        f = metrics.get("ranking_funnel")
        if f:
            print(
                f"      funnel: {f['queries']} queries -> "
                f"{f['with_any_candidate']} with a candidate -> "
                f"{f['true_compound_among_candidates']} with the true compound among them "
                f"-> {f['true_compound_ranked_first']} ranked first"
                f" (rank-1 rate among reachable: "
                f"{f['ranked_first_of_those_reachable']:.4f})"
                if f["ranked_first_of_those_reachable"] is not None
                else ""
            )

    out = args.workdir / "report.json"
    if baseline_report is not None:
        results["comparison_to_baseline"] = compare_benchmark_reports(
            baseline_report, results
        )
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
