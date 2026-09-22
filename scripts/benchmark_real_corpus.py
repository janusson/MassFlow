#!/usr/bin/env python3
"""Real-data, objectively-labelled benchmark for MassFlow annotation accuracy.

Replaces the synthetic benchmark's biggest weakness (made-up spectra) with real
acquired spectra and labels that come from the data itself rather than from
construction.

**Design: cross-instrument compound identification.**
The corpus contains the same Vaniya-Fiehn natural-products library acquired on
three different instruments (LTQ, QTOF, QExactive). Using one instrument as the
reference library and another as the queries is a genuine cross-instrument
library-matching task — the same design used in the Spec2Vec evaluation — and
the label is objective: a hit is correct iff the query's and the hit's InChIKey
connectivity block agree.

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
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
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
    stats: dict[str, Any] = {
        "library_file": library_file,
        "library_records": len(lib_records),
        "library_distinct_connectivity": len(lib_by_key),
        "library_duplicate_spectra": len(lib_records) - len(lib_hashes),
        "query_sources": {},
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
            labels[qid] = {
                "inchikey": ik,
                "connectivity": ik[:14],
                "true_lib_records": [f"LIB_{i:06d}" for i in lib_by_key[ik[:14]]],
                "precursor_mz": precursor_of(rec["meta"]),
                "source": qf,
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

    (workdir / "queries.mgf").write_text("\n".join(blocks) + "\n")
    stats["queries_write_check"] = verify_written_spectra(
        workdir / "queries.mgf", "mgf", selected_records
    )
    (workdir / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    return {"stats": stats, "labels": labels}


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
            "min_peaks": 3,
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
    proc = subprocess.run(
        cmd + ["annotate", "--config", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=14400,
    )
    csvs = sorted(run_dir.rglob("*_results.csv")) if run_dir.exists() else []
    return {
        "exit_code": proc.returncode,
        "results_csv": str(csvs[0]) if csvs else None,
        "stdout_tail": proc.stdout[-1500:],
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
) -> dict[str, Any]:
    """Top-1 + weighted P/R/F1 with InChIKey connectivity as the label.

    ``q_thresholds=inf`` means "ignore the q column", which is what a single
    permissive run wants: the score sweep then shows where the pipeline should
    be operated instead of us guessing a threshold up front.

    ``min_matched_peaks`` is applied at evaluation time so two engines run at
    different CLI gates can still be compared at one common operating point.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    ref_col = _col(rows, "reference_name")
    query_col = _col(rows, "query_id")
    score_col = _col(rows, "score")
    q_col = _col(rows, "q_value")
    if not query_col:
        raise RuntimeError(
            f"no query column in results: {list(rows[0]) if rows else 'no rows'}"
        )
    if not (ref_col and score_col):
        # The pipeline exported only no-hit placeholder rows: no candidate passed
        # the configured gates. That is a result, not a harness failure.
        return {
            "n_rows": 0,
            "n_queries": len(labels),
            "hits_passed_gates": 0,
            "top1_accuracy_connectivity": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "f1_micro": 0.0,
            "multi_candidate_queries": 0,
            "multi_candidate_top1_accuracy": None,
            "note": "no candidate pair passed min_score/min_matched_peaks for any query",
        }
    assert ref_col and query_col and score_col

    lib_of = {q: set(v["true_lib_records"]) for q, v in labels.items()}
    conn_of = {q: v["connectivity"] for q, v in labels.items()}

    # Annotated so the heterogeneous literal is typed Any-valued: without this
    # mypy widens the dict to dict[str, object] and every arithmetic use of a
    # field fails.
    parsed: list[dict[str, Any]] = [
        {
            "query": str(r[query_col]),
            "ref": str(r[ref_col]),
            "score": float(r[score_col]),
            "peaks": int(float(r["matched_peaks"])) if r.get("matched_peaks") else 0,
            "q": float(r[q_col]) if q_col and r.get(q_col) else float("nan"),
        }
        for r in rows
        if r.get(ref_col)
    ]
    parsed = [p for p in parsed if p["peaks"] >= min_matched_peaks]
    if not parsed:
        return {
            "n_rows": 0,
            "n_queries": len(labels),
            "hits_passed_gates": 0,
            "top1_accuracy_connectivity": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "f1_micro": 0.0,
            "multi_candidate_queries": 0,
            "multi_candidate_top1_accuracy": None,
            "sweep": [],
            "note": "no candidate pair passed min_score/min_matched_peaks for any query",
        }

    hits_by_query: dict[str, list[dict[str, Any]]] = {}
    for p in parsed:
        hits_by_query.setdefault(p["query"], []).append(p)
    # NB: bound as `hit_list`, not `v`: mypy types the whole function scope, and
    # reusing `v` below for a label dict would inherit this list type.
    for hit_list in hits_by_query.values():
        hit_list.sort(key=lambda p: -p["score"])

    out: dict[str, Any] = {"n_rows": len(parsed), "n_queries": len(labels), "sweep": []}
    best: Optional[dict[str, Any]] = None

    for tau in score_thresholds:
        for alpha in q_thresholds:
            y_true: list[str] = []
            y_pred: list[str] = []
            multi_true = multi_correct = n_called = 0
            for qid in sorted(labels):
                hits = [
                    h
                    for h in hits_by_query.get(qid, [])
                    if h["score"] >= tau and (alpha == float("inf") or h["q"] <= alpha)
                ]
                y_true.append(conn_of[qid])
                if not hits:
                    y_pred.append("NO_CALL")
                    continue
                n_called += 1
                top = hits[0]
                correct = top["ref"] in lib_of[qid]
                y_pred.append(conn_of[qid] if correct else f"WRONG:{top['ref']}")
                if len({h["ref"] for h in hits}) > 1:
                    multi_true += 1
                    multi_correct += int(correct)

            acc = float(accuracy_score(y_true, y_pred))
            p_w, r_w, f_w, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )  # type: ignore[arg-type]
            _, _, f_mi, _ = precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0
            )  # type: ignore[arg-type]
            row: dict[str, Any] = {
                "min_score": tau,
                "q_threshold": None if alpha == float("inf") else alpha,
                "queries_called": n_called,
                "top1_accuracy_connectivity": acc,
                "precision_weighted": float(p_w),
                "recall_weighted": float(r_w),
                "f1_weighted": float(f_w),
                "f1_micro": float(f_mi),
                "multi_candidate_queries": multi_true,
                "multi_candidate_top1_accuracy": (
                    multi_correct / multi_true if multi_true else None
                ),
            }
            out["sweep"].append(row)
            if best is None or row["f1_weighted"] > best["f1_weighted"]:
                best = row

    out["best_f1_operating_point"] = best

    # Ranking funnel: separates the two ways a query can fail. "The true
    # compound was never a candidate" is a gating/prefilter problem; "it was a
    # candidate but ranked below something else" is a scoring/ranking problem.
    # Without this split, a low top-1 number says nothing about which to fix.
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

    # Per query-source breakdown: LTQ (low-resolution ion trap) and QTOF differ
    # enough in fragmentation that pooling them hides which source is failing.
    by_source: dict[str, dict[str, Any]] = {}
    for qid, v in labels.items():
        src = str(v.get("source", "unknown"))
        # NB: named `entry`, not `acc`. `acc` is already bound to the float
        # accuracy inside the sweep loop above, and mypy types the whole
        # function scope, so reusing the name makes the dict itself a float.
        entry = by_source.setdefault(
            src,
            {
                "queries": 0,
                "with_any_candidate": 0,
                "true_compound_among_candidates": 0,
                "true_compound_ranked_first": 0,
                "top1_accuracy": 0.0,
                "reachable_fraction": 0.0,
                "rank1_of_reachable": None,
            },
        )
        entry["queries"] += 1
        hits = hits_by_query.get(qid, [])
        if hits:
            entry["with_any_candidate"] += 1
        first_rank = None
        for rank, h in enumerate(hits, start=1):
            if h["ref"] in lib_of[qid]:
                first_rank = rank
                break
        if first_rank is not None:
            entry["true_compound_among_candidates"] += 1
            if first_rank == 1:
                entry["true_compound_ranked_first"] += 1
    for entry in by_source.values():
        n = entry["queries"] or 1
        entry["top1_accuracy"] = entry["true_compound_ranked_first"] / n
        entry["reachable_fraction"] = entry["true_compound_among_candidates"] / n
        if entry["true_compound_among_candidates"]:
            entry["rank1_of_reachable"] = (
                entry["true_compound_ranked_first"]
                / entry["true_compound_among_candidates"]
            )
    out["by_source"] = by_source
    out["ranking_funnel"] = funnel
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/massflow-real"))
    ap.add_argument("--library-file", default=DEFAULT_LIBRARY)
    ap.add_argument("--query-files", nargs="*", default=DEFAULT_QUERIES)
    ap.add_argument("--max-queries", type=int, default=400)
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
    args = ap.parse_args(argv)

    print("=" * 80)
    print("MassFlow real-corpus benchmark (real spectra, InChIKey labels)")
    print("=" * 80)

    if args.workdir.exists() and not args.keep_data:
        shutil.rmtree(args.workdir)
    labels_file = args.workdir / "labels.json"
    if args.keep_data and labels_file.exists():
        # Reuse the prepared dataset: parsing ~400 MB of MSP is the slow part.
        labels = json.loads(labels_file.read_text())
        stats_file = args.workdir / "stats.json"
        s = json.loads(stats_file.read_text()) if stats_file.exists() else {}
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

    results: dict[str, Any] = {"dataset": s, "runs": {}}
    for algorithm in args.algorithms:
        name = f"real_{algorithm}{args.run_tag}"
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
            )
            print(
                f"\n[run] {algorithm} over {s.get('library_records', '?')} references "
                f"(min_score={args.min_score}, min_matched_peaks={args.min_matched_peaks}, "
                f"q<={args.fdr_threshold}) …",
                flush=True,
            )
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
            score_thresholds=args.sweep_scores or (0.0,),
            q_thresholds=args.sweep_q or (float("inf"),),
        )
        results["runs"][name] = metrics
        if metrics.get("hits_passed_gates") == 0 and metrics["n_rows"] == 0:
            print(f"      {metrics['note']}")
            continue
        print(f"      rows={metrics['n_rows']} queries={metrics['n_queries']}")
        print(
            f"      {'min_score':>9} {'called':>7} {'top1':>7} {'prec':>7} "
            f"{'recall':>7} {'wF1':>7} {'multi':>6} {'multi_top1':>10}"
        )
        for row in metrics["sweep"]:
            mca = row["multi_candidate_top1_accuracy"]
            print(
                f"      {row['min_score']:>9.2f} {row['queries_called']:>7} "
                f"{row['top1_accuracy_connectivity']:>7.4f} "
                f"{row['precision_weighted']:>7.4f} {row['recall_weighted']:>7.4f} "
                f"{row['f1_weighted']:>7.4f} {row['multi_candidate_queries']:>6} "
                f"{(f'{mca:.4f}' if mca is not None else 'n/a'):>10}"
            )
        best = metrics.get("best_f1_operating_point")
        if best:
            print(
                f"      best weighted F1: {best['f1_weighted']:.4f} at "
                f"min_score={best['min_score']:.2f} "
                f"(top-1 {best['top1_accuracy_connectivity']:.4f}, "
                f"{best['queries_called']}/{metrics['n_queries']} called)"
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
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
