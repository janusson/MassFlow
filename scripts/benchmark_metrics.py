#!/usr/bin/env python3
"""Baseline accuracy benchmark + independent FDR calibration audit for MassFlow.

Fills two gaps in the repository:

1. **No classifier metrics exist anywhere in the repo.** ``scripts/benchmark_*.py``
   measure wall-clock and memory only, and ``tests/scientific_validation/`` is a
   known-answer suite over 6 hand-built spectra with no labelled false negatives
   (so recall is not computable there). This script generates a *labelled*
   synthetic benchmark whose hard case is isomer/analogue separation — several
   compounds share a precursor m/z, so the MS1 prefilter cannot separate them and
   only MS2 decides — runs the real ``massflow annotate`` CLI over it, and reports
   top-1 accuracy plus Precision / Recall / weighted F1 (sklearn, support-weighted).

2. **The shipped ground-truth fixture cannot exercise FDR.** Its decoy null is
   empty (``n_decoy_competitions: 0``, ``degraded_mode_flags: ['decoy_null_empty',
   'fdr_uncalibrated']``), so every p-value is 1.0 by contract. This script
   maximises decoy competitiveness (permissive gates + dense spectra, per
   ``docs/user-guide/scoring_logic.md`` §5.1) to obtain a **populated decoy null**
   and then audits the q/p mathematics three independent ways:

   a. **Formula re-derivation** — every per-query q/p from the real engine is
      recomputed with a from-scratch transcription of the documented formula and
      diffed against ``MassFlow.similarity.calibrate_query_level_fdr``.
   b. **Invariant checks** — monotone closure, decoy-first ties, the ``+1``
      pseudo-count, the empty-null ``1/N`` rank bound, and empty-target behaviour.
   c. **Monte-Carlo conservativeness** — exchangeable null target/decoy scores;
      proves the estimator controls FDR at its nominal level (observed FDP <= alpha).

Usage
-----
::

    uv run python scripts/benchmark_metrics.py
    uv run python scripts/benchmark_metrics.py --workdir /tmp/mfbench --groups 40
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Isomers within a group share a precursor m/z; groups are spaced far wider
# than any tolerance so the MS1 prefilter cleanly separates groups.
GROUP_PRECURSOR_START = 220.0
GROUP_PRECURSOR_SPACING = 6.0


# ---------------------------------------------------------------------------
# 1. Synthetic labelled benchmark
# ---------------------------------------------------------------------------


def _compound_fragments(
    group: int, isomer: int, rng: random.Random, dense: bool
) -> list[tuple[float, float]]:
    """A fragment pattern with a shared core plus **one** diagnostic peak.

    All isomers of a group share the entire core, so the cosine between any two
    siblings is high (~0.88 here) and every sibling survives the MS1 prefilter.
    The only thing separating an isomer from its siblings is a single diagnostic
    peak at a high m/z. That is the realistic isomeric/analogue case, and it
    gives the benchmark a genuine accuracy ceiling:

    * a query that retains the diagnostic peak is separable (its own reference
      scores 1.0 against ~0.88 for the siblings);
    * a query that loses it is *information-theoretically* ambiguous — its own
      reference and every sibling tie at the same score, so the correct answer
      cannot be recovered at all.

    ``dense`` adds many extra core peaks, which raises the probability that a
    randomly jittered decoy peak lands inside the scoring tolerance of a query
    peak. That is what populates a decoy null in the real engine.
    """
    core_base = 100.0 + 7.0 * (group % 12)
    n_core = 14 if dense else 6
    peaks: list[tuple[float, float]] = []
    for k in range(n_core):
        mz = round(core_base + 4.5 * k + rng.uniform(-0.002, 0.002), 4)
        peaks.append((mz, 600.0 - 8.0 * k))
    # The single diagnostic peak: highest m/z in the spectrum, so it is always
    # the last element after sorting and can be dropped unambiguously.
    peaks.append((round(300.0 + 3.0 * isomer, 4), 500.0))
    peaks.sort(key=lambda p: p[0])
    return [(mz, max(int_, 5.0)) for mz, int_ in peaks]


def _fmt_msp(name: str, precursor_mz: float, peaks: list[tuple[float, float]]) -> str:
    lines = [
        f"NAME: {name}",
        f"ID: {name}",
        f"PRECURSOR_MZ: {precursor_mz:.6f}",
        "CHARGE: 1",
        "IONMODE: Positive",
        "ADDUCT: [M+H]+",
        f"NUM PEAKS: {len(peaks)}",
    ]
    lines += [f"{mz:.4f}\t{int_:.1f}" for mz, int_ in peaks]
    return "\n".join(lines) + "\n"


def _fmt_mgf(name: str, precursor_mz: float, peaks: list[tuple[float, float]]) -> str:
    lines = [
        "BEGIN IONS",
        f"TITLE={name}",
        f"ID={name}",
        f"PEPMASS={precursor_mz:.6f}",
        "CHARGE=1",
        "MSLEVEL=2",
        "IONMODE=Positive",
        "ADDUCT=[M+H]+",
        "RTINSECONDS=60.0",
    ]
    lines += [f"{mz:.4f} {int_:.1f}" for mz, int_ in peaks]
    lines.append("END IONS")
    return "\n".join(lines) + "\n"


def _perturb(
    peaks: list[tuple[float, float]], kind: str, rng: random.Random
) -> list[tuple[float, float]]:
    """Apply a realistic acquisition perturbation.

    ``dropout`` deliberately removes the **diagnostic** peak (the last element
    by construction), which is the information-destroying case: the query then
    becomes indistinguishable from its siblings and no scorer can recover the
    right answer. Keeping this in the mix is what stops the benchmark from
    reporting a meaningless 1.000.
    """
    out = list(peaks)
    if kind == "perfect":
        return out
    if kind == "dropout":
        return [(mz, int_ * rng.uniform(0.7, 1.0)) for mz, int_ in out[:-1]]
    if kind == "jitter":
        return [
            (mz + rng.uniform(-0.006, 0.006), int_ * rng.uniform(0.6, 1.0))
            for mz, int_ in out
        ]
    if kind == "noise":
        extras = [
            (round(rng.uniform(80.0, 400.0), 4), rng.uniform(15.0, 60.0))
            for _ in range(12)
        ]
        return sorted(out + extras)
    raise ValueError(kind)


def build_dataset(
    workdir: Path,
    groups: int,
    isomers: int,
    queries_per_compound: int,
    foreign_queries: int,
    seed: int,
    dense: bool = False,
) -> dict[str, Any]:
    rng = random.Random(seed)
    workdir.mkdir(parents=True, exist_ok=True)

    library_entries: list[tuple[str, float, list[tuple[float, float]]]] = []
    labels: dict[str, Any] = {}

    for g in range(groups):
        precursor = GROUP_PRECURSOR_START + GROUP_PRECURSOR_SPACING * g
        for i in range(isomers):
            ref_id = f"REF_G{g:03d}_I{i:02d}"
            library_entries.append(
                (ref_id, precursor, _compound_fragments(g, i, rng, dense))
            )

    with (workdir / "library.msp").open("w") as fh:
        for ref_id, precursor, peaks in library_entries:
            fh.write(_fmt_msp(ref_id, precursor, peaks))

    kinds = ["perfect", "dropout", "jitter", "noise"]
    query_blocks: list[str] = []
    n_positive = 0
    for g in range(groups):
        precursor = GROUP_PRECURSOR_START + GROUP_PRECURSOR_SPACING * g
        for i in range(isomers):
            ref_id = f"REF_G{g:03d}_I{i:02d}"
            base_peaks = _compound_fragments(g, i, rng, dense)
            for k in range(queries_per_compound):
                kind = kinds[k % len(kinds)]
                q_id = f"Q_G{g:03d}_I{i:02d}_{kind}_{k}"
                query_blocks.append(
                    _fmt_mgf(q_id, precursor, _perturb(base_peaks, kind, rng))
                )
                labels[q_id] = {
                    "true_ref": ref_id,
                    "kind": kind,
                    "group": g,
                    "isomer": i,
                    "is_positive": True,
                }
                n_positive += 1

    foreign_base = GROUP_PRECURSOR_START + GROUP_PRECURSOR_SPACING * groups + 25.0
    for f in range(foreign_queries):
        q_id = f"Q_FOREIGN_{f:03d}"
        peaks = sorted(
            (round(rng.uniform(80.0, 500.0), 4), rng.uniform(50.0, 900.0))
            for _ in range(6)
        )
        query_blocks.append(_fmt_mgf(q_id, foreign_base + 3.0 * f, peaks))
        labels[q_id] = {
            "true_ref": None,
            "kind": "foreign",
            "group": None,
            "isomer": None,
            "is_positive": False,
        }

    with (workdir / "experiment.mgf").open("w") as fh:
        fh.write("\n".join(query_blocks))

    import yaml  # noqa: PLC0415

    def _config(name: str, min_score: float, min_peaks: int, fdr: float) -> dict:
        return {
            "project": {"name": name, "output_directory": str(workdir / name)},
            "input": {
                "input_path": str(workdir / "experiment.mgf"),
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
                "algorithm": "cosine",
                "ms1_tolerance": 0.02,
                "ms2_tolerance": 0.02,
                "min_score": min_score,
                "min_matched_peaks": min_peaks,
                "fdr_threshold": fdr,
            },
            "export": {"format": "csv"},
        }

    # 1. `candidates`: permissive gates so every candidate pair reaches the CSV,
    #    letting the metrics be swept over thresholds without re-running.
    # 2. `production`: the gates a user would actually ship.
    # 3. `dense`: same permissive gates; used only with --dense data, where many
    #    near-coincident peaks populate the decoy null for the FDR audit.
    configs = {
        "candidates": _config("run_candidates", 0.0, 1, 1.0),
        "production": _config("run_production", 0.7, 3, 0.05),
        "dense": _config("run_dense", 0.0, 1, 1.0),
    }
    for name, cfg in configs.items():
        (workdir / f"config_{name}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False)
        )
    (workdir / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))

    return {
        "labels": labels,
        "n_library": len(library_entries),
        "n_queries": len(query_blocks),
        "n_positive": n_positive,
        "n_foreign": foreign_queries,
        "groups": groups,
        "isomers": isomers,
        "dense": dense,
        "mean_peaks_per_spectrum": statistics.fmean(
            len(p) for _, _, p in library_entries
        ),
    }


# ---------------------------------------------------------------------------
# 2. Run the real CLI
# ---------------------------------------------------------------------------


def run_annotate(config_path: Path, run_dir: Path) -> dict[str, Any]:
    """Run ``massflow annotate`` and return its status plus the results CSV path."""
    console_script = REPO_ROOT / ".venv" / "bin" / "massflow"
    if console_script.exists():
        cmd = [str(console_script), "annotate", "--config", str(config_path)]
    else:  # pragma: no cover - fallback for non-uv environments
        cmd = [
            sys.executable,
            "-c",
            "import sys; from MassFlow.cli import main; sys.exit(main())",
            "annotate",
            "--config",
            str(config_path),
        ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=7200
    )
    csvs = sorted(run_dir.rglob("*_results.csv")) if run_dir.exists() else []
    return {
        "exit_code": proc.returncode,
        "results_csv": str(csvs[0]) if csvs else None,
        "n_results_csv": len(csvs),
        "stdout_tail": proc.stdout[-2500:],
        "stderr_tail": proc.stderr[-2500:],
    }


def _read_run_report(run_dir: Path) -> dict[str, Any]:
    """Read the run's own YAML provenance report (status, FDR summary, flags)."""
    import yaml  # noqa: PLC0415

    reports = sorted(run_dir.rglob("*_results.report.yaml")) if run_dir.exists() else []
    if not reports:
        return {}
    doc = yaml.safe_load(reports[0].read_text()) or {}
    return {
        "status": doc.get("status"),
        "spectra_loaded": doc.get("spectra_loaded"),
        "spectra_rejected": doc.get("spectra_rejected"),
        "hits_produced": doc.get("hits_produced"),
        "degraded_mode_flags": doc.get("degraded_mode_flags"),
        "warnings": doc.get("warnings"),
        "fdr": doc.get("fdr"),
        "n_warnings": len(doc.get("warnings") or []),
    }


def _read_results_csv(run_dir: Path) -> list[dict[str, Any]]:
    csvs = sorted(run_dir.rglob("*_results.csv")) if run_dir.exists() else []
    if not csvs:
        return []
    with csvs[0].open(newline="") as fh:
        return [dict(r) for r in _csv.DictReader(fh)]


def _col(rows: list[dict[str, Any]], *names: str) -> Optional[str]:
    if not rows:
        return None
    keys = {k.lower(): k for k in rows[0]}
    for n in names:
        if n.lower() in keys:
            return keys[n.lower()]
    return None


# ---------------------------------------------------------------------------
# 3. Precision / Recall / weighted F1
# ---------------------------------------------------------------------------


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def evaluate_accuracy(
    rows: list[dict[str, Any]],
    labels: dict[str, Any],
    score_thresholds: Iterable[float],
    q_thresholds: Iterable[float],
) -> dict[str, Any]:
    """Top-1 multiclass **and** pair-level binary P/R/F1 over a threshold grid."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    q_col = _col(rows, "q_value")
    score_col = _col(rows, "score")
    ref_col = _col(rows, "reference_name")
    query_col = _col(rows, "query_id")
    if not all([q_col, score_col, ref_col, query_col]):
        raise RuntimeError(
            f"result columns not recognised; got {list(rows[0]) if rows else 'no rows'}"
        )
    assert q_col and score_col and ref_col and query_col

    parsed: list[dict[str, Any]] = [
        {
            "query": str(r[query_col]),
            "ref": str(r[ref_col]),
            "score": _f(r[score_col]),
            "q": _f(r[q_col]),
        }
        for r in rows
        if r.get(ref_col)
    ]
    n_rows_missing_q = sum(1 for p in parsed if not np.isfinite(p["q"]))

    positive_queries = sorted(q for q, v in labels.items() if v.get("is_positive"))
    negative_queries = sorted(q for q, v in labels.items() if not v.get("is_positive"))
    negatives = set(negative_queries)

    out: dict[str, Any] = {
        "n_exported_rows": len(parsed),
        "n_rows_missing_q": n_rows_missing_q,
        "n_positive_queries": len(positive_queries),
        "n_foreign_queries": len(negative_queries),
        "thresholds": [],
    }

    for tau in score_thresholds:
        for alpha in q_thresholds:
            kept = [r for r in parsed if r["score"] >= tau and r["q"] <= alpha]
            best: dict[str, dict[str, Any]] = {}
            for r in kept:
                cur = best.get(r["query"])
                if cur is None or r["score"] > cur["score"]:
                    best[r["query"]] = r

            y_true = [labels[q]["true_ref"] for q in positive_queries]
            y_pred = [
                best[q]["ref"] if q in best else "NO_CALL" for q in positive_queries
            ]

            if y_true:
                acc = float(accuracy_score(y_true, y_pred))

                def _prf(avg: str) -> tuple[float, float, float]:
                    p, r, f, _ = precision_recall_fscore_support(
                        y_true,
                        y_pred,
                        average=avg,
                        zero_division=0,  # type: ignore[arg-type]
                    )
                    return float(p), float(r), float(f)

                p_w, r_w, f_w = _prf("weighted")
                p_mi, r_mi, f_mi = _prf("micro")
                p_ma, r_ma, f_ma = _prf("macro")
            else:  # pragma: no cover
                acc = p_w = r_w = f_w = p_mi = r_mi = f_mi = p_ma = r_ma = f_ma = float(
                    "nan"
                )

            tp = sum(
                1
                for r in kept
                if r["query"] in labels
                and labels[r["query"]]["is_positive"]
                and labels[r["query"]]["true_ref"] == r["ref"]
            )
            fp = len(kept) - tp
            fn = len(positive_queries) - tp
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = (
                2 * prec * rec / (prec + rec)
                if (prec == prec and rec == rec and (prec + rec) > 0)
                else float("nan")
            )
            foreign_called = {r["query"] for r in kept if r["query"] in negatives}

            # per-perturbation breakdown: shows *why* accuracy falls — the
            # dropout class is information-limited, not a scorer failure.
            per_kind: dict[str, dict[str, Any]] = {}
            for q in positive_queries:
                kind = labels[q].get("kind", "?")
                bucket = per_kind.setdefault(kind, {"n": 0, "n_correct": 0})
                bucket["n"] += 1
                if q in best and best[q]["ref"] == labels[q]["true_ref"]:
                    bucket["n_correct"] += 1
            for bucket in per_kind.values():
                bucket["accuracy"] = (
                    bucket["n_correct"] / bucket["n"] if bucket["n"] else float("nan")
                )

            out["thresholds"].append(
                {
                    "min_score": tau,
                    "q_threshold": alpha,
                    "n_kept_rows": len(kept),
                    "top1": {
                        "accuracy": acc,
                        "precision_weighted": p_w,
                        "recall_weighted": r_w,
                        "f1_weighted": f_w,
                        "precision_micro": p_mi,
                        "recall_micro": r_mi,
                        "f1_micro": f_mi,
                        "precision_macro": p_ma,
                        "recall_macro": r_ma,
                        "f1_macro": f_ma,
                    },
                    "pair_level": {
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision": float(prec),
                        "recall": float(rec),
                        "f1": float(f1),
                    },
                    "foreign_queries_with_a_call": len(foreign_called),
                    "n_foreign_queries": len(negative_queries),
                    "per_kind": per_kind,
                }
            )
    return out


def evaluate_fixture_ground_truth(manifest_path: Path) -> dict[str, Any]:
    """Contrast case: the shipped fixture. Shows precision only, no recall."""
    manifest = json.loads(manifest_path.read_text())
    caffeine_refs = {"REF_CAFFEINE", "REF_CAFFEINE_DUPLICATE", "REF_CAFFEINE_PARTIAL"}
    negatives = {
        "REF_CAFFEINE_PRECURSOR_VIOLATION",
        "REF_CAFFEINE_ADDUCT_VIOLATION",
        "REF_NOISE",
    }
    per_run: dict[str, Any] = {}
    for run_name in ("cosine", "modified_cosine", "consensus", "cascade", "cosine_rt"):
        run = manifest.get("runs", {}).get(run_name)
        if not run or "exported" not in run:
            continue
        rt_gated = (
            run.get("settings", {}).get("similarity", {}).get("rt_tolerance")
            is not None
        )
        negs = set(negatives) | ({"REF_CAFFEINE_RT_VIOLATION"} if rt_gated else set())
        tp = fp = 0
        for q_name, hits in run["exported"].items():
            if not isinstance(hits, list):
                continue
            is_pos_q = (
                q_name.startswith("Q_CAFFEINE") or q_name == "Q_NO_RT"
            ) and q_name != "Q_MISSING_PRECURSOR"
            for h in hits:
                ref = h.get("reference_name")
                if ref in caffeine_refs:
                    if is_pos_q:
                        tp += 1
                    else:
                        fp += 1
                elif ref in negs:
                    fp += 1
        per_run[run_name] = {
            "n_true_positive_hits": tp,
            "n_false_positive_hits": fp,
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "recall_computable": False,
            "recall_note": (
                "the fixture contains no labelled false negatives, so recall "
                "(and therefore F1) has an undefined denominator; only a "
                "precision proxy can be read off it"
            ),
            "status": run.get("status"),
            "degraded_mode_flags": run.get("degraded_mode_flags"),
            "fdr_summary": run.get("fdr_summary"),
        }
    return {"per_run": per_run}


# ---------------------------------------------------------------------------
# 4. Independent FDR / p-value audit
# ---------------------------------------------------------------------------


def reference_q_p(
    target_scores: list[float], decoy_scores: list[float]
) -> dict[float, tuple[float, float]]:
    """From-scratch transcription of the documented formula. No MassFlow code.

        FDR(t) = (1 + #{D >= t}) / #{T >= t}, clipped to [0, 1]
        q(s)   = min over t <= s of FDR(t)
        p(s)   = (1 + #{D >= s}) / (1 + #{D})
    Ties rank decoy-first (conservative). Empty decoy set -> q = 1/N for all
    targets, p = 1.0.
    """
    decoys = np.asarray(decoy_scores, dtype=float)
    targets = np.asarray(target_scores, dtype=float)
    if targets.size == 0:
        return {}
    if decoys.size == 0:
        return {float(s): (1.0 / targets.size, 1.0) for s in targets}
    out: dict[float, tuple[float, float]] = {}
    for s in targets:
        candidates = [t for t in np.unique(np.concatenate([targets, decoys])) if t <= s]
        q = 1.0
        for t in candidates:
            n_tgt = int((targets >= t).sum())
            if n_tgt == 0:
                continue
            q = min(q, min((1 + int((decoys >= t).sum())) / n_tgt, 1.0))
        p = (1 + int((decoys >= s).sum())) / (1 + decoys.size)
        out[float(s)] = (q, p)
    return out


def audit_fdr_math(workdir: Path, decoy_mz_shift_da: float = 1.0) -> dict[str, Any]:
    """Re-derive q/p from real engine output and diff against MassFlow's own."""
    from MassFlow.config import ProcessingConfig, SimilarityConfig
    from MassFlow.io import load_spectra
    from MassFlow.processing import process_spectra
    from MassFlow.similarity import calibrate_query_level_fdr, get_similarity_engine

    lib_spectra = list(
        process_spectra(
            load_spectra(workdir / "library.msp", file_format="msp"),
            ProcessingConfig(min_peaks=1),
        )
    )
    qry_spectra = list(
        process_spectra(
            load_spectra(workdir / "experiment.mgf", file_format="mgf"),
            ProcessingConfig(min_peaks=1),
        )
    )
    cfg = SimilarityConfig(
        algorithm="cosine",
        ms1_tolerance=0.02,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        fdr_threshold=1.0,
    )
    engine = get_similarity_engine(cfg)
    results = engine.search(
        qry_spectra,
        lib_spectra,
        include_decoys=True,
        decoy_min_relative_intensity=0.01,
        decoy_mz_shift_da=decoy_mz_shift_da,
    )

    best_target: dict[str, float] = {}
    best_decoy: dict[str, float] = {}
    for r in results:
        qid, score = r.get("query_id"), float(r["score"])
        if qid is None:
            continue
        if r.get("is_decoy", False):
            best_decoy[qid] = max(best_decoy.get(qid, -np.inf), score)
        else:
            best_target[qid] = max(best_target.get(qid, -np.inf), score)

    q_mf, p_mf, summary = calibrate_query_level_fdr(results)
    independent = reference_q_p(list(best_target.values()), list(best_decoy.values()))

    rows: list[dict[str, Any]] = []
    dq = dp = 0.0
    for qid, tscore in sorted(best_target.items()):
        if tscore not in independent:
            continue
        q_ref, p_ref = independent[tscore]
        q_here, p_here = q_mf.get(qid, float("nan")), p_mf.get(qid, float("nan"))
        d_q = abs(q_here - q_ref)
        d_p = abs(p_here - p_ref)
        dq, dp = max(dq, d_q), max(dp, d_p)
        if d_q > 1e-12 or d_p > 1e-12:
            rows.append(
                {
                    "query_id": qid,
                    "score": tscore,
                    "q_massflow": q_here,
                    "q_independent": q_ref,
                    "p_massflow": p_here,
                    "p_independent": p_ref,
                }
            )

    return {
        "engine": type(engine).__name__,
        "n_queries_scored": len(qry_spectra),
        "n_reference_spectra": len(lib_spectra),
        "n_results": len(results),
        "n_target_competitions": summary.get("n_target_competitions"),
        "n_decoy_competitions": summary.get("n_decoy_competitions"),
        "n_competing_queries": summary.get("n_competing_queries"),
        "decoy_null_empty": (summary.get("n_decoy_competitions") or 0) == 0,
        "n_decoy_scores_above_min_score": getattr(engine, "decoy_diagnostics", {}).get(
            "n_decoy_scores_above_min_score"
        )
        if isinstance(getattr(engine, "decoy_diagnostics", None), dict)
        else None,
        "n_queries_compared": len(best_target),
        "max_abs_q_error": dq,
        "max_abs_p_error": dp,
        "n_mismatches": len(rows),
        "mismatches": rows[:10],
        "distinct_q_values": sorted({round(v, 10) for v in q_mf.values()}),
        "distinct_p_values": sorted({round(v, 10) for v in p_mf.values()}),
    }


def audit_invariants() -> dict[str, Any]:
    """Verify the documented algebraic properties of ``calculate_fdr`` directly."""
    from MassFlow.similarity import calculate_fdr, calculate_empirical_p_values

    rng = np.random.default_rng(20260921)
    checks: dict[str, Any] = {}

    # (a) Monotone closure: q ranks are non-decreasing as score decreases, i.e.
    #     q[i] <= q[i+1] along the descending-score order.
    ok = True
    for _ in range(500):
        t = rng.uniform(0, 1, size=int(rng.integers(1, 40)))
        d = rng.uniform(0, 1, size=int(rng.integers(1, 40)))
        _, q, _ = calculate_fdr(t, d)
        if len(q) > 1 and np.any(np.diff(q) < -1e-15):
            ok = False
            break
    checks["monotone_q_non_decreasing_as_score_decreases"] = ok

    # (b) q must be non-increasing in the score itself: a better hit can never
    #     carry a worse q than a weaker one.
    ok = True
    for _ in range(200):
        t = rng.uniform(0, 1, size=30)
        d = rng.uniform(0, 1, size=30)
        scores, q, is_t = calculate_fdr(t, d)
        tgt = [(s, qq) for s, qq, it in zip(scores, q, is_t) if it]
        for i in range(len(tgt) - 1):
            if tgt[i][1] > tgt[i + 1][1] + 1e-15:
                ok = False
                break
    checks["q_nondecreasing_with_weaker_target_scores"] = ok

    # (c) decoy-first ties
    _, q, is_t = calculate_fdr(np.array([0.5]), np.array([0.5]))
    checks["tie_ranks_decoy_first"] = bool(not is_t[0] and is_t[1])
    checks["tie_fdr_is_conservative"] = bool(q[1] >= 0.5)

    # (d) +1 pseudo-count: no target can receive q == 0.0
    _, q, _ = calculate_fdr(np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.1]))
    checks["q_never_zero_pseudo_count"] = bool(np.all(q > 0.0))

    # (e) empty decoy null -> the 1/N rank bound for every competing query
    t = np.array([0.9, 0.8, 0.7, 0.6])
    _, q, _ = calculate_fdr(t, np.array([]))
    checks["empty_null_q_is_1_over_N"] = bool(np.allclose(q, 1.0 / len(t)))
    checks["empty_null_p_is_one"] = bool(
        np.allclose(calculate_empirical_p_values(t, np.array([])), 1.0)
    )

    # (f) empty target set -> decoys carry q == 1.0 and are not targets
    _, q2, is_t2 = calculate_fdr(np.array([]), np.array([0.4, 0.5]))
    checks["empty_targets_q_is_one"] = bool(np.allclose(q2, 1.0) and not is_t2.any())

    # (g) formula spot-check: a hand-computable case.
    #     targets 1.0/0.9/0.8, decoy 0.5 ->
    #       FDR ranks: 1/1, 1/2, 1/3, (1+1)/3
    #       suffix-min over ranks -> 1/3 for each target rank
    scores, q, is_t = calculate_fdr(np.array([1.0, 0.9, 0.8]), np.array([0.5]))
    q_of_targets = q[is_t]
    checks["hand_computed_case_matches"] = bool(
        q_of_targets.size == 3 and np.allclose(q_of_targets, [1 / 3, 1 / 3, 1 / 3])
    )
    return checks


def conservativeness_study(
    replicates: int = 400, n_null: int = 500, n_signal: int = 100, seed: int = 7
) -> dict[str, Any]:
    """Monte-Carlo check that the TDC estimator controls FDR at its nominal level.

    Under the null a query's best target score and best decoy score are
    **exchangeable** (both from F). Signal queries draw their target score from a
    stochastically larger G. Exchangeability is the estimator's only assumption,
    so the study tests it rather than assuming it.
    """
    from MassFlow.similarity import calculate_fdr

    rng = np.random.default_rng(seed)
    alphas = [0.01, 0.05, 0.10, 0.20]
    observed: dict[float, list[float]] = {a: [] for a in alphas}
    covered: dict[float, int] = {a: 0 for a in alphas}

    for _ in range(replicates):
        t_null = rng.beta(2.0, 5.0, size=n_null)
        d_null = rng.beta(2.0, 5.0, size=n_null)
        t_sig = np.maximum(
            rng.beta(6.0, 2.0, size=n_signal), rng.beta(2.0, 5.0, size=n_signal)
        )
        d_sig = rng.beta(2.0, 5.0, size=n_signal)

        targets = np.concatenate([t_null, t_sig])
        decoys = np.concatenate([d_null, d_sig])
        is_null = np.concatenate(
            [np.ones(n_null, dtype=bool), np.zeros(n_signal, dtype=bool)]
        )
        sorted_targets = targets

        scores, q_values, is_target = calculate_fdr(targets, decoys)
        # q is a function of the score; look it up per target query.
        q_of: dict[float, float] = {}
        for s, q, it in zip(scores, q_values, is_target):
            if it:
                q_of[float(s)] = float(q)

        for a in alphas:
            accepted_null = accepted_total = 0
            for s, null in zip(sorted_targets, is_null):
                q = q_of.get(float(s))
                if q is not None and q <= a:
                    accepted_total += 1
                    accepted_null += int(null)
            if accepted_total:
                fdp = accepted_null / accepted_total
                observed[a].append(fdp)
                if fdp <= a:
                    covered[a] += 1

    out: dict[str, Any] = {}
    for a in alphas:
        vals = observed[a]
        if not vals:
            out[f"alpha={a}"] = {
                "replicates_with_any_accept": 0,
                "mean_observed_fdp": None,
                "median_observed_fdp": None,
                "p95_observed_fdp": None,
                "max_observed_fdp": None,
                "nominal_alpha": a,
                "fraction_of_replicates_with_fdp_le_alpha": None,
                "fdr_controlled_on_average": None,
                "note": (
                    "no query was accepted at this threshold in any replicate — "
                    "with this signal/null mix the smallest attainable q-value "
                    "exceeds alpha, so the guarantee is vacuous here"
                ),
            }
            continue
        out[f"alpha={a}"] = {
            "replicates_with_any_accept": len(vals),
            "mean_observed_fdp": float(statistics.fmean(vals)),
            "median_observed_fdp": float(statistics.median(vals)),
            "p95_observed_fdp": float(np.percentile(vals, 95)),
            "max_observed_fdp": float(max(vals)),
            "nominal_alpha": a,
            "fraction_of_replicates_with_fdp_le_alpha": (
                covered[a] / replicates if replicates else float("nan")
            ),
            "fdr_controlled_on_average": bool(statistics.fmean(vals) <= a),
        }
    return out


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/massflow-benchmark"))
    parser.add_argument(
        "--groups",
        type=int,
        default=25,
        help="isomer groups (each group shares a precursor m/z)",
    )
    parser.add_argument("--isomers", type=int, default=4, help="compounds per group")
    parser.add_argument("--queries-per-compound", type=int, default=4)
    parser.add_argument("--foreign-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument(
        "--dense",
        action="store_true",
        help="pack more peaks per spectrum to populate the decoy null",
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-monte-carlo", action="store_true")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {}

    print("=" * 80)
    print("MassFlow baseline accuracy benchmark + FDR calibration audit")
    print("=" * 80)

    if not args.keep and args.workdir.exists():
        shutil.rmtree(args.workdir)

    # Accuracy data is deliberately NOT dense: dense spectra are near-identical
    # by construction (which is what populates the decoy null) and would make the
    # accuracy numbers uninformative. The two questions need different data.
    data_dir = args.workdir / "data"
    meta = build_dataset(
        data_dir,
        groups=args.groups,
        isomers=args.isomers,
        queries_per_compound=args.queries_per_compound,
        foreign_queries=args.foreign_queries,
        seed=args.seed,
        dense=False,
    )
    report["dataset"] = meta
    print(
        f"\n[dataset] library={meta['n_library']} refs  "
        f"queries={meta['n_queries']} ({meta['n_positive']} positive / "
        f"{meta['n_foreign']} foreign)  "
        f"mean peaks/spectrum={meta['mean_peaks_per_spectrum']:.1f}"
    )
    print(
        f"          hard case: {args.isomers} isomers share each precursor m/z and "
        f"the whole fragment core; only ONE diagnostic peak separates them"
    )

    null_dir: Optional[Path] = None
    if args.dense:
        dense_dir = args.workdir / "null_data"
        null_dir = dense_dir
        null_meta = build_dataset(
            dense_dir,
            groups=args.groups,
            isomers=args.isomers,
            queries_per_compound=args.queries_per_compound,
            foreign_queries=args.foreign_queries,
            seed=args.seed + 1,
            dense=True,
        )
        report["null_dataset"] = null_meta
        print(
            f"\n[null data] {null_meta['n_library']} refs, "
            f"mean peaks/spectrum={null_meta['mean_peaks_per_spectrum']:.1f} "
            f"— built only to populate the decoy null for the FDR audit"
        )

    # ---- CLI runs ----------------------------------------------------------
    # output_directory in the generated configs is <data_dir>/run_<name>
    run_dirs = {
        "candidates": data_dir / "run_candidates",
        "production": data_dir / "run_production",
    }

    runs: dict[str, Any] = {}
    for name, run_dir in run_dirs.items():
        if args.skip_pipeline:
            runs[name] = {
                "exit_code": 0,
                "results_csv": None,
                "stdout_tail": "(skipped)",
            }
        else:
            print(f"\n[run] massflow annotate --config {name} …", flush=True)
            runs[name] = run_annotate(data_dir / f"config_{name}.yaml", run_dir)
            print(
                f"      exit_code={runs[name]['exit_code']}  "
                f"csv={runs[name]['results_csv']}"
            )
            if runs[name]["exit_code"] != 0:
                print(runs[name]["stderr_tail"][-1200:])
        rep = _read_run_report(run_dir)
        runs[name]["report"] = rep
        if rep:
            print(
                f"      status={rep['status']} loaded={rep['spectra_loaded']} "
                f"rejected={rep['spectra_rejected']} hits={rep['hits_produced']} "
                f"flags={rep['degraded_mode_flags']}"
            )
            print(f"      fdr={rep['fdr']}")
    report["runs"] = runs

    # ---- Part A: accuracy --------------------------------------------------
    labels = meta["labels"]
    cand_rows = _read_results_csv(run_dirs["candidates"])
    report["accuracy"] = {"rows_source": "candidates config (permissive gates)"}
    if cand_rows:
        acc = evaluate_accuracy(
            cand_rows,
            labels,
            score_thresholds=[0.0, 0.3, 0.5, 0.7, 0.9],
            q_thresholds=[0.05, 0.2, 1.0],
        )
        report["accuracy"].update(acc)
        print(
            f"\n[accuracy] threshold sweep — {len(cand_rows)} exported rows, "
            f"{acc['n_positive_queries']} positive / {acc['n_foreign_queries']} "
            f"foreign queries"
        )
        hdr = (
            f"\n{'min_score':>9} {'q<=a':>6} {'rows':>6} {'acc':>7} "
            f"{'P_w':>7} {'R_w':>7} {'F1_w':>7} {'F1_mi':>7} {'F1_ma':>7} "
            f"{'P_pair':>7} {'R_pair':>7} {'F1_pair':>7} {'FP_fgn':>7}"
        )
        print(hdr)
        print("-" * len(hdr))
        for t in acc["thresholds"]:
            m, p = t["top1"], t["pair_level"]
            print(
                f"{t['min_score']:>9.2f} {t['q_threshold']:>6.2f} "
                f"{t['n_kept_rows']:>6} {m['accuracy']:>7.3f} "
                f"{m['precision_weighted']:>7.3f} {m['recall_weighted']:>7.3f} "
                f"{m['f1_weighted']:>7.3f} {m['f1_micro']:>7.3f} "
                f"{m['f1_macro']:>7.3f} "
                f"{p['precision']:>7.3f} {p['recall']:>7.3f} {p['f1']:>7.3f} "
                f"{t['foreign_queries_with_a_call']:>7}"
            )

        # shipped production gates: one headline operating point
        prod_rows = _read_results_csv(run_dirs["production"])
        if prod_rows:
            prod = evaluate_accuracy(prod_rows, labels, [0.0], [1.0])
            one = prod["thresholds"][0]
            report["production_gates"] = {
                "csv_rows_total": len(prod_rows),
                "rows_with_reference": one["n_kept_rows"],
                "no_hit_placeholder_rows": len(prod_rows) - one["n_kept_rows"],
                "hits_per_query": (
                    one["n_kept_rows"]
                    / len({r["query_id"] for r in prod_rows if r.get("reference_name")})
                    if prod_rows
                    else None
                ),
                "top1": one["top1"],
                "pair_level": one["pair_level"],
                "foreign_queries_with_a_call": one["foreign_queries_with_a_call"],
                "settings": "min_score=0.7, min_matched_peaks=3, fdr_threshold=0.05",
            }
            print(
                f"\n[production gates · min_score=0.7, min_matched_peaks=3, "
                f"q<=0.05] {one['n_kept_rows']} hits "
                f"({len(prod_rows)} CSV rows incl. {len(prod_rows) - one['n_kept_rows']} "
                f"no-hit placeholders)"
            )
            print(
                f"    top-1     accuracy={one['top1']['accuracy']:.3f}  "
                f"P_w={one['top1']['precision_weighted']:.3f} "
                f"R_w={one['top1']['recall_weighted']:.3f} "
                f"F1_w={one['top1']['f1_weighted']:.3f} "
                f"F1_mi={one['top1']['f1_micro']:.3f}"
            )
            print(
                f"    pair-level P={one['pair_level']['precision']:.3f} "
                f"R={one['pair_level']['recall']:.3f} "
                f"F1={one['pair_level']['f1']:.3f}   "
                f"foreign queries with a call: "
                f"{one['foreign_queries_with_a_call']}/{one['n_foreign_queries']}"
            )
            print("    per-perturbation top-1 accuracy (why the headline is not 1.0):")
            for kind, b in sorted(one["per_kind"].items()):
                print(f"      {kind:<9} n={b['n']:<5} acc={b['accuracy']:.3f}")
    else:
        print("\n[accuracy] NO ROWS EXPORTED — harness error, not a scientific result")
        report["accuracy"]["error"] = "no rows"

    # ---- shipped fixture for contrast --------------------------------------
    fixture = REPO_ROOT / "tests/scientific_validation/ground_truth_results.json"
    if fixture.exists():
        report["fixture_ground_truth"] = evaluate_fixture_ground_truth(fixture)
        print(
            "\n[fixture] shipped scientific-validation ground truth "
            "(6 hand-built spectra):"
        )
        for name, d in report["fixture_ground_truth"]["per_run"].items():
            prec = d["precision"]
            print(
                f"    {name:<16} TP={d['n_true_positive_hits']:<3} "
                f"FP={d['n_false_positive_hits']:<3} "
                f"prec={prec if prec == prec else float('nan'):.3f}  "
                f"recall=n/a  flags={d['degraded_mode_flags']}"
            )
        print(
            "    -> the fixture cannot yield recall or F1: it has no labelled "
            "false negatives"
        )

    # ---- Part B: FDR audit -------------------------------------------------
    print("\n" + "=" * 80)
    print("FDR target-decoy calibration audit")
    print("=" * 80)

    report["invariants"] = audit_invariants()
    print("\n[invariants] documented algebraic properties of calculate_fdr()")
    for k, v in report["invariants"].items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")

    try:
        audit_dir = null_dir if null_dir is not None else data_dir
        audit = audit_fdr_math(audit_dir)
        report["independent_recheck"] = audit
        print(
            "\n[recheck] recomputed q/p from real engine output and diffed "
            "against calibrate_query_level_fdr()"
        )
        print(
            f"    engine={audit['engine']}  refs={audit['n_reference_spectra']}  "
            f"queries={audit['n_queries_scored']}  results={audit['n_results']}"
        )
        print(
            f"    target competitions={audit['n_target_competitions']}  "
            f"decoy competitions={audit['n_decoy_competitions']}  "
            f"decoy_null_empty={audit['decoy_null_empty']}"
        )
        print(
            f"    queries compared={audit['n_queries_compared']}  "
            f"mismatches={audit['n_mismatches']}  "
            f"max|dq|={audit['max_abs_q_error']:.3g}  "
            f"max|dp|={audit['max_abs_p_error']:.3g}"
        )
        print(
            f"    distinct q={audit['distinct_q_values'][:6]}  "
            f"distinct p={audit['distinct_p_values'][:6]}"
        )
        if audit["decoy_null_empty"]:
            print(
                "    NOTE: decoy null empty at this configuration -> q is the "
                "conservative 1/N rank bound, not a decoy estimate. "
                "Re-run with --dense for a populated null."
            )
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        report["independent_recheck"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"\n[recheck] FAILED: {type(exc).__name__}: {exc}")

    if not args.skip_monte_carlo:
        print(
            f"\n[Monte-Carlo] {args.replicates} replicates, exchangeable null "
            f"target/decoy scores"
        )
        report["conservativeness"] = conservativeness_study(replicates=args.replicates)
        hdr = (
            f"\n{'alpha':>6} {'mean FDP':>9} {'median':>8} {'p95':>8} "
            f"{'max':>8} {'reps FDP<=a':>12} {'controlled':>11}"
        )
        print(hdr)
        print("-" * len(hdr))
        for d in report["conservativeness"].values():
            mean = d["mean_observed_fdp"]
            mean_s = f"{mean:.4f}" if mean is not None else "n/a"
            med = d["median_observed_fdp"]
            med_s = f"{med:.4f}" if med is not None else "n/a"
            p95 = d["p95_observed_fdp"]
            p95_s = f"{p95:.4f}" if p95 is not None else "n/a"
            mx = d["max_observed_fdp"]
            mx_s = f"{mx:.4f}" if mx is not None else "n/a"
            fr = d["fraction_of_replicates_with_fdp_le_alpha"]
            fr_s = f"{fr:.3f}" if fr is not None else "n/a"
            ctrl = d["fdr_controlled_on_average"]
            ctrl_s = str(ctrl) if ctrl is not None else "n/a"
            print(
                f"{d['nominal_alpha']:>6.2f} {mean_s:>9} {med_s:>8} {p95_s:>8} "
                f"{mx_s:>8} {fr_s:>12} {ctrl_s:>11}"
            )
        print(
            "\n  'controlled' = mean observed false-discovery proportion across "
            "replicates <= nominal alpha (the FDR guarantee is on the mean, not "
            "per replicate)."
        )

    out_path = args.workdir / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
