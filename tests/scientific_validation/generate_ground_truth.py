"""
Ground-truth fixture generator for the MassFlow scientific-validation suite.

This script is the *provenance* of ``tests/scientific_validation/*``:

* it defines the known-answer spectra (literature-anchored caffeine fragments
  plus clearly-labeled synthetic spectra),
* it writes the MSP fixture files (stable ``ID:`` metadata, so fixture
  assertions never depend on file stems),
* it verifies the CURRENT pipeline against an INDEPENDENT reference
  implementation of the published formulas (Watrous et al. 2012 cosine /
  modified cosine; Elias & Gygi 2007 target-decoy competition),
* and only then writes ``ground_truth_results.json`` — the recorded ground
  truth the test suite asserts against.

Regeneration policy: the fixtures change ONLY when the scientific contract
intentionally changes. If the pipeline diverges from the reference formulas
the script FAILS and refuses to write the manifest.

Reference formulas implemented here (not reusing MassFlow internals):

* cosine: dot product of matched-peak intensities (greedy assignment within
  tolerance; unambiguous in these fixtures) divided by the L2 norms of the
  full peak lists of both spectra;
* modified cosine: if ``|precursor_ref - precursor_query| <= tolerance`` the
  plain cosine applies; otherwise peaks are matched in the frame shifted by
  ``precursor_ref - precursor_query`` (the Watrous 2012 precursor-alignment),
  with norms still computed over the unshifted peak lists;
* q-value: ``q(s) = min over t <= s of FDR(t)`` with
  ``FDR(t) = (1 + #{queries: D_q >= t}) / #{queries: T_q >= t}``, ties
  ranked decoy-first (conservative), clipped to [0, 1];
* empirical p-value: ``p(s) = (1 + #{D_q >= s}) / (1 + #{queries with D_q})``
  (diagnostic only).

Run: ``uv run python tests/scientific_validation/generate_ground_truth.py``
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from matchms import Spectrum

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.workflow import run_annotation_pipeline

HERE = Path(__file__).resolve().parent
LIBRARY_FILE = HERE / "ground_truth_library.msp"
EXPERIMENT_FILE = HERE / "ground_truth_experiment.msp"
MANIFEST_FILE = HERE / "ground_truth_results.json"

# ---------------------------------------------------------------------------
# Known-answer spectra
# ---------------------------------------------------------------------------
# The caffeine [M+H]+ ion (C8H11N4O2+, m/z 195.0877) and its fragment series
# 138.0662 (loss of C2H3NO), 110.0717 (loss of CO), 83.0608 (loss of HCN)
# are the well-documented caffeine MS/MS fragments reported in public
# libraries (e.g. MassBank / GNPS records) and the caffeine metabolism
# literature. All other fixtures are explicitly synthetic.

CAFFEINE_PRECURSOR = 195.0877
CAFFEINE_PEAKS = [
    (138.0662, 999.0),
    (110.0717, 850.0),
    (83.0608, 420.0),
    (42.0344, 150.0),
]

LIBRARY_SPECTRA: list[dict[str, Any]] = [
    {
        "id": "REF_CAFFEINE",
        "name": "REF_CAFFEINE",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 2.0,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS,
    },
    {
        "id": "REF_CAFFEINE_DUPLICATE",
        "name": "REF_CAFFEINE_DUPLICATE",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 2.5,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS,
    },
    {
        "id": "REF_CAFFEINE_PARTIAL",
        "name": "REF_CAFFEINE_PARTIAL",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 2.1,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS[:2],
    },
    {
        # Identical fragments, precursor 10 Da away: excluded by the cosine
        # MS1 prefilter; excluded by modified cosine's shift alignment.
        "id": "REF_CAFFEINE_PRECURSOR_VIOLATION",
        "name": "REF_CAFFEINE_PRECURSOR_VIOLATION",
        "precursor_mz": CAFFEINE_PRECURSOR + 10.0,
        "retention_time": 2.0,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS,
    },
    {
        # Same fragments, opposite ionization mode: excluded by the adduct
        # gate even though the spectra themselves match perfectly.
        "id": "REF_CAFFEINE_ADDUCT_VIOLATION",
        "name": "REF_CAFFEINE_ADDUCT_VIOLATION",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 2.0,
        "adduct": "[M-H]-",
        "charge": -1,
        "peaks": CAFFEINE_PEAKS,
    },
    {
        # Same fragments, RT 18 min away: excluded only when rt_tolerance
        # is configured.
        "id": "REF_CAFFEINE_RT_VIOLATION",
        "name": "REF_CAFFEINE_RT_VIOLATION",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 20.0,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS,
    },
    {
        # Chemically unrelated fragments; same precursor (passes MS1, scores
        # ~0).
        "id": "REF_NOISE",
        "name": "REF_NOISE",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 3.0,
        "adduct": "[M+H]+",
        "peaks": [(500.1, 100.0), (600.2, 50.0), (700.3, 20.0)],
    },
]

EXPERIMENT_SPECTRA: list[dict[str, Any]] = [
    {
        # Exact match to REF_CAFFEINE / REF_CAFFEINE_DUPLICATE (score 1.0).
        "id": "Q_CAFFEINE_PERFECT",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 2.0,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS,
    },
    {
        # Near match: 3 of 4 fragments.
        "id": "Q_CAFFEINE_NEAR",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 2.0,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS[:3],
    },
    {
        # The same fragmentation shifted by +100 Da precursor: invisible to
        # cosine (MS1 gate), matched at 1.0 by modified cosine via the
        # precursor-shift alignment.
        "id": "Q_CAFFEINE_SHIFTED",
        "precursor_mz": CAFFEINE_PRECURSOR + 100.0,
        "retention_time": 2.0,
        "adduct": "[M+H]+",
        "peaks": [(mz + 100.0, intensity) for mz, intensity in CAFFEINE_PEAKS],
    },
    {
        # Weak match: 2 moderate-intensity fragments + a non-matching peak.
        "id": "Q_CAFFEINE_WEAK",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 2.0,
        "adduct": "[M+H]+",
        "peaks": [(138.0662, 500.0), (110.0717, 200.0), (250.0, 50.0)],
    },
    {
        # Large peak count (60 peaks): 4 matching + 56 non-matching peaks.
        "id": "Q_MANY_PEAKS",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": 2.0,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS + [(400.0 + 5.0 * i, 5.0) for i in range(56)],
    },
    {
        # No retention time: bypasses the RT filter when it is configured.
        "id": "Q_NO_RT",
        "precursor_mz": CAFFEINE_PRECURSOR,
        "retention_time": None,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS[:2],
    },
    {
        # Missing precursor: rejected by the I/O validation layer and counted
        # in spectra_rejected (observable, never silently analyzed).
        "id": "Q_MISSING_PRECURSOR",
        "precursor_mz": None,
        "retention_time": 2.0,
        "adduct": "[M+H]+",
        "peaks": CAFFEINE_PEAKS,
    },
]

FIXTURE_CONFIG: dict[str, dict[str, Any]] = {
    "processing": {"min_peaks": 1},
    "similarity": {
        "min_score": 0.0,
        "min_matched_peaks": 1,
        "fdr_threshold": 1.0,
        "ms1_tolerance": 0.5,
        "ms2_tolerance": 0.02,
        "rt_tolerance": None,
    },
}

RT_CONFIG: dict[str, dict[str, Any]] = {
    "processing": {"min_peaks": 1},
    "similarity": {
        "min_score": 0.0,
        "min_matched_peaks": 1,
        "fdr_threshold": 1.0,
        "ms1_tolerance": 0.5,
        "ms2_tolerance": 0.02,
        "rt_tolerance": 1.0,
    },
}

CASCADE_CONFIG: dict[str, dict[str, Any]] = {
    "processing": {"min_peaks": 1},
    "similarity": {
        "min_score": 0.0,
        "min_matched_peaks": 1,
        "fdr_threshold": 1.0,
        "ms1_tolerance": 0.5,
        "ms2_tolerance": 0.02,
        "rt_tolerance": None,
        "cascade_lower_bound": 0.0,
        "cascade_upper_bound": 0.3,
    },
}


# ---------------------------------------------------------------------------
# Independent reference implementations (published formulas, no MassFlow
# internals)
# ---------------------------------------------------------------------------


def _match_pairs(query_peaks, ref_peaks, tolerance: float, shift: float):
    """Greedy tolerance matching in (possibly shifted) m/z frame.

    Fixtures are constructed so no peak is within tolerance of two peaks of
    the other spectrum, making greedy == optimal assignment.
    """
    pairs = []
    ref_used = [False] * len(ref_peaks)
    for q_mz, q_int in query_peaks:
        best = None
        best_delta = None
        for ref_index, (r_mz, r_int) in enumerate(ref_peaks):
            if ref_used[ref_index]:
                continue
            delta = abs((q_mz + shift) - r_mz)
            if delta <= tolerance and (best_delta is None or delta < best_delta):
                best = ref_index
                best_delta = delta
        if best is not None:
            ref_used[best] = True
            pairs.append((q_mz, q_int, ref_peaks[best][0], ref_peaks[best][1]))
    return pairs


def reference_cosine(
    query_peaks,
    ref_peaks,
    tolerance: float,
    mz_power: float = 0.0,
    intensity_power: float = 1.0,
) -> tuple[float, int]:
    """Watrous/Stein cosine on matched peaks (norms over ALL peaks)."""
    pairs = _match_pairs(query_peaks, ref_peaks, tolerance, shift=0.0)
    if not pairs:
        return 0.0, 0
    dot = sum(
        (q_mz**mz_power * q_int**intensity_power)
        * (r_mz**mz_power * r_int**intensity_power)
        for q_mz, q_int, r_mz, r_int in pairs
    )
    q_norm = (
        sum(
            (mz**mz_power * intensity**intensity_power) ** 2
            for mz, intensity in query_peaks
        )
        ** 0.5
    )
    r_norm = (
        sum(
            (mz**mz_power * intensity**intensity_power) ** 2
            for mz, intensity in ref_peaks
        )
        ** 0.5
    )
    if q_norm == 0 or r_norm == 0:
        return 0.0, 0
    return dot / (q_norm * r_norm), len(pairs)


def reference_modified_cosine(
    query_peaks,
    query_precursor: float | None,
    ref_peaks,
    ref_precursor: float | None,
    tolerance: float,
    mz_power: float = 0.0,
    intensity_power: float = 1.0,
) -> tuple[float, int]:
    """Modified cosine with precursor-shift alignment (Watrous 2012).

    matchms semantics: when ``|precursor_ref - precursor_query| <= tolerance``
    the plain cosine applies. Otherwise peaks are matched in BOTH frames —
    the exact m/z frame and the frame shifted by ``precursor_ref -
    precursor_query`` — and the union is scored (greedy assignment over the
    combined pairs sorted by weight product). An MS1 window is therefore not
    a gate: a reference whose fragments coincide exactly is matched even
    when its precursor is far away.
    """
    if query_precursor is None or ref_precursor is None:
        return reference_cosine(
            query_peaks, ref_peaks, tolerance, mz_power, intensity_power
        )
    mass_shift = ref_precursor - query_precursor
    if abs(mass_shift) <= tolerance:
        return reference_cosine(
            query_peaks, ref_peaks, tolerance, mz_power, intensity_power
        )

    # Greedy assignment over the union of both frames, ordered by weight
    # product (mirrors matchms ``score_best_matches``).
    def weight(mz, intensity):
        return mz**mz_power * intensity**intensity_power

    pairs_zero = _match_pairs(query_peaks, ref_peaks, tolerance, shift=0.0)
    pairs_shift = _match_pairs(query_peaks, ref_peaks, tolerance, shift=mass_shift)
    all_pairs = [
        (q_mz, q_int, r_mz, r_int)
        for q_mz, q_int, r_mz, r_int in pairs_zero + pairs_shift
    ]
    if not all_pairs:
        return 0.0, 0
    # Sort by weight product descending; each peak usable once.
    all_pairs.sort(key=lambda p: weight(p[0], p[1]) * weight(p[2], p[3]), reverse=True)
    used_query = set()
    used_ref = set()
    matched: list[tuple] = []
    for q_mz, q_int, r_mz, r_int in all_pairs:
        q_key = (q_mz, q_int)
        r_key = (r_mz, r_int)
        if q_key in used_query or r_key in used_ref:
            continue
        used_query.add(q_key)
        used_ref.add(r_key)
        matched.append((q_mz, q_int, r_mz, r_int))

    dot = sum(
        weight(q_mz, q_int) * weight(r_mz, r_int)
        for q_mz, q_int, r_mz, r_int in matched
    )
    q_norm = sum(weight(mz, intensity) ** 2 for mz, intensity in query_peaks) ** 0.5
    r_norm = sum(weight(mz, intensity) ** 2 for mz, intensity in ref_peaks) ** 0.5
    if q_norm == 0 or r_norm == 0:
        return 0.0, 0
    return dot / (q_norm * r_norm), len(matched)


def reference_tdc(
    target_scores: list[float], decoy_scores: list[float]
) -> tuple[dict[float, float], dict[float, float]]:
    """Per-score q-values and p-values from the documented TDC formula.

    Ties rank decoys before targets (conservative). Returns mappings from
    each distinct target score to (q, p).
    """
    targets = np.array(target_scores, dtype=np.float64)
    decoys = np.array(decoy_scores, dtype=np.float64)

    if targets.size == 0:
        return {}, {}

    if decoys.size == 0:
        order = np.argsort(targets)[::-1]
        sorted_targets = targets[order]
        cum_targets = np.arange(1, sorted_targets.size + 1)
        fdr = np.minimum(1.0 / cum_targets, 1.0)
        q_sorted = np.minimum.accumulate(fdr[::-1])[::-1]
        rank_q: dict[float, float] = {}
        for score, q in zip(sorted_targets, q_sorted):
            rank_q[float(score)] = max(rank_q.get(float(score), 0.0), float(q))
        return rank_q, {float(s): 1.0 for s in sorted_targets}

    scores = np.concatenate([targets, decoys])
    is_target = np.concatenate(
        [np.ones(targets.size, dtype=bool), np.zeros(decoys.size, dtype=bool)]
    )
    # Descending by score; decoys first within ties.
    order = np.lexsort((is_target, -scores))
    sorted_scores = scores[order]
    sorted_is_target = is_target[order]
    cum_targets = np.cumsum(sorted_is_target)
    cum_decoys = np.cumsum(~sorted_is_target)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr_raw = (cum_decoys + 1.0) / cum_targets
    fdr = np.minimum(np.where(cum_targets > 0, fdr_raw, 1.0), 1.0)
    q_values = np.minimum.accumulate(fdr[::-1])[::-1]

    # q-value of a target score = q of its LAST (lowest-ranked) occurrence.
    ascending_scores = sorted_scores[::-1]
    ascending_q = q_values[::-1]
    q_by_score: dict[float, float] = {}
    for score in np.unique(targets):
        index = int(np.searchsorted(ascending_scores, float(score), side="right")) - 1
        q_by_score[float(score)] = float(ascending_q[index]) if index >= 0 else 1.0

    sorted_decoys = np.sort(decoys)
    positions = np.searchsorted(sorted_decoys, targets, side="left")
    greater_equal = decoys.size - positions
    p_values = (greater_equal.astype(float) + 1.0) / (decoys.size + 1.0)
    p_by_score = dict(zip((float(s) for s in targets), (float(p) for p in p_values)))
    return q_by_score, p_by_score


# ---------------------------------------------------------------------------
# Fixture file writing
# ---------------------------------------------------------------------------


def write_msp(path: Path, spectra: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for spec in spectra:
        lines.append(f"NAME: {spec.get('name', spec['id'])}")
        lines.append(f"ID: {spec['id']}")
        if spec.get("precursor_mz") is not None:
            lines.append(f"PRECURSOR_MZ: {spec['precursor_mz']}")
        charge = spec.get("charge", 1)
        lines.append(f"CHARGE: {charge}")
        if spec.get("retention_time") is not None:
            lines.append(f"RT: {spec['retention_time']}")
        lines.append(f"ADDUCT: {spec['adduct']}")
        lines.append(f"NUM PEAKS: {len(spec['peaks'])}")
        for mz, intensity in spec["peaks"]:
            lines.append(f"{mz}\t{intensity}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def build_config(
    output_directory: Path, algorithm: str, settings: dict
) -> MassFlowConfig:
    similarity = SimilarityConfig(algorithm=algorithm, **settings["similarity"])
    return MassFlowConfig(
        project=ProjectConfig(output_directory=output_directory),
        input=InputConfig(
            input_path=EXPERIMENT_FILE,
            library_path=LIBRARY_FILE,
            format="msp",
        ),
        processing=ProcessingConfig(**settings["processing"]),
        similarity=similarity,
    )


def load_processed(path: Path, processing_settings: dict) -> list[Spectrum]:
    """Load + process a fixture file exactly as the pipeline does."""
    from MassFlow.io import load_spectra
    from MassFlow.processing import process_spectra

    return list(
        process_spectra(
            load_spectra(path, file_format="msp"),
            ProcessingConfig(**processing_settings),
        )
    )


def reference_candidates(
    engine: str,
    queries: list[Spectrum],
    references: list[Spectrum],
    settings: dict,
) -> dict[str, list[dict[str, Any]]]:
    """Per-query expected candidate rows from the reference formulas.

    Mirrors each engine's actual sub-pipeline:

    * ``cosine``: MS1 + adduct + RT gates, then cosine;
    * ``modified_cosine``: adduct + RT gates (NO MS1 gate), then the
      two-frame modified cosine;
    * ``consensus``: each sub-engine collects with ``min_matched_peaks=0``
      and ``min_score=0.0`` (so non-matching pairs appear at score 0.0);
      the consensus score is the weight-normalized mean of the available
      sub-scores;
    * ``cascade``: stage 1 cosine (MS1 + adduct + RT gates, threshold
      ``cascade_lower_bound``) winnows the reference set; stage 2 modified
      cosine re-scores survivors against ``cascade_upper_bound``.
    """
    tolerance = settings["similarity"]["ms2_tolerance"]
    ms1_tolerance = settings["similarity"]["ms1_tolerance"]
    min_score = settings["similarity"]["min_score"]
    min_matched = settings["similarity"]["min_matched_peaks"]

    def passes_gates(q, r, apply_ms1: bool) -> bool:
        q_precursor = q.get("precursor_mz")
        r_precursor = r.get("precursor_mz")
        if (
            apply_ms1
            and q_precursor is not None
            and r_precursor is not None
            and abs(float(q_precursor) - float(r_precursor)) > ms1_tolerance
        ):
            return False
        q_adduct = q.get("adduct")
        r_adduct = r.get("adduct")
        if q_adduct is not None and r_adduct is not None:
            q_pos, q_neg = "+" in str(q_adduct), "-" in str(q_adduct)
            r_pos, r_neg = "+" in str(r_adduct), "-" in str(r_adduct)
            if (r_pos and q_neg) or (r_neg and q_pos):
                return False
        if settings["similarity"].get("rt_tolerance") is not None:
            q_rt = q.get("retention_time")
            r_rt = r.get("retention_time")
            if q_rt is not None and r_rt is not None:
                if (
                    abs(float(q_rt) - float(r_rt))
                    > settings["similarity"]["rt_tolerance"]
                ):
                    return False
        return True

    consensus_weights = settings["similarity"].get(
        "consensus_weights", {"cosine": 0.5, "modified_cosine": 0.5}
    )
    consensus_min_engines = settings["similarity"].get("consensus_min_engines", 1)
    cascade_lower = settings["similarity"].get("cascade_lower_bound", 0.0)
    cascade_upper = settings["similarity"].get("cascade_upper_bound", 0.0)

    def score_pair(q, r, algo: str) -> tuple[float, int]:
        q_peaks = [
            (float(m), float(i)) for m, i in zip(q.peaks.mz, q.peaks.intensities)
        ]
        r_peaks = [
            (float(m), float(i)) for m, i in zip(r.peaks.mz, r.peaks.intensities)
        ]
        q_precursor = q.get("precursor_mz")
        r_precursor = r.get("precursor_mz")
        if algo == "modified_cosine":
            return reference_modified_cosine(
                q_peaks,
                float(q_precursor) if q_precursor is not None else None,
                r_peaks,
                float(r_precursor) if r_precursor is not None else None,
                tolerance,
            )
        return reference_cosine(q_peaks, r_peaks, tolerance)

    per_query: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        q_id = str(query.get("id"))
        candidates: list[dict[str, Any]] = []

        for ref in references:
            r_id = str(ref.get("id"))
            if r_id.endswith("_decoy"):
                continue
            r_name = str(ref.get("compound_name") or ref.get("name"))

            if engine == "cosine":
                if not passes_gates(query, ref, apply_ms1=True):
                    continue
                score, matched = score_pair(query, ref, "cosine")
                if score < min_score or matched < min_matched:
                    continue
                candidates.append(
                    {
                        "reference_id": r_id,
                        "reference_name": r_name,
                        "score": float(score),
                        "matched_peaks": int(matched),
                    }
                )

            elif engine == "modified_cosine":
                if not passes_gates(query, ref, apply_ms1=False):
                    continue
                score, matched = score_pair(query, ref, "modified_cosine")
                if score < min_score or matched < min_matched:
                    continue
                candidates.append(
                    {
                        "reference_id": r_id,
                        "reference_name": r_name,
                        "score": float(score),
                        "matched_peaks": int(matched),
                    }
                )

            elif engine == "consensus":
                # Sub-engines collect with min_score=0.0, min_matched_peaks=0:
                # non-matching pairs therefore appear at score 0.0 / 0 peaks.
                # The cosine sub-engine's MS1 gate manifests as a 0.0
                # sentinel row (pinned contract: an MS1-excluded pair scores
                # 0.0), so the pair still enters the consensus bucket and
                # dilutes the weighted mean.
                if not passes_gates(query, ref, apply_ms1=False):
                    continue
                sub: dict[str, tuple[float, int]] = {}
                if passes_gates(query, ref, apply_ms1=True):
                    sub["cosine"] = score_pair(query, ref, "cosine")
                else:
                    sub["cosine"] = (0.0, 0)
                sub["modified_cosine"] = score_pair(query, ref, "modified_cosine")
                if len(sub) < consensus_min_engines:
                    continue
                total_weight = sum(consensus_weights.get(algo, 0.0) for algo in sub)
                if total_weight <= 0:
                    continue
                consensus_score = (
                    sum(sub[algo][0] * consensus_weights.get(algo, 0.0) for algo in sub)
                    / total_weight
                )
                if consensus_score < min_score:
                    continue
                # matched_peaks comes from the first sub-engine's row (the
                # consensus template): cosine when present, else modified.
                template_matched = (
                    sub["cosine"][1] if "cosine" in sub else sub["modified_cosine"][1]
                )
                candidates.append(
                    {
                        "reference_id": r_id,
                        "reference_name": r_name,
                        "score": float(consensus_score),
                        "matched_peaks": int(template_matched),
                    }
                )

            elif engine == "cascade":
                # Stage 1 (cosine, threshold cascade_lower): the reference
                # survives if it passes the adduct/RT gates AND its stage-1
                # row scores >= lower. MS1-excluded pairs survive via the
                # 0.0 sentinel when lower == 0.0.
                if not passes_gates(query, ref, apply_ms1=False):
                    continue
                if passes_gates(query, ref, apply_ms1=True):
                    stage1_score, _ = score_pair(query, ref, "cosine")
                else:
                    stage1_score = 0.0  # MS1 sentinel
                if stage1_score < cascade_lower:
                    continue
                # Stage 2 (modified cosine, threshold cascade_upper).
                score, matched = score_pair(query, ref, "modified_cosine")
                if score < cascade_upper:
                    continue
                candidates.append(
                    {
                        "reference_id": r_id,
                        "reference_name": r_name,
                        "score": float(score),
                        "matched_peaks": int(matched),
                    }
                )

        candidates.sort(key=lambda row: (-row["score"], row["reference_id"]))
        per_query[q_id] = candidates
    return per_query


def run_pipeline(
    output_directory: Path, algorithm: str, settings: dict
) -> tuple[MassFlowConfig, Any, Path]:
    config = build_config(output_directory, algorithm, settings)
    results = run_annotation_pipeline(config)
    csv_path = output_directory / f"{EXPERIMENT_FILE.stem}_results.csv"
    return config, results[0], csv_path


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    write_msp(LIBRARY_FILE, LIBRARY_SPECTRA)
    write_msp(EXPERIMENT_FILE, EXPERIMENT_SPECTRA)

    manifest: dict[str, Any] = {
        "description": (
            "Ground truth for the MassFlow scientific-validation suite. "
            "Generated by tests/scientific_validation/generate_ground_truth.py; "
            "every expected value is verified against an independent reference "
            "implementation of the published cosine / modified-cosine (Watrous "
            "2012) and target-decoy competition (Elias & Gygi 2007) formulas."
        ),
        "fixture_files": {
            "library": str(LIBRARY_FILE.relative_to(HERE)),
            "experiment": str(EXPERIMENT_FILE.relative_to(HERE)),
        },
        "config": FIXTURE_CONFIG,
        "cascade_config": CASCADE_CONFIG,
        "rt_config": RT_CONFIG,
        "runs": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # In-process processed spectra (shared by reference scoring and the
        # engine-level decoy extraction).
        queries = load_processed(EXPERIMENT_FILE, FIXTURE_CONFIG["processing"])
        references = load_processed(LIBRARY_FILE, FIXTURE_CONFIG["processing"])
        queries_by_id = {str(q.get("id")): q for q in queries}
        references_by_id = {str(r.get("id")): r for r in references}

        for algorithm, settings, label in [
            ("cosine", FIXTURE_CONFIG, "cosine"),
            ("modified_cosine", FIXTURE_CONFIG, "modified_cosine"),
            ("consensus", FIXTURE_CONFIG, "consensus"),
            ("cascade", CASCADE_CONFIG, "cascade"),
            ("cosine", RT_CONFIG, "cosine_rt"),
            ("cosine", FIXTURE_CONFIG, "cosine_zarr"),
        ]:
            run_dir = tmp_path / label
            run_dir.mkdir()
            if label == "cosine_zarr":
                config = build_config(run_dir, "cosine", FIXTURE_CONFIG)
                config.input.storage_backend = "zarr"
                run_annotation_pipeline(config)
                csv_path = run_dir / f"{EXPERIMENT_FILE.stem}_results.csv"
            else:
                config, result, csv_path = run_pipeline(run_dir, algorithm, settings)

            csv_rows = read_csv_rows(csv_path)
            digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            print(
                f"[{label}] status={result.status} hits={result.hits_produced} "
                f"loaded={result.spectra_loaded} rejected={result.spectra_rejected} "
                f"csv_sha256={digest[:16]}..."
            )

            # Per-query export from the CSV.
            exported: dict[str, list[dict[str, Any]]] = {}
            for row in csv_rows:
                q_id = row["query_id"]
                if not row.get("reference_name"):
                    continue  # unmatched-query base row
                exported.setdefault(q_id, []).append(
                    {
                        "reference_name": row["reference_name"],
                        "score": float(row["score"]),
                        "matched_peaks": int(row["matched_peaks"]),
                        "q_value": float(row["q_value"])
                        if row.get("q_value")
                        else None,
                        "p_value": float(row["p_value"])
                        if row.get("p_value")
                        else None,
                        "annotation_status": row["Annotation_Status"],
                        "score_breakdown": row.get("score_breakdown"),
                    }
                )

            if label == "cosine_zarr":
                # Storage-backend equivalence: identical CSV bytes.
                baseline = manifest["runs"]["cosine"]["csv_sha256"]
                assert digest == baseline, (
                    f"zarr run diverged from sqlite run: {digest} != {baseline}"
                )
                manifest["runs"]["cosine_zarr"] = {
                    "csv_sha256": digest,
                    "equivalent_to": "cosine",
                }
                continue

            # ---- Reference candidate sets ----
            expected_candidates = reference_candidates(
                algorithm, queries, references, settings
            )
            expected_export: dict[str, list[dict[str, Any]]] = {}
            for q_id, candidates in expected_candidates.items():
                expected_export[q_id] = [
                    {
                        "reference_name": references_by_id[c["reference_id"]].get(
                            "compound_name"
                        ),
                        "score": c["score"],
                        "matched_peaks": c["matched_peaks"],
                    }
                    for c in candidates
                ]

            # ---- FDR reference: per-query best target/decoy scores from the
            # real engine output (deterministic decoys). ----
            from MassFlow.similarity import get_similarity_engine

            engine = get_similarity_engine(
                SimilarityConfig(algorithm=algorithm, **settings["similarity"])
            )
            engine_hits = engine.search(
                queries,
                references,
                include_decoys=True,
                decoy_min_relative_intensity=0.01,
                decoy_mz_shift_da=1.0,
            )
            best_target: dict[str, float] = {}
            best_decoy: dict[str, float] = {}
            for hit in engine_hits:
                q_id = hit["query_id"]
                if hit.get("is_decoy"):
                    best_decoy[q_id] = max(
                        best_decoy.get(q_id, -np.inf), float(hit["score"])
                    )
                else:
                    best_target[q_id] = max(
                        best_target.get(q_id, -np.inf), float(hit["score"])
                    )

            # Reference TDC from the documented formula.
            target_list = [
                best_target[q]
                for q in queries_by_id
                if np.isfinite(best_target.get(q, -np.inf))
            ]
            decoy_list = [
                best_decoy[q]
                for q in queries_by_id
                if np.isfinite(best_decoy.get(q, -np.inf))
            ]
            q_by_score, p_by_score = reference_tdc(target_list, decoy_list)

            per_query_truth: dict[str, Any] = {}
            for q_id in queries_by_id:
                target = best_target.get(q_id, np.inf)
                decoy = best_decoy.get(q_id, np.inf)
                finite_target = np.isfinite(target)
                if finite_target:
                    q_val = q_by_score[float(target)]
                    p_val = p_by_score[float(target)]
                else:
                    q_val, p_val = 1.0, 1.0
                per_query_truth[q_id] = {
                    "best_target_score": float(target) if finite_target else None,
                    "best_decoy_score": float(decoy) if np.isfinite(decoy) else None,
                    "q_value": q_val,
                    "p_value": p_val,
                }

            # ---- Verify pipeline output against the reference ----
            for q_id, truth in per_query_truth.items():
                expected_rows = expected_export.get(q_id, [])
                actual_rows = exported.get(q_id, [])
                assert len(actual_rows) == len(expected_rows), (
                    f"[{label}] {q_id}: exported {len(actual_rows)} rows, "
                    f"reference predicts {len(expected_rows)}"
                )
                # Score ordering: rows must be sorted by score descending.
                scores = [row["score"] for row in actual_rows]
                assert scores == sorted(scores, reverse=True), (
                    f"[{label}] {q_id}: rows not sorted by score descending"
                )

                # Within equal-score groups the row order is not part of the
                # scientific contract (numpy quicksort tie order is
                # deterministic but arbitrary): compare groups as sets.
                def score_groups(rows):
                    groups: dict[float, set[tuple[str, int]]] = {}
                    for row in rows:
                        groups.setdefault(round(float(row["score"]), 12), set()).add(
                            (row["reference_name"], int(row["matched_peaks"]))
                        )
                    return groups

                assert score_groups(actual_rows) == score_groups(expected_rows), (
                    f"[{label}] {q_id}: candidate sets differ:\n"
                    f"  actual:   {score_groups(actual_rows)}\n"
                    f"  expected: {score_groups(expected_rows)}"
                )
                # Per-row score equality (scores are score-group-keyed, so
                # verify each expected row has a matching actual score).
                actual_by_name = {row["reference_name"]: row for row in actual_rows}
                for expected in expected_rows:
                    actual = actual_by_name[expected["reference_name"]]
                    assert actual["matched_peaks"] == expected["matched_peaks"], (
                        f"[{label}] {q_id} vs {expected['reference_name']}: "
                        f"{actual['matched_peaks']} != {expected['matched_peaks']} matched peaks"
                    )
                    assert abs(actual["score"] - expected["score"]) <= 1e-12, (
                        f"[{label}] {q_id} vs {expected['reference_name']}: "
                        f"score {actual['score']} != reference {expected['score']}"
                    )
                # q-value traceability: every exported row carries the query's
                # reference q-value.
                if actual_rows:
                    for row in actual_rows:
                        assert abs(row["q_value"] - truth["q_value"]) <= 1e-12, (
                            f"[{label}] {q_id}: row q={row['q_value']} != "
                            f"reference q={truth['q_value']}"
                        )
                    # Annotation status of the best row.
                    best_row = actual_rows[0]
                    expected_status = (
                        "Matched" if best_row["score"] >= 0.9 else "Putative"
                    )
                    assert best_row["annotation_status"] == expected_status, (
                        f"[{label}] {q_id}: status {best_row['annotation_status']} "
                        f"!= {expected_status}"
                    )

            # Unmatched queries must have a base row with Unknown status.
            for q_id in queries_by_id:
                if not per_query_truth[q_id]["best_target_score"]:
                    continue
                has_hit_rows = bool(exported.get(q_id))
                if not has_hit_rows:
                    base_rows = [r for r in csv_rows if r["query_id"] == q_id]
                    assert base_rows and not base_rows[0].get("reference_name"), (
                        f"[{label}] {q_id}: missing Unknown base row"
                    )
                    assert base_rows[0]["Annotation_Status"] == "Unknown"

            manifest["runs"][label] = {
                "algorithm": algorithm,
                "settings": settings,
                "csv_sha256": digest,
                "status": result.status,
                "spectra_loaded": result.spectra_loaded,
                "spectra_rejected": result.spectra_rejected,
                "hits_produced": result.hits_produced,
                "library_size": result.fdr_summary.get("library_size"),
                "fdr_summary": result.fdr_summary,
                "warnings": list(result.warnings),
                "degraded_mode_flags": list(result.degraded_mode_flags),
                "queries": per_query_truth,
                "exported": exported,
            }
            print(
                f"    ok: reference formulas match pipeline for all {len(queries_by_id)} queries"
            )

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"Wrote {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
