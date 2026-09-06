"""
Golden scientific-validation suite for MassFlow.

These tests assert SCIENTIFIC MEANING, not merely code output. Every
expected value is a known answer derived from the published formulas —
cosine / modified cosine (Watrous et al. 2012, PNAS), target-decoy
competition (Elias & Gygi 2007, Nat. Methods), spectral-entropy decoys
(Li et al. 2021, Nat. Methods) — and cross-checked against an independent
reference implementation in
``tests/scientific_validation/generate_ground_truth.py``.

The fixture set (``tests/scientific_validation/ground_truth_*.msp``) is
anchored to the well-documented caffeine [M+H]+ fragment series (m/z
195.0877 -> 138.0662, loss of C2H3NO; -> 110.0717, loss of CO; ->
83.0608, loss of HCN; plus the characteristic 42.0344 fragment), as
reported in public MS/MS libraries (MassBank/GNPS) and the caffeine
metabolism literature. All other fixture spectra are explicitly synthetic
and their expected scores are hand-computed from the formulas.

``ground_truth_results.json`` is the recorded ground truth: it was written
by the generator ONLY after the pipeline reproduced the reference formulas
for every engine, every query, every candidate, and every q-value. A code
change that alters any scientific output (scoring, candidate sets, matched
peak counts, FDR calibration, ranking, export) breaks these tests loudly.
Fixtures are regenerated only when the scientific contract intentionally
changes: ``uv run python tests/scientific_validation/generate_ground_truth.py``
(fails loudly on any divergence).

Coverage of the contract (see ``docs/user-guide/scoring_logic.md``):

* perfect matches, near matches, unmatched queries;
* precursor-mass violations, adduct violations, RT filtering, missing
  precursors, missing RT (filter bypass);
* modified-cosine precursor-shift alignment (both frames);
* duplicate scores and ties (conservative decoy-first ranking);
* decoy generation (determinism, entropy preservation, decoy competition);
* per-query target-decoy FDR: q-value traceability, competition unit,
  1/N bound when the decoy null is empty, p-value as diagnostic only;
* large peak counts; multi-engine runs (cosine, modified_cosine,
  consensus, cascade); SQLite/Zarr backend equivalence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.io import load_spectra
from MassFlow.processing import process_spectra
from MassFlow.similarity import (
    SimilarityEngine,
    generate_decoys,
    get_similarity_engine,
    spectral_entropy,
)
from MassFlow.workflow import run_annotation_pipeline

pytestmark = pytest.mark.scientific

FIXTURE_DIR = Path(__file__).parent / "scientific_validation"
LIBRARY_FILE = FIXTURE_DIR / "ground_truth_library.msp"
EXPERIMENT_FILE = FIXTURE_DIR / "ground_truth_experiment.msp"
MANIFEST_FILE = FIXTURE_DIR / "ground_truth_results.json"

MANIFEST = json.loads(MANIFEST_FILE.read_text())

CAFFEINE_PRECURSOR = 195.0877
CAFFEINE_FRAGMENTS = [138.0662, 110.0717, 83.0608, 42.0344]

# Literature-anchored known-answer scores (Watrous cosine; see generator).
SCORE_PERFECT = 1.0
SCORE_NEAR = 0.9941215181521759  # 3 of 4 fragments
SCORE_PARTIAL_VS_FULL = 0.9467702576772113  # 2 fragments vs full spectrum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fixture_config(
    algorithm: str, tmp_path: Path, settings: dict[str, Any] | None = None
) -> MassFlowConfig:
    """Build the fixture pipeline config exactly as recorded in the manifest."""
    settings = settings or MANIFEST["config"]
    return MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path),
        input=InputConfig(
            input_path=EXPERIMENT_FILE,
            library_path=LIBRARY_FILE,
            format="msp",
        ),
        processing=ProcessingConfig(**settings["processing"]),
        similarity=SimilarityConfig(algorithm=algorithm, **settings["similarity"]),
    )


def run_fixture_pipeline(
    tmp_path: Path, algorithm: str, settings: dict[str, Any] | None = None
):
    """Run the pipeline on the fixture files; return (file_result, csv_rows)."""
    config = fixture_config(algorithm, tmp_path, settings)
    results = run_annotation_pipeline(config)
    assert len(results) == 1
    csv_path = tmp_path / f"{EXPERIMENT_FILE.stem}_results.csv"
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return results[0], rows, csv_path


def load_processed(path: Path) -> list[Spectrum]:
    return list(
        process_spectra(
            load_spectra(path, file_format="msp"), ProcessingConfig(min_peaks=1)
        )
    )


def rows_for(rows: list[dict[str, str]], query_id: str) -> list[dict[str, str]]:
    return [r for r in rows if r["query_id"] == query_id and r.get("reference_name")]


def csv_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_tdc(
    target_scores: list[float], decoy_scores: list[float]
) -> dict[float, tuple[float, float]]:
    """Independent TDC recomputation (documented formula, decoy-first ties).

    Returns {target score: (q, p)}. This mirrors the definition in
    ``docs/user-guide/scoring_logic.md``, implemented here from scratch so
    the test does not share code with the implementation under test.
    """
    targets = np.array(target_scores, dtype=np.float64)
    decoys = np.array(decoy_scores, dtype=np.float64)
    if targets.size == 0:
        return {}

    if decoys.size == 0:
        order = np.argsort(targets)[::-1]
        sorted_targets = targets[order]
        fdr = np.minimum(1.0 / np.arange(1, sorted_targets.size + 1), 1.0)
        q_sorted = np.minimum.accumulate(fdr[::-1])[::-1]
        return {float(s): (float(q), 1.0) for s, q in zip(sorted_targets, q_sorted)}

    scores = np.concatenate([targets, decoys])
    is_target = np.concatenate(
        [np.ones(targets.size, dtype=bool), np.zeros(decoys.size, dtype=bool)]
    )
    order = np.lexsort((is_target, -scores))  # decoys first within ties
    sorted_scores = scores[order]
    sorted_is_target = is_target[order]
    cum_targets = np.cumsum(sorted_is_target)
    cum_decoys = np.cumsum(~sorted_is_target)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr_raw = (cum_decoys + 1.0) / cum_targets
    fdr = np.minimum(np.where(cum_targets > 0, fdr_raw, 1.0), 1.0)
    q_values = np.minimum.accumulate(fdr[::-1])[::-1]

    ascending_scores = sorted_scores[::-1]
    ascending_q = q_values[::-1]
    q_by_score: dict[float, float] = {}
    for score in np.unique(targets):
        index = int(np.searchsorted(ascending_scores, float(score), side="right")) - 1
        q_by_score[float(score)] = float(ascending_q[index]) if index >= 0 else 1.0

    sorted_decoys = np.sort(decoys)
    positions = np.searchsorted(sorted_decoys, targets, side="left")
    p_values = (decoys.size - positions + 1.0) / (decoys.size + 1.0)
    p_by_score = dict(zip((float(s) for s in targets), (float(p) for p in p_values)))
    return {float(s): (q_by_score[float(s)], p_by_score[float(s)]) for s in targets}


# ---------------------------------------------------------------------------
# 1. Fixture and manifest integrity
# ---------------------------------------------------------------------------


class TestManifestIntegrity:
    """The recorded ground truth must be self-consistent before it is used."""

    def test_fixture_files_exist_and_are_nonempty(self) -> None:
        for path in (LIBRARY_FILE, EXPERIMENT_FILE, MANIFEST_FILE):
            assert path.is_file() and path.stat().st_size > 0

    def test_library_contains_expected_references(self) -> None:
        ids = {str(s.get("id")) for s in load_processed(LIBRARY_FILE)}
        assert {
            "REF_CAFFEINE",
            "REF_CAFFEINE_DUPLICATE",
            "REF_CAFFEINE_PARTIAL",
            "REF_CAFFEINE_PRECURSOR_VIOLATION",
            "REF_CAFFEINE_ADDUCT_VIOLATION",
            "REF_CAFFEINE_RT_VIOLATION",
            "REF_NOISE",
        } <= ids
        assert len(ids) == 7

    def test_caffeine_fragments_are_literature_anchored(self) -> None:
        """The anchor fragments must carry the documented neutral losses."""
        losses = [
            CAFFEINE_PRECURSOR - CAFFEINE_FRAGMENTS[0],  # C2H3NO = 57.0215
            CAFFEINE_FRAGMENTS[0] - CAFFEINE_FRAGMENTS[1],  # CO = 27.9949
            CAFFEINE_FRAGMENTS[1] - CAFFEINE_FRAGMENTS[2],  # HCN = 27.0109
        ]
        expected = [57.0215, 27.9949, 27.0109]
        for actual, wanted in zip(losses, expected):
            assert abs(actual - wanted) < 0.001, (
                f"fragment series no longer matches the documented caffeine "
                f"losses: {losses}"
            )

    def test_manifest_records_all_engine_runs(self) -> None:
        assert set(MANIFEST["runs"]) == {
            "cosine",
            "modified_cosine",
            "consensus",
            "cascade",
            "cosine_rt",
            "cosine_zarr",
        }

    def test_manifest_q_values_match_independent_tdc(self) -> None:
        """Recompute every recorded q/p from the recorded per-query best
        scores with an independent implementation of the documented formula."""
        for label in ("cosine", "modified_cosine", "consensus", "cascade", "cosine_rt"):
            run = MANIFEST["runs"][label]
            targets = [
                q["best_target_score"]
                for q in run["queries"].values()
                if q["best_target_score"] is not None
            ]
            decoys = [
                q["best_decoy_score"]
                for q in run["queries"].values()
                if q["best_decoy_score"] is not None
            ]
            expected = reference_tdc(targets, decoys)
            for query_truth in run["queries"].values():
                score = query_truth["best_target_score"]
                if score is None:
                    assert query_truth["q_value"] == 1.0
                    assert query_truth["p_value"] == 1.0
                    continue
                q, p = expected[float(score)]
                assert abs(query_truth["q_value"] - q) < 1e-12
                assert abs(query_truth["p_value"] - p) < 1e-12


# ---------------------------------------------------------------------------
# 2. Engine-level known answers (self-contained, from the published formulas)
# ---------------------------------------------------------------------------


class TestKnownAnswerScores:
    """Hand-computed scores from the Watrous/Stein cosine definition."""

    @pytest.fixture(scope="class")
    def spectra(self):
        return {
            "queries": load_processed(EXPERIMENT_FILE),
            "references": load_processed(LIBRARY_FILE),
        }

    def _search(self, algorithm: str, query: Spectrum, ref: Spectrum, **overrides):
        config = SimilarityConfig(
            algorithm=algorithm,
            min_score=0.0,
            min_matched_peaks=0,
            ms1_tolerance=0.5,
            ms2_tolerance=0.02,
            **overrides,
        )
        engine = get_similarity_engine(config)
        return engine.search(
            [query],
            [ref],
            include_decoys=False,
            decoy_min_relative_intensity=0.01,
            decoy_mz_shift_da=1.0,
        )

    @pytest.fixture(scope="class")
    def q_perfect(self, spectra) -> Spectrum:
        return next(
            s for s in spectra["queries"] if s.get("id") == "Q_CAFFEINE_PERFECT"
        )

    @pytest.fixture(scope="class")
    def ref_caffeine(self, spectra) -> Spectrum:
        return next(s for s in spectra["references"] if s.get("id") == "REF_CAFFEINE")

    @pytest.fixture(scope="class")
    def ref_partial(self, spectra) -> Spectrum:
        return next(
            s for s in spectra["references"] if s.get("id") == "REF_CAFFEINE_PARTIAL"
        )

    def test_perfect_match_scores_1_with_4_matched_peaks(
        self, q_perfect, ref_caffeine
    ) -> None:
        """Identical peak lists → cosine = 1.0, all 4 peaks matched."""
        for algorithm in ("cosine", "modified_cosine"):
            results = self._search(algorithm, q_perfect, ref_caffeine)
            assert len(results) == 1
            assert results[0]["score"] == pytest.approx(1.0, abs=1e-12)
            assert results[0]["matched_peaks"] == 4
            assert not results[0]["is_decoy"]

    def test_near_match_score_is_hand_computed(self, spectra) -> None:
        """3 of 4 fragments → cos = sqrt(1896901/1919401) (Watrous formula)."""
        query = next(s for s in spectra["queries"] if s.get("id") == "Q_CAFFEINE_NEAR")
        ref = next(s for s in spectra["references"] if s.get("id") == "REF_CAFFEINE")
        results = self._search("cosine", query, ref)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(SCORE_NEAR, abs=1e-12)
        assert results[0]["matched_peaks"] == 3

    def test_partial_reference_vs_full_query(self, q_perfect, ref_partial) -> None:
        """2 matched peaks; the query's unmatched peaks lower the norm."""
        results = self._search("cosine", q_perfect, ref_partial)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(SCORE_PARTIAL_VS_FULL, abs=1e-12)
        assert results[0]["matched_peaks"] == 2

    def test_precursor_violation_excluded_by_cosine_ms1(
        self, spectra, q_perfect
    ) -> None:
        """Identical fragments 10 Da outside the MS1 window: cosine gates the
        pair (0.0), modified cosine still matches in the exact frame."""
        ref = next(
            s
            for s in spectra["references"]
            if s.get("id") == "REF_CAFFEINE_PRECURSOR_VIOLATION"
        )
        cosine = self._search("cosine", q_perfect, ref)
        assert cosine[0]["score"] == pytest.approx(0.0, abs=1e-12)
        assert cosine[0]["matched_peaks"] == 0
        modified = self._search("modified_cosine", q_perfect, ref)
        assert modified[0]["score"] == pytest.approx(1.0, abs=1e-12)
        assert modified[0]["matched_peaks"] == 4

    def test_adduct_violation_excluded_by_both_engines(
        self, spectra, q_perfect
    ) -> None:
        """Same fragments, [M-H]- vs [M+H]+: the adduct gate drops the pair."""
        ref = next(
            s
            for s in spectra["references"]
            if s.get("id") == "REF_CAFFEINE_ADDUCT_VIOLATION"
        )
        for algorithm in ("cosine", "modified_cosine"):
            results = self._search(algorithm, q_perfect, ref)
            assert results == [], f"{algorithm} must reject the adduct violation"

    def test_rt_violation_excluded_only_when_rt_tolerance_set(
        self, spectra, q_perfect
    ) -> None:
        ref = next(
            s
            for s in spectra["references"]
            if s.get("id") == "REF_CAFFEINE_RT_VIOLATION"
        )
        without_rt = self._search("cosine", q_perfect, ref)
        assert len(without_rt) == 1
        with_rt = self._search("cosine", q_perfect, ref, rt_tolerance=1.0)
        assert with_rt == []

    def test_shifted_query_invisible_to_cosine_found_by_modified(
        self, spectra, ref_caffeine
    ) -> None:
        """The +100 Da precursor-shifted query: cosine (MS1-gated) cannot see
        it; modified cosine aligns it via the precursor shift to 1.0."""
        query = next(
            s for s in spectra["queries"] if s.get("id") == "Q_CAFFEINE_SHIFTED"
        )
        cosine = self._search("cosine", query, ref_caffeine)
        assert cosine[0]["score"] == pytest.approx(0.0, abs=1e-12)
        modified = self._search("modified_cosine", query, ref_caffeine)
        assert len(modified) == 1
        assert modified[0]["score"] == pytest.approx(1.0, abs=1e-12)
        assert modified[0]["matched_peaks"] == 4

    def test_duplicate_reference_scores_identically(self, spectra, q_perfect) -> None:
        """A duplicate library entry must score byte-identically."""
        refs = [
            s
            for s in spectra["references"]
            if s.get("id") in ("REF_CAFFEINE", "REF_CAFFEINE_DUPLICATE")
        ]
        scores = {
            str(r.get("id")): self._search("cosine", q_perfect, r)[0]["score"]
            for r in refs
        }
        assert scores["REF_CAFFEINE"] == scores["REF_CAFFEINE_DUPLICATE"] == 1.0

    def test_large_peak_count_round_trip(self, spectra, ref_caffeine) -> None:
        """60-peak query: no peak loss through loading+processing; self-match
        counts all 60 peaks; the 4 real fragments match the reference."""
        query = next(s for s in spectra["queries"] if s.get("id") == "Q_MANY_PEAKS")
        assert len(query.peaks.mz) == 60
        self_match = self._search("cosine", query, query)
        assert self_match[0]["score"] == pytest.approx(1.0, abs=1e-12)
        assert self_match[0]["matched_peaks"] == 60
        vs_caffeine = self._search("cosine", query, ref_caffeine)
        assert vs_caffeine[0]["matched_peaks"] == 4
        assert vs_caffeine[0]["score"] == pytest.approx(
            MANIFEST["runs"]["cosine"]["queries"]["Q_MANY_PEAKS"]["best_target_score"],
            abs=1e-12,
        )

    def test_missing_precursor_query_is_rejected_observably(self) -> None:
        """A query without a precursor is rejected by the validation layer
        (documented contract) and never silently analyzed."""
        rejected: list[str] = []
        list(
            load_spectra(
                EXPERIMENT_FILE,
                file_format="msp",
                rejection_reporter=rejected.append,
            )
        )
        assert any("Missing precursor_mz" in reason for reason in rejected), rejected


# ---------------------------------------------------------------------------
# 3. Golden pipeline runs (manifest-driven known answers)
# ---------------------------------------------------------------------------


class TestGoldenPipelineRuns:
    """The current pipeline must reproduce the recorded ground truth
    byte-for-byte (CSV digest) and row-for-row (candidates, scores, matched
    peak counts, q/p values, annotation statuses)."""

    @pytest.mark.parametrize(
        "label,algorithm",
        [
            ("cosine", "cosine"),
            ("modified_cosine", "modified_cosine"),
            ("consensus", "consensus"),
            ("cascade", "cascade"),
            ("cosine_rt", "cosine"),
        ],
    )
    def test_pipeline_reproduces_recorded_ground_truth(
        self, tmp_path, label: str, algorithm: str
    ) -> None:
        settings = MANIFEST["runs"][label]["settings"]
        result, rows, csv_path = run_fixture_pipeline(tmp_path, algorithm, settings)

        assert csv_sha256(csv_path) == MANIFEST["runs"][label]["csv_sha256"], (
            f"[{label}] CSV diverged from the recorded ground truth. A scoring, "
            "FDR, ranking, or export change must be a deliberate scientific "
            "decision (regenerate fixtures via generate_ground_truth.py)."
        )
        assert result.status == MANIFEST["runs"][label]["status"]
        assert result.spectra_loaded == MANIFEST["runs"][label]["spectra_loaded"]
        assert result.spectra_rejected == MANIFEST["runs"][label]["spectra_rejected"]
        assert result.hits_produced == MANIFEST["runs"][label]["hits_produced"]
        assert result.fdr_summary == MANIFEST["runs"][label]["fdr_summary"]
        assert list(result.warnings) == MANIFEST["runs"][label]["warnings"]
        assert (
            list(result.degraded_mode_flags)
            == MANIFEST["runs"][label]["degraded_mode_flags"]
        )

        recorded_export = MANIFEST["runs"][label]["exported"]
        for query_id, recorded_rows in recorded_export.items():
            actual_rows = rows_for(rows, query_id)
            assert len(actual_rows) == len(recorded_rows), (
                f"[{label}] {query_id}: {len(actual_rows)} rows exported, "
                f"{len(recorded_rows)} recorded"
            )
            for actual, expected in zip(actual_rows, recorded_rows):
                assert actual["reference_name"] == expected["reference_name"]
                assert float(actual["score"]) == pytest.approx(
                    expected["score"], abs=1e-12
                )
                assert int(actual["matched_peaks"]) == expected["matched_peaks"]
                assert float(actual["q_value"]) == pytest.approx(
                    expected["q_value"], abs=1e-12
                )
                assert float(actual["p_value"]) == pytest.approx(
                    expected["p_value"], abs=1e-12
                )
                assert actual["Annotation_Status"] == expected["annotation_status"]
                if expected["score_breakdown"]:
                    assert actual["score_breakdown"] == expected["score_breakdown"]

    def test_sqlite_and_zarr_backends_are_scientifically_equivalent(
        self,
        tmp_path,
    ) -> None:
        """storage_backend=zarr must not change a single byte of the results."""
        config = fixture_config("cosine", tmp_path)
        config.input.storage_backend = "zarr"
        results = run_annotation_pipeline(config)
        csv_path = tmp_path / f"{EXPERIMENT_FILE.stem}_results.csv"
        assert csv_sha256(csv_path) == MANIFEST["runs"]["cosine"]["csv_sha256"]
        assert csv_sha256(csv_path) == MANIFEST["runs"]["cosine_zarr"]["csv_sha256"]
        assert results[0].hits_produced == MANIFEST["runs"]["cosine"]["hits_produced"]

    def test_runs_are_deterministic(self, tmp_path) -> None:
        """Two identical runs produce byte-identical CSVs and identical
        provenance except for explicitly time-varying fields."""
        first, _, first_csv = run_fixture_pipeline(tmp_path / "a", "cosine")
        second, _, second_csv = run_fixture_pipeline(tmp_path / "b", "cosine")
        assert csv_sha256(first_csv) == csv_sha256(second_csv)
        assert first.fdr_summary == second.fdr_summary


# ---------------------------------------------------------------------------
# 4. FDR scientific interpretation
# ---------------------------------------------------------------------------


class TestFDRInterpretation:
    """What the reported q-values mean, and how they must be used."""

    def test_q_values_are_traceable_to_competition_units(self, tmp_path) -> None:
        """Every exported q-value equals the reference TDC q of the query's
        best target score; the competition unit is the query spectrum."""
        result, rows, _ = run_fixture_pipeline(tmp_path, "cosine")
        run = MANIFEST["runs"]["cosine"]
        targets = [
            q["best_target_score"]
            for q in run["queries"].values()
            if q["best_target_score"] is not None
        ]
        decoys = [
            q["best_decoy_score"]
            for q in run["queries"].values()
            if q["best_decoy_score"] is not None
        ]
        reference = reference_tdc(targets, decoys)

        for row in rows:
            if not row.get("reference_name"):
                continue
            query_truth = run["queries"][row["query_id"]]
            q, p = reference[float(query_truth["best_target_score"])]
            assert float(row["q_value"]) == pytest.approx(q, abs=1e-12)
            assert float(row["p_value"]) == pytest.approx(p, abs=1e-12)

        # Competition unit: exactly one target competition per query with a
        # best target score.
        assert run["fdr_summary"]["n_target_competitions"] == len(targets) == 5
        assert run["fdr_summary"]["n_competing_queries"] == 5

    def test_unmatched_query_has_no_q_value_and_unknown_status(
        self,
        tmp_path,
    ) -> None:
        """Q_CAFFEINE_SHIFTED is MS1-invisible in the cosine run: it must not
        be calibrated (no target competition) and exports as Unknown."""
        run = MANIFEST["runs"]["cosine"]
        shifted = run["queries"]["Q_CAFFEINE_SHIFTED"]
        assert shifted["best_target_score"] is None
        assert shifted["q_value"] == 1.0
        result, rows, _ = run_fixture_pipeline(tmp_path, "cosine")
        shifted_rows = [r for r in rows if r["query_id"] == "Q_CAFFEINE_SHIFTED"]
        assert len(shifted_rows) == 1
        assert shifted_rows[0]["Annotation_Status"] == "Unknown"
        assert not shifted_rows[0].get("reference_name")

    def test_empty_decoy_null_uses_rank_bound_and_flags_degraded(
        self,
        tmp_path,
    ) -> None:
        """Zero decoy hits: q = 1/N rank bound (N = competing queries),
        p = 1.0, the run is explicitly degraded (fdr_uncalibrated), and the
        exported CSV must carry that warning in provenance."""
        run = MANIFEST["runs"]["cosine"]
        assert run["fdr_summary"]["n_decoy_competitions"] == 0
        for query_truth in run["queries"].values():
            if query_truth["best_target_score"] is not None:
                assert query_truth["q_value"] == pytest.approx(1 / 5, abs=1e-12)
                assert query_truth["p_value"] == 1.0
        assert "fdr_uncalibrated" in run["degraded_mode_flags"]
        assert any("uncalibrated" in w for w in run["warnings"])

        result, _, _ = run_fixture_pipeline(tmp_path, "cosine")
        assert result.status == "degraded"
        report_path = tmp_path / f"{EXPERIMENT_FILE.stem}_results.report.yaml"
        assert report_path.exists()
        assert "fdr_uncalibrated" in report_path.read_text()

    def test_decoy_present_null_gives_formula_p_values(self) -> None:
        """Consensus run: every query competes with a decoy (0.0 sentinel),
        so p = (1 + #{D >= s}) / (1 + #{D}) and q follows the TDC formula."""
        run = MANIFEST["runs"]["consensus"]
        assert run["fdr_summary"]["n_decoy_competitions"] == 6
        for query_truth in run["queries"].values():
            if query_truth["best_target_score"] is not None:
                assert query_truth["p_value"] == pytest.approx(1 / 7, abs=1e-12)

    def test_q_value_is_the_only_filter(self, tmp_path) -> None:
        """fdr_threshold filters on q only; p is never compared to it."""
        cosine_manifest = MANIFEST["runs"]["cosine"]
        for threshold, expected_hits in [(0.1, 0), (0.2, 20), (0.5, 20)]:
            settings = cosine_manifest["settings"]
            settings = {
                "processing": settings["processing"],
                "similarity": {**settings["similarity"], "fdr_threshold": threshold},
            }
            result, rows, _ = run_fixture_pipeline(
                tmp_path / f"t{threshold}", "cosine", settings
            )
            assert result.hits_produced == expected_hits, (
                f"threshold {threshold}: expected {expected_hits} hits"
            )
            if expected_hits == 0:
                for row in rows:
                    assert not row.get("reference_name"), (
                        "no hit may be exported when every q > threshold"
                    )

    def test_duplicate_scores_across_queries_share_q_value(self) -> None:
        """Q_CAFFEINE_PERFECT and Q_NO_RT both have best target 1.0: identical
        scores must map to identical q-values (score-level calibration)."""
        cosine = MANIFEST["runs"]["cosine"]["queries"]
        assert cosine["Q_CAFFEINE_PERFECT"]["best_target_score"] == 1.0
        assert cosine["Q_NO_RT"]["best_target_score"] == 1.0
        assert cosine["Q_CAFFEINE_PERFECT"]["q_value"] == cosine["Q_NO_RT"]["q_value"]


# ---------------------------------------------------------------------------
# 5. Decoy generation and competition
# ---------------------------------------------------------------------------


class TestDecoyGeneration:
    """Entropy-preserving decoys (Li et al. 2021): deterministic, entropy
    preserving, precursor preserving, and never identical to their source."""

    @pytest.fixture(scope="class")
    def references(self):
        return load_processed(LIBRARY_FILE)

    def test_decoys_are_deterministic_and_seed_dependent(self, references) -> None:
        first = generate_decoys(references, random_seed=42)
        second = generate_decoys(references, random_seed=42)
        other = generate_decoys(references, random_seed=7)
        assert len(first) == len(references) == 7
        for a, b, c in zip(first, second, other):
            np.testing.assert_array_equal(a.peaks.mz, b.peaks.mz)
            np.testing.assert_array_equal(a.peaks.intensities, b.peaks.intensities)
            assert not np.array_equal(a.peaks.mz, c.peaks.mz)

    def test_decoys_preserve_entropy_and_precursor(self, references) -> None:
        for target, decoy in zip(references, generate_decoys(references)):
            assert decoy.get("precursor_mz") == target.get("precursor_mz")
            assert decoy.get("id") == f"{target.get('id')}_decoy"
            assert decoy.get("is_decoy") is True
            assert decoy.get("spectral_entropy") == pytest.approx(
                spectral_entropy(np.asarray(target.peaks.intensities)), abs=1e-9
            )
            assert spectral_entropy(
                np.asarray(decoy.peaks.intensities)
            ) == pytest.approx(
                spectral_entropy(np.asarray(target.peaks.intensities)), abs=1e-9
            )

    def test_decoy_never_identical_to_source(self, references) -> None:
        for target, decoy in zip(references, generate_decoys(references)):
            assert not np.array_equal(target.peaks.mz, decoy.peaks.mz)
            assert not np.array_equal(target.peaks.intensities, decoy.peaks.intensities)

    def test_query_matching_a_decoy_is_tagged_as_decoy(self, references) -> None:
        """A query built from a decoy's peaks must produce an is_decoy=True
        hit at 1.0, so target-decoy competition can reject it (the mechanism
        that prevents decoy matches from becoming annotations)."""
        decoy = generate_decoys(references, random_seed=42)[0]
        query = Spectrum(
            mz=np.array(decoy.peaks.mz, dtype=np.float64),
            intensities=np.array(decoy.peaks.intensities, dtype=np.float64),
            metadata={
                "id": "Q_DECOY_MATCH",
                "precursor_mz": decoy.get("precursor_mz"),
            },
        )
        engine = SimilarityEngine(
            SimilarityConfig(
                algorithm="cosine",
                min_score=0.0,
                min_matched_peaks=1,
                ms1_tolerance=100.0,
                ms2_tolerance=0.02,
            )
        )
        hits = engine.search(
            [query],
            references,
            include_decoys=True,
            decoy_min_relative_intensity=0.01,
            decoy_mz_shift_da=1.0,
        )
        decoy_hits = [h for h in hits if h["is_decoy"]]
        assert decoy_hits, "the decoy match must be found and tagged is_decoy"
        assert max(h["score"] for h in decoy_hits) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. Engine contrasts
# ---------------------------------------------------------------------------


class TestEngineContrasts:
    """Engines are not interchangeable; each has a documented scope."""

    def test_consensus_dilutes_ms1_gated_pairs(self, tmp_path) -> None:
        """The cosine sub-engine's MS1 gate surfaces as a 0.0 sentinel, so the
        consensus of (cosine 0.0, modified 1.0) = 0.5 for the precursor-
        violating reference — pinned engine contract."""
        result, rows, _ = run_fixture_pipeline(tmp_path, "consensus")
        row = next(
            r
            for r in rows_for(rows, "Q_CAFFEINE_PERFECT")
            if r["reference_name"] == "REF_CAFFEINE_PRECURSOR_VIOLATION"
        )
        assert float(row["score"]) == pytest.approx(0.5, abs=1e-12)
        assert json.loads(row["score_breakdown"]) == {
            "cosine": 0.0,
            "modified_cosine": 1.0,
        }
        assert row["Annotation_Status"] == "Putative"

    def test_cascade_equals_final_stage_at_zero_lower_bound(self) -> None:
        """With cascade_lower_bound=0.0 every adduct/RT-passing reference
        survives stage 1 (the MS1 sentinel scores 0.0), so the cascade output
        equals the modified-cosine final stage over all of them."""
        assert (
            MANIFEST["runs"]["cascade"]["csv_sha256"]
            == MANIFEST["runs"]["modified_cosine"]["csv_sha256"]
        )

    def test_cascade_lower_bound_gates_shifted_queries(self) -> None:
        """At a positive lower bound the cosine stage really gates: the
        shifted query has no cosine candidates, so the cascade (unlike
        modified cosine) returns nothing for it."""
        queries = load_processed(EXPERIMENT_FILE)
        query = next(s for s in queries if s.get("id") == "Q_CAFFEINE_SHIFTED")
        references = load_processed(LIBRARY_FILE)
        cascade = get_similarity_engine(
            SimilarityConfig(
                algorithm="cascade",
                min_score=0.0,
                min_matched_peaks=0,
                ms1_tolerance=0.5,
                ms2_tolerance=0.02,
                cascade_lower_bound=0.3,
                cascade_upper_bound=0.0,
            )
        )
        hits = cascade.search([query], references, include_decoys=False)
        assert hits == []

    def test_rt_tolerance_removes_exactly_the_rt_violation(self, tmp_path) -> None:
        """The RT run exports 16 of 20 rows: every query loses exactly its
        REF_CAFFEINE_RT_VIOLATION row; the RT-less query keeps it (missing RT
        bypasses the filter)."""
        cosine = MANIFEST["runs"]["cosine"]
        rt = MANIFEST["runs"]["cosine_rt"]
        assert cosine["hits_produced"] == 20
        assert rt["hits_produced"] == 16
        for query_id in (
            "Q_CAFFEINE_PERFECT",
            "Q_CAFFEINE_NEAR",
            "Q_CAFFEINE_WEAK",
            "Q_MANY_PEAKS",
        ):
            names_rt = {r["reference_name"] for r in rt["exported"].get(query_id, [])}
            assert "REF_CAFFEINE_RT_VIOLATION" not in names_rt
        # Q_NO_RT has no retention time: the filter is bypassed for it.
        no_rt_names = {r["reference_name"] for r in rt["exported"]["Q_NO_RT"]}
        assert "REF_CAFFEINE_RT_VIOLATION" in no_rt_names
        # Q_CAFFEINE_SHIFTED is MS1-invisible in cosine: unaffected by RT.
        assert "Q_CAFFEINE_SHIFTED" not in rt["exported"]


# ---------------------------------------------------------------------------
# 7. Provenance of the golden runs
# ---------------------------------------------------------------------------


class TestGoldenRunProvenance:
    def test_run_provenance_records_engine_backend_and_config(
        self,
        tmp_path,
    ) -> None:
        run_fixture_pipeline(tmp_path, "consensus")
        provenance_path = tmp_path / "run_provenance.json"
        assert provenance_path.exists()
        payload = json.loads(provenance_path.read_text())
        assert payload["schema_version"] == 2
        assert payload["backend"] == "sqlite"
        assert payload["results"]["files_total"] == 1
        assert payload["results"]["files_succeeded"] == 1
        assert payload["decoy_seed"] == 42
        assert payload["engine"]["algorithm"] == "consensus"
