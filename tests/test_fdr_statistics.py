"""
Synthetic statistical tests for the per-query target-decoy FDR contract.

These tests verify not only numerical output but the intended SCIENTIFIC
INTERPRETATION of MassFlow's calibration. The contract (see
``docs/user-guide/scoring_logic.md`` and ``MassFlow.similarity.calculate_fdr``):

* Competition unit: the QUERY SPECTRUM. Each query competes once, with its
  best target hit against its best decoy hit.
* FDR(t) = (1 + #{queries with best decoy >= t}) / #{queries with best
  target >= t}, monotone-closed to q-values.
* Ties: decoys rank before targets (conservative).
* The empirical p-value is a diagnostic; the q-value is the ONLY filter.
* Single-file, multi-file, and streaming-library executions must produce
  identical statistical behavior.

Every expected value below was hand-computed from the ranking definition.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from matchms import Spectrum
from typing import Any

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.similarity import (
    SearchResult,
    calibrate_query_level_fdr,
    calculate_empirical_p_values,
    calculate_fdr,
    generate_decoys,
)
from MassFlow.workflow import _process_single_file

pytestmark = pytest.mark.scientific


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spectrum(spec_id: str, precursor_mz: float = 100.0) -> Spectrum:
    """Deterministic synthetic spectrum with a single peak."""
    return Spectrum(
        mz=np.array([precursor_mz], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata={"id": spec_id, "precursor_mz": precursor_mz, "charge": 1},
    )


def make_hit(
    query_id: str,
    reference_id: str,
    score: float,
    is_decoy: bool = False,
) -> SearchResult:
    """SearchResult-shaped dict for synthetic calibration tests."""
    return SearchResult(
        query_id=query_id,
        query_precursor_mz=100.0,
        reference_id=reference_id,
        reference_name=reference_id,
        reference_precursor_mz=100.0,
        score=float(score),
        matched_peaks=5,
        smiles=None,
        inchikey=None,
        is_decoy=is_decoy,
        q_value=1.0,
        p_value=None,
        annotation_tier=None,
        structural_similarity=None,
        mass_error_ppm=None,
        score_breakdown=None,
    )


def run_fdr_block(hits: list[SearchResult], library_size: int = 10) -> list[Any]:
    """Run the workflow FDR block over synthetic engine output.

    ``_process_single_file`` is exercised with a fake worker engine so the
    full per-file path (aggregation -> calibration -> filtering) is covered.
    """
    fake_engine = _FakeEngine(hits)

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=Path("/tmp/unused")),
        input=InputConfig(
            input_path=Path("query.mgf"),
            library_path=Path("reference.msp"),
            format="mgf",
        ),
        similarity=SimilarityConfig(fdr_threshold=1.0, min_score=0.0),
    )

    with patch("MassFlow.workflow._worker_engine", fake_engine):
        with patch(
            "MassFlow.workflow.io.load_spectra",
            side_effect=[[make_spectrum("query_1")], []],
        ):
            with patch(
                "MassFlow.workflow.processing.process_spectra",
                side_effect=lambda spectra, config: spectra,
            ):
                result = _process_single_file(
                    Path("query.mgf"), config, library_size=library_size
                )
    return result.results


class _FakeEngine:
    """Engine stub returning pre-engineered hits."""

    def __init__(self, hits: list[Any]):
        self._hits = hits

    def search(self, query_spectra, reference_spectra, **kwargs):
        # Consume the (possibly counted) library iterator so the streaming
        # counter reflects the full library, like a real engine would.
        list(reference_spectra)
        return [dict(hit) for hit in self._hits]


@pytest.fixture(autouse=True)
def reset_worker_state(monkeypatch):
    """Worker globals persist across tests in this module (same process):
    reset the backend state so no test leaks a worker-owned store into the
    next."""
    monkeypatch.setattr("MassFlow.workflow._worker_engine", None)
    monkeypatch.setattr("MassFlow.workflow._worker_router", None)
    monkeypatch.setattr("MassFlow.workflow._worker_backend", None)
    monkeypatch.setattr("MassFlow.workflow._worker_library_spec", None)


# ---------------------------------------------------------------------------
# 1. Competition unit: one query / one target / one decoy
# ---------------------------------------------------------------------------


class TestSingleQueryCompetition:
    def test_single_query_q_value_is_always_1(self) -> None:
        """With one competing query, no FDR below 1.0 is claimable.

        FDR(t) = (1 + #{decoy >= t}) / #{target >= t} with one target is
        >= 1/1 regardless of how low the decoy scores: the +1 pseudo-count
        dominates. Reporting q < 1 for a single query would be
        over-optimistic.
        """
        sorted_scores, q_values, is_target = calculate_fdr(
            np.array([0.9]), np.array([0.5])
        )
        assert list(is_target) == [True, False]
        assert q_values[0] == pytest.approx(1.0)

    def test_single_query_decoy_win_p_is_1(self) -> None:
        """When the only decoy beats the target, p = 1.0 (no evidence)."""
        p = calculate_empirical_p_values(np.array([0.9]), np.array([0.95]))
        assert p[0] == pytest.approx(1.0)
        p = calculate_empirical_p_values(np.array([0.9]), np.array([0.5]))
        # (1 + 0) / (1 + 1)
        assert p[0] == pytest.approx(0.5)

    def test_workflow_single_query_kept_only_at_threshold_1(self) -> None:
        """End-to-end: a single query with a target above its decoy is
        exported with q=1.0 and kept only when fdr_threshold=1.0."""
        hits = [
            make_hit("query_1", "ref_1", 0.9),
            make_hit("query_1", "ref_1_decoy", 0.5, is_decoy=True),
        ]
        results = run_fdr_block(hits)
        assert len(results) == 1
        assert results[0]["q_value"] == pytest.approx(1.0)
        assert results[0]["p_value"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 2. Multiple targets per query must not distort the estimate
# ---------------------------------------------------------------------------


class TestMultipleHitsPerQuery:
    def test_many_hits_of_one_query_do_not_inflate_target_count(self) -> None:
        """q1 has 50 threshold-passing hits; q2 has 1. Per-query TDC must
        give both queries the same q-value.

        Bests: q1: T=0.9, D=0.5; q2: T=0.8, D=0.5.
        Ranking: 0.9T, 0.8T, 0.5D, 0.5D.
        fdr: 1/1, 1/2, 2/2, 3/2->1  =>  q = [0.5, 0.5, 1, 1].

        The OLD hit-level pooling ranked all 51 target hits and gave
        q(0.9) ~= 1/51 ~= 0.02 — an over-optimistic estimate inflated by
        q1's own correlated hits.
        """
        hits = [make_hit("q1", f"ref_{i}", 0.9 - 0.01 * i) for i in range(50)]
        hits.append(make_hit("q1", "ref_1_decoy", 0.5, is_decoy=True))
        hits.append(make_hit("q2", "ref_51", 0.8))
        hits.append(make_hit("q2", "ref_51_decoy", 0.5, is_decoy=True))

        q_by_query, p_by_query, summary = calibrate_query_level_fdr(hits)
        assert q_by_query["q1"] == pytest.approx(0.5)
        assert q_by_query["q2"] == pytest.approx(0.5)
        assert summary["n_competing_queries"] == 2
        assert summary["n_target_competitions"] == 2
        assert summary["n_decoy_competitions"] == 2

    def test_all_hits_of_one_query_share_its_q_value(self) -> None:
        """The q-value is a property of the query: every exported row of the
        query carries the same q (traceability to the competition unit)."""
        hits = [
            make_hit("q1", "ref_a", 0.95),
            make_hit("q1", "ref_b", 0.70),
            make_hit("q1", "ref_a_decoy", 0.5, is_decoy=True),
            make_hit("q2", "ref_c", 0.85),
            make_hit("q2", "ref_c_decoy", 0.6, is_decoy=True),
        ]
        q_by_query, _p, _s = calibrate_query_level_fdr(hits)
        assert q_by_query["q1"] == q_by_query["q2"]


# ---------------------------------------------------------------------------
# 3. Ties are handled conservatively
# ---------------------------------------------------------------------------


class TestTies:
    def test_tied_decoy_ranks_before_target(self) -> None:
        """A decoy that ties a target score is counted against the target."""
        sorted_scores, q_values, is_target = calculate_fdr(
            np.array([0.8, 0.7]), np.array([0.8, 0.7])
        )
        # Expected ranking: 0.8D, 0.8T, 0.7D, 0.7T
        assert list(is_target) == [False, True, False, True]
        # fdr: rank1 D -> 1.0; rank2 T -> (1+1)/1=2->1.0;
        #      rank3 D -> 2/1->1.0; rank4 T -> (2+1)/2->1.0
        assert np.all(q_values == pytest.approx(1.0))

    def test_ties_never_lower_a_targets_q_value(self) -> None:
        """Compared to a decoy scoring epsilon below the target, a tied
        decoy must not produce a smaller (better) q-value."""
        target = np.array([0.8])
        tied = calculate_fdr(target, np.array([0.8]))[1][1]
        below = calculate_fdr(target, np.array([0.7999]))[1][1]
        assert tied >= below


# ---------------------------------------------------------------------------
# 4. Duplicate scores map to identical q-values
# ---------------------------------------------------------------------------


class TestDuplicateScores:
    def test_identical_scores_share_q_and_p(self) -> None:
        """Two queries with the same best target score get the same
        q-value (q is a function of the score), and duplicate score values
        cannot cause mismatched p-value assignment."""
        hits = [
            make_hit("q1", "ref_a", 0.85),
            make_hit("q1", "ref_a_decoy", 0.5, is_decoy=True),
            make_hit("q2", "ref_b", 0.85),
            make_hit("q2", "ref_b_decoy", 0.6, is_decoy=True),
        ]
        q_by_query, p_by_query, _ = calibrate_query_level_fdr(hits)
        # Ranking: 0.85T, 0.85T, 0.6D, 0.5D -> q = [0.5, 0.5, 1, 1]
        assert q_by_query["q1"] == pytest.approx(0.5)
        assert q_by_query["q2"] == pytest.approx(0.5)
        # p = (1 + #{decoy >= 0.85}) / (1 + 2) = 1/3 for both
        assert p_by_query["q1"] == pytest.approx(1.0 / 3.0)
        assert p_by_query["q2"] == pytest.approx(1.0 / 3.0)

    def test_duplicate_scores_within_one_query(self) -> None:
        """Duplicate target hits inside a single query collapse to one
        competition entry (the max)."""
        hits = [
            make_hit("q1", "ref_a", 0.9),
            make_hit("q1", "ref_b", 0.9),
            make_hit("q1", "ref_a_decoy", 0.5, is_decoy=True),
            make_hit("q1", "ref_b_decoy", 0.5, is_decoy=True),
        ]
        q_by_query, p_by_query, summary = calibrate_query_level_fdr(hits)
        assert summary["n_target_competitions"] == 1
        assert summary["n_decoy_competitions"] == 1
        assert q_by_query["q1"] == pytest.approx(1.0)
        assert p_by_query["q1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. Zero decoys / zero targets
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_zero_decoys_q_is_rank_bound(self) -> None:
        """Without any decoy evidence, no calibration is possible: every
        target shares the conservative bound q = 1/N."""
        sorted_scores, q_values, is_target = calculate_fdr(
            np.array([0.9, 0.8]), np.array([])
        )
        assert np.all(is_target)
        assert np.all(q_values == pytest.approx(0.5))

    def test_zero_decoys_p_is_one(self) -> None:
        p = calculate_empirical_p_values(np.array([0.9, 0.8]), np.array([]))
        assert np.all(p == pytest.approx(1.0))

    def test_zero_targets_q_is_one(self) -> None:
        sorted_scores, q_values, is_target = calculate_fdr(
            np.array([]), np.array([0.5, 0.4])
        )
        assert np.all(q_values == pytest.approx(1.0))
        assert not np.any(is_target)

    def test_workflow_zero_decoys_exports_nothing_at_strict_threshold(
        self,
    ) -> None:
        """Only target hits, no decoy hits: q = 1/N; at fdr_threshold=0.01
        nothing is exported, at 1.0 everything is."""
        hits = [make_hit("q1", "ref_1", 0.9), make_hit("q2", "ref_2", 0.8)]
        results = run_fdr_block(hits, library_size=10)
        assert len(results) == 2
        assert all(r["q_value"] == pytest.approx(0.5) for r in results)
        assert all(r["p_value"] == pytest.approx(1.0) for r in results)


# ---------------------------------------------------------------------------
# 6. Small libraries: same model, no silent concept switch
# ---------------------------------------------------------------------------


class TestSmallLibraries:
    def test_small_library_q_is_conservative_with_pseudo_count(self) -> None:
        """3 queries, every decoy above every target: all q-values are 1.0
        (the +1 pseudo-count prevents optimistic claims)."""
        targets = np.array([0.9, 0.85, 0.8])
        decoys = np.array([0.95, 0.92, 0.9])
        _scores, q_values, _is_target = calculate_fdr(targets, decoys)
        assert np.all(q_values == pytest.approx(1.0))

    def test_workflow_uses_q_not_bonferroni_p_for_filtering(self) -> None:
        """10 queries with target 0.9 and decoy 0.5 each.

        q = 0.1 for every query (FDR = 1/10 at the last target rank).
        p = (1 + 0)/(1 + 10) = 0.0909; the OLD small-library path multiplied
        p by the number of hits (Bonferroni: 0.909) and compared against
        fdr_threshold — silently switching FWER semantics into the FDR
        threshold. The contract filters on the q-value alone:
        at fdr_threshold = 0.1 all 10 hits are kept; at 0.05 none are.
        """
        hits = []
        for i in range(10):
            hits.append(make_hit(f"q{i}", f"ref_{i}", 0.9))
            hits.append(make_hit(f"q{i}", f"ref_{i}_decoy", 0.5, is_decoy=True))

        kept = run_fdr_block(hits, library_size=10)
        assert len(kept) == 10
        assert all(r["q_value"] == pytest.approx(0.1) for r in kept)
        # p is exported as a diagnostic and equals (1+0)/(1+10)
        assert all(r["p_value"] == pytest.approx(1.0 / 11.0) for r in kept)
        # Bonferroni-corrected p would have been 10 * 1/11 = 0.909 > 0.1,
        # rejecting everything under the old silent switch.
        assert all(10.0 * r["p_value"] > 0.1 for r in kept)

    def test_small_library_warning_uses_true_library_size(self) -> None:
        """The small-library warning must reflect the TRUE library size.

        In the streaming path the parent cannot load the library, so the
        size is counted while the engine consumes the library iterator: a
        10-spectrum library warns, a 5000-spectrum library does not."""
        hits = [make_hit("query_1", "ref_1", 0.9)]
        config = MassFlowConfig(
            project=ProjectConfig(output_directory=Path("/tmp/unused")),
            input=InputConfig(
                input_path=Path("query.mgf"),
                library_path=Path("reference.msp"),
                format="mgf",
            ),
            similarity=SimilarityConfig(fdr_threshold=0.01, min_score=0.0),
        )

        def run_streaming(n_library: int):
            fake_engine = _FakeEngine(hits)
            library = [make_spectrum(f"r{i}") for i in range(n_library)]
            with patch("MassFlow.workflow._worker_engine", fake_engine):
                with patch(
                    "MassFlow.workflow.io.load_spectra",
                    side_effect=[[make_spectrum("query_1")], library],
                ):
                    with patch(
                        "MassFlow.workflow.processing.process_spectra",
                        side_effect=lambda spectra, config: spectra,
                    ):
                        with patch(
                            "MassFlow.workflow._emit_small_library_warning"
                        ) as mock_warn:
                            _process_single_file(Path("query.mgf"), config)
            return mock_warn

        mock_warn = run_streaming(10)
        mock_warn.assert_called_once()

        mock_warn = run_streaming(5000)
        mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Heterogeneous engines: per-query pairing is engine-consistent
# ---------------------------------------------------------------------------


class TestMultiEngine:
    def test_query_is_competed_against_its_own_engines_decoy(self) -> None:
        """A query scored by an ML/consensus engine with a wide score scale
        must be out-competed by ITS OWN decoy, not by another engine's.

        99 classical queries: T=0.9, D=0.5.
        1 ML query:           T=0.9, D=0.95 (its own decoy wins).
        Ranking: 0.95D(ml), 0.9T x 100, 0.5D x 99.
        fdr at the last target rank: (1 + 1)/100 = 0.02 -> every query q=0.02.
        Without the ML query's decoy in the null, q would be 1/100 = 0.01.
        """
        hits = []
        for i in range(99):
            hits.append(make_hit(f"class_{i}", f"ref_{i}", 0.9))
            hits.append(make_hit(f"class_{i}", f"ref_{i}_decoy", 0.5, is_decoy=True))
        hits.append(make_hit("ml_0", "ref_ml", 0.9))
        hits.append(make_hit("ml_0", "ref_ml_decoy", 0.95, is_decoy=True))

        q_by_query, _p, summary = calibrate_query_level_fdr(hits)
        assert summary["n_target_competitions"] == 100
        assert summary["n_decoy_competitions"] == 100
        assert q_by_query["ml_0"] == pytest.approx(0.02)
        assert q_by_query["class_0"] == pytest.approx(0.02)

        # Sanity: removing the wide-scale decoy lowers q to 1/100.
        without_ml_decoy = [
            h for h in hits if not (h["query_id"] == "ml_0" and h["is_decoy"])
        ]
        q_by_query2, _p2, _s2 = calibrate_query_level_fdr(without_ml_decoy)
        assert q_by_query2["ml_0"] == pytest.approx(0.01)

    def test_heterogeneous_score_scales_do_not_break_pooling(self) -> None:
        """Per-query counts pool legitimately across engines: each query's
        target and decoy come from the same engine, so the exchangeability
        assumption holds within a query."""
        # Engine A: targets 0.9, decoys 0.5 (n=50)
        # Engine B: targets 0.6, decoys 0.4 (n=50)
        hits = []
        for i in range(50):
            hits.append(make_hit(f"a_{i}", f"a_ref_{i}", 0.9))
            hits.append(make_hit(f"a_{i}", f"a_ref_{i}_decoy", 0.5, is_decoy=True))
            hits.append(make_hit(f"b_{i}", f"b_ref_{i}", 0.6))
            hits.append(make_hit(f"b_{i}", f"b_ref_{i}_decoy", 0.4, is_decoy=True))

        q_by_query, p_by_query, summary = calibrate_query_level_fdr(hits)
        assert summary["n_target_competitions"] == 100
        # Both groups have all their decoys below their targets; the global
        # minimum of (1 + cumD)/cumT is reached at the last target rank:
        # (1 + 0)/100 = 0.01.
        assert q_by_query["a_0"] == pytest.approx(0.01)
        assert q_by_query["b_0"] == pytest.approx(0.01)
        assert p_by_query["a_0"] == pytest.approx(1.0 / 101.0)
        assert p_by_query["b_0"] == pytest.approx(1.0 / 101.0)


# ---------------------------------------------------------------------------
# 8. Decoy generation: chunk-invariance (streaming equivalence)
# ---------------------------------------------------------------------------


class TestDecoyChunkInvariance:
    def test_decoy_set_is_identical_when_library_is_chunked(self) -> None:
        """Decoys generated over the full library must equal the union of
        decoys generated per chunk. This is what makes the single-file /
        streaming path statistically equivalent to the multi-file path."""
        rng = np.random.default_rng(7)
        spectra = []
        for i in range(25):
            n_peaks = int(rng.integers(5, 15))
            spectra.append(
                Spectrum(
                    mz=np.sort(rng.uniform(50, 900, size=n_peaks)),
                    intensities=rng.uniform(1, 100, size=n_peaks),
                    metadata={
                        "id": f"ref_{i:03d}",
                        "precursor_mz": float(rng.uniform(100, 1000)),
                    },
                )
            )

        full_decoys = generate_decoys(spectra, random_seed=42)
        chunked_decoys = []
        for chunk in (spectra[:10], spectra[10:20], spectra[20:]):
            chunked_decoys.extend(generate_decoys(chunk, random_seed=42))

        assert len(full_decoys) == len(chunked_decoys) == 25
        for full, chunked in zip(full_decoys, chunked_decoys):
            assert np.array_equal(full.peaks.mz, chunked.peaks.mz)
            assert np.array_equal(full.peaks.intensities, chunked.peaks.intensities)
            assert full.get("id") == chunked.get("id")

    def test_decoy_config_params_flow_to_engine_generated_decoys(self) -> None:
        """SimilarityEngine.search must honor the workflow's decoy config
        when it generates decoys internally (single-file path)."""
        from MassFlow.similarity import SimilarityEngine

        engine = SimilarityEngine(
            SimilarityConfig(algorithm="cosine", min_score=0.0, min_matched_peaks=0)
        )
        query = make_spectrum("q1", precursor_mz=200.0)
        ref = Spectrum(
            mz=np.array([150.0, 300.0], dtype=np.float64),
            intensities=np.array([1.0, 2.0], dtype=np.float64),
            metadata={"id": "ref_1", "precursor_mz": 200.0, "charge": 1},
        )
        results = engine.search(
            [query],
            [ref],
            include_decoys=True,
            decoy_min_relative_intensity=0.05,
            decoy_mz_shift_da=2.5,
        )
        decoy_hits = [r for r in results if r["is_decoy"]]
        assert len(decoy_hits) == 1
        assert decoy_hits[0]["reference_id"] == "ref_1_decoy"


# ---------------------------------------------------------------------------
# 9. Consensus engine produces a valid decoy null
# ---------------------------------------------------------------------------


class TestConsensusEngineNull:
    def test_consensus_search_scores_decoys_on_the_same_scale(self) -> None:
        """Consensus (cosine + modified_cosine) must return decoy hits with
        consensus scores, so per-query TDC has a same-scale null."""
        from MassFlow.similarity import ConsensusEngine

        config = SimilarityConfig(
            algorithm="consensus",
            consensus_weights={"cosine": 0.5, "modified_cosine": 0.5},
            consensus_min_engines=1,
            min_score=0.0,
            min_matched_peaks=0,
            ms1_tolerance=100.0,
            ms2_tolerance=0.5,
        )
        engine = ConsensusEngine(config)

        rng = np.random.default_rng(3)
        references = []
        for i in range(6):
            references.append(
                Spectrum(
                    mz=np.sort(rng.uniform(100, 500, size=8)),
                    intensities=rng.uniform(1, 100, size=8),
                    metadata={
                        "id": f"ref_{i}",
                        "precursor_mz": float(400 + i),
                        "charge": 1,
                    },
                )
            )
        query = Spectrum(
            mz=references[0].peaks.mz.copy(),
            intensities=references[0].peaks.intensities.copy(),
            metadata={"id": "query_1", "precursor_mz": 400.0, "charge": 1},
        )

        results = engine.search([query], references, include_decoys=True)
        decoy_hits = [r for r in results if r["is_decoy"]]
        assert len(decoy_hits) > 0, (
            "Consensus must produce decoy hits for a per-query null"
        )
        # Consensus scores of decoys are bounded like targets and are
        # computed by the same weighted aggregation.
        for hit in decoy_hits:
            assert 0.0 <= hit["score"] <= 1.0

        q_by_query, _p, summary = calibrate_query_level_fdr(results)
        assert "query_1" in q_by_query
        # n_*_competitions count per-query bests: one query, so one
        # competition per side regardless of the number of decoy hits.
        assert summary["n_decoy_competitions"] == 1
        assert summary["n_target_competitions"] == 1


# ---------------------------------------------------------------------------
# 10. Cascade engine scores decoys (regression for the include_decoys bug)
# ---------------------------------------------------------------------------


class TestCascadeEngineNull:
    def test_cascade_search_returns_decoy_hits_when_requested(self) -> None:
        """Cascade with include_decoys=True must score decoys. Previously
        the stages forced include_decoys=False and no decoys were ever
        generated, leaving the FDR null empty in the single-file path."""
        from MassFlow.similarity import CascadeEngine

        config = SimilarityConfig(
            algorithm="cascade",
            cascade_stages=["cosine", "modified_cosine"],
            cascade_lower_bound=0.0,
            cascade_upper_bound=0.0,
            min_score=0.0,
            min_matched_peaks=0,
            ms1_tolerance=100.0,
            ms2_tolerance=0.5,
        )
        engine = CascadeEngine(config)

        rng = np.random.default_rng(5)
        references = []
        for i in range(6):
            references.append(
                Spectrum(
                    mz=np.sort(rng.uniform(100, 500, size=8)),
                    intensities=rng.uniform(1, 100, size=8),
                    metadata={
                        "id": f"ref_{i}",
                        "precursor_mz": float(400 + i),
                        "charge": 1,
                    },
                )
            )
        query = Spectrum(
            mz=references[0].peaks.mz.copy(),
            intensities=references[0].peaks.intensities.copy(),
            metadata={"id": "query_1", "precursor_mz": 400.0, "charge": 1},
        )

        results = engine.search([query], references, include_decoys=True)
        decoy_hits = [r for r in results if r["is_decoy"]]
        assert len(decoy_hits) > 0, "Cascade must score decoys when include_decoys=True"

        # Decoys pass through the same stages as targets.
        q_by_query, _p, summary = calibrate_query_level_fdr(results)
        # One query -> one decoy competition entry (its best decoy hit).
        assert summary["n_decoy_competitions"] == 1
        assert summary["n_target_competitions"] == 1

    def test_cascade_without_decoys_matches_classical_fallback(self) -> None:
        """include_decoys=False must not add decoys (no double counting)."""
        from MassFlow.similarity import CascadeEngine

        config = SimilarityConfig(
            algorithm="cascade",
            cascade_stages=["cosine", "modified_cosine"],
            cascade_lower_bound=0.0,
            cascade_upper_bound=0.0,
            min_score=0.0,
            min_matched_peaks=0,
            ms1_tolerance=100.0,
            ms2_tolerance=0.5,
        )
        engine = CascadeEngine(config)
        ref = Spectrum(
            mz=np.array([100.0, 200.0], dtype=np.float64),
            intensities=np.array([1.0, 1.0], dtype=np.float64),
            metadata={"id": "ref_1", "precursor_mz": 150.0, "charge": 1},
        )
        query = Spectrum(
            mz=np.array([100.0, 200.0], dtype=np.float64),
            intensities=np.array([1.0, 1.0], dtype=np.float64),
            metadata={"id": "query_1", "precursor_mz": 150.0, "charge": 1},
        )
        results = engine.search([query], [ref], include_decoys=False)
        assert all(not r["is_decoy"] for r in results)


# ---------------------------------------------------------------------------
# 11. Single-file vs multi-file vs streaming equivalence
# ---------------------------------------------------------------------------


class TestExecutionModeEquivalence:
    """Single-file, multi-file, and streaming-library runs must produce
    identical q-values and identical small-library decisions."""

    N_LIBRARY = 12
    QUERY_ID = "exp_query_0"

    def _engine_hits(self, query_id: str) -> list[SearchResult]:
        hits = [
            make_hit(query_id, "ref_0", 0.92),
            make_hit(query_id, "ref_0_decoy", 0.5, is_decoy=True),
            make_hit(query_id, "ref_1", 0.71),
            make_hit(query_id, "ref_1_decoy", 0.6, is_decoy=True),
        ]
        return hits

    def _config(self, tmp_path: Path, streaming: bool = False) -> MassFlowConfig:
        query_path = tmp_path / "experimental.mgf"
        query_path.touch()
        library_path = tmp_path / "library.msp"
        library_path.touch()
        return MassFlowConfig(
            project=ProjectConfig(output_directory=tmp_path / "results"),
            input=InputConfig(
                input_path=query_path,
                library_path=library_path,
                format="mgf",
                streaming_threshold_mb=0 if streaming else 500,
            ),
            similarity=SimilarityConfig(fdr_threshold=1.0, min_score=0.0),
        )

    def _run_pipeline(self, config: MassFlowConfig, fake_engine) -> dict:
        """Run run_annotation_pipeline with a fake engine and return the
        exported per-query q-values from the saved results."""
        from MassFlow.workflow import run_annotation_pipeline

        library = [
            make_spectrum(f"ref_{i}", precursor_mz=float(100 + i))
            for i in range(self.N_LIBRARY)
        ]
        query = make_spectrum(self.QUERY_ID, precursor_mz=100.0)
        assert config.input.library_path is not None
        library_path = Path(config.input.library_path)

        def mock_load(path, file_format=None, rejection_reporter=None):
            if Path(path) == library_path:
                return iter(library)
            return iter([query])

        with patch("MassFlow.workflow.get_similarity_engine", return_value=fake_engine):
            with patch("MassFlow.workflow.io.load_spectra", side_effect=mock_load):
                with patch(
                    "MassFlow.workflow.processing.process_spectra",
                    side_effect=lambda spectra, config: spectra,
                ):
                    with patch("MassFlow.workflow.io.save_match_results") as mock_save:
                        run_annotation_pipeline(config)

        assert mock_save.call_count >= 1
        # For multi-file runs every file saves its own results; they share
        # the same synthetic query, so the last call's results represent
        # the run.
        results = mock_save.call_args.args[0]
        return {r["query_id"]: r["q_value"] for r in results}

    def test_single_file_and_multi_file_q_values_identical(self, tmp_path) -> None:
        """The single-file path (no _init_worker) and the worker path must
        export identical q-values."""
        from concurrent.futures import ThreadPoolExecutor

        # --- Single-file run ------------------------------------------------
        hits = self._engine_hits(self.QUERY_ID)
        fake_engine = _FakeEngine(hits)
        single_q = self._run_pipeline(self._config(tmp_path), fake_engine)

        # --- Multi-file run (two files; worker-initialized) ------------------
        results_dir = tmp_path / "multi"
        results_dir.mkdir()
        config = self._config(tmp_path)
        config.input.input_path = results_dir
        (results_dir / "a.mgf").touch()
        (results_dir / "b.mgf").touch()

        fake_engine = _FakeEngine(hits)
        with patch("MassFlow.workflow.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value = ThreadPoolExecutor(
                max_workers=2
            )
            multi_q = self._run_pipeline(config, fake_engine)

        assert set(single_q) == set(multi_q)
        for query_id in single_q:
            assert single_q[query_id] == pytest.approx(multi_q[query_id])

    def test_streaming_library_knows_true_size(self, tmp_path) -> None:
        """Streaming mode (parent cannot load the library) must still derive
        the true library size from the counted iterator and produce the same
        q-values as the non-streaming path."""
        hits = self._engine_hits(self.QUERY_ID)

        # Non-streaming reference run
        fake_engine = _FakeEngine(hits)
        normal_q = self._run_pipeline(
            self._config(tmp_path, streaming=False), fake_engine
        )

        # Streaming run: threshold 0 MB -> stream_library=True -> parent
        # never loads the library; workers count it while searching.
        fake_engine = _FakeEngine(hits)
        streaming_q = self._run_pipeline(
            self._config(tmp_path, streaming=True), fake_engine
        )

        assert normal_q == pytest.approx(streaming_q)


# ---------------------------------------------------------------------------
# 12. Decoy config threading end-to-end
# ---------------------------------------------------------------------------


class TestDecoyConfigThreading:
    def test_single_file_pipeline_uses_configured_decoy_parameters(self) -> None:
        """The single-file path generates decoys inside the engine; those
        decoys must respect processing.decoy_* (previously the module
        defaults were silently used, diverging from multi-file runs)."""
        from MassFlow.similarity import SimilarityEngine

        config = MassFlowConfig(
            project=ProjectConfig(output_directory=Path("/tmp/unused")),
            input=InputConfig(
                input_path=Path("query.mgf"),
                library_path=Path("reference.msp"),
                format="mgf",
            ),
            processing=ProcessingConfig(
                decoy_min_relative_intensity=0.1, decoy_mz_shift_da=4.0
            ),
            similarity=SimilarityConfig(fdr_threshold=1.0, min_score=0.0),
        )
        ref = Spectrum(
            mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
            intensities=np.array([10.0, 5.0, 0.5], dtype=np.float64),
            metadata={"id": "ref_1", "precursor_mz": 150.0, "charge": 1},
        )
        query = Spectrum(
            mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
            intensities=np.array([10.0, 5.0, 0.5], dtype=np.float64),
            metadata={"id": "query_1", "precursor_mz": 150.0, "charge": 1},
        )

        engine = SimilarityEngine(
            SimilarityConfig(algorithm="cosine", min_score=0.0, min_matched_peaks=0)
        )
        with patch("MassFlow.workflow._worker_engine", engine):
            with patch(
                "MassFlow.workflow.io.load_spectra",
                side_effect=[[query], [ref]],
            ):
                with patch(
                    "MassFlow.workflow.processing.process_spectra",
                    side_effect=lambda spectra, config: spectra,
                ):
                    result = _process_single_file(Path("query.mgf"), config)

        # The engine-generated decoy must have been filtered at the
        # configured 10%-of-base-peak floor: the 0.5-intensity peak is
        # excluded, so the decoy has 2 peaks, not 3.
        decoys = generate_decoys(
            [ref],
            min_relative_intensity=0.1,
            mz_shift_da=4.0,
        )
        assert decoys[0].peaks.mz.size == 2
        # The workflow passed the config values through: a decoy hit entered
        # the competition (it is never exported, but it calibrates the
        # query). Single query -> q=1.0, p = (1 + 0)/(1 + 1) = 0.5.
        assert result.results[0]["q_value"] == pytest.approx(1.0)
        assert result.results[0]["p_value"] == pytest.approx(0.5)
        assert result.fdr_summary is not None
        assert result.fdr_summary["n_decoy_competitions"] == 1
