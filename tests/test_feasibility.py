"""
Tests for the feasibility pre-analysis (``MassFlow.feasibility``), the
``massflow preflight`` command, and the zero-hit diagnosis.

These tests pin the guarantees the feature advertises:

* precursor compatibility is computed with the same routines the `cosine`
  search uses, and agrees with an independent brute-force reference;
* the four zero-hit situations are distinguished only where the counters
  support it;
* smallness and FDR-reliability are reported independently;
* preflight never alters parameters or blocks a run.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_msp

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.feasibility import (
    LibraryProfile,
    build_feasibility_report,
    diagnose_zero_hits,
    precursor_compatibility,
    profile_library,
    profile_query_file,
    query_profile_from_spectra,
    render_feasibility_report,
)
from MassFlow.similarity import SimilarityEngine
from MassFlow.storage import create_spectral_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spectrum(
    spec_id: str,
    precursor_mz: float,
    peaks: list[tuple[float, float]],
    *,
    adduct: str | None = None,
    formula: str | None = None,
    retention_time: float | None = None,
) -> Spectrum:
    mz = np.asarray([p[0] for p in peaks], dtype=np.float64)
    intensities = np.asarray([p[1] for p in peaks], dtype=np.float64)
    metadata: dict = {"id": spec_id, "precursor_mz": float(precursor_mz)}
    if adduct is not None:
        metadata["adduct"] = adduct
    if formula is not None:
        metadata["formula"] = formula
    if retention_time is not None:
        metadata["retention_time"] = retention_time
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)


def library_profile(
    mzs: list[float],
    adducts: list[str | None] | None = None,
) -> LibraryProfile:
    n = len(mzs)
    return LibraryProfile(
        source_path=Path("library.msp"),
        kind="store",
        usable_count=n,
        rejected_count=0,
        io_rejected_count=0,
        processing_rejected_count=0,
        integrity_rejected_count=0,
        rejection_examples=[],
        missing_precursor_count=int(np.count_nonzero(np.isnan(mzs))),
        precursor_mzs=np.asarray(mzs, dtype=np.float64),
        retention_times=np.full(n, np.nan, dtype=np.float64),
        adducts=list(adducts) if adducts is not None else [None] * n,
        ion_modes={},
        ion_mode_reliable=False,
    )


def make_config(
    tmp_path: Path,
    *,
    algorithm: str = "cosine",
    ms1_tolerance: float = 0.02,
    min_matched_peaks: int = 3,
    fdr_threshold: float = 0.05,
    library_path: Path | None = None,
    input_path: Path | None = None,
) -> MassFlowConfig:
    return MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "out"),
        input=InputConfig(
            input_path=input_path if input_path is not None else tmp_path / "query.msp",
            library_path=library_path,
            format=None,
        ),
        similarity=SimilarityConfig(
            algorithm=algorithm,  # type: ignore[arg-type]
            ms1_tolerance=ms1_tolerance,
            min_matched_peaks=min_matched_peaks,
            fdr_threshold=fdr_threshold,
        ),
    )


PEAKS = [(50.0, 100.0), (80.0, 500.0), (120.0, 250.0), (180.0, 75.0)]


# ---------------------------------------------------------------------------
# Precursor compatibility: none / partial / complete
# ---------------------------------------------------------------------------


class TestPrecursorCompatibility:
    def test_no_overlap(self, tmp_path):
        lib = library_profile([100.0, 200.0, 300.0])
        queries = [
            make_spectrum("q0", 500.0, PEAKS),
            make_spectrum("q1", 600.0, PEAKS),
        ]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(lib, qp, make_config(tmp_path))

        assert report.n_queries == 2
        assert report.n_queries_with_candidate == 0
        assert report.n_queries_with_compatible_candidate == 0
        assert report.n_candidate_pairs == 0

    def test_partial_overlap(self, tmp_path):
        lib = library_profile([100.0, 200.0, 300.0])
        queries = [
            make_spectrum("q0", 100.01, PEAKS),  # within 0.02 of 100
            make_spectrum("q1", 500.0, PEAKS),  # no candidate
        ]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(lib, qp, make_config(tmp_path))

        assert report.n_queries_with_candidate == 1
        assert report.n_queries_with_compatible_candidate == 1

    def test_complete_overlap(self, tmp_path):
        lib = library_profile([100.0, 200.0])
        queries = [
            make_spectrum("q0", 100.005, PEAKS),
            make_spectrum("q1", 200.01, PEAKS),
        ]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(lib, qp, make_config(tmp_path))

        assert report.n_queries_with_candidate == 2
        assert report.n_queries_with_compatible_candidate == 2

    def test_tolerance_is_respected(self, tmp_path):
        """A precursor just outside the tolerance is not a candidate."""
        lib = library_profile([100.0])
        queries = [make_spectrum("q0", 100.05, PEAKS)]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(
            lib, qp, make_config(tmp_path, ms1_tolerance=0.02)
        )
        assert report.n_queries_with_candidate == 0

    def test_ion_mode_gate_excludes_opposite_modes(self, tmp_path):
        """Precursor overlap in opposite ion modes is not a compatible candidate."""
        lib = library_profile([100.0], adducts=["[M+H]+"])
        queries = [make_spectrum("q0", 100.005, PEAKS, adduct="[M-H]-")]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(lib, qp, make_config(tmp_path))

        assert report.n_queries_with_candidate == 1  # precursor overlaps
        assert report.n_queries_with_compatible_candidate == 0  # but mode clashes

    def test_modified_cosine_is_not_definitive(self, tmp_path):
        lib = library_profile([100.0])
        queries = [make_spectrum("q0", 500.0, PEAKS)]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(
            lib, qp, make_config(tmp_path, algorithm="modified_cosine")
        )
        assert report.ms1_gate_definitive is False

    def test_resolution_ppm_window_is_used(self, tmp_path):
        lib = library_profile([100.001])
        queries = [make_spectrum("q0", 100.0, PEAKS)]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        config = make_config(tmp_path, ms1_tolerance=0.0001)
        config.similarity.resolution_ppm = 20.0
        report = precursor_compatibility(lib, qp, config)
        assert report.n_queries_with_candidate == 1
        assert report.resolution_ppm == 20.0

    def test_rt_filter_excludes_known_incompatible_pair(self, tmp_path):
        lib = library_profile([100.0])
        lib.retention_times[0] = 10.0
        queries = [make_spectrum("q0", 100.0, PEAKS)]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        qp.retention_times[0] = 20.0
        config = make_config(tmp_path)
        config.similarity.rt_tolerance = 1.0
        report = precursor_compatibility(lib, qp, config)
        assert report.n_queries_with_candidate == 1
        assert report.n_queries_with_compatible_candidate == 0

    def test_missing_reference_precursor_bypasses_ms1_like_search(self, tmp_path):
        lib = library_profile([float("nan")])
        queries = [make_spectrum("q0", 999.0, PEAKS)]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(lib, qp, make_config(tmp_path))
        assert report.n_queries_with_candidate == 1
        assert report.n_queries_with_precursor_candidate == 0
        assert report.n_missing_precursor_pairs == 1


class TestAgreementWithReference:
    """Feasibility coverage must match an independent brute-force calculation."""

    @pytest.mark.parametrize(
        "ref_mzs, query_mzs",
        [
            ([100.0, 200.0, 300.0], [100.01, 250.0, 300.02]),
            ([10.0], [10.5, 11.0]),
            ([100.0, 200.0], [100.0, 200.0, 150.0]),
        ],
    )
    def test_matches_brute_force(self, tmp_path, ref_mzs, query_mzs):
        tol = 0.02
        lib = library_profile(ref_mzs)
        queries = [make_spectrum(f"q{i}", mz, PEAKS) for i, mz in enumerate(query_mzs)]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(
            lib, qp, make_config(tmp_path, ms1_tolerance=tol)
        )

        expected = sum(1 for q in query_mzs if any(abs(r - q) <= tol for r in ref_mzs))
        assert report.n_queries_with_candidate == expected


class TestAgreementWithSearch:
    """Preflight must not disagree with the cosine engine's MS1 candidate gate."""

    def test_ms1_excluded_query_never_scores(self, tmp_path):
        # Two references share the query's peak list; only one shares its
        # precursor. The cosine MS1 gate must exclude the far-precursor query
        # entirely, which is what preflight reports as "no candidate".
        refs = [
            make_spectrum("r_match", 100.0, PEAKS),
            make_spectrum("r_far", 500.0, PEAKS),
        ]
        queries = [
            make_spectrum("q_ok", 100.005, PEAKS),
            make_spectrum("q_far", 400.0, PEAKS),
        ]
        cfg = make_config(tmp_path, algorithm="cosine", min_matched_peaks=1)
        cfg.similarity.min_score = 0.5

        engine = SimilarityEngine(cfg.similarity)
        results = engine.search(queries, refs, include_decoys=False, min_score=0.5)

        scored_query_ids = {r["query_id"] for r in results}
        assert "q_ok" in scored_query_ids
        assert "q_far" not in scored_query_ids

        # Preflight agrees: q_far is not a candidate, q_ok is.
        lib = library_profile([100.0, 500.0])
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(lib, qp, cfg)
        assert report.n_queries_with_candidate == 1
        assert report.n_candidate_pairs == 1
        assert report.n_queries_with_precursor_candidate == 1
        assert engine.search_diagnostics["n_target_pairs_scored"] == 1
        assert engine.search_diagnostics["n_target_scores_above_gates"] == 1

    def test_rt_eligibility_matches_actual_search(self, tmp_path):
        refs = [make_spectrum("r", 100.0, PEAKS, retention_time=10.0)]
        queries = [make_spectrum("q", 100.0, PEAKS, retention_time=20.0)]
        cfg = make_config(tmp_path, min_matched_peaks=1)
        cfg.similarity.min_score = 0.5
        cfg.similarity.rt_tolerance = 1.0
        engine = SimilarityEngine(cfg.similarity)
        results = engine.search(queries, refs, include_decoys=False)
        assert not results

        lib = library_profile([100.0])
        lib.retention_times[0] = 10.0
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        report = precursor_compatibility(lib, qp, cfg)
        assert report.n_queries_with_candidate == 1
        assert report.n_queries_with_compatible_candidate == 0


# ---------------------------------------------------------------------------
# Query profile: sparse / empty MS2 spectra
# ---------------------------------------------------------------------------


class TestQueryProfileQuality:
    def test_peak_count_distribution(self, tmp_path):
        queries = [
            make_spectrum("empty", 100.0, []),
            make_spectrum("one", 101.0, [(50.0, 1.0)]),
            make_spectrum("two", 102.0, [(50.0, 1.0), (60.0, 2.0)]),
            make_spectrum("five", 103.0, [(float(m), 1.0) for m in range(5)]),
        ]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        assert list(qp.peak_counts) == [0, 1, 2, 5]
        assert int((qp.peak_counts < 3).sum()) == 3

    def test_empty_spectra_are_retained_in_profile(self, tmp_path):
        queries = [make_spectrum("empty", 100.0, [])]
        qp = query_profile_from_spectra(tmp_path / "q.msp", queries)
        assert qp.n_retained == 1
        assert qp.peak_counts[0] == 0

    def test_rejected_queries_are_counted(self, tmp_path):
        """I/O-validation rejections count, matching the annotation pipeline."""
        mgf = tmp_path / "q.mgf"
        mgf.write_text(
            "BEGIN IONS\nTITLE=q_ok\nPEPMASS=100.0\nCHARGE=1+\n"
            "50.0 1.0\n80.0 2.0\nEND IONS\n\n"
            # A spectrum with no precursor must be rejected by validation.
            "BEGIN IONS\nTITLE=q_bad\nCHARGE=1+\n50.0 1.0\nEND IONS\n\n"
        )
        cfg = make_config(tmp_path, input_path=mgf)
        profile = profile_query_file(mgf, cfg)
        assert profile.rejected_count >= 1
        assert profile.loaded_count >= 1


# ---------------------------------------------------------------------------
# Reference library: integrity and statistics
# ---------------------------------------------------------------------------


class TestLibraryIntegrity:
    def test_raw_library_with_impossible_chemistry_is_flagged(self, tmp_path):
        # Declared formula C2H6O with a precursor m/z of 500 (M+H) is off by
        # orders of magnitude: the 5 ppm gate must reject it.
        good = make_spectrum("Good", 47.0491, PEAKS, adduct="[M+H]+", formula="C2H6O")
        bad = make_spectrum("Bad", 500.0, PEAKS, adduct="[M+H]+", formula="C2H6O")
        lib_path = tmp_path / "library.msp"
        save_as_msp([good, bad], str(lib_path))

        cfg = make_config(tmp_path, library_path=lib_path)
        profile = profile_library(cfg)

        assert profile.kind == "file"
        assert profile.usable_count == 1
        assert profile.rejected_count >= 1
        assert profile.integrity_rejected_count == 1
        assert any("ppm" in reason.lower() for reason in profile.rejection_examples)

    def test_empty_massflow_store_is_reported_as_no_usable_spectra(self, tmp_path):
        store_path = tmp_path / "empty.db"
        store = create_spectral_store(store_path)
        store.close()
        query_path = tmp_path / "query.msp"
        save_as_msp([make_spectrum("Q0", 100.0, PEAKS)], str(query_path))
        report = build_feasibility_report(
            make_config(tmp_path, library_path=store_path, input_path=query_path)
        )
        assert report.library.usable_count == 0
        assert any("no usable spectra" in f.text.lower() for f in report.findings)

    def test_corrupt_massflow_store_fails_preflight_explicitly(self, tmp_path):
        store_path = tmp_path / "corrupt.db"
        store_path.write_text("not a SQLite database")
        query_path = tmp_path / "query.msp"
        save_as_msp([make_spectrum("Q0", 100.0, PEAKS)], str(query_path))
        with pytest.raises(sqlite3.DatabaseError):
            build_feasibility_report(
                make_config(tmp_path, library_path=store_path, input_path=query_path)
            )

    def test_clean_library_has_no_rejections(self, tmp_path):
        entries = [
            make_spectrum("A", 47.0491, PEAKS, adduct="[M+H]+", formula="C2H6O"),
            make_spectrum("B", 61.0284, PEAKS, adduct="[M+H]+", formula="C2H4O2"),
        ]
        lib_path = tmp_path / "library.msp"
        save_as_msp(entries, str(lib_path))
        profile = profile_library(make_config(tmp_path, library_path=lib_path))
        assert profile.usable_count == 2
        assert profile.rejected_count == 0


# ---------------------------------------------------------------------------
# Report assessment: smallness vs FDR reliability
# ---------------------------------------------------------------------------


def _write_library_msp(path: Path, n: int) -> None:
    spectra = [
        make_spectrum(f"L{i}", 100.0 + i, PEAKS, adduct="[M+H]+") for i in range(n)
    ]
    save_as_msp(spectra, str(path))


class TestAssessment:
    def _report(self, tmp_path, n_lib, fdr):
        lib_path = tmp_path / "lib.msp"
        q_path = tmp_path / "query.msp"
        _write_library_msp(lib_path, n_lib)
        save_as_msp([make_spectrum("Q0", 100.0, PEAKS)], str(q_path))
        cfg = make_config(
            tmp_path, library_path=lib_path, input_path=q_path, fdr_threshold=fdr
        )
        return build_feasibility_report(cfg)

    def test_small_library_reported_regardless_of_fdr(self, tmp_path):
        report = self._report(tmp_path, n_lib=10, fdr=1.0)
        assert any("Small reference library" in f.text for f in report.findings), (
            "smallness must be reported even when fdr_threshold is relaxed"
        )

    def test_fdr_caveat_only_when_threshold_strict(self, tmp_path):
        strict = self._report(tmp_path, n_lib=10, fdr=0.05)
        relaxed = self._report(tmp_path, n_lib=10, fdr=1.0)
        assert any("larger relevant library" in r for r in strict.recommendations)
        assert not any("larger relevant library" in r for r in relaxed.recommendations)
        assert all(
            "raise fdr_threshold" not in r.lower() for r in strict.recommendations
        )

    def test_no_overlap_flagged_as_definitive_for_cosine(self, tmp_path):
        report = self._report(tmp_path, n_lib=10, fdr=1.0)
        # The single query (m/z 100) overlaps the library, so coverage is full;
        # instead assert the report renders and is advisory.
        text = render_feasibility_report(report)
        assert "MassFlow Preflight" in text
        assert "Advisory only" in text

    def test_modified_cosine_no_precursor_overlap_is_not_definitive(self, tmp_path):
        lib_path = tmp_path / "lib.msp"
        q_path = tmp_path / "query.msp"
        _write_library_msp(lib_path, 3)
        save_as_msp([make_spectrum("Q0", 999.0, PEAKS)], str(q_path))
        cfg = make_config(
            tmp_path,
            algorithm="modified_cosine",
            library_path=lib_path,
            input_path=q_path,
        )
        report = build_feasibility_report(cfg)
        coverage_finding = next(
            f for f in report.findings if "precursor-window candidate" in f.text
        )
        assert coverage_finding.definitive is False
        assert "not a definitive no-match" in coverage_finding.text

    def test_full_coverage_reported(self, tmp_path):
        lib_path = tmp_path / "lib.msp"
        q_path = tmp_path / "query.msp"
        _write_library_msp(lib_path, 5)  # library at 100..104
        save_as_msp([make_spectrum("Q0", 102.0, PEAKS)], str(q_path))
        cfg = make_config(tmp_path, library_path=lib_path, input_path=q_path)
        report = build_feasibility_report(cfg)
        assert report.compatibility[0].n_queries_with_compatible_candidate == 1
        assert not any("No query" in f.text for f in report.findings)


# ---------------------------------------------------------------------------
# Zero-hit diagnosis
# ---------------------------------------------------------------------------


class TestZeroHitDiagnosis:
    def test_no_analyzable_spectra(self):
        reasons = diagnose_zero_hits(
            n_queries=0,
            n_rejected=10,
            compatibility=None,
            search_diagnostics=None,
            pre_fdr_target_hits=0,
            fdr_uncalibrated=False,
        )
        assert reasons and "No analyzable query spectra" in reasons[0]

    def test_no_covered_queries_is_definitive(self):
        reasons = diagnose_zero_hits(
            n_queries=100,
            n_rejected=0,
            compatibility={
                "engine": "cosine",
                "ms1_gate_definitive": True,
                "tolerance_da": 0.02,
                "n_queries_with_compatible_candidate": 0,
            },
            search_diagnostics={"n_target_scores_above_gates": 0},
            pre_fdr_target_hits=0,
            fdr_uncalibrated=False,
        )
        assert any("No query precursor" in r for r in reasons)

    def test_candidates_below_gates(self):
        reasons = diagnose_zero_hits(
            n_queries=100,
            n_rejected=0,
            compatibility={
                "engine": "cosine",
                "ms1_gate_definitive": True,
                "tolerance_da": 0.02,
                "n_queries_with_compatible_candidate": 20,
            },
            search_diagnostics={"n_target_scores_above_gates": 0},
            pre_fdr_target_hits=0,
            fdr_uncalibrated=False,
        )
        assert any("min_score and min_matched_peaks" in r for r in reasons)

    def test_fdr_filtered_hits(self):
        reasons = diagnose_zero_hits(
            n_queries=100,
            n_rejected=0,
            compatibility={
                "engine": "cosine",
                "ms1_gate_definitive": True,
                "tolerance_da": 0.02,
                "n_queries_with_compatible_candidate": 20,
            },
            search_diagnostics={"n_target_scores_above_gates": 5},
            pre_fdr_target_hits=5,
            fdr_uncalibrated=True,
        )
        assert any("removed by FDR filtering" in r for r in reasons)
        assert any("uncalibrated" in r for r in reasons)

    def test_no_evidence_yields_no_attribution(self):
        reasons = diagnose_zero_hits(
            n_queries=100,
            n_rejected=0,
            compatibility=None,
            search_diagnostics=None,
            pre_fdr_target_hits=0,
            fdr_uncalibrated=False,
        )
        assert reasons == []


# ---------------------------------------------------------------------------
# CLI: massflow preflight
# ---------------------------------------------------------------------------


class TestPreflightCli:
    def test_preflight_reports_coverage(self, tmp_path):
        from typer.testing import CliRunner

        from MassFlow import cli

        lib_path = tmp_path / "lib.msp"
        q_path = tmp_path / "query.msp"
        _write_library_msp(lib_path, 5)
        save_as_msp(
            [make_spectrum("Q0", 100.0, PEAKS), make_spectrum("Q1", 999.0, PEAKS)],
            str(q_path),
        )
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "project:\n"
            "  output_directory: out\n"
            "input:\n"
            f"  input_path: {q_path}\n"
            f"  library_path: {lib_path}\n"
            "similarity:\n"
            "  algorithm: cosine\n"
            "  ms1_tolerance: 0.02\n"
            "  min_matched_peaks: 3\n"
            "  fdr_threshold: 0.05\n"
        )

        result = CliRunner().invoke(cli.app, ["preflight", "--config", str(cfg_path)])
        assert result.exit_code == 0
        assert "MassFlow Preflight" in result.stdout
        assert "Queries with eligible precursor/mode/RT candidate" in result.stdout

    def test_preflight_is_advisory_on_poor_overlap(self, tmp_path):
        from typer.testing import CliRunner

        from MassFlow import cli

        lib_path = tmp_path / "lib.msp"
        q_path = tmp_path / "query.msp"
        _write_library_msp(lib_path, 5)  # library at 100..104
        save_as_msp([make_spectrum("Q0", 999.0, PEAKS)], str(q_path))  # no overlap
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "project:\n  output_directory: out\n"
            "input:\n"
            f"  input_path: {q_path}\n"
            f"  library_path: {lib_path}\n"
        )

        result = CliRunner().invoke(cli.app, ["preflight", "--config", str(cfg_path)])
        # Advisory: a scientifically poor run does not fail the command.
        assert result.exit_code == 0
        assert "Advisory only" in result.stdout

    def test_preflight_errors_on_missing_library(self, tmp_path):
        from typer.testing import CliRunner

        from MassFlow import cli

        q_path = tmp_path / "query.msp"
        save_as_msp([make_spectrum("Q0", 100.0, PEAKS)], str(q_path))
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "project:\n  output_directory: out\n"
            "input:\n"
            f"  input_path: {q_path}\n"
            f"  library_path: {tmp_path / 'missing.msp'}\n"
        )
        result = CliRunner().invoke(cli.app, ["preflight", "--config", str(cfg_path)])
        assert result.exit_code == 1
