"""
Regression tests for the decoy -> target-decoy-competition path.

Motivation (issue #68: ``fdr_uncalibrated`` results & zero decoy hits):
real-world runs report ``n_decoy_competitions == 0``. These tests pin down the
full decoy path stage by stage, so a future change cannot silently lose decoys
again, and so the *reason* for an empty decoy null stays diagnosable:

1. generation — decoys keep their source's complete peak list, precursor m/z,
   entropy, and metadata (a decoy must be as matchable as the library spectrum
   it stands in for);
2. filtering — decoys pass the MS1 precursor window (Da and 5 ppm modes) and
   the ion-mode gate exactly like targets, while out-of-window/incompatible
   references are still rejected;
3. scoring — decoy hits that clear the configured gates enter target-decoy
   competition as decoy competitions (``n_decoy_competitions > 0``);
4. physics gate — genuinely invalid reference spectra are still rejected and
   decoys inherit only validated metadata;
5. execution modes — chunked (streaming) library searching generates and
   competes exactly the same decoys as in-memory searching;
6. diagnostics — an empty decoy null is reported (engine flag, workflow flag,
   warning with the measured best decoy score) and never exported as if it were
   an FDR-controlled result.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp

import MassFlow.similarity as similarity_module
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.processing import physical_integrity_reason
from MassFlow.similarity import (
    SimilarityEngine,
    calibrate_query_level_fdr,
    generate_decoys,
    spectral_entropy,
)
from MassFlow.workflow import run_annotation_pipeline

pytestmark = pytest.mark.scientific


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Caffeine fragmentation series (public MS/MS libraries; also used by the
# scientific-validation fixture): precursor 195.0877 [M+H]+, m/z ascending as
# required by matchms.
CAFFEINE_MZ = [42.0344, 83.0608, 110.0717, 138.0662]
CAFFEINE_INTENSITIES = [0.15, 0.42, 0.85, 1.0]
CAFFEINE_PRECURSOR = 195.0877
CAFFEINE_MH = 195.087652  # theoretical [M+H]+ m/z (formula C8H10N4O2)


def _spectrum(
    spec_id: str,
    mz: list[float],
    intensities: list[float],
    precursor_mz: float,
    **extra_meta: Any,
) -> Spectrum:
    """Deterministic float64 spectrum with standard metadata."""
    metadata: dict[str, Any] = {"id": spec_id, "precursor_mz": precursor_mz}
    metadata.update(extra_meta)
    return Spectrum(
        mz=np.asarray(mz, dtype=np.float64),
        intensities=np.asarray(intensities, dtype=np.float64),
        metadata=metadata,
    )


def _caffeine_spectrum(spec_id: str = "ref_caffeine") -> Spectrum:
    return _spectrum(
        spec_id,
        CAFFEINE_MZ,
        CAFFEINE_INTENSITIES,
        CAFFEINE_PRECURSOR,
        compound_name="Caffeine",
    )


def _query_matching_decoy(
    decoy: Spectrum, spec_id: str = "query_from_decoy"
) -> Spectrum:
    """A query whose peaks are exactly a decoy's peaks.

    Such a query is matched by that decoy at cosine 1.0 with every decoy peak
    matched, so the decoy's score *warrants* identification and the decoy must
    reach target-decoy competition. Used to probe the plumbing of the decoy
    path without changing any production threshold.
    """
    metadata = {
        key: value
        for key, value in decoy.metadata.items()
        if key not in ("id", "is_decoy", "spectral_entropy")
    }
    metadata["id"] = spec_id
    return Spectrum(
        mz=np.asarray(decoy.peaks.mz, dtype=np.float64).copy(),
        intensities=np.asarray(decoy.peaks.intensities, dtype=np.float64).copy(),
        metadata=metadata,
    )


def _engine(**overrides: Any) -> SimilarityEngine:
    settings: dict[str, Any] = {
        "algorithm": "cosine",
        "ms1_tolerance": 0.02,
        "ms2_tolerance": 0.02,
        "min_score": 0.6,
        "min_matched_peaks": 3,
    }
    settings.update(overrides)
    return SimilarityEngine(SimilarityConfig(**settings))


# ---------------------------------------------------------------------------
# 1. Generation: decoys stay matchable and faithful to their source
# ---------------------------------------------------------------------------


class TestDecoyGenerationFidelity:
    def test_decoy_keeps_full_peak_list(self) -> None:
        """The baseline filter selects intensities to permute, not peaks to drop.

        A decoy that lost its sub-1%-of-base-peak peaks would be systematically
        less matchable than the library spectra it stands in for — a
        target/decoy asymmetry that biases the FDR estimate.
        """
        source = _spectrum(
            "noisy",
            [100.0, 150.0, 200.0, 300.0, 400.0],
            [1000.0, 500.0, 200.0, 5.0, 2.0],  # 5.0 / 2.0 are below 1% of 1000
            CAFFEINE_PRECURSOR,
        )
        decoy = generate_decoys([source], random_seed=7)[0]

        source_mz = np.asarray(source.peaks.mz, dtype=np.float64)
        decoy_mz = np.asarray(decoy.peaks.mz, dtype=np.float64)
        decoy_intensities = np.asarray(decoy.peaks.intensities, dtype=np.float64)

        assert decoy_mz.size == source_mz.size == 5
        # Baseline-noise peaks keep their own intensities...
        assert sorted(decoy_intensities)[:2] == [2.0, 5.0]
        # ...so the entropy-relevant profile is still a permutation of the
        # source's three-signal-peak profile, and the measured entropy matches.
        assert sorted(decoy_intensities)[2:] == [200.0, 500.0, 1000.0]
        source_entropy = spectral_entropy(
            np.asarray(source.peaks.intensities, dtype=np.float64)
        )
        assert spectral_entropy(decoy_intensities) == pytest.approx(
            source_entropy, abs=1e-12
        )

    def test_decoy_preserves_precursor_identity_and_metadata(self) -> None:
        source = _caffeine_spectrum()
        decoy = generate_decoys([source], random_seed=11)[0]
        assert float(decoy.get("precursor_mz")) == pytest.approx(
            CAFFEINE_PRECURSOR, rel=1e-15
        )
        assert decoy.get("is_decoy") is True
        assert str(decoy.get("id")).endswith("_decoy")
        assert str(decoy.get("compound_name")).endswith("_decoy")

    def test_peak_count_matches_source_for_every_representative_spectrum(self) -> None:
        rng = np.random.default_rng(2024)
        sources = [
            _spectrum(
                f"lib{i}",
                np.sort(rng.uniform(50.0, 900.0, n)).tolist(),
                np.maximum(rng.lognormal(0.0, 1.2, n) / 5.0, 0.01).tolist(),
                400.0 + i,
            )
            for i, n in enumerate((5, 20, 100, 301))
        ]
        decoys = generate_decoys(sources, random_seed=42)
        for source, decoy in zip(sources, decoys):
            assert np.asarray(decoy.peaks.mz).size == np.asarray(source.peaks.mz).size
            # Positions are jittered, never dropped; the mass range is kept.
            assert np.asarray(decoy.peaks.mz).min() > 0.0
            assert float(decoy.peaks.mz.max()) == pytest.approx(
                float(source.peaks.mz.max()), abs=1.5
            )


# ---------------------------------------------------------------------------
# 2. Filtering: the precursor/ion-mode path treats decoys like targets
# ---------------------------------------------------------------------------


class TestDecoyFilterPath:
    @pytest.mark.parametrize("resolution_ppm", [None, 5.0])
    def test_decoy_is_admitted_by_the_precursor_window_it_inherits(
        self, resolution_ppm: float | None
    ) -> None:
        """The 5 ppm precursor gate does NOT reject decoys.

        Decoys preserve the source's ``precursor_mz`` bit-for-bit, so they pass
        the same MS1 window as their target, in Da mode and in strict ppm mode.
        """
        target = _caffeine_spectrum()
        decoy = generate_decoys([target], random_seed=42)[0]
        query = _query_matching_decoy(decoy)

        engine = _engine(resolution_ppm=resolution_ppm)
        results = engine.search([query], [target], include_decoys=True)

        decoy_hits = [r for r in results if r["is_decoy"]]
        assert len(decoy_hits) == 1, (
            "the decoy must be a candidate under the same precursor window as "
            "its target"
        )
        assert decoy_hits[0]["score"] == pytest.approx(1.0)
        assert decoy_hits[0]["matched_peaks"] == len(CAFFEINE_MZ)

    def test_precursor_window_still_rejects_out_of_window_references(self) -> None:
        """The MS1 window is a real physical gate: it still filters candidates.

        The same query shifted 20 ppm away from the target/decoy precursor must
        find neither (the gate is not bypassed for any spectrum).
        """
        target = _caffeine_spectrum()
        decoy = generate_decoys([target], random_seed=42)[0]
        query = _query_matching_decoy(decoy)
        query.set(
            "precursor_mz", CAFFEINE_PRECURSOR * (1.0 + 20e-6)
        )  # 20 ppm: outside 5 ppm but inside 0.02 Da

        strict = _engine(resolution_ppm=5.0)
        assert strict.search([query], [target], include_decoys=True) == []

    def test_ion_mode_gate_applies_identically_to_target_and_decoy(self) -> None:
        """An ion-mode conflict rejects target and decoy alike."""
        target = _caffeine_spectrum()
        target.set("adduct", "[M+H]+")
        decoy = generate_decoys([target], random_seed=42)[0]
        assert decoy.get("adduct") == "[M+H]+"

        query = _query_matching_decoy(decoy)
        query.set("adduct", "[M-H]-")  # negative mode vs positive-mode library

        results = _engine().search([query], [target], include_decoys=True)
        assert results == [], "opposite ion modes must match neither targets nor decoys"

        query.set("adduct", "[M+H]+")
        results = _engine().search([query], [target], include_decoys=True)
        assert sum(1 for r in results if r["is_decoy"]) == 1


# ---------------------------------------------------------------------------
# 3. Scoring: a decoy whose score warrants it reaches competition
# ---------------------------------------------------------------------------


class TestDecoyReachesCompetition:
    def test_decoy_hit_enters_target_decoy_competition(self) -> None:
        """Both sides of the competition are populated from engine output.

        One query matches the target exactly (a target competition), one query
        is built from the decoy's peaks (a decoy competition): the decoy side
        must be represented in target-decoy competition, not dropped.
        """
        target = _caffeine_spectrum()
        decoy = generate_decoys([target], random_seed=42)[0]
        query_target = _caffeine_spectrum("query_target")
        query_decoy = _query_matching_decoy(decoy)

        results = _engine().search(
            [query_target, query_decoy], [target], include_decoys=True
        )

        assert [r["reference_id"] for r in results if r["is_decoy"]] == [
            str(decoy.get("id"))
        ]
        decoy_hit = next(r for r in results if r["is_decoy"])
        assert decoy_hit["query_id"] == "query_from_decoy"
        assert decoy_hit["matched_peaks"] == len(CAFFEINE_MZ)

        _q_by_query, _p_by_query, summary = calibrate_query_level_fdr(results)
        assert summary["n_target_competitions"] == 1
        assert summary["n_decoy_competitions"] == 1, (
            "a decoy hit that clears the configured gates must enter "
            "target-decoy competition"
        )

    def test_decoys_are_absent_when_not_requested(self) -> None:
        """The same search without decoys has no decoy rows at all."""
        target = _caffeine_spectrum()
        query = _caffeine_spectrum("query_exact")

        results = _engine().search([query], [target], include_decoys=False)
        assert all(not r["is_decoy"] for r in results)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(1.0)

    def test_engine_reports_decoy_side_statistics(self) -> None:
        """The engine exposes what the decoy side achieved, emptied or not."""
        target = _caffeine_spectrum()
        decoy = generate_decoys([target], random_seed=42)[0]
        query = _query_matching_decoy(decoy)
        engine = _engine()

        engine.search([query], [target], include_decoys=True)
        diagnostics = engine.decoy_diagnostics

        assert diagnostics["n_decoy_pairs_scored"] == 1
        assert diagnostics["n_decoy_hits"] == 1
        assert diagnostics["decoy_null_empty"] is False
        assert diagnostics["best_decoy_score"] == pytest.approx(1.0)
        assert engine.degraded_mode_flags == []

    def test_empty_decoy_null_is_reported_by_the_engine(self) -> None:
        """A decoy that cannot reach the gates is reported, not silently lost."""
        target = _caffeine_spectrum()
        query = _caffeine_spectrum("query_exact")
        engine = _engine(min_score=0.7)

        results = engine.search(
            [query], [target], include_decoys=True, decoy_mz_shift_da=1.0
        )
        diagnostics = engine.decoy_diagnostics

        assert [r for r in results if r["is_decoy"]] == []
        assert diagnostics["decoy_null_empty"] is True
        assert diagnostics["n_decoy_pairs_scored"] == 1
        assert diagnostics["best_decoy_score"] < 0.7
        assert "decoy_null_empty" in engine.degraded_mode_flags

    def test_decoy_displacement_inside_tolerance_is_flagged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A near-copy decoy (displacement below the MS2 tolerance) is warned about.

        Such decoys coincide with their source at scoring tolerance, so the
        true match's own decoy competes with it and the null is inflated.
        """
        target = _caffeine_spectrum()
        query = _caffeine_spectrum("query_exact")
        engine = _engine()

        with caplog.at_level("WARNING", logger="MassFlow.similarity"):
            engine.search(
                [query], [target], include_decoys=True, decoy_mz_shift_da=0.005
            )

        assert any(
            "below the MS2 scoring tolerance" in record.message
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# 4. The 5 ppm physical-integrity gate is untouched
# ---------------------------------------------------------------------------


class TestPhysicsGatePreserved:
    def test_invalid_reference_spectrum_is_still_rejected(self) -> None:
        """A >5 ppm precursor/formula conflict is still a hard rejection.

        The gate needs a complete context (structure + charge + adduct/ion
        mode); with one, a 10 ppm deviation must still be rejected.
        """
        conflicting = _spectrum(
            "bad_precursor",
            CAFFEINE_MZ,
            CAFFEINE_INTENSITIES,
            CAFFEINE_MH * (1.0 + 10e-6),  # 10 ppm off the [M+H]+ formula mass
            formula="C8H10N4O2",
            charge=1,
            adduct="[M+H]+",
            ionmode="positive",
        )
        reason = physical_integrity_reason(conflicting)
        assert reason is not None and "ppm" in reason

        valid = _spectrum(
            "good_precursor",
            CAFFEINE_MZ,
            CAFFEINE_INTENSITIES,
            CAFFEINE_MH,
            formula="C8H10N4O2",
            charge=1,
            adduct="[M+H]+",
            ionmode="positive",
        )
        assert physical_integrity_reason(valid) is None

    def test_decoys_inherit_only_validated_metadata(self) -> None:
        """Decoys never invent structural claims; they copy validated metadata.

        The decoy's precursor m/z is the target's, so it cannot pass the gate
        for a spectrum whose source could not.
        """
        valid = _spectrum(
            "good_precursor",
            CAFFEINE_MZ,
            CAFFEINE_INTENSITIES,
            CAFFEINE_MH,
            formula="C8H10N4O2",
            charge=1,
            adduct="[M+H]+",
            ionmode="positive",
        )
        decoy = generate_decoys([valid], random_seed=13)[0]
        decoy.set("id", "good_precursor_decoy")
        assert float(decoy.get("precursor_mz")) == pytest.approx(CAFFEINE_MH, rel=1e-15)
        assert physical_integrity_reason(decoy) is None


# ---------------------------------------------------------------------------
# 5. Execution modes: chunked streaming == in-memory
# ---------------------------------------------------------------------------


class TestStreamingDecoyEquivalence:
    def test_chunked_streaming_matches_in_memory_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunk boundaries must not change which decoys compete.

        Decoys are generated per chunk from content-hashed per-spectrum seeds,
        so a streamed library (small chunks forced here) must produce exactly
        the same decoy hits as the in-memory list.
        """
        library = [
            _spectrum(
                f"lib{i}",
                [100.0 + i, 150.0 + i, 200.0 + i, 250.0 + i],
                [1.0, 0.7, 0.4, 0.2],
                300.0 + i,
            )
            for i in range(5)
        ]
        queries = [
            _query_matching_decoy(
                generate_decoys([spec], random_seed=42, mz_shift_da=0.001)[0],
                spec_id=f"query_{i}",
            )
            for i, spec in enumerate(library)
        ]

        engine = _engine(min_score=0.5)
        in_memory = engine.search(
            queries, list(library), include_decoys=True, decoy_mz_shift_da=0.001
        )

        def _small_chunks(spectra, chunk_size=10000):
            chunk: list[Spectrum] = []
            for spectrum in spectra:
                chunk.append(spectrum)
                if len(chunk) >= 2:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

        monkeypatch.setattr(similarity_module, "yield_fixed_chunks", _small_chunks)
        streamed = engine.search(
            queries, iter(library), include_decoys=True, decoy_mz_shift_da=0.001
        )

        def _decoy_signature(results):
            return sorted(
                (
                    r["query_id"],
                    r["reference_id"],
                    round(r["score"], 12),
                    int(r["matched_peaks"]),
                )
                for r in results
                if r["is_decoy"]
            )

        assert _decoy_signature(in_memory) == _decoy_signature(streamed)
        assert _decoy_signature(in_memory), "the fixture must produce decoy hits"
        assert engine.decoy_diagnostics["n_decoy_hits"] == len(
            _decoy_signature(streamed)
        )


# ---------------------------------------------------------------------------
# 6. End-to-end: an empty decoy null is flagged, explained, and never
#    exported as an FDR-controlled result
# ---------------------------------------------------------------------------


def _write_fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    query = _caffeine_spectrum("query_caffeine")
    reference = _caffeine_spectrum("ref_caffeine")
    unused = _spectrum("ref_unrelated", [500.0, 600.0], [1.0, 0.5], 700.0)

    query_path = tmp_path / "experiment.mgf"
    library_path = tmp_path / "library.msp"
    save_as_mgf([query], str(query_path))
    save_as_msp([reference, unused], str(library_path))
    return query_path, library_path


def _run_pipeline(
    tmp_path: Path, **similarity_overrides: Any
) -> tuple[Any, list[dict[str, str]]]:
    query_path, library_path = _write_fixture_files(tmp_path)
    output_dir = tmp_path / "results"
    similarity_settings: dict[str, Any] = {
        "algorithm": "cosine",
        "ms1_tolerance": 0.02,
        "ms2_tolerance": 0.02,
        "min_matched_peaks": 3,
        "min_score": 0.7,
        "fdr_threshold": 1.0,
    }
    processing_settings: dict[str, Any] = {
        "min_peaks": 1,
        "noise_threshold": 0.0,
        "min_intensity": 0.0,
    }
    for key, value in similarity_overrides.items():
        # Decoy-generation parameters live in the processing config (they
        # describe the decoy spectra, not the search).
        if key.startswith("decoy_"):
            processing_settings[key] = value
        else:
            similarity_settings[key] = value
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=output_dir),
        input=InputConfig(input_path=query_path, library_path=library_path),
        processing=ProcessingConfig(**processing_settings),
        similarity=SimilarityConfig(**similarity_settings),
    )
    results = run_annotation_pipeline(config)
    assert len(results) == 1
    result = results[0]
    csv_path = next(iter(sorted(output_dir.glob("*_results.csv"))))
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return result, rows


class TestEmptyDecoyNullDiagnostics:
    def test_empty_decoy_null_is_flagged_explained_and_labeled(
        self, tmp_path: Path
    ) -> None:
        """Zero decoy hits: explicit flags, an explanatory warning, and rows
        that do not read as FDR-controlled."""
        result, rows = _run_pipeline(tmp_path, decoy_mz_shift_da=1.0)

        assert result.fdr_summary is not None
        assert result.fdr_summary["n_decoy_competitions"] == 0
        assert "fdr_uncalibrated" in result.degraded_mode_flags
        assert "decoy_null_empty" in result.degraded_mode_flags
        assert any("best decoy score" in w for w in result.warnings), (
            "the warning must explain that decoys were scored but stayed below "
            "the gates"
        )

        matched_rows = [row for row in rows if row.get("score")]
        assert matched_rows
        assert all(row["Annotation_Status"] == "Uncalibrated" for row in matched_rows)

    def test_decoy_evidence_clears_the_uncalibrated_marker(
        self, tmp_path: Path
    ) -> None:
        """Decoy hits that clear the gates are reported and un-mark the run.

        With a displacement inside the MS2 tolerance the decoy coincides with
        its source's fragments and therefore competes (the documented
        near-copy regime); the exported rows are then labeled normally.
        """
        result, rows = _run_pipeline(tmp_path, decoy_mz_shift_da=0.001, min_score=0.5)

        assert result.fdr_summary is not None
        assert result.fdr_summary["n_decoy_competitions"] == 1
        assert "fdr_uncalibrated" not in result.degraded_mode_flags
        assert "decoy_null_empty" not in result.degraded_mode_flags

        matched_rows = [row for row in rows if row.get("score")]
        assert matched_rows
        assert all(
            row["Annotation_Status"] in {"Matched", "Putative"} for row in matched_rows
        )
