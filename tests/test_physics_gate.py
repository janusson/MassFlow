"""
Tests for the strict 5 ppm physical-integrity gate in the classical pipeline.

Contract under test (see ``docs/CAPABILITY_MATRIX.md``, open decision D-8 and
§Scientific Data Integrity):

* Spectra that declare structural metadata (``formula`` / ``smiles`` /
  ``inchi``) are checked with the existing ``SpectrumMetadata`` /
  ``MolecularStructure`` contracts (``MassFlow.models``) at processing time.
  Malformed chemistry metadata and >5 ppm precursor deviations are rejected,
  never searched.
* Query files: rejections are counted per file (``spectra_rejected``) with
  human-readable reasons; a file whose spectra are all rejected is an
  explicit ``failed`` result with a failure report -- never an empty success.
* Raw reference libraries: any physically invalid entry aborts the run with a
  ``PhysicalIntegrityError`` before any file is processed, because a silently
  shrunk target pool would change every query's FDR calibration.
* Spectra without structural claims are exempt (the dominant case for raw
  experimental files) and pay no validation cost.
"""

from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp

from MassFlow import processing
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.processing import physical_integrity_reason
from MassFlow.workflow import run_annotation_pipeline

# ---------------------------------------------------------------------------
# Chemistry helpers (all masses derived from the same pyteomics SSOT the
# models use, so fixtures are exact by construction)
# ---------------------------------------------------------------------------

CAFFEINE_FORMULA = "C8H10N4O2"
CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CAFFEINE_MH = 195.087652  # [M+H]+ theoretical m/z of caffeine (monoisotopic)

SALICYLIC_FORMULA = "C7H6O3"
SALICYLIC_MH = 137.024416  # [M-H]- theoretical m/z of salicylic acid


def _precursor_at_ppm(theoretical_mz: float, ppm_error: float) -> float:
    """Precursor m/z that deviates from ``theoretical_mz`` by ``ppm_error``."""
    return theoretical_mz * (1.0 + ppm_error / 1e6)


def make_spectrum(
    spec_id: str,
    precursor_mz: float,
    peaks: list[tuple[float, float]],
    *,
    formula: str | None = None,
    smiles: str | None = None,
    inchi: str | None = None,
    exact_mass: float | None = None,
    charge: int | None = 1,
    adduct: str | None = None,
    ionmode: str | None = None,
) -> Spectrum:
    """Build a spectrum carrying the given chemistry claims."""
    metadata: dict = {
        "id": spec_id,
        "compound_name": spec_id,
        "precursor_mz": precursor_mz,
    }
    if charge is not None:
        metadata["charge"] = charge
    if adduct is not None:
        metadata["adduct"] = adduct
    if ionmode is not None:
        metadata["ionmode"] = ionmode
    if formula is not None:
        metadata["formula"] = formula
    if smiles is not None:
        metadata["smiles"] = smiles
    if inchi is not None:
        metadata["inchi"] = inchi
    if exact_mass is not None:
        metadata["exact_mass"] = exact_mass
    mz = np.asarray([p[0] for p in peaks], dtype=np.float64)
    intensities = np.asarray([p[1] for p in peaks], dtype=np.float64)
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)


def _pipeline_config(
    tmp_path: Path,
    query_path: Path,
    library_path: Path,
    *,
    output_name: str = "results",
) -> MassFlowConfig:
    return MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / output_name),
        input=InputConfig(input_path=query_path, library_path=library_path),
        processing=ProcessingConfig(min_peaks=1, noise_threshold=0.0),
        similarity=SimilarityConfig(
            fdr_threshold=1.0,
            min_score=0.0,
            min_matched_peaks=1,
            ms1_tolerance=100.0,
        ),
    )


# ---------------------------------------------------------------------------
# Gate unit tests (processing.physical_integrity_reason)
# ---------------------------------------------------------------------------


class TestGateUnit:
    def test_no_structural_claim_is_exempt(self):
        """Spectra without formula/smiles/inchi claims pass untouched."""
        s = make_spectrum("q1", 195.0877, [(100.0, 1.0)], charge=1)
        assert physical_integrity_reason(s) is None

    def test_missing_context_disables_strict_check(self):
        """A structure claim without charge (or adduct/ion mode) cannot be
        mass-verified; the model contract skips it gracefully."""
        s = make_spectrum(
            "q2", 999.0, [(100.0, 1.0)], formula=CAFFEINE_FORMULA, charge=None
        )
        assert physical_integrity_reason(s) is None

    def test_consistent_full_context_passes(self):
        s = make_spectrum(
            "ok",
            CAFFEINE_MH,
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            adduct="[M+H]+",
            ionmode="positive",
        )
        assert physical_integrity_reason(s) is None

    def test_imputed_adduct_from_ionmode(self):
        """Positive mode without an explicit adduct imputes [M+H]+: a
        matching precursor passes, a contradicting one is rejected."""
        passing = make_spectrum(
            "imp_ok", CAFFEINE_MH, [(100.0, 1.0)], formula=CAFFEINE_FORMULA
        )
        # metadata_processing derives ionmode=positive from charge=1, so the
        # gate sees the ionmode claim even when the file omitted it.
        passing.set("ionmode", "positive")
        assert physical_integrity_reason(passing) is None
        failing = make_spectrum(
            "imp_bad",
            _precursor_at_ppm(CAFFEINE_MH, 50.0),
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
        )
        failing.set("ionmode", "positive")
        reason = physical_integrity_reason(failing)
        assert reason is not None
        assert "precursor m/z" in reason and "ppm" in reason

    def test_over_5ppm_precursor_deviation_rejected(self):
        s = make_spectrum(
            "off_by_ten",
            _precursor_at_ppm(CAFFEINE_MH, 10.0),
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            adduct="[M+H]+",
            ionmode="positive",
        )
        reason = physical_integrity_reason(s)
        assert reason is not None
        assert "off_by_ten" in reason
        assert "10.0" in reason or "10.0" in reason or "ppm" in reason

    def test_5ppm_boundary(self):
        """4.999 ppm passes; 5.001 ppm is rejected (strict >5.0 gate)."""
        ok = make_spectrum(
            "b4999",
            _precursor_at_ppm(CAFFEINE_MH, 4.999),
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            adduct="[M+H]+",
            ionmode="positive",
        )
        assert physical_integrity_reason(ok) is None
        bad = make_spectrum(
            "b5001",
            _precursor_at_ppm(CAFFEINE_MH, 5.001),
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            adduct="[M+H]+",
            ionmode="positive",
        )
        reason = physical_integrity_reason(bad)
        assert reason is not None
        assert "5.0 ppm" in reason

    def test_negative_mode_over_5ppm_rejected(self):
        ok = make_spectrum(
            "neg_ok",
            SALICYLIC_MH,
            [(100.0, 1.0)],
            formula=SALICYLIC_FORMULA,
            charge=-1,
            adduct="[M-H]-",
            ionmode="negative",
        )
        assert physical_integrity_reason(ok) is None
        bad = make_spectrum(
            "neg_bad",
            _precursor_at_ppm(SALICYLIC_MH, 9.0),
            [(100.0, 1.0)],
            formula=SALICYLIC_FORMULA,
            charge=-1,
            adduct="[M-H]-",
            ionmode="negative",
        )
        reason = physical_integrity_reason(bad)
        assert reason is not None
        assert "neg_bad" in reason

    def test_unparseable_smiles_rejected(self):
        s = make_spectrum(
            "smiles_junk",
            195.0877,
            [(100.0, 1.0)],
            smiles="NOT_A_SMILES",
            adduct="[M+H]+",
            ionmode="positive",
        )
        reason = physical_integrity_reason(s)
        assert reason is not None
        assert "could not be parsed" in reason

    def test_exact_mass_conflict_rejected(self):
        s = make_spectrum(
            "mass_junk",
            CAFFEINE_MH,
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            exact_mass=999.0,  # matchms normalizes this key to parent_mass
            adduct="[M+H]+",
            ionmode="positive",
        )
        reason = physical_integrity_reason(s)
        assert reason is not None
        assert "exact mass 999.0 conflicts" in reason

    def test_unsupported_adduct_rejected(self):
        s = make_spectrum(
            "adduct_junk",
            CAFFEINE_MH,
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            adduct="[M+Weird]+",
            ionmode="positive",
        )
        reason = physical_integrity_reason(s)
        assert reason is not None
        assert "not supported" in reason

    def test_malformed_formula_rejected(self):
        s = make_spectrum(
            "formula_junk",
            CAFFEINE_MH,
            [(100.0, 1.0)],
            formula="C8H10N4O2ZZ",
            adduct="[M+H]+",
            ionmode="positive",
        )
        reason = physical_integrity_reason(s)
        assert reason is not None
        assert "malformed structural metadata" in reason

    def test_no_structural_claim_never_reported(self):
        """Spectra without claims produce no reporter events (fast path)."""
        reasons: list[str] = []
        spectra = [
            make_spectrum("a", 100.0, [(100.0, 1.0)], charge=1),
            make_spectrum("b", 200.0, [(100.0, 1.0)]),
        ]
        out = processing.process_spectra_batch(
            spectra, ProcessingConfig(min_peaks=1, noise_threshold=0.0), reasons.append
        )
        assert len(out) == 2
        assert reasons == []

    def test_process_batch_drops_invalid_and_reports(self):
        """One invalid claim among valid spectra: the invalid spectrum is
        dropped, the rest survive, and the reporter receives the reason."""
        reasons: list[str] = []
        spectra = [
            make_spectrum("good_a", 100.0, [(100.0, 1.0)], charge=1),
            make_spectrum(
                "bad_ppm",
                _precursor_at_ppm(CAFFEINE_MH, 20.0),
                [(100.0, 1.0)],
                formula=CAFFEINE_FORMULA,
                adduct="[M+H]+",
                ionmode="positive",
            ),
            make_spectrum("good_b", 200.0, [(100.0, 1.0)], charge=1),
        ]
        out = processing.process_spectra_batch(
            spectra, ProcessingConfig(min_peaks=1, noise_threshold=0.0), reasons.append
        )
        assert [s.get("id") for s in out] == ["good_a", "good_b"]
        assert len(reasons) == 1
        assert "bad_ppm" in reasons[0]


# ---------------------------------------------------------------------------
# Classical annotate pipeline: query-file semantics
# ---------------------------------------------------------------------------


class TestQueryFileSemantics:
    def test_mixed_file_rejects_invalid_and_succeeds(self, tmp_path):
        """One >5 ppm query among valid ones: the invalid spectrum is
        counted in spectra_rejected and absent from the export; the file
        still succeeds."""
        good1 = make_spectrum("good_caff", CAFFEINE_MH, [(100.0, 1.0)], charge=1)
        good2 = make_spectrum("good_plain", 200.0, [(100.0, 1.0)], charge=1)
        bad = make_spectrum(
            "bad_caff",
            _precursor_at_ppm(CAFFEINE_MH, 25.0),
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            adduct="[M+H]+",
            ionmode="positive",
        )
        query_path = tmp_path / "queries.mgf"
        save_as_mgf([good1, good2, bad], str(query_path))
        library_path = tmp_path / "lib.msp"
        save_as_msp(
            [
                make_spectrum("ref_caff", CAFFEINE_MH, [(100.0, 1.0)], charge=1),
                make_spectrum("ref_plain", 200.0, [(100.0, 1.0)], charge=1),
            ],
            str(library_path),
        )

        results = run_annotation_pipeline(
            _pipeline_config(tmp_path, query_path, library_path)
        )
        assert len(results) == 1
        result = results[0]
        assert result.status in ("success", "degraded")
        assert result.spectra_loaded == 3
        assert result.spectra_rejected == 1
        csv_text = (tmp_path / "results" / "queries_results.csv").read_text()
        assert "bad_caff" not in csv_text
        assert "good_caff" in csv_text

    def test_file_with_only_physically_invalid_spectra_fails(self, tmp_path):
        """A query file whose spectra all fail the >5 ppm gate is an
        explicit failure with the gate reason, a failure report, and no
        results CSV."""
        bad = make_spectrum(
            "all_bad",
            _precursor_at_ppm(CAFFEINE_MH, 500.0),
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            adduct="[M+H]+",
            ionmode="positive",
        )
        query_path = tmp_path / "queries.mgf"
        save_as_mgf([bad], str(query_path))
        library_path = tmp_path / "lib.msp"
        save_as_msp(
            [make_spectrum("ref_caff", CAFFEINE_MH, [(100.0, 1.0)], charge=1)],
            str(library_path),
        )

        results = run_annotation_pipeline(
            _pipeline_config(tmp_path, query_path, library_path)
        )
        assert len(results) == 1
        result = results[0]
        assert result.status == "failed"
        joined = "; ".join(result.fatal_errors)
        assert "physical-integrity" in joined
        assert "all_bad" in joined
        assert "ppm" in joined
        assert not (tmp_path / "results" / "queries_results.csv").exists()
        assert (tmp_path / "results" / "queries_failed.report.yaml").exists()

    def test_file_with_malformed_structure_claims_fails(self, tmp_path):
        """Malformed structural metadata (an unparseable formula claim that
        survives matchms harmonization) fails the file explicitly when
        nothing else is analyzable."""
        bad = make_spectrum(
            "junk_formula",
            195.0877,
            [(100.0, 1.0)],
            formula="C8H10N4O2ZZ",  # unknown element 'Z': unparseable
            adduct="[M+H]+",
            ionmode="positive",
        )
        query_path = tmp_path / "q.mgf"
        save_as_mgf([bad], str(query_path))
        library_path = tmp_path / "lib.msp"
        save_as_msp(
            [make_spectrum("ref_caff", CAFFEINE_MH, [(100.0, 1.0)], charge=1)],
            str(library_path),
        )
        results = run_annotation_pipeline(
            _pipeline_config(tmp_path, query_path, library_path)
        )
        result = results[0]
        assert result.status == "failed"
        joined = "; ".join(result.fatal_errors)
        assert "malformed structural metadata" in joined
        assert "junk_formula" in joined

    def test_file_with_unparseable_smiles_fails_when_repairs_disabled(self, tmp_path):
        """With structural repairs disabled, an unparseable SMILES claim
        reaches the gate and fails the file explicitly (the default
        harmonization layer erases such claims before the gate)."""
        bad = make_spectrum(
            "junk_smiles",
            195.0877,
            [(100.0, 1.0)],
            smiles="NOT_A_SMILES",
            adduct="[M+H]+",
            ionmode="positive",
        )
        query_path = tmp_path / "q.mgf"
        save_as_mgf([bad], str(query_path))
        library_path = tmp_path / "lib.msp"
        save_as_msp(
            [make_spectrum("ref_caff", CAFFEINE_MH, [(100.0, 1.0)], charge=1)],
            str(library_path),
        )
        config = _pipeline_config(tmp_path, query_path, library_path)
        config.processing.repair_inchi_inchikey_smiles = False
        results = run_annotation_pipeline(config)
        result = results[0]
        assert result.status == "failed"
        assert "could not be parsed" in "; ".join(result.fatal_errors)

    def test_consistent_claimed_query_succeeds(self, tmp_path):
        """A query file that carries full, consistent chemistry claims is
        annotated normally (the gate is not a structure-claim ban)."""
        q = make_spectrum(
            "claimed",
            CAFFEINE_MH,
            [(100.0, 1.0), (CAFFEINE_MH, 1.0)],
            formula=CAFFEINE_FORMULA,
            smiles=CAFFEINE_SMILES,
            adduct="[M+H]+",
            ionmode="positive",
        )
        query_path = tmp_path / "q.mgf"
        save_as_mgf([q], str(query_path))
        library_path = tmp_path / "lib.msp"
        save_as_msp(
            [
                make_spectrum(
                    "ref_caff",
                    CAFFEINE_MH,
                    [(100.0, 1.0), (CAFFEINE_MH, 1.0)],
                    charge=1,
                )
            ],
            str(library_path),
        )
        results = run_annotation_pipeline(
            _pipeline_config(tmp_path, query_path, library_path)
        )
        assert results[0].status in ("success", "degraded")
        assert results[0].spectra_rejected == 0
        assert (tmp_path / "results" / "q_results.csv").exists()


# ---------------------------------------------------------------------------
# Classical annotate pipeline: raw reference-library semantics (strict)
# ---------------------------------------------------------------------------


class TestLibraryStrictness:
    def _query_file(self, tmp_path: Path) -> Path:
        q = make_spectrum("query_caff", CAFFEINE_MH, [(100.0, 1.0)], charge=1)
        query_path = tmp_path / "query.mgf"
        save_as_mgf([q], str(query_path))
        return query_path

    def test_library_over_5ppm_aborts_run(self, tmp_path):
        """A raw reference library containing a >5 ppm entry aborts the run
        before any file is processed (silently shrinking the target pool
        would corrupt every FDR calibration)."""
        library_path = tmp_path / "lib.msp"
        save_as_msp(
            [
                make_spectrum(
                    "ref_good",
                    CAFFEINE_MH,
                    [(100.0, 1.0)],
                    formula=CAFFEINE_FORMULA,
                    adduct="[M+H]+",
                    ionmode="positive",
                ),
                make_spectrum(
                    "ref_off",
                    _precursor_at_ppm(CAFFEINE_MH, 40.0),
                    [(100.0, 1.0)],
                    formula=CAFFEINE_FORMULA,
                    adduct="[M+H]+",
                    ionmode="positive",
                ),
            ],
            str(library_path),
        )
        config = _pipeline_config(
            tmp_path, self._query_file(tmp_path), library_path, output_name="out_strict"
        )
        with pytest.raises(processing.PhysicalIntegrityError) as excinfo:
            run_annotation_pipeline(config)
        message = str(excinfo.value)
        assert "ref_off" in message
        assert "5 ppm" in message or "ppm" in message
        # The partial store must not linger (a re-run would otherwise reuse
        # a silently shrunk pool).
        assert list((tmp_path / "out_strict").glob("*.db")) == []

    def test_library_unparseable_claim_aborts_run(self, tmp_path):
        """Malformed structural metadata in a library entry is a strict
        failure, not a silent drop. An unparseable formula claim survives
        harmonization and trips the gate."""
        library_path = tmp_path / "lib.msp"
        save_as_msp(
            [
                make_spectrum(
                    "junk_formula",
                    195.0877,
                    [(100.0, 1.0)],
                    formula="C8H10N4O2ZZ",
                    adduct="[M+H]+",
                    ionmode="positive",
                )
            ],
            str(library_path),
        )
        config = _pipeline_config(
            tmp_path, self._query_file(tmp_path), library_path, output_name="out2"
        )
        with pytest.raises(processing.PhysicalIntegrityError) as excinfo:
            run_annotation_pipeline(config)
        assert "malformed structural metadata" in str(excinfo.value)

    def test_library_unparseable_smiles_aborts_when_repairs_disabled(self, tmp_path):
        """With structural repairs disabled, unparseable SMILES claims reach
        the gate and abort the run (default harmonization erases them)."""
        library_path = tmp_path / "lib.msp"
        save_as_msp(
            [
                make_spectrum(
                    "junk_smiles",
                    195.0877,
                    [(100.0, 1.0)],
                    smiles="NOT_A_SMILES",
                    adduct="[M+H]+",
                    ionmode="positive",
                )
            ],
            str(library_path),
        )
        config = _pipeline_config(
            tmp_path, self._query_file(tmp_path), library_path, output_name="out3"
        )
        config.processing.repair_inchi_inchikey_smiles = False
        with pytest.raises(processing.PhysicalIntegrityError) as excinfo:
            run_annotation_pipeline(config)
        assert "could not be parsed" in str(excinfo.value)

    def test_library_with_consistent_claims_succeeds(self, tmp_path):
        """A library whose full-context claims are physically consistent is
        stored and annotated normally."""
        library_path = tmp_path / "lib.msp"
        save_as_msp(
            [
                make_spectrum(
                    "ref_good",
                    CAFFEINE_MH,
                    [(100.0, 1.0)],
                    formula=CAFFEINE_FORMULA,
                    smiles=CAFFEINE_SMILES,
                    adduct="[M+H]+",
                    ionmode="positive",
                )
            ],
            str(library_path),
        )
        results = run_annotation_pipeline(
            _pipeline_config(tmp_path, self._query_file(tmp_path), library_path)
        )
        assert results[0].status in ("success", "degraded")
        assert results[0].spectra_rejected == 0
        assert (tmp_path / "results" / "query_results.csv").exists()

    def test_library_5ppm_boundary_is_strict(self, tmp_path):
        """4.999 ppm library entries pass the gate; 5.001 ppm entries abort
        the run. The strict >5.0 boundary is enforced in the annotate path,
        not only in the model layer."""
        entry = make_spectrum(
            "ref_boundary",
            _precursor_at_ppm(CAFFEINE_MH, 4.999),
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            adduct="[M+H]+",
            ionmode="positive",
        )
        library_path = tmp_path / "lib_ok.msp"
        save_as_msp([entry], str(library_path))
        results = run_annotation_pipeline(
            _pipeline_config(tmp_path, self._query_file(tmp_path), library_path)
        )
        assert results[0].status in ("success", "degraded")
        assert results[0].spectra_rejected == 0

        entry_off = make_spectrum(
            "ref_boundary_off",
            _precursor_at_ppm(CAFFEINE_MH, 5.001),
            [(100.0, 1.0)],
            formula=CAFFEINE_FORMULA,
            adduct="[M+H]+",
            ionmode="positive",
        )
        library_bad = tmp_path / "lib_bad.msp"
        save_as_msp([entry_off], str(library_bad))
        with pytest.raises(processing.PhysicalIntegrityError):
            run_annotation_pipeline(
                _pipeline_config(
                    tmp_path, self._query_file(tmp_path), library_bad, output_name="o2"
                )
            )


class TestCliLibraryStrictness:
    """The CLI surfaces a strict library failure as exit code 1 with the
    gate reasons visible to the user."""

    def _cli_config(self, tmp_path: Path, library_path: Path) -> Path:
        query_path = tmp_path / "query.mgf"
        save_as_mgf(
            [make_spectrum("query_caff", CAFFEINE_MH, [(100.0, 1.0)], charge=1)],
            str(query_path),
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""
project:
  name: "physics_gate_cli"
  output_directory: "{tmp_path / "results"}"

input:
  input_path: "{query_path}"
  library_path: "{library_path}"
  format: "mgf"

processing:
  min_peaks: 1
  noise_threshold: 0.0

similarity:
  algorithm: "cosine"
  min_score: 0.0
  fdr_threshold: 1.0
"""
        )
        return config_path

    def test_library_gate_failure_exits_nonzero_with_message(self, tmp_path):
        from typer.testing import CliRunner

        from MassFlow import cli

        library_path = tmp_path / "lib_bad.msp"
        save_as_msp(
            [
                make_spectrum(
                    "ref_off",
                    _precursor_at_ppm(CAFFEINE_MH, 40.0),
                    [(100.0, 1.0)],
                    formula=CAFFEINE_FORMULA,
                    adduct="[M+H]+",
                    ionmode="positive",
                )
            ],
            str(library_path),
        )
        config_path = self._cli_config(tmp_path, library_path)

        runner = CliRunner()
        result = runner.invoke(cli.app, ["annotate", "--config", str(config_path)])

        assert result.exit_code == 1
        assert "Annotation failed" in result.output
        assert "5 ppm" in result.output or "ppm" in result.output
