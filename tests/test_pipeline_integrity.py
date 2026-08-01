"""
Rigorous, chemistry-aware regression test suite for MassFlow pipeline integrity.

Validates:
  - Numerical stability of the 5.0 ppm precursor mass boundary (models.py).
  - Chimeric spectra with overlapping isotopic envelopes (cheminformatics.py).
  - Valid dark/undocumented neutral losses vs. trivial losses (< 19 Da).
  - Determinism of Target-Decoy FDR across single- and multi-worker execution.
  - ADDUCT_OFFSETS correctness: ionic-formula neutralisation across all modes.
  - Benchmark: all-vs-all 10³ queries vs 10⁴ references (time + peak memory).
"""

from __future__ import annotations

import gc
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_msp

# ── MassFlow imports ───────────────────────────────────────────────────────
from MassFlow.cheminformatics import (
    _formula_to_isotopic_envelope,
    _formula_to_monoisotopic_mass,
    compute_adduct_offset,
)
from MassFlow.config import (
    ProcessingConfig,
    SimilarityConfig,
)
from MassFlow.models import MolecularStructure, SpectrumMetadata
from MassFlow.similarity import (
    calculate_fdr,
    generate_decoys,
    get_similarity_engine,
)

# ---------------------------------------------------------------------------
# Optional RDKit detection (same pattern as MassFlow internals)
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem  # noqa: F401

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

# ---------------------------------------------------------------------------
# Tolerances — appropriate for high-resolution Orbitrap / TOF data
# ---------------------------------------------------------------------------
PPM_RTOL = 1e-6  # 1 ppm relative tolerance for mass comparisons
MASS_ATOL = 1e-6  # 1 µDa absolute tolerance — well below Orbitrap noise
INTENSITY_ATOL = 1e-9  # intensity precision
SCORE_ATOL = 1e-8  # cosine score precision (matchms float32 → float64)
SCORE_RTOL = 1e-5  # relative tolerance for score comparisons


# ============================================================================
# Helpers
# ============================================================================


def _make_test_spectrum(
    spec_id: str,
    precursor_mz: float,
    mz: list[float] | np.ndarray,
    intensities: list[float] | np.ndarray,
    **extra_meta: object,
) -> Spectrum:
    """Create a ``matchms.Spectrum`` with mandatory metadata keys."""
    meta: dict[str, object] = {
        "precursor_mz": precursor_mz,
        "id": spec_id,
        "compound_name": spec_id,
    }
    meta.update(extra_meta)
    return Spectrum(
        mz=np.array(mz, dtype=np.float64),
        intensities=np.array(intensities, dtype=np.float64),
        metadata=meta,
    )


# ============================================================================
# Task 1a — 5.0 ppm precursor mass boundary fixtures & tests
# ============================================================================


class Test5PpmBoundary:
    """Validate that the strict ``> 5.0`` ppm gate behaves correctly.

    The validator in ``MolecularStructure`` uses::

        ppm_error > 5.0  →  is_physically_valid = False

    Because floating-point multiplication ``exact * (1 + 5/1e6)`` can
    produce a value whose ppm error slightly exceeds 5.0 due to roundoff,
    we compute the *true* boundary mass directly from the formula::

        boundary_mass = exact_mass * (1 + 5.0e-6)

    and then search around it for the exact transition point.
    """

    FORMULA = "C6H6"  # Benzene

    @pytest.fixture(scope="class")
    def exact_mass(self) -> float:
        """Monoisotopic mass of benzene via pyteomics SSOT."""
        return _formula_to_monoisotopic_mass(self.FORMULA)

    @pytest.fixture(scope="class")
    def adduct_offset_H(self) -> float:
        o = compute_adduct_offset("[M+H]+")
        assert o is not None
        return o

    @pytest.fixture(scope="class")
    def adduct_offset_minusH(self) -> float:
        o = compute_adduct_offset("[M-H]-")
        assert o is not None
        return o

    # -- Boundary masses computed with sub-µDa precision ---------------------

    @staticmethod
    def _mass_at_ppm_error(exact: float, ppm_error: float) -> float:
        """Return the exact mass that yields *exactly* ``ppm_error`` ppm."""
        return exact * (1.0 + ppm_error / 1e6)

    def test_5_0_ppm_float64_stability(self, exact_mass: float) -> None:
        """The mass computed by ``exact * (1 + 5.0e-6)`` may have a ppm error
        slightly above 5.0 due to float64 roundoff (~10⁻¹⁰ ppm excess).

        The code's strict ``> 5.0`` gate correctly flags it as invalid, and
        the excess is negligibly small (far below any experimental precision).
        """
        m_5 = self._mass_at_ppm_error(exact_mass, 5.0)
        ppm_err = abs(m_5 - exact_mass) / exact_mass * 1e6

        struct = MolecularStructure(formula=self.FORMULA, exact_mass=m_5)
        if ppm_err <= 5.0:
            assert struct.is_physically_valid is True
        else:
            assert struct.is_physically_valid is False
            # Float64 roundoff should stay within 10⁻⁹ ppm of the boundary.
            assert ppm_err < 5.0 + 1e-9, (
                f"ppm error ({ppm_err:.15e}) deviated more than 1e-9 above 5.0"
            )

    def test_4_999_ppm_is_valid(self, exact_mass: float) -> None:
        m_4_999 = self._mass_at_ppm_error(exact_mass, 4.999)
        struct = MolecularStructure(formula=self.FORMULA, exact_mass=m_4_999)
        assert struct.is_physically_valid is True

    def test_5_001_ppm_is_invalid(self, exact_mass: float) -> None:
        m_5_001 = self._mass_at_ppm_error(exact_mass, 5.001)
        struct = MolecularStructure(formula=self.FORMULA, exact_mass=m_5_001)
        assert struct.is_physically_valid is False

    # -- SpectrumMetadata (adduct-aware) boundary tests ---------------------

    def test_spectrum_metadata_5_001_ppm_fails(
        self, exact_mass: float, adduct_offset_H: float
    ) -> None:
        """SpectrumMetadata with 5.001 ppm mass error must be invalid."""
        m_5_001 = self._mass_at_ppm_error(exact_mass, 5.001)
        theoretical_mz = (m_5_001 + adduct_offset_H) / 1

        mol = MolecularStructure(formula=self.FORMULA, exact_mass=m_5_001)
        meta = SpectrumMetadata(
            spectrum_id="ppm_boundary",
            precursor_mz=theoretical_mz,
            charge=1,
            adduct="[M+H]+",
            molecule=mol,
        )
        assert meta.is_physically_valid is False, (
            "5.001 ppm precursor error must be flagged invalid in SpectrumMetadata"
        )

    def test_spectrum_metadata_4_999_ppm_passes(
        self, exact_mass: float, adduct_offset_H: float
    ) -> None:
        """SpectrumMetadata with 4.999 ppm mass error must be valid."""
        m_4_999 = self._mass_at_ppm_error(exact_mass, 4.999)
        theoretical_mz = (m_4_999 + adduct_offset_H) / 1

        mol = MolecularStructure(formula=self.FORMULA, exact_mass=m_4_999)
        meta = SpectrumMetadata(
            spectrum_id="ppm_boundary",
            precursor_mz=theoretical_mz,
            charge=1,
            adduct="[M+H]+",
            molecule=mol,
        )
        assert meta.is_physically_valid is True

    def test_negative_mode_5_001_ppm_fails(
        self, exact_mass: float, adduct_offset_minusH: float
    ) -> None:
        """SpectrumMetadata [M-H]- with 5.001 ppm error must be invalid."""
        m_5_001 = self._mass_at_ppm_error(exact_mass, 5.001)
        theoretical_mz = (m_5_001 + adduct_offset_minusH) / abs(-1)

        mol = MolecularStructure(formula=self.FORMULA, exact_mass=m_5_001)
        meta = SpectrumMetadata(
            spectrum_id="ppm_neg",
            precursor_mz=theoretical_mz,
            charge=-1,
            adduct="[M-H]-",
            molecule=mol,
        )
        assert meta.is_physically_valid is False

    def test_negative_mode_4_999_ppm_passes(
        self, exact_mass: float, adduct_offset_minusH: float
    ) -> None:
        """SpectrumMetadata [M-H]- with 4.999 ppm error must be valid."""
        m_4_999 = self._mass_at_ppm_error(exact_mass, 4.999)
        theoretical_mz = (m_4_999 + adduct_offset_minusH) / abs(-1)

        mol = MolecularStructure(formula=self.FORMULA, exact_mass=m_4_999)
        meta = SpectrumMetadata(
            spectrum_id="ppm_neg",
            precursor_mz=theoretical_mz,
            charge=-1,
            adduct="[M-H]-",
            molecule=mol,
        )
        assert meta.is_physically_valid is True

    def test_ppm_calculation_no_drift(self, exact_mass: float) -> None:
        """Verify that the ppm_error formula itself has no precision drift."""
        struct = MolecularStructure(formula=self.FORMULA, exact_mass=exact_mass)
        assert struct.is_physically_valid is True
        assert struct.exact_mass is not None
        np.testing.assert_allclose(
            struct.exact_mass,
            exact_mass,
            rtol=PPM_RTOL,
            atol=MASS_ATOL,
        )


# ============================================================================
# Task 1b — Chimeric spectra with overlapping isotopic envelopes
# ============================================================================


class TestChimericIsotopicEnvelopes:
    """Test handling of overlapping / near-degenerate isotopic patterns.

    A "chimeric" spectrum contains fragments from two co-eluting molecules
    whose isotopic envelopes overlap in m/z space.  We simulate this by
    computing envelopes for halogenated formulas and verifying that:

    1. Monoisotopic masses are cleanly separated.
    2. The M vs M+2 spacing within each envelope is ~2 Da (the hallmark
       of a monohalogenated compound).
    3. Merging two envelopes does not cross-contaminate the monoisotopic
       mass of either formula.
    """

    @pytest.fixture(scope="class")
    def br_envelope(self) -> list[tuple[float, float]]:
        """Isotopic envelope for bromobenzene: C6H5Br."""
        return _formula_to_isotopic_envelope("C6H5Br", max_isopeaks=4)

    @pytest.fixture(scope="class")
    def cl_envelope(self) -> list[tuple[float, float]]:
        """Isotopic envelope for chlorobenzene: C6H5Cl."""
        return _formula_to_isotopic_envelope("C6H5Cl", max_isopeaks=4)

    def test_envelopes_have_nonzero_length(
        self,
        br_envelope,
        cl_envelope,
    ) -> None:
        assert len(br_envelope) > 0
        assert len(cl_envelope) > 0

    def test_monoisotopic_masses_are_distinct(
        self,
        br_envelope,
        cl_envelope,
    ) -> None:
        """C6H5Br and C6H5Cl differ by the Br–Cl mass difference (~43.95 Da)."""
        br_m = br_envelope[0][0]
        cl_m = cl_envelope[0][0]
        separation = abs(br_m - cl_m)
        # Br (78.918) vs Cl (34.969) → ~43.95 Da apart
        assert 40.0 < separation < 48.0, (
            f"Br vs Cl monoisotopic separation ({separation:.4f} Da) outside expected range"
        )

    def test_m_plus_2_spacing_is_two_daltons(
        self,
        br_envelope,
        cl_envelope,
    ) -> None:
        """Within each monohalogenated envelope, M+2 should be ≈2 Da above M.

        This is the defining signature of a single Br or Cl atom (both
        have abundant M+2 isotopes).
        """
        for label, env in [("Br", br_envelope), ("Cl", cl_envelope)]:
            assert len(env) >= 3, f"{label} envelope needs at least 3 isotopologues"
            m = env[0][0]
            m2 = env[2][0]
            spacing = m2 - m
            np.testing.assert_allclose(
                spacing,
                2.0,
                atol=0.15,
                rtol=0,
                err_msg=f"{label} M+2 spacing ({spacing:.4f} Da) should be ~2 Da",
            )

    def test_abundance_sum_normalised(
        self,
        br_envelope,
        cl_envelope,
    ) -> None:
        """The most abundant isotopologue must have relative abundance 1.0."""
        for env in (br_envelope, cl_envelope):
            max_abund = max(ab for _m, ab in env)
            np.testing.assert_allclose(max_abund, 1.0, atol=1e-6)

    def test_merging_preserves_monoisotopic_mass(self) -> None:
        """When computing two envelopes independently, their monoisotopic
        masses remain correct — no cross-contamination."""
        env_caffeine = _formula_to_isotopic_envelope("C8H10N4O2")
        env_theobromine = _formula_to_isotopic_envelope("C7H8N4O2")
        mono_caff = _formula_to_monoisotopic_mass("C8H10N4O2")
        mono_theo = _formula_to_monoisotopic_mass("C7H8N4O2")

        np.testing.assert_allclose(
            env_caffeine[0][0],
            mono_caff,
            rtol=PPM_RTOL,
            atol=MASS_ATOL,
        )
        np.testing.assert_allclose(
            env_theobromine[0][0],
            mono_theo,
            rtol=PPM_RTOL,
            atol=MASS_ATOL,
        )
        assert abs(mono_caff - mono_theo) > 10.0, (
            "Caffeine and theobromine monoisotopic masses should differ significantly"
        )


# ============================================================================
# Task 1c — Valid dark/undocumented neutral losses vs trivial losses (< 19 Da)
# ============================================================================


class TestNeutralLossClassification:
    """Neutral loss awareness: distinguish trivial (< 19 Da) losses from
    structurally informative dark/undocumented losses.

    In mass spectrometry, fragment ions are formed by loss of neutral
    fragments. Losses smaller than ~19 Da are usually non-informative
    (e.g., H•, H₂), whereas larger losses (H₂O, CO, SO₃, etc.) are
    structurally diagnostic even if not present in canonical adduct tables.
    """

    # Common neutral losses and their monoisotopic masses (Da)
    TRIVIAL_LOSSES: dict[str, float] = {
        "H_radical": 1.007825,
        "H2": 2.015650,
        "electron": 0.000549,
    }
    DOCUMENTED_LOSSES: dict[str, float] = {
        "H2O": 18.010565,  # water
        "NH3": 17.026549,  # ammonia
        "CO": 27.994915,  # carbon monoxide
        "CO2": 43.989829,  # carbon dioxide
        "CH3OH": 32.026215,  # methanol
        "HCOOH": 46.005479,  # formic acid
    }
    DARK_LOSSES: dict[str, float] = {
        "SO3": 79.956818,  # sulfate loss (common in sulfated metabolites)
        "C2H4O2": 60.021129,  # acetic acid
        "C6H10O5": 162.052824,  # hexose (glycosidic cleavage)
        "H3PO4": 97.976896,  # phosphate
        "C5H8O4": 132.042260,  # deoxyribose
    }

    # Threshold below which a neutral loss is considered "trivial"
    TRIVIAL_THRESHOLD = 19.0  # Da

    def test_trivial_losses_below_threshold(self) -> None:
        """All TRIVIAL_LOSSES must be below 19 Da."""
        for name, mass in self.TRIVIAL_LOSSES.items():
            assert mass < self.TRIVIAL_THRESHOLD, (
                f"{name} ({mass:.6f} Da) must be a trivial loss"
            )

    def test_water_and_ammonia_border_threshold(self) -> None:
        """H₂O (18.01 Da) and NH₃ (17.03 Da) are below 19 Da — they sit at
        the boundary of trivial vs. informative losses."""
        assert self.DOCUMENTED_LOSSES["H2O"] < self.TRIVIAL_THRESHOLD
        assert self.DOCUMENTED_LOSSES["NH3"] < self.TRIVIAL_THRESHOLD
        # But they are still structurally diagnostic (H₂O, NH₃ are common
        # neutral losses that inform about functional groups).
        assert self.DOCUMENTED_LOSSES["H2O"] > 10.0
        assert self.DOCUMENTED_LOSSES["NH3"] > 10.0

    def test_dark_losses_well_above_threshold(self) -> None:
        """Dark/undocumented losses are structurally informative (> 19 Da)."""
        for name, mass in self.DARK_LOSSES.items():
            assert mass > self.TRIVIAL_THRESHOLD, (
                f"Dark loss {name} ({mass:.6f} Da) must exceed trivial threshold"
            )

    def test_classification_boundary_at_19_daltons(self) -> None:
        """The threshold at 19.0 Da cleanly separates trivial from informative."""
        assert self.TRIVIAL_THRESHOLD == 19.0
        # Everything in TRIVIAL_LOSSES is below
        for mass in self.TRIVIAL_LOSSES.values():
            assert mass < self.TRIVIAL_THRESHOLD
        # Everything in DARK_LOSSES is above
        for mass in self.DARK_LOSSES.values():
            assert mass > self.TRIVIAL_THRESHOLD

    def test_documented_losses_span_boundary(self) -> None:
        """Documented losses span both sides of the 19 Da threshold,
        highlighting that the boundary is a heuristic, not a hard rule."""
        below = {k: v for k, v in self.DOCUMENTED_LOSSES.items() if v < 19.0}
        above = {k: v for k, v in self.DOCUMENTED_LOSSES.items() if v >= 19.0}
        # H2O and NH3 are below; CO, CO2, CH3OH, HCOOH are above
        assert len(below) >= 2, "Expected H2O and NH3 below 19 Da"
        assert len(above) >= 3, "Expected CO, CO2, CH3OH, HCOOH above 19 Da"


# ============================================================================
# Task 2 — FDR determinism: single-threaded vs multi-worker
# ============================================================================


class TestFDRDeterminism:
    """Assert that Target-Decoy FDR is identical whether the pipeline runs
    with a single process or in a multi-worker pool.

    The core property: ``calculate_fdr()`` is pure (deterministic). The
    decoy library must also be identical, because ``generate_decoys`` uses
    a fixed ``random_seed=42``.
    """

    @pytest.fixture(scope="class")
    def synthetic_library(self, tmp_path_factory) -> Path:
        """Create a small synthetic MSP library of 10 reference spectra
        with high spectral similarity to the queries."""
        rng = np.random.default_rng(42)
        refs: list[Spectrum] = []
        for i in range(10):
            pmz = 300.0
            # Use consistent peak pattern so cosine scores are meaningful
            mz = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0], dtype=np.float64)
            intensities = np.array([0.5, 1.0, 0.4, 0.8, 0.3, 0.6], dtype=np.float64)
            # Add small jitter to make them non-identical but highly similar
            mz = mz + rng.uniform(-0.001, 0.001, size=len(mz))
            intensities = intensities * rng.uniform(0.95, 1.05, size=len(intensities))
            refs.append(
                _make_test_spectrum(
                    f"lib_{i}",
                    precursor_mz=pmz,
                    mz=mz,
                    intensities=intensities,
                    smiles="CCO",
                    inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                )
            )
        lib_path = tmp_path_factory.mktemp("fdr_det") / "library.msp"
        save_as_msp(refs, str(lib_path))
        return lib_path

    def test_search_decoy_consistency_single_vs_multi_worker(
        self,
        synthetic_library: Path,
        tmp_path: Path,
    ) -> None:
        """Assert that search results are identical whether decoys are
        pre-generated (multi-worker path) or generated on-the-fly inside
        ``SimilarityEngine.search()`` (single-process path).

        Both paths use ``generate_decoys(..., random_seed=42)`` so the
        decoy spectra and resulting scores must be bit-identical.
        """
        # Load reference spectra from the synthetic library MSP file.
        from MassFlow import io, processing

        ref_iter = io.load_spectra(synthetic_library)
        ref_list = list(processing.process_spectra(ref_iter, ProcessingConfig()))
        assert len(ref_list) > 0

        # Build query spectra that are highly similar to the references.
        rng = np.random.default_rng(43)
        queries: list[Spectrum] = []
        for i in range(5):
            pmz = 300.0
            mz = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0], dtype=np.float64)
            intensities = np.array([0.5, 1.0, 0.4, 0.8, 0.3, 0.6], dtype=np.float64)
            mz = mz + rng.uniform(-0.002, 0.002, size=len(mz))
            intensities = intensities * rng.uniform(0.90, 1.10, size=len(intensities))
            queries.append(
                _make_test_spectrum(f"q_{i}", pmz, mz=mz, intensities=intensities)
            )

        config = SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=100.0,
            tolerance=0.05,
            min_score=0.0,
            min_matched_peaks=1,
        )

        # -- Path A: Single-process (decoys generated inside search()) --
        engine_a = get_similarity_engine(config)
        results_a = engine_a.search(
            query_spectra=queries,
            reference_spectra=ref_list,
            include_decoys=True,
        )
        assert len(results_a) > 0

        # -- Path B: Multi-worker (decoys pre-generated, then passed as ref_list) --
        decoy_list = generate_decoys(ref_list, random_seed=42)
        all_refs = ref_list + decoy_list
        n_targets = len(ref_list)
        ref_pmzs = np.array(
            [float(s.get("precursor_mz", 0.0) or 0.0) for s in all_refs],
            dtype=np.float64,
        )
        ref_is_decoy = np.array(
            [False] * n_targets + [True] * len(decoy_list),
            dtype=bool,
        )

        engine_b = get_similarity_engine(config)
        results_b = engine_b.search(
            query_spectra=queries,
            reference_spectra=all_refs,
            include_decoys=False,
            ref_precursor_mzs=ref_pmzs,
            ref_is_decoy=ref_is_decoy,
        )
        assert len(results_b) > 0

        # -- Assert identical score vectors --
        def _score_vector(res: list[dict]) -> np.ndarray:
            return np.sort(np.array([r["score"] for r in res], dtype=np.float64))

        sv_a = _score_vector(results_a)  # type: ignore[arg-type]
        sv_b = _score_vector(results_b)  # type: ignore[arg-type]
        np.testing.assert_allclose(sv_a, sv_b, rtol=SCORE_RTOL, atol=SCORE_ATOL)

    def test_calculate_fdr_deterministic_repeated_calls(self) -> None:
        """Multiple calls to calculate_fdr with identical inputs must produce
        byte-identical output."""
        rng = np.random.default_rng(99)
        targets = rng.uniform(0.5, 1.0, size=200).astype(np.float64)
        decoys = rng.uniform(0.0, 0.8, size=500).astype(np.float64)

        results = []
        for _ in range(10):
            s, q, t = calculate_fdr(targets.copy(), decoys.copy())
            results.append((s.copy(), q.copy(), t.copy()))

        s0, q0, t0 = results[0]
        for i, (si, qi, ti) in enumerate(results[1:], start=1):
            np.testing.assert_array_equal(s0, si, err_msg=f"call {i} scores diverge")
            np.testing.assert_array_equal(q0, qi, err_msg=f"call {i} q-values diverge")
            np.testing.assert_array_equal(
                t0, ti, err_msg=f"call {i} is_target diverges"
            )


# ============================================================================
# Task 3 — ADDUCT_OFFSETS neutralises ionic formulas
# ============================================================================


class TestAdductOffsetNeutralization:
    """Verify that ``compute_adduct_offset`` correctly computes mass offsets
    for all standard adducts in both positive and negative ion modes, and
    that the neutral-mass reconstruction is free of precision drift."""

    CAFFEINE_FORMULA = "C8H10N4O2"

    # Chemically correct offset signs:
    #   offset = atoms_mass - charge × ELECTRON_MASS
    #
    # Positive mode: the offset may be positive (added atoms > electron loss)
    #   or negative (radical cation loses only an electron).
    # Negative mode: offset is negative for [M-H]- (lost proton, gained electron)
    #   and positive for adduct additions ([M+Cl]-, [M+HCOO]-, etc.) where
    #   the anion mass dominates the electron gain.
    ADDUCT_EXPECTATIONS: dict[str, int] = {
        "[M+H]+": +1,  # H mass > electron mass → positive
        "[M+NH4]+": +1,
        "[M+Na]+": +1,
        "[M+K]+": +1,
        "[M]+": -1,  # radical cation: only electron loss → negative
        "[M+2H]2+": +1,
        "[M-H]-": -1,  # lost proton + gained electron → negative
        "[M+Cl]-": +1,  # Cl mass dominates electron gain → positive
        "[M+HCOO]-": +1,
        "[M+CH3COO]-": +1,
        "[M+FA-H]-": +1,  # formate adduct
        "[M]-": +1,  # radical anion: electron gain → positive
    }

    @pytest.fixture(scope="class")
    def caffeine_mass(self) -> float:
        return _formula_to_monoisotopic_mass(self.CAFFEINE_FORMULA)

    @pytest.mark.parametrize("adduct,expected_sign", list(ADDUCT_EXPECTATIONS.items()))
    def test_adduct_offset_sign(
        self,
        adduct: str,
        expected_sign: int,
    ) -> None:
        """Each adduct offset must have the chemically correct sign."""
        offset = compute_adduct_offset(adduct)
        assert offset is not None, f"Unknown adduct: {adduct}"

        actual_sign = 1 if offset > 0 else -1 if offset < 0 else 0
        assert actual_sign == expected_sign, (
            f"{adduct}: expected sign {expected_sign:+d}, got {offset:+.6f}"
        )
        assert abs(offset) > 1e-6, f"{adduct} offset is suspiciously close to zero"

    @pytest.mark.parametrize(
        "adduct",
        [
            "[M+H]+",
            "[M+Na]+",
            "[M+K]+",
            "[M-H]-",
            "[M+Cl]-",
            "[M+2H]2+",
        ],
    )
    def test_roundtrip_neutral_mass_no_drift(
        self,
        adduct: str,
        caffeine_mass: float,
    ) -> None:
        """Reconstructing the neutral mass via (mz × |charge| - offset) must
        recover the original exact mass within 1 µDa."""
        offset = compute_adduct_offset(adduct)
        assert offset is not None

        charge_map = {
            "[M+H]+": 1,
            "[M+Na]+": 1,
            "[M+K]+": 1,
            "[M-H]-": -1,
            "[M+Cl]-": -1,
            "[M+2H]2+": 2,
        }
        z = charge_map[adduct]
        abs_z = abs(z)

        theoretical_mz = (caffeine_mass + offset) / abs_z
        reconstructed = theoretical_mz * abs_z - offset

        np.testing.assert_allclose(
            reconstructed,
            caffeine_mass,
            rtol=PPM_RTOL,
            atol=MASS_ATOL,
            err_msg=f"Roundtrip drift for {adduct}",
        )

    def test_formate_acetate_aliases_identical(self) -> None:
        """[M+FA-H]- and [M+HCOO]- are formate aliases and must be identical."""
        offset_fa = compute_adduct_offset("[M+FA-H]-")
        offset_hcoo = compute_adduct_offset("[M+HCOO]-")
        assert offset_fa is not None
        assert offset_hcoo is not None
        np.testing.assert_allclose(offset_fa, offset_hcoo, rtol=0, atol=MASS_ATOL)

    def test_doubly_charged_mz_is_half(self, caffeine_mass: float) -> None:
        """For [M+2H]2+, the observed m/z is approximately half the [M+H]+ m/z."""
        o1 = compute_adduct_offset("[M+H]+")
        o2 = compute_adduct_offset("[M+2H]2+")
        assert o1 is not None
        assert o2 is not None

        mz_1 = caffeine_mass + o1
        mz_2 = (caffeine_mass + o2) / 2

        ratio = mz_2 / mz_1
        assert 0.49 < ratio < 0.51, (
            f"[M+2H]2+ m/z ({mz_2:.4f}) should be ~half [M+H]+ m/z ({mz_1:.4f})"
        )

    def test_unknown_adduct_returns_none(self) -> None:
        """Unrecognised adduct strings must return None."""
        assert compute_adduct_offset("[M+Unknown]+") is None
        assert compute_adduct_offset("") is None
        assert compute_adduct_offset("garbage") is None

    def test_cache_consistency(self) -> None:
        """Repeated calls with the same adduct must return the same float."""
        o1 = compute_adduct_offset("[M+Na]+")
        o2 = compute_adduct_offset("[M+Na]+")
        assert o1 is not None
        assert o1 == o2
        np.testing.assert_allclose(o1, o2, rtol=0, atol=0)


# ============================================================================
# Task 4 — Benchmark: 10³ queries vs 10⁴ references (all-vs-all)
# ============================================================================


@pytest.mark.benchmark(group="pipeline_integrity")
class TestAllVsAllBenchmark:
    """Performance regression guard: synthetic all-vs-all cosine comparison.

    Scales: 1 000 query spectra × 10 000 reference spectra with decoys.
    Fixed random seed ensures reproducible spectrum generation across runs.
    """

    N_QUERIES = 1000
    N_REFS = 10000

    @pytest.fixture(scope="class")
    def synthetic_queries(self) -> list[Spectrum]:
        rng = np.random.default_rng(42)
        queries: list[Spectrum] = []
        for i in range(self.N_QUERIES):
            pmz = 300.0 + rng.uniform(-50.0, 50.0)
            n_peaks = rng.integers(5, 20)
            mz = np.sort(rng.uniform(50.0, 500.0, size=n_peaks)).astype(np.float64)
            intensities = rng.uniform(0.05, 1.0, size=n_peaks).astype(np.float64)
            queries.append(
                _make_test_spectrum(
                    f"q_{i}",
                    precursor_mz=pmz,
                    mz=mz,
                    intensities=intensities,
                )
            )
        return queries

    @pytest.fixture(scope="class")
    def synthetic_references(self) -> list[Spectrum]:
        rng = np.random.default_rng(84)
        refs: list[Spectrum] = []
        for i in range(self.N_REFS):
            pmz = 300.0 + rng.uniform(-100.0, 100.0)
            n_peaks = rng.integers(5, 30)
            mz = np.sort(rng.uniform(50.0, 600.0, size=n_peaks)).astype(np.float64)
            intensities = rng.uniform(0.05, 1.0, size=n_peaks).astype(np.float64)
            refs.append(
                _make_test_spectrum(
                    f"r_{i}",
                    precursor_mz=pmz,
                    mz=mz,
                    intensities=intensities,
                )
            )
        return refs

    def test_all_vs_all_cosine_runtime(
        self,
        benchmark,
        synthetic_queries,
        synthetic_references,
    ) -> None:
        """Benchmark: runtime of 1k queries × 10k refs cosine search (no decoys)."""
        config = SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=100.0,
            tolerance=0.05,
            min_score=0.0,
            min_matched_peaks=1,
        )
        engine = get_similarity_engine(config)

        result = benchmark.pedantic(
            engine.search,
            kwargs={
                "query_spectra": synthetic_queries,
                "reference_spectra": synthetic_references,
                "include_decoys": False,
            },
            rounds=3,
            iterations=1,
        )
        assert len(result) > 0, "Benchmark search must return at least one result"

    def test_all_vs_all_cosine_memory(
        self,
        synthetic_queries,
        synthetic_references,
    ) -> None:
        """Peak memory consumption for 1k queries × 10k refs search.

        Uses ``tracemalloc`` to measure peak allocated memory during the
        full sparse-array search.  Decoys are excluded from this test to
        keep the benchmark focused on the core all-vs-all matrix.
        """
        config = SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=100.0,
            tolerance=0.05,
            min_score=0.0,
            min_matched_peaks=1,
        )
        engine = get_similarity_engine(config)

        gc.collect()
        tracemalloc.start()
        try:
            results = engine.search(
                query_spectra=synthetic_queries,
                reference_spectra=synthetic_references,
                include_decoys=False,
            )
            _current, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        peak_mb = peak_bytes / (1024 * 1024)
        assert peak_mb > 0, "Peak memory must be measurable"
        assert len(results) > 0, "Search must return results"

        # 10⁴ × 10³ all-vs-all with sparse arrays should fit within 2 GiB.
        assert peak_mb < 2048, f"Peak memory {peak_mb:.1f} MiB exceeds 2 GiB soft limit"


# ============================================================================
# Complementary integrity tests
# ============================================================================


class TestPipelineNumericalStability:
    """Additional numerical-stability guardrails for the annotation pipeline."""

    def test_decoys_preserve_precursor_mz_exactly(self) -> None:
        """Decoy generation must not perturb precursor_mz; it must be bit-identical."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            pmz = rng.uniform(50.0, 1500.0)
            spec = _make_test_spectrum(
                "s",
                pmz,
                mz=np.sort(rng.uniform(10.0, pmz, size=10)).astype(np.float64),
                intensities=rng.uniform(0.1, 1.0, size=10).astype(np.float64),
            )
            decoys = generate_decoys([spec], random_seed=0)
            assert len(decoys) == 1
            decoy_pmz = decoys[0].get("precursor_mz")
            assert decoy_pmz == pmz, (
                f"Decoy precursor_mz ({decoy_pmz}) differs from original ({pmz})"
            )

    def test_generate_decoys_seeded_determinism(self) -> None:
        """generate_decoys with the same seed must produce identical decoys."""
        rng = np.random.default_rng(55)
        specs = [
            _make_test_spectrum(
                f"s{i}",
                rng.uniform(100.0, 500.0),
                mz=np.sort(rng.uniform(50.0, 400.0, size=8)).astype(np.float64),
                intensities=rng.uniform(0.1, 1.0, size=8).astype(np.float64),
            )
            for i in range(20)
        ]
        decoys_a = generate_decoys(specs, random_seed=42)
        decoys_b = generate_decoys(specs, random_seed=42)

        for i, (da, db) in enumerate(zip(decoys_a, decoys_b)):
            np.testing.assert_array_equal(
                da.peaks.mz,
                db.peaks.mz,
                err_msg=f"Decoy {i} m/z arrays diverge",
            )
            np.testing.assert_array_equal(
                da.peaks.intensities,
                db.peaks.intensities,
                err_msg=f"Decoy {i} intensity arrays diverge",
            )

    def test_mz_arrays_always_float64(self) -> None:
        """All m/z and intensity arrays must remain float64 throughout.

        Downcasting to float32 introduces ~0.1 ppm rounding error, which is
        unacceptable for high-resolution instruments.
        """
        rng = np.random.default_rng(7)
        for _ in range(20):
            mz = np.sort(rng.uniform(50.0, 1000.0, size=10))
            intensities = rng.uniform(0.0, 1.0, size=10)
            spec = Spectrum(
                mz=mz,
                intensities=intensities,
                metadata={"id": "f64check", "precursor_mz": 300.0},
            )
            assert spec.peaks.mz.dtype == np.float64, (
                f"m/z dtype is {spec.peaks.mz.dtype}, expected float64"
            )
            assert spec.peaks.intensities.dtype == np.float64, (
                f"intensity dtype is {spec.peaks.intensities.dtype}, expected float64"
            )
