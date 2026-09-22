"""
Adduct-registry notation handling and the physical-integrity gate.

Real ``.msp``/``.mgf`` libraries spell the same ion many ways -- ``[M+H]+``,
``M+H``, ``[M+H]1+``, ``[m+h]+`` -- and the strict 5 ppm gate rejects any
adduct it cannot resolve. A formatting difference therefore quarantined an
otherwise correct spectrum (issue #72: ingestion rejections caused by adduct
notation, not by physical mass error).

Contract under test:

* ``normalize_adduct`` maps the notational variants onto the canonical
  ``_ADDUCT_SPECS`` keys, and fails closed for anything it cannot resolve --
  including notation that declares a charge contradicting the chemistry.
* The registry covers the adducts that dominate real library exports, each
  derived through pyteomics (the project's single mass source of truth).
* ``compute_adduct_offset`` / ``calculate_theoretical_mass`` /
  ``SpectrumMetadata`` / ``physical_integrity_reason`` all agree on the
  normalised form, so a correctly-notated ``M+H`` spectrum is no longer
  rejected at ingestion.
"""

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import processing
from MassFlow.cheminformatics import (
    _ADDUCT_SPECS,
    calculate_theoretical_mass,
    compute_adduct_offset,
    normalize_adduct,
)
from MassFlow.config import ProcessingConfig
from MassFlow.models import MolecularStructure, SpectrumMetadata
from MassFlow.processing import metadata_processing, physical_integrity_reason

pytestmark = pytest.mark.scientific

# Caffeine: the reference molecule used across the adduct suite.
CAFFEINE_FORMULA = "C8H10N4O2"
CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CAFFEINE_MH = 195.087652  # [M+H]+ theoretical m/z of caffeine

# Salicylic acid: a negative-mode companion for [M-H]- checks.
SALICYLIC_FORMULA = "C7H6O3"


def make_spectrum(
    spec_id: str,
    precursor_mz: float,
    *,
    formula: str | None = None,
    charge: int | None = 1,
    adduct: str | None = None,
    ionmode: str | None = None,
    compound_name: str | None = None,
) -> Spectrum:
    """Build a minimal single-peak spectrum carrying the given metadata."""
    metadata: dict[str, object] = {"id": spec_id, "precursor_mz": precursor_mz}
    if formula is not None:
        metadata["formula"] = formula
    if charge is not None:
        metadata["charge"] = charge
    if adduct is not None:
        metadata["adduct"] = adduct
    if ionmode is not None:
        metadata["ionmode"] = ionmode
    if compound_name is not None:
        metadata["compound_name"] = compound_name
    return Spectrum(
        mz=np.array([100.0, 150.0], dtype="float64"),
        intensities=np.array([10.0, 20.0], dtype="float64"),
        metadata=metadata,
        metadata_harmonization=False,
    )


# ---------------------------------------------------------------------------
# normalize_adduct: notation variants
# ---------------------------------------------------------------------------


class TestNormalizeAdductVariants:
    """Every spelling of an ion must resolve to one canonical registry key."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Bracketing and charge-suffix placement
            ("[M+H]+", "[M+H]+"),
            ("M+H", "[M+H]+"),
            ("[M+H]1+", "[M+H]+"),
            ("[M+H]+1", "[M+H]+"),
            ("[M+H]+ ", "[M+H]+"),
            # Interior whitespace
            ("[M + H]+", "[M+H]+"),
            ("  [M+NH4]+  ", "[M+NH4]+"),
            # Case variations (library exports are inconsistent)
            ("[m+h]+", "[M+H]+"),
            ("m+na", "[M+Na]+"),
            ("[M+NA]+", "[M+Na]+"),
            # Bare bodies keep their sign
            ("M-H", "[M-H]-"),
            ("[M-H]1-", "[M-H]-"),
            ("[M-2H]2-", "[M-2H]2-"),
            ("[M+2H]2+", "[M+2H]2+"),
            # Alias chemistry
            ("[M+FA-H]-", "[M+HCOO]-"),
            ("[M+FA-H] -", "[M+HCOO]-"),  # legacy key with interior whitespace
            ("M+HCOOH-H", "[M+HCOO]-"),
            ("M+OAc", "[M+CH3COO]-"),
            ("M+Ac-H", "[M+CH3COO]-"),
            ("M+CF3COO", "[M+TFA-H]-"),
            ("M+ACN+H", "[M+CH3CN+H]+"),
            ("M+MeOH+H", "[M+CH3OH+H]+"),
            ("M-H2O+H", "[M+H-H2O]+"),
        ],
    )
    def test_variants_resolve_to_canonical_key(self, raw, expected):
        assert normalize_adduct(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "M+H",  # charge-less notation still resolves via the sign of the terms
            "[M+H]1+",
            "[m+h]+",
            "[M + H]+",
        ],
    )
    def test_all_variants_share_one_offset(self, raw):
        """Normalisation must not change the computed physics."""
        assert compute_adduct_offset(raw) == compute_adduct_offset("[M+H]+")


class TestNormalizeAdductFailsClosed:
    """Unresolvable or self-contradictory notation must not be guessed at."""

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "   ",
            "not-an-adduct",
            "M",  # ambiguous: shared body of [M]+ and [M]-
            "[M+Weird]+",
            "[M+Unknown]+",
            "[M+H]2+",  # declares a charge the chemistry does not support
            "[M-H]+",  # sign conflict
            "[M++]",
            "1+",
        ],
    )
    def test_unresolvable_notation_returns_none(self, raw):
        assert normalize_adduct(raw) is None
        assert compute_adduct_offset(raw) is None

    @pytest.mark.parametrize("raw", ["[M+H", "M+H]", "[M + H"])
    def test_bracket_formatting_is_lenient(self, raw):
        """Brackets are pure formatting, so unbalanced variants still resolve.

        This cannot create a false positive: the resolved key still has to
        clear the 5 ppm precursor gate before a spectrum is accepted.
        """
        assert normalize_adduct(raw) == "[M+H]+"

    def test_unknown_adduct_does_not_reach_the_registry(self):
        """The legacy failure mode must be preserved: junk stays junk."""
        assert _ADDUCT_SPECS.get("[M+Weird]+") is None
        assert "[M+H]+" in _ADDUCT_SPECS


# ---------------------------------------------------------------------------
# Registry coverage: offsets derived independently via pyteomics
# ---------------------------------------------------------------------------


class TestRegistryCoverage:
    """Every registry entry must resolve to a physically sensible offset."""

    @pytest.mark.parametrize(
        ("adduct", "expected_offset"),
        [
            # (atoms added/removed) - charge x electron, computed from pyteomics
            ("[M+H]+", 1.007276),
            ("[M+Na]+", 22.989221),
            ("[M+NH4]+", 18.033826),
            ("[M+K]+", 38.963158),
            ("[M+2H]2+", 2.014553),
            ("[M+3H]3+", 3.021829),
            ("[M+2Na-H]+", 44.971165),
            ("[M+H-H2O]+", -17.003288),
            ("[M+CH3CN+H]+", 42.033826),
            ("[M+CH3OH+H]+", 33.033491),
            ("[M]+", -0.000549),
            ("[M]-", 0.000549),
            ("[M-H]-", -1.007276),
            ("[M+Cl]-", 34.969401),
            ("[M+HCOO]-", 44.998203),
            ("[M+CH3COO]-", 59.013853),
            ("[M+TFA-H]-", 112.985587),
            ("[M+HCO3]-", 60.993117),
            ("[M+NO3]-", 61.988366),
            ("[M+Br]-", 78.918886),
            ("[M+I]-", 126.905022),
            ("[M-2H]2-", -2.014553),
        ],
    )
    def test_offset_matches_expected_mass_shift(self, adduct, expected_offset):
        offset = compute_adduct_offset(adduct)
        assert offset is not None
        assert offset == pytest.approx(expected_offset, abs=1e-5)

    def test_every_registry_entry_is_self_consistent(self):
        """Every key must round-trip through normalisation and yield an offset."""
        for key in _ADDUCT_SPECS:
            assert normalize_adduct(key) == key
            assert compute_adduct_offset(key) is not None

    def test_common_library_adducts_are_all_covered(self):
        """Adducts that dominate real exports must resolve, not quarantine."""
        common = [
            "[M+H]+",
            "M+H",
            "M+Na",
            "M+NH4",
            "M+K",
            "[M-H]-",
            "M-H",
            "[M+HCOO]-",
            "[M+CH3COO]-",
            "[M+Cl]-",
            "[M+2H]2+",
            "[M+FA-H]-",
            "[M+TFA-H]-",
            "[M+H-H2O]+",
            "[M+2Na-H]+",
        ]
        unresolved = [a for a in common if normalize_adduct(a) is None]
        assert unresolved == []


# ---------------------------------------------------------------------------
# calculate_theoretical_mass accepts the same notation
# ---------------------------------------------------------------------------


class TestTheoreticalMassAcceptsVariants:
    def test_alias_notation_matches_canonical_result(self):
        baseline = calculate_theoretical_mass(formula=CAFFEINE_FORMULA, adduct="[M+H]+")
        assert baseline == pytest.approx(CAFFEINE_MH, abs=1e-5)
        for alias in ("M+H", "[M+H]1+", "[m+h]+"):
            assert calculate_theoretical_mass(
                formula=CAFFEINE_FORMULA, adduct=alias
            ) == pytest.approx(baseline)

    def test_smiles_and_formula_paths_agree(self):
        from_formula = calculate_theoretical_mass(
            formula=CAFFEINE_FORMULA, adduct="M+H"
        )
        from_smiles = calculate_theoretical_mass(smiles=CAFFEINE_SMILES, adduct="M+H")
        assert from_formula == pytest.approx(CAFFEINE_MH, abs=1e-5)
        # SMILES depends on RDKit; skip the cross-check when it is unavailable.
        if from_smiles is not None:
            assert from_smiles == pytest.approx(CAFFEINE_MH, abs=1e-5)

    def test_negative_mode_alias(self):
        theo = calculate_theoretical_mass(formula=SALICYLIC_FORMULA, adduct="M-H")
        assert theo == pytest.approx(137.024416, abs=1e-5)

    def test_unknown_adduct_still_raises(self):
        with pytest.raises(ValueError, match="is not supported"):
            calculate_theoretical_mass(formula=CAFFEINE_FORMULA, adduct="[M+Weird]+")


# ---------------------------------------------------------------------------
# The model contract: alias notation no longer fails the physics gate
# ---------------------------------------------------------------------------


class TestSpectrumMetadataNotationHandling:
    def test_alias_notation_is_canonicalised_and_valid(self):
        """Regression: ``M+H`` must validate, not quarantine (issue #72)."""
        meta = SpectrumMetadata(
            spectrum_id="alias",
            precursor_mz=CAFFEINE_MH,
            charge=1,
            adduct="M+H",
            molecule=MolecularStructure(formula=CAFFEINE_FORMULA),
        )
        assert meta.adduct == "[M+H]+"
        assert meta.is_physically_valid is True

    def test_canonicalised_adduct_is_persisted(self):
        meta = SpectrumMetadata(
            spectrum_id="alias_persist",
            precursor_mz=CAFFEINE_MH,
            charge=1,
            adduct="[M+H]1+",
            molecule=MolecularStructure(formula=CAFFEINE_FORMULA),
        )
        assert meta.adduct == "[M+H]+"

    def test_unresolvable_adduct_is_preserved_and_invalid(self):
        meta = SpectrumMetadata(
            spectrum_id="junk",
            precursor_mz=CAFFEINE_MH,
            charge=1,
            adduct="[M+Weird]+",
            molecule=MolecularStructure(formula=CAFFEINE_FORMULA),
        )
        assert meta.adduct == "[M+Weird]+"
        assert meta.is_physically_valid is False

    def test_alias_notation_does_not_bypass_the_ppm_gate(self):
        """Normalisation must not weaken the 5 ppm check itself."""
        meta = SpectrumMetadata(
            spectrum_id="alias_drift",
            precursor_mz=CAFFEINE_MH * (1 + 5.1 / 1e6),
            charge=1,
            adduct="M+H",
            molecule=MolecularStructure(formula=CAFFEINE_FORMULA),
        )
        assert meta.is_physically_valid is False


# ---------------------------------------------------------------------------
# Ingestion pipeline: harmonisation and the gate agree on one spelling
# ---------------------------------------------------------------------------


class TestIngestionPath:
    def test_metadata_processing_canonicalises_adduct(self):
        spectrum = make_spectrum(
            "ingest_alias",
            CAFFEINE_MH,
            formula=CAFFEINE_FORMULA,
            adduct="M+H",
            ionmode="positive",
        )
        processed = metadata_processing(spectrum)
        assert processed is not None
        assert processed.get("adduct") == "[M+H]+"

    def test_unknown_adduct_is_left_untouched_by_harmonisation(self):
        spectrum = make_spectrum(
            "ingest_junk",
            CAFFEINE_MH,
            formula=CAFFEINE_FORMULA,
            adduct="[M+Weird]+",
            ionmode="positive",
        )
        processed = metadata_processing(spectrum)
        assert processed is not None
        # Only resolvable notation is rewritten; the gate owns the verdict.
        assert processed.get("adduct") == "[M+Weird]+"

    def test_ingestion_gate_accepts_alias_notation(self):
        """Regression for the reported ingestion rejections (issue #72)."""
        spectrum = make_spectrum(
            "ingest_alias_gate",
            CAFFEINE_MH,
            formula=CAFFEINE_FORMULA,
            adduct="M+H",
            ionmode="positive",
        )
        assert physical_integrity_reason(spectrum) is None

    def test_ingestion_gate_still_rejects_unknown_adduct(self):
        spectrum = make_spectrum(
            "ingest_junk_gate",
            CAFFEINE_MH,
            formula=CAFFEINE_FORMULA,
            adduct="[M+Weird]+",
            ionmode="positive",
        )
        reason = physical_integrity_reason(spectrum)
        assert reason is not None
        assert "not supported" in reason

    def test_processing_pipeline_keeps_alias_notation_spectrum(self):
        """End of the ingestion path: the spectrum survives processing."""
        spectrum = make_spectrum(
            "pipeline_alias",
            CAFFEINE_MH,
            formula=CAFFEINE_FORMULA,
            adduct="M+H",
            ionmode="positive",
            compound_name="Caffeine",
        )
        processed = processing.process_spectra_batch(
            [spectrum], ProcessingConfig(min_peaks=1, noise_threshold=0.0)
        )
        assert [s.get("id") for s in processed] == ["pipeline_alias"]
        assert processed[0].get("adduct") == "[M+H]+"
