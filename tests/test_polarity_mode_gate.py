"""Regression tests for ionization-polarity gating in similarity search.

Verifies that :class:`MassFlow.similarity.SimilarityEngine` enforces polarity
compatibility using ``adduct``, ``ionmode``, and signed ``charge`` metadata
across classical scoring modes and reference-delivery patterns, while
preserving permissive semantics when polarity is genuinely unknown.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
from matchms import Spectrum
from pyteomics import mass

from MassFlow.cheminformatics import ELECTRON_MASS, compute_adduct_offset
from MassFlow.config import SimilarityConfig
from MassFlow.diagnostics import DiagnosticReason
from MassFlow.models import MolecularStructure, SpectrumMetadata
from MassFlow.similarity import SimilarityEngine

Algorithm = Literal["cosine", "modified_cosine"]
ReferenceMode = Literal["list", "iterator", "cached"]


def _caffeine_precursor_mz(adduct: str = "[M+H]+") -> float:
    """Compute exact caffeine precursor m/z from formula and adduct chemistry."""
    molecule = MolecularStructure(formula="C8H10N4O2")
    assert molecule.is_physically_valid
    offset = compute_adduct_offset(adduct)
    assert offset is not None
    expected_charge = 1 if adduct.endswith("+") else -1
    mz_val = float(mass.calculate_mass(formula="C8H10N4O2")) + offset
    metadata = SpectrumMetadata(
        spectrum_id=f"caffeine_{adduct}",
        precursor_mz=mz_val,
        charge=expected_charge,
        adduct=adduct,
        molecule=molecule,
    )
    assert metadata.is_physically_valid
    return mz_val


def _caffeine_fragments(charge_sign: int = 1) -> np.ndarray:
    """Return exact float64 caffeine fragment m/z values."""
    return (
        np.array(
            [
                mass.calculate_mass(formula=formula)
                for formula in ("C2H4N", "C4H7N2", "C6H8N3O")
            ],
            dtype=np.float64,
        )
        - charge_sign * ELECTRON_MASS
    )


def _make_caffeine_spectrum(
    spectrum_id: str,
    *,
    precursor_adduct_for_mass: str = "[M+H]+",
    adduct: str | None = None,
    ionmode: str | None = None,
    charge: Any = None,
    intensities: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fragment_sign: int = 1,
) -> Spectrum:
    """Build a deterministic float64 Spectrum with caller-specified polarity tags."""
    metadata: dict[str, Any] = {
        "id": spectrum_id,
        "compound_name": spectrum_id,
        "formula": "C8H10N4O2",
        "precursor_mz": _caffeine_precursor_mz(precursor_adduct_for_mass),
    }
    if adduct is not None:
        metadata["adduct"] = adduct
    if ionmode is not None:
        metadata["ionmode"] = ionmode
    if charge is not None:
        metadata["charge"] = charge
    return Spectrum(
        mz=_caffeine_fragments(fragment_sign),
        intensities=np.array(intensities, dtype=np.float64),
        metadata=metadata,
    )


def _run_search(
    engine: SimilarityEngine,
    query: Spectrum,
    references: list[Spectrum],
    reference_mode: ReferenceMode,
) -> list[dict[str, Any]]:
    """Run SimilarityEngine.search across list, iterator, and cached-m/z modes."""
    if reference_mode == "list":
        return engine.search([query], references, include_decoys=False)
    if reference_mode == "iterator":
        return engine.search([query], iter(references), include_decoys=False)
    ref_mzs = np.array(
        [float(ref.get("precursor_mz")) for ref in references], dtype=np.float64
    )
    return engine.search(
        [query],
        references,
        include_decoys=False,
        ref_precursor_mzs=ref_mzs,
    )


@pytest.mark.core
@pytest.mark.scientific
@pytest.mark.parametrize("algorithm", ["cosine", "modified_cosine"])
@pytest.mark.parametrize("reference_mode", ["list", "iterator", "cached"])
def test_opposite_ionmode_without_adduct_rejected(
    algorithm: Algorithm, reference_mode: ReferenceMode
) -> None:
    """Opposite ionmode must block matches even when adduct metadata is absent."""
    query = _make_caffeine_spectrum(
        "query_pos", ionmode="positive", precursor_adduct_for_mass="[M+H]+"
    )
    # Use matching peak m/z and wide MS1 tolerance so only the polarity gate acts.
    reference = _make_caffeine_spectrum(
        "ref_neg", ionmode="negative", precursor_adduct_for_mass="[M-H]-"
    )
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm=algorithm,
            ms1_tolerance=3.0,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
            collect_retrieval_diagnostics=True,
        )
    )

    results = _run_search(engine, query, [reference], reference_mode)

    assert results == []
    diag = engine.retrieval_diagnostics["queries"]["query_pos"]
    assert diag["target"]["matched_peak_pass_pairs"] == 1
    assert diag["target"]["polarity_compatible_pairs"] == 0
    assert diag["target"]["rejection_counts"] == {"polarity_incompatible": 1}
    assert diag["terminal_outcome"] == DiagnosticReason.POLARITY_INCOMPATIBLE.value


@pytest.mark.core
@pytest.mark.scientific
@pytest.mark.parametrize("algorithm", ["cosine", "modified_cosine"])
def test_adduct_vs_opposite_ionmode_without_adduct_rejected(
    algorithm: Algorithm,
) -> None:
    """Adduct on one side and opposite ionmode on the other must not match."""
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm=algorithm,
            ms1_tolerance=3.0,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
        )
    )

    q_adduct_pos = _make_caffeine_spectrum("q_pos", adduct="[M+H]+")
    ref_ionmode_neg = _make_caffeine_spectrum(
        "r_neg", ionmode="negative", precursor_adduct_for_mass="[M-H]-"
    )
    assert engine.search([q_adduct_pos], [ref_ionmode_neg], include_decoys=False) == []

    q_ionmode_pos = _make_caffeine_spectrum("q_pos_im", ionmode="positive")
    ref_adduct_neg = _make_caffeine_spectrum(
        "r_neg_ad", adduct="[M-H]-", precursor_adduct_for_mass="[M-H]-"
    )
    assert engine.search([q_ionmode_pos], [ref_adduct_neg], include_decoys=False) == []


@pytest.mark.core
@pytest.mark.scientific
@pytest.mark.parametrize("algorithm", ["cosine", "modified_cosine"])
def test_opposite_signed_charge_without_adduct_or_ionmode_rejected(
    algorithm: Algorithm,
) -> None:
    """Opposite signed charges must block matches when adduct and ionmode are missing."""
    query = _make_caffeine_spectrum("q_charge_pos", charge=1)
    reference = _make_caffeine_spectrum(
        "r_charge_neg", charge=-1, precursor_adduct_for_mass="[M-H]-"
    )
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm=algorithm,
            ms1_tolerance=3.0,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
        )
    )

    assert engine.search([query], [reference], include_decoys=False) == []


@pytest.mark.core
@pytest.mark.scientific
def test_top_n_not_displaced_by_opposite_ionmode_perfect_score() -> None:
    """Opposite-ionmode high-scoring hit must not consume top_n slots."""
    query = _make_caffeine_spectrum(
        "query_pos", ionmode="positive", intensities=(1.0, 1.0, 1.0)
    )
    neg_perfect = _make_caffeine_spectrum(
        "neg_perfect",
        ionmode="negative",
        precursor_adduct_for_mass="[M-H]-",
        intensities=(1.0, 1.0, 1.0),
    )
    pos_eligible = _make_caffeine_spectrum(
        "pos_eligible",
        ionmode="positive",
        intensities=(1.0, 1.0, 0.5),
    )
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=3.0,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
        )
    )

    ranked = engine.search(
        [query],
        [neg_perfect, pos_eligible],
        top_n=1,
        include_decoys=False,
    )

    assert len(ranked) == 1
    assert ranked[0]["reference_id"] == "pos_eligible"
    assert ranked[0]["score"] == pytest.approx(0.9622504486493763, abs=1e-12)


@pytest.mark.core
@pytest.mark.scientific
@pytest.mark.parametrize(
    "unknown_meta",
    [
        {},
        {"ionmode": ""},
        {"ionmode": "n/a"},
        {"ionmode": "neutral"},
        {"charge": 0},
    ],
)
def test_missing_or_neutral_polarity_remains_permissive(
    unknown_meta: dict[str, Any],
) -> None:
    """Missing, zero, or neutral polarity metadata must not block candidates."""
    query = _make_caffeine_spectrum("query_pos", ionmode="positive", charge=1)
    reference = _make_caffeine_spectrum("ref_unknown", **unknown_meta)
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=0.02,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
        )
    )

    results = engine.search([query], [reference], include_decoys=False)

    assert len(results) == 1
    assert results[0]["reference_id"] == "ref_unknown"


@pytest.mark.core
@pytest.mark.scientific
def test_same_polarity_variants_and_aliases_remain_compatible() -> None:
    """Same-polarity aliases, adducts, and multi-charge states remain compatible."""
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm="modified_cosine",
            ms1_tolerance=30.0,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
        )
    )

    q_pos = _make_caffeine_spectrum("q_pos", ionmode="POS", charge=1)
    r_na = _make_caffeine_spectrum(
        "r_na",
        adduct="[M+Na]+",
        precursor_adduct_for_mass="[M+Na]+",
        charge=1,
    )
    r_charge2 = _make_caffeine_spectrum("r_charge2", charge=2)

    results = engine.search([q_pos], [r_na, r_charge2], include_decoys=False)
    assert {row["reference_id"] for row in results} == {"r_na", "r_charge2"}

    r_neg_alias = _make_caffeine_spectrum(
        "r_neg_alias", ionmode="NEG", precursor_adduct_for_mass="[M-H]-"
    )
    assert engine.search([q_pos], [r_neg_alias], include_decoys=False) == []


@pytest.mark.core
@pytest.mark.scientific
def test_adduct_and_ionmode_take_precedence_over_stale_charge() -> None:
    """Explicit adduct or ionmode overrides stale default charge=1 metadata."""
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=3.0,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
        )
    )
    query_pos = _make_caffeine_spectrum("q_pos", adduct="[M+H]+", charge=1)
    query_neg = _make_caffeine_spectrum(
        "q_neg", adduct="[M-H]-", precursor_adduct_for_mass="[M-H]-", charge=-1
    )

    stale_neg_adduct = _make_caffeine_spectrum(
        "r_neg_adduct_stale_charge",
        adduct="[M-H]-",
        precursor_adduct_for_mass="[M-H]-",
        charge=1,
    )
    stale_neg_ionmode = _make_caffeine_spectrum(
        "r_neg_ionmode_stale_charge",
        ionmode="negative",
        precursor_adduct_for_mass="[M-H]-",
        charge=1,
    )

    assert (
        engine.search(
            [query_pos], [stale_neg_adduct, stale_neg_ionmode], include_decoys=False
        )
        == []
    )
    matched_neg = engine.search(
        [query_neg], [stale_neg_adduct, stale_neg_ionmode], include_decoys=False
    )
    assert {row["reference_id"] for row in matched_neg} == {
        "r_neg_adduct_stale_charge",
        "r_neg_ionmode_stale_charge",
    }


@pytest.mark.core
@pytest.mark.scientific
def test_conflicting_adduct_and_ionmode_fails_closed() -> None:
    """A spectrum claiming contradictory adduct and ionmode matches neither mode."""
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=3.0,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.9,
        )
    )
    corrupted = _make_caffeine_spectrum(
        "corrupted", adduct="[M+H]+", ionmode="negative"
    )
    query_pos = _make_caffeine_spectrum("q_pos", adduct="[M+H]+", ionmode="positive")
    query_neg = _make_caffeine_spectrum(
        "q_neg",
        adduct="[M-H]-",
        ionmode="negative",
        precursor_adduct_for_mass="[M-H]-",
    )

    assert engine.search([query_pos], [corrupted], include_decoys=False) == []
    assert engine.search([query_neg], [corrupted], include_decoys=False) == []


@pytest.mark.core
def test_batch_score_rejects_opposite_ionmode() -> None:
    """batch_score must return NaN rather than scoring opposite-ionmode pairs."""
    engine = SimilarityEngine(
        SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=3.0,
            ms2_tolerance=0.02,
            min_matched_peaks=3,
            min_score=0.0,
        )
    )
    query_pos = _make_caffeine_spectrum("q_pos", ionmode="positive")
    ref_pos = _make_caffeine_spectrum("r_pos", ionmode="positive")
    ref_neg = _make_caffeine_spectrum(
        "r_neg", ionmode="negative", precursor_adduct_for_mass="[M-H]-"
    )

    scores = engine.batch_score([query_pos, query_pos], [ref_pos, ref_neg])

    assert scores[0] == pytest.approx(1.0, abs=1e-12)
    assert np.isnan(scores[1])


@pytest.mark.core
@pytest.mark.parametrize(
    ("raw_ionmode", "expected"),
    [
        ("positive", 1),
        (" POS ", 1),
        ("P", 1),
        ("+", 1),
        ("negative", -1),
        ("neg", -1),
        ("n", -1),
        ("-", -1),
        ("neutral", None),
        ("n/a", None),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_ionmode_sign_normalization(raw_ionmode: Any, expected: int | None) -> None:
    """_ionmode_sign parses explicit polarity labels and ignores non-ionic values."""
    from MassFlow.similarity import _ionmode_sign

    assert _ionmode_sign(raw_ionmode) == expected


@pytest.mark.core
@pytest.mark.parametrize(
    ("raw_charge", "expected"),
    [
        (1, 1),
        (-2, -1),
        (np.int64(-1), -1),
        (0, None),
        (0.0, None),
        (True, None),
        (False, None),
        (float("nan"), None),
        (float("inf"), None),
        (None, None),
        ("1+", 1),
        ("2-", -1),
        ("-1", -1),
        ("+2", 1),
        ("0+", None),
        ("-1+", None),
        ("", None),
        ("not-a-charge", None),
        (object(), None),
    ],
)
def test_charge_sign_parsing(raw_charge: Any, expected: int | None) -> None:
    """_charge_sign handles ints, floats, string notations, booleans, and invalids."""
    from MassFlow.similarity import _charge_sign

    assert _charge_sign(raw_charge) == expected


@pytest.mark.core
def test_spectrum_polarity_state_handles_raw_ion_mode_and_string_charge() -> None:
    """Un-harmonized raw ion_mode and string charge metadata still gate search."""
    from MassFlow.similarity import _spectrum_polarity_state

    pos_raw = Spectrum(
        mz=_caffeine_fragments(1),
        intensities=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        metadata={"id": "pos_raw", "precursor_mz": 195.08765, "ion_mode": "positive"},
        metadata_harmonization=False,
    )
    neg_raw = Spectrum(
        mz=_caffeine_fragments(1),
        intensities=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        metadata={"id": "neg_raw", "precursor_mz": 195.08765, "charge": "1-"},
        metadata_harmonization=False,
    )

    neg_raw_ion_mode_alias = Spectrum(
        mz=_caffeine_fragments(1),
        intensities=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        metadata={
            "id": "neg_raw_alias",
            "precursor_mz": 195.08765,
            "ion_mode": "N",
        },
        metadata_harmonization=False,
    )

    assert _spectrum_polarity_state(pos_raw) == 1
    assert _spectrum_polarity_state(neg_raw) == -1
    assert _spectrum_polarity_state(neg_raw_ion_mode_alias) == -1
