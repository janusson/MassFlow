"""
Stage-1 precursor-candidacy tests (R-2).

The candidate stage is pure, deterministic and numpy-only, so the window
arithmetic is tested directly. The graph-level tests then prove the two
properties the review decision asks for: a dedicated candidacy window can never
be more selective than the similarity engine's own MS1 prefilter, and widening it
adds *scored pairs*, never edges.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchms import Spectrum
from pydantic import ValidationError

from MassFlow.config import NetworkConfig
from MassFlow.network import build_spectral_graph
from MassFlow.network.candidates import generate_candidate_pairs
from MassFlow.network.spectral import _candidacy_window

pytestmark = pytest.mark.unit

_PEAKS_MZ = np.array(
    [50.0, 80.0, 110.0, 140.0, 170.0, 200.0, 230.0, 260.0, 290.0, 320.0],
    dtype=np.float64,
)
_PEAKS_INT = np.array(
    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float64
)
# Disjoint fragments: a candidate pair that cannot match a single peak.
_FOREIGN_MZ = np.array(
    [61.0, 91.0, 121.0, 151.0, 181.0, 211.0, 241.0, 271.0, 301.0, 331.0],
    dtype=np.float64,
)


def _spectrum(
    precursor_mz: float, spec_id: str, *, mz: np.ndarray = _PEAKS_MZ
) -> Spectrum:
    return Spectrum(
        mz=mz.copy(),
        intensities=_PEAKS_INT.copy(),
        metadata={
            "id": spec_id,
            "precursor_mz": precursor_mz,
            "retention_time": 100.0,
            "adduct": "[M+H]+",
            "charge": 1,
        },
    )


def _cfg(**overrides: object) -> NetworkConfig:
    payload: dict[str, object] = {
        "enabled": True,
        "min_score": 0.7,
        "min_matched_peaks": 3,
        "top_k_per_node": None,
    }
    payload.update(overrides)
    return NetworkConfig(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Default behaviour is unchanged, and the window can never shrink
# --------------------------------------------------------------------------


def test_default_candidacy_equals_the_engine_ms1_prefilter() -> None:
    """With no dedicated knob, candidacy stays exactly ms1_tolerance (Da)."""
    cfg = _cfg()
    assert cfg.precursor_candidacy_tolerance is None
    assert _candidacy_window(cfg) == (cfg.ms1_tolerance, None)


def test_narrower_da_candidacy_is_floored_to_the_engine_prefilter() -> None:
    """A candidacy window below ms1_tolerance cannot drop an engine-valid pair."""
    mzs = np.array([1000.0, 1000.005, 1000.015, 1000.028], dtype=np.float64)
    floored = _candidacy_window(_cfg(precursor_candidacy_tolerance=0.001))
    assert floored == (0.02, None)
    assert generate_candidate_pairs(
        mzs, tolerance=floored[0], tolerance_ppm=floored[1]
    ) == generate_candidate_pairs(mzs, tolerance=0.02)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"precursor_candidacy_tolerance": 0.001},
        {"precursor_candidacy_tolerance": 0.5},
        {"precursor_candidacy_tolerance": 5.0, "precursor_candidacy_unit": "ppm"},
        {"precursor_candidacy_tolerance": 200.0, "precursor_candidacy_unit": "ppm"},
    ],
)
def test_candidacy_is_a_superset_of_the_engine_prefilter(
    overrides: dict[str, object],
) -> None:
    """Every pair the engine's MS1 prefilter accepts must still be a candidate."""
    mzs = np.array(
        [100.0, 100.015, 250.0, 250.03, 999.99, 1000.0, 1000.019, 1500.0, 1500.05],
        dtype=np.float64,
    )
    cfg = _cfg(**overrides)
    floor, ppm = _candidacy_window(cfg)
    candidates = set(generate_candidate_pairs(mzs, tolerance=floor, tolerance_ppm=ppm))
    engine_accepted = set(generate_candidate_pairs(mzs, tolerance=cfg.ms1_tolerance))
    assert engine_accepted <= candidates


def test_a_wider_da_window_only_adds_pairs() -> None:
    mzs = np.array([1000.0, 1000.015, 1000.05], dtype=np.float64)
    default = set(generate_candidate_pairs(mzs, tolerance=0.02))
    wider = set(generate_candidate_pairs(mzs, tolerance=0.1))
    assert default < wider


# --------------------------------------------------------------------------
# The ppm unit scales with mass (the physical point of the change)
# --------------------------------------------------------------------------


def test_ppm_window_grows_with_mass() -> None:
    """25 ppm at m/z 1500 is 0.0375 Da: admitted by 20 ppm, missed by 0.02 Da."""
    pair = np.array([1500.0, 1500.025], dtype=np.float64)
    assert generate_candidate_pairs(pair, tolerance=0.02) == []
    assert generate_candidate_pairs(pair, tolerance=0.02, tolerance_ppm=20.0) == [
        (0, 1)
    ]


def test_ppm_window_still_bounds_the_pair() -> None:
    """The ppm window is a window, not an open door."""
    pair = np.array([1500.0, 1500.040], dtype=np.float64)
    assert generate_candidate_pairs(pair, tolerance=0.02, tolerance_ppm=20.0) == []


def test_da_floor_governs_at_low_mass() -> None:
    """At m/z 100, 20 ppm is 0.002 Da — the ms1_tolerance floor still applies."""
    pair = np.array([100.0, 100.015], dtype=np.float64)
    assert generate_candidate_pairs(pair, tolerance=0.0, tolerance_ppm=20.0) == []
    assert generate_candidate_pairs(pair, tolerance=0.02, tolerance_ppm=20.0) == [
        (0, 1)
    ]


def test_ppm_only_window_is_accepted_without_a_da_tolerance() -> None:
    pair = np.array([1000.0, 1000.010], dtype=np.float64)
    assert generate_candidate_pairs(pair, tolerance=0.0, tolerance_ppm=20.0) == [(0, 1)]


def test_candidacy_is_deterministic_under_both_units() -> None:
    mzs = np.array([300.0, 100.0, 100.015, 900.0, 900.02, 500.0], dtype=np.float64)
    assert generate_candidate_pairs(mzs, tolerance=0.02) == generate_candidate_pairs(
        mzs, tolerance=0.02
    )
    assert generate_candidate_pairs(
        mzs, tolerance=0.02, tolerance_ppm=30.0
    ) == generate_candidate_pairs(mzs, tolerance=0.02, tolerance_ppm=30.0)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_invalid_windows_raise() -> None:
    mzs = [100.0, 100.01]
    with pytest.raises(ValueError):
        generate_candidate_pairs(mzs, tolerance=0.0)  # no window at all
    with pytest.raises(ValueError):
        generate_candidate_pairs(mzs, tolerance=-0.02)
    with pytest.raises(ValueError):
        generate_candidate_pairs(mzs, tolerance=0.02, tolerance_ppm=0.0)
    with pytest.raises(ValueError):
        generate_candidate_pairs(mzs, tolerance=0.02, tolerance_ppm=-5.0)


def test_candidacy_unit_is_validated() -> None:
    with pytest.raises(ValidationError):
        NetworkConfig(precursor_candidacy_unit="Th")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        NetworkConfig(precursor_candidacy_tolerance=0.0)
    with pytest.raises(ValidationError):
        NetworkConfig(precursor_candidacy_tolerance=-1.0)


# --------------------------------------------------------------------------
# Graph level: widening candidacy adds scored pairs, never edges
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_widened_candidacy_adds_no_edge_for_a_failing_pair() -> None:
    """A new candidate pair that cannot match a peak still yields no edge."""
    spectra = [
        _spectrum(1000.0, "a"),
        _spectrum(1000.025, "b", mz=_FOREIGN_MZ),
    ]
    default = build_spectral_graph(spectra, _cfg())
    widened = build_spectral_graph(
        spectra,
        _cfg(precursor_candidacy_tolerance=30.0, precursor_candidacy_unit="ppm"),
    )
    # 30 ppm at m/z 1000 is 0.03 Da, so the pair *is* now a candidate...
    assert generate_candidate_pairs(
        np.array([1000.0, 1000.025]), tolerance=0.02, tolerance_ppm=30.0
    ) == [(0, 1)]
    # ...but it matches no fragment, so no edge appears in either graph.
    assert default.relationships == []
    assert widened.relationships == []


@pytest.mark.integration
def test_widened_candidacy_adds_a_real_edge_when_the_pair_matches() -> None:
    """Positive control: the same window change does connect a matching pair."""
    spectra = [_spectrum(1000.0, "a"), _spectrum(1000.025, "b")]
    assert build_spectral_graph(spectra, _cfg()).relationships == []
    widened = build_spectral_graph(
        spectra,
        _cfg(precursor_candidacy_tolerance=30.0, precursor_candidacy_unit="ppm"),
    )
    assert len(widened.relationships) == 1
    edge = widened.relationships[0]
    # The edge still records the *scoring* tolerances, not the candidacy window.
    assert edge.ms1_tolerance == 0.02
    assert edge.tolerance_unit == "Da"


@pytest.mark.integration
def test_default_run_is_unchanged_by_the_new_config_fields() -> None:
    """The new fields at their defaults leave the graph document identical."""
    spectra = [
        _spectrum(1000.0, "a"),
        _spectrum(1000.01, "b"),
        _spectrum(500.0, "c"),
    ]
    implicit = build_spectral_graph(spectra, _cfg(), created_at="2026-01-01T00:00:00Z")
    explicit = build_spectral_graph(
        spectra,
        _cfg(precursor_candidacy_tolerance=None, precursor_candidacy_unit="Da"),
        created_at="2026-01-01T00:00:00Z",
    )
    assert implicit.to_json() == explicit.to_json()
    assert len(implicit.relationships) == 1
    # Spec 9: the candidacy configuration is recorded on edge provenance.
    parameters = implicit.relationships[0].provenance.parameters
    assert parameters["precursor_candidacy_unit"] == "Da"
    assert parameters["precursor_candidacy_tolerance"] is None
