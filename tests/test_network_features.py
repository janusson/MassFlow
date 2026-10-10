"""
Stage 2 feature-identity tests.

Deterministic grouping of experimental spectra into LC-MS features by ion
channel, precursor m/z window and retention-time window, with a
highest-TIC representative.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import NetworkConfig
from MassFlow.network.build import build_spectral_graph
from MassFlow.network.features import build_features
from MassFlow.network.models import MolecularGraph, spectrum_node_id

pytestmark = pytest.mark.unit

_BASE_MZ = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0])
_BASE_INT = np.array([1.0, 0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])


def _spectrum(
    spec_id: str,
    precursor_mz: float,
    *,
    retention_time: float | None = 100.0,
    tic: float = 1.0,
    ion_mode: str = "positive",
    adduct: str = "[M+H]+",
    charge: int = 1,
) -> Spectrum:
    metadata: dict[str, object] = {
        "id": spec_id,
        "precursor_mz": precursor_mz,
        "adduct": adduct,
        "charge": charge,
        "ionmode": ion_mode,
    }
    if retention_time is not None:
        metadata["retention_time"] = retention_time
    return Spectrum(
        mz=_BASE_MZ.copy(),
        intensities=(_BASE_INT * tic).astype(np.float64),
        metadata=metadata,
    )


def _cfg(**overrides: object) -> NetworkConfig:
    payload: dict[str, object] = {"enabled": True}
    payload.update(overrides)
    return NetworkConfig(**payload)  # type: ignore[arg-type]


def _feature_of(spectra: list[Spectrum], cfg: NetworkConfig):
    node_ids = [str(spectrum_node_id(s)) for s in spectra]
    return build_features(spectra, node_ids, cfg), node_ids


def test_groups_close_ions_into_one_feature() -> None:
    spectra = [_spectrum("a", 500.0), _spectrum("b", 500.005)]
    features, _ = _feature_of(spectra, _cfg())
    assert len(features) == 1
    assert len(features[0].spectrum_node_ids) == 2


def test_representative_is_the_highest_tic_member() -> None:
    spectra = [_spectrum("a", 500.0, tic=1.0), _spectrum("b", 500.005, tic=3.0)]
    features, node_ids = _feature_of(spectra, _cfg())
    assert features[0].representative_spectrum_node_id == node_ids[1]


def test_different_adduct_is_a_different_feature() -> None:
    spectra = [
        _spectrum("a", 500.0, adduct="[M+H]+"),
        _spectrum("b", 500.005, adduct="[M+Na]+"),
    ]
    features, _ = _feature_of(spectra, _cfg())
    assert len(features) == 2


def test_different_charge_is_a_different_feature() -> None:
    spectra = [
        _spectrum("a", 500.0, charge=1),
        _spectrum("b", 500.005, charge=2),
    ]
    features, _ = _feature_of(spectra, _cfg())
    assert len(features) == 2


def test_retention_time_window_splits_features() -> None:
    spectra = [
        _spectrum("a", 500.0, retention_time=100.0),
        _spectrum("b", 500.005, retention_time=200.0),
    ]
    features, _ = _feature_of(spectra, _cfg(feature_rt_tolerance=30.0))
    assert len(features) == 2


def test_missing_rt_does_not_block_mz_grouping() -> None:
    spectra = [
        _spectrum("a", 500.0, retention_time=None),
        _spectrum("b", 500.005, retention_time=None),
    ]
    features, _ = _feature_of(spectra, _cfg(feature_rt_tolerance=30.0))
    assert len(features) == 1


def test_precursor_window_is_respected() -> None:
    spectra = [_spectrum("a", 500.0), _spectrum("b", 500.5)]
    features, _ = _feature_of(spectra, _cfg(feature_precursor_tolerance=0.01))
    assert len(features) == 2


def test_feature_construction_is_deterministic() -> None:
    spectra = [
        _spectrum("a", 500.0),
        _spectrum("b", 500.005, tic=2.0),
        _spectrum("c", 700.0),
    ]
    first, _ = _feature_of(spectra, _cfg())
    second, _ = _feature_of(spectra, _cfg())
    assert [f.feature_id for f in first] == [f.feature_id for f in second]
    assert [f.spectrum_node_ids for f in first] == [f.spectrum_node_ids for f in second]


def test_disabled_feature_building_returns_nothing() -> None:
    spectra = [_spectrum("a", 500.0), _spectrum("b", 500.005)]
    features, _ = _feature_of(spectra, _cfg(build_features=False))
    assert features == []


def test_node_ids_must_align_with_spectra() -> None:
    with pytest.raises(ValueError):
        build_features([_spectrum("a", 500.0)], [], _cfg())


def test_graph_attaches_feature_membership() -> None:
    spectra = [_spectrum("a", 500.0), _spectrum("b", 500.005)]
    graph = build_spectral_graph(spectra, _cfg())
    assert isinstance(graph, MolecularGraph)
    assert len(graph.features) == 1
    feature_id = graph.features[0].feature_id
    assert all(node.feature_id == feature_id for node in graph.nodes)
    assert graph.features[0].feature_id in {
        feature.feature_id for feature in graph.features
    }


def test_graph_with_features_round_trips() -> None:
    spectra = [
        _spectrum("a", 500.0, retention_time=100.0),
        _spectrum("b", 500.005, retention_time=101.0),
        _spectrum("c", 500.0, retention_time=220.0),
    ]
    graph = build_spectral_graph(spectra, _cfg())
    assert len(graph.features) == 2
    restored = MolecularGraph.from_json(graph.to_json())
    assert restored.to_dict() == graph.to_dict()
