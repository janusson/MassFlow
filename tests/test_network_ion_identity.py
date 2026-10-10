"""
Stage 3 ion-identity tests (adduct / same-molecule relationships).

Uses caffeine as the known-answer molecule: ``[M+H]+`` (195.0877) and ``[M+Na]+``
(217.0696) must be linked as the same molecule, with a neutral-mass error of
~0 ppm, while unknown adducts, inconsistent charges, cross-mode pairs and
unrelated masses must not link.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.cheminformatics import adduct_charge, compute_adduct_offset
from MassFlow.config import NetworkConfig
from MassFlow.network.build import build_spectral_graph
from MassFlow.network.ion_identity import build_ion_identity_relationships
from MassFlow.network.models import (
    Feature,
    IonIdentityRelationship,
    MolecularGraph,
    RelationshipProvenance,
    feature_node_id,
    library_node_id,
)

pytestmark = pytest.mark.unit

# Caffeine (C8H10N4O2) monoisotopic mass, Da.
_CAFFEINE_MONOISOTOPIC_MASS = 194.0803755


def _precursor_mz(neutral_mass: float, adduct: str) -> float:
    offset = compute_adduct_offset(adduct)
    charge = adduct_charge(adduct)
    assert offset is not None and charge is not None
    return (neutral_mass + offset) / abs(charge)


def _feature(
    neutral_mass: float,
    adduct: str,
    *,
    index: int,
    ion_mode: str = "positive",
    retention_time: float = 100.0,
    mass_shift: float = 0.0,
) -> Feature:
    precursor_mz = _precursor_mz(neutral_mass, adduct) + mass_shift
    charge = adduct_charge(adduct)
    member = str(library_node_id(1, f"member_{index}"))
    return Feature(
        feature_id=str(
            feature_node_id(
                precursor_mz=precursor_mz,
                retention_time_seconds=retention_time,
                charge=charge,
                adduct=adduct,
                ion_mode=ion_mode,
            )
        ),
        precursor_mz=precursor_mz,
        retention_time_seconds=retention_time,
        ion_mode=ion_mode,
        charge=charge,
        adduct=adduct,
        spectrum_node_ids=(member,),
        representative_spectrum_node_id=member,
    )


def _raw_feature(
    precursor_mz: float,
    adduct: str,
    charge: int,
    *,
    index: int,
    ion_mode: str = "positive",
) -> Feature:
    member = str(library_node_id(1, f"raw_{index}"))
    return Feature(
        feature_id=str(
            feature_node_id(
                precursor_mz=precursor_mz,
                retention_time_seconds=None,
                charge=charge,
                adduct=adduct,
                ion_mode=ion_mode,
            )
        ),
        precursor_mz=precursor_mz,
        retention_time_seconds=None,
        ion_mode=ion_mode,
        charge=charge,
        adduct=adduct,
        spectrum_node_ids=(member,),
        representative_spectrum_node_id=member,
    )


def _cfg(**overrides: object) -> NetworkConfig:
    payload: dict[str, object] = {"enabled": True}
    payload.update(overrides)
    return NetworkConfig(**payload)  # type: ignore[arg-type]


def _provenance() -> RelationshipProvenance:
    return RelationshipProvenance(algorithm="adduct_offset")


def _build(
    features: list[Feature], cfg: NetworkConfig
) -> list[IonIdentityRelationship]:
    return build_ion_identity_relationships(features, cfg, provenance=_provenance())


def test_known_answer_caffeine_adducts_are_same_molecule() -> None:
    features = [
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+H]+", index=0),
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+Na]+", index=1),
    ]
    relationships = _build(features, _cfg())
    assert len(relationships) == 1
    relationship = relationships[0]
    assert relationship.relationship_type == "ion_identity"
    assert relationship.relationship_kind == "adduct"
    assert relationship.molecule_relationship == "same_molecule"
    assert relationship.adduct_source == "[M+H]+"
    assert relationship.adduct_target == "[M+Na]+"
    assert relationship.delta_mass_error_ppm == pytest.approx(0.0, abs=1e-6)
    assert relationship.delta_mass_da == pytest.approx(
        relationship.expected_delta_mass_da, abs=1e-6
    )


def test_unrelated_masses_do_not_link() -> None:
    features = [
        _feature(194.08, "[M+H]+", index=0),
        _feature(300.10, "[M+Na]+", index=1),
    ]
    assert _build(features, _cfg()) == []


def test_unknown_adduct_fails_closed() -> None:
    features = [
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+H]+", index=0),
        _raw_feature(217.06, "[M+Weird]+", 1, index=1),
    ]
    assert _build(features, _cfg()) == []


def test_inconsistent_declared_charge_fails_closed() -> None:
    # [M+H]+ is singly positive; a declared charge of 2 is contradictory.
    features = [
        _raw_feature(195.0877, "[M+H]+", 1, index=0),
        _raw_feature(195.0877, "[M+H]+", 2, index=1),
    ]
    assert _build(features, _cfg()) == []


def test_cross_mode_pairs_do_not_link() -> None:
    features = [
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+H]+", index=0, ion_mode="positive"),
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M-H]-", index=1, ion_mode="negative"),
    ]
    assert _build(features, _cfg()) == []


def test_same_adduct_duplicate_ion_is_not_an_adduct_link() -> None:
    features = [
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+H]+", index=0, retention_time=100.0),
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+H]+", index=1, retention_time=500.0),
    ]
    assert _build(features, _cfg()) == []


def test_ppm_tolerance_is_respected() -> None:
    features = [
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+H]+", index=0),
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+Na]+", index=1, mass_shift=0.01),
    ]
    # 0.01 Da on a ~194 Da neutral mass is ~50 ppm, well beyond the 5 ppm default.
    assert _build(features, _cfg()) == []


def test_order_independent_and_deterministic() -> None:
    features = [
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+H]+", index=0),
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+Na]+", index=1),
    ]
    forward = _build(features, _cfg())
    reverse = _build(list(reversed(features)), _cfg())
    assert [r.relationship_id for r in forward] == [r.relationship_id for r in reverse]


def test_disabled_ion_identity_returns_nothing() -> None:
    features = [
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+H]+", index=0),
        _feature(_CAFFEINE_MONOISOTOPIC_MASS, "[M+Na]+", index=1),
    ]
    assert _build(features, _cfg(build_ion_identity=False)) == []


def _spectrum(spec_id: str, precursor_mz: float, adduct: str) -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0]),
        intensities=np.array([1.0, 0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]),
        metadata={
            "id": spec_id,
            "precursor_mz": precursor_mz,
            "retention_time": 100.0,
            "adduct": adduct,
            "charge": 1,
            "ionmode": "positive",
        },
    )


def test_graph_includes_ion_identity_relationship_between_features() -> None:
    spectra = [
        _spectrum("h", _precursor_mz(_CAFFEINE_MONOISOTOPIC_MASS, "[M+H]+"), "[M+H]+"),
        _spectrum(
            "na", _precursor_mz(_CAFFEINE_MONOISOTOPIC_MASS, "[M+Na]+"), "[M+Na]+"
        ),
    ]
    cfg = NetworkConfig(enabled=True, min_score=0.9, min_matched_peaks=8)
    graph = build_spectral_graph(spectra, cfg)

    ion_edges = [
        rel for rel in graph.relationships if rel.relationship_type == "ion_identity"
    ]
    assert len(ion_edges) == 1
    feature_ids = {feature.feature_id for feature in graph.features}
    assert {ion_edges[0].source_node_id, ion_edges[0].target_node_id} <= feature_ids
    assert graph.fdr_assessments == {}

    restored = MolecularGraph.from_json(graph.to_json())
    assert restored.to_dict() == graph.to_dict()


def test_ion_identity_relationship_has_no_confidence_field() -> None:
    for field in ("q_value", "p_value", "fdr", "confidence"):
        assert field not in IonIdentityRelationship.model_fields
