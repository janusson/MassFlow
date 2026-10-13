"""
Stage 4 chemical-relationship tests (neutral-loss hypotheses).

These use caffeine (C8H10N4O2, 194.0803755 Da) and its dehydration product as the
known-answer pair, plus synthetic Δmass fixtures. A neutral-loss relationship must
be labeled as an explicit *hypothesis* and must never be presented as a
structural claim or a confidence value.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.cheminformatics import adduct_charge, compute_adduct_offset
from MassFlow.config import NetworkConfig
from MassFlow.network.build import build_spectral_graph
from MassFlow.network.chemical import (
    NEUTRAL_LOSSES,
    build_chemical_relationships,
    neutral_loss_mass,
)
from MassFlow.network.models import (
    ChemicalRelationship,
    Feature,
    MolecularGraph,
    RelationshipProvenance,
    feature_node_id,
    library_node_id,
)

pytestmark = pytest.mark.unit

_CAFFEINE = 194.0803755
_WATER = neutral_loss_mass("H2O")


def _precursor_mz(neutral_mass: float, adduct: str) -> float:
    offset = compute_adduct_offset(adduct)
    charge = adduct_charge(adduct)
    assert offset is not None and charge is not None
    return (neutral_mass + offset) / abs(charge)


def _feature(
    neutral_mass: float,
    adduct: str = "[M+H]+",
    *,
    index: int,
    ion_mode: str = "positive",
    mass_shift: float = 0.0,
) -> Feature:
    precursor_mz = _precursor_mz(neutral_mass, adduct) + mass_shift
    charge = adduct_charge(adduct)
    member = str(library_node_id(1, f"member_{index}"))
    return Feature(
        feature_id=str(
            feature_node_id(
                precursor_mz=precursor_mz,
                retention_time_seconds=100.0,
                charge=charge,
                adduct=adduct,
                ion_mode=ion_mode,
            )
        ),
        precursor_mz=precursor_mz,
        retention_time_seconds=100.0,
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


def _build(features: list[Feature], cfg: NetworkConfig) -> list[ChemicalRelationship]:
    return build_chemical_relationships(
        features, cfg, provenance=RelationshipProvenance(algorithm="neutral_loss_table")
    )


def _loss_feature(
    parent_mass: float, fragment_mass: float, ppm_error: float, *, index: int
) -> Feature:
    """A feature whose neutral mass sits off a ``fragment_mass`` loss by *ppm_error*.

    ``observed_loss = parent_mass - neutral_mass = fragment_mass * (1 + ppm*1e-6)``,
    so the exact ppm check reports ``ppm_error``.
    """
    return _feature(
        parent_mass - fragment_mass,
        index=index,
        mass_shift=-fragment_mass * ppm_error * 1e-6,
    )


def test_neutral_loss_table_masses_are_physical() -> None:
    assert neutral_loss_mass("H2O") == pytest.approx(18.010565, abs=1e-4)
    assert neutral_loss_mass("CO2") == pytest.approx(43.989829, abs=1e-4)
    labels = {label for label, _ in NEUTRAL_LOSSES}
    assert {"H2O", "NH3", "CO2", "C6H12O6"} <= labels


def test_known_answer_caffeine_dehydration_is_a_hypothesis() -> None:
    parent = _feature(_CAFFEINE, index=0)
    fragment = _feature(_CAFFEINE - _WATER, index=1)

    relationships = _build([parent, fragment], _cfg())

    assert len(relationships) == 1
    relationship = relationships[0]
    assert relationship.relationship_type == "chemical"
    assert relationship.interpretation_status == "hypothesis"
    assert relationship.molecule_relationship == "related_molecule"
    assert relationship.transformation == "loss:H2O"
    assert relationship.neutral_loss is not None
    assert relationship.neutral_loss.label == "H2O"
    assert relationship.neutral_loss.formula == "H2O"
    assert relationship.neutral_loss.mass_error_ppm == pytest.approx(0.0, abs=1e-6)
    # Orientation: larger (parent) -> smaller (fragment); delta is negative.
    assert relationship.source_node_id == parent.feature_id
    assert relationship.target_node_id == fragment.feature_id
    assert relationship.delta_mass_da == pytest.approx(-_WATER, abs=1e-6)


def test_co2_loss_is_recognized() -> None:
    parent = _feature(300.0, index=0)
    fragment = _feature(300.0 - neutral_loss_mass("CO2"), index=1)
    relationships = _build([parent, fragment], _cfg())
    assert len(relationships) == 1
    assert relationships[0].transformation == "loss:CO2"


def test_uninterpretable_delta_yields_no_relationship() -> None:
    parent = _feature(300.0, index=0)
    other = _feature(250.0, index=1)  # 50 Da: not in the curated loss table
    assert _build([parent, other], _cfg()) == []


def test_ppm_tolerance_is_respected() -> None:
    parent = _feature(300.0, index=0)
    # 0.01 Da error on an 18 Da loss is ~555 ppm, far beyond the 5 ppm default.
    fragment = _feature(300.0 - _WATER + 0.01, index=1)
    assert _build([parent, fragment], _cfg()) == []


def test_ppm_prefilter_is_a_superset_of_the_exact_check() -> None:
    """Regression: the prefilter window must be sized on the *fragment* mass.

    The window used to be ``target_mass * tolerance_ppm * 1e-6`` where
    ``target_mass = parent - fragment``; for an H2O loss from M = 30.0106 the
    target is only ~12.0 Da, so the window admitted ~3.3 ppm instead of 5 and a
    4.9 ppm loss was silently dropped.
    """
    parent_mass = 30.0106
    parent = _feature(parent_mass, index=0)

    inside = _loss_feature(parent_mass, _WATER, 4.9, index=1)
    found = _build([parent, inside], _cfg())
    assert len(found) == 1
    assert found[0].transformation == "loss:H2O"
    assert found[0].neutral_loss is not None
    assert found[0].neutral_loss.mass_error_ppm == pytest.approx(4.9, abs=1e-6)

    beyond = _loss_feature(parent_mass, _WATER, 5.1, index=2)
    assert _build([parent, beyond], _cfg()) == []


def test_hexose_loss_from_a_glycoside_is_not_dropped_by_the_prefilter() -> None:
    """A hexose loss from a ~300 Da precursor must survive the ppm prefilter.

    The target for a 180 Da hexose loss off a 300 Da precursor is only ~120 Da,
    so the old window admitted ~3.3 ppm and a 4.5 ppm loss was silently dropped.
    """
    hexose = neutral_loss_mass("C6H12O6")
    parent_mass = 300.0
    parent = _feature(parent_mass, index=0)
    fragment = _loss_feature(parent_mass, hexose, 4.5, index=1)

    found = _build([parent, fragment], _cfg())
    assert len(found) == 1
    assert found[0].transformation == "loss:C6H12O6"
    assert found[0].neutral_loss is not None
    assert found[0].neutral_loss.mass_error_ppm == pytest.approx(4.5, abs=1e-6)


def test_unknown_adduct_fails_closed() -> None:
    parent = _feature(300.0, index=0)
    fragment = _raw_feature(281.99, "[M+Weird]+", 1, index=1)
    assert _build([parent, fragment], _cfg()) == []


def test_cross_mode_pairs_do_not_link() -> None:
    parent = _feature(300.0, index=0, ion_mode="positive")
    fragment = _feature(300.0 - _WATER, index=1, ion_mode="negative")
    assert _build([parent, fragment], _cfg()) == []


def test_multiply_charged_adducts_use_neutral_masses() -> None:
    # With |z| = 2 the precursor m/z difference is half the neutral-mass
    # difference, so a naive m/z subtraction would miss the H2O loss.
    parent = _feature(300.0, "[M+2H]2+", index=0)
    fragment = _feature(300.0 - _WATER, "[M+2H]2+", index=1)
    relationships = _build([parent, fragment], _cfg())
    assert len(relationships) == 1
    assert relationships[0].transformation == "loss:H2O"
    assert relationships[0].delta_mass_da == pytest.approx(-_WATER, abs=1e-6)


def test_order_independent_and_deterministic() -> None:
    features = [_feature(300.0, index=0), _feature(300.0 - _WATER, index=1)]
    forward = _build(features, _cfg())
    reverse = _build(list(reversed(features)), _cfg())
    assert [r.relationship_id for r in forward] == [r.relationship_id for r in reverse]


def test_disabled_chemical_returns_nothing() -> None:
    features = [_feature(300.0, index=0), _feature(300.0 - _WATER, index=1)]
    assert _build(features, _cfg(build_chemical=False)) == []


def test_chemical_relationship_has_no_confidence_field() -> None:
    for field in ("q_value", "p_value", "fdr", "confidence"):
        assert field not in ChemicalRelationship.model_fields


def _spectrum(spec_id: str, precursor_mz: float) -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0]),
        intensities=np.array([1.0, 0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]),
        metadata={
            "id": spec_id,
            "precursor_mz": precursor_mz,
            "retention_time": 100.0,
            "adduct": "[M+H]+",
            "charge": 1,
            "ionmode": "positive",
        },
    )


def test_graph_includes_chemical_relationship_between_features() -> None:
    spectra = [
        _spectrum("parent", _precursor_mz(_CAFFEINE, "[M+H]+")),
        _spectrum("fragment", _precursor_mz(_CAFFEINE - _WATER, "[M+H]+")),
    ]
    cfg = NetworkConfig(enabled=True, min_score=0.9, min_matched_peaks=8)
    graph = build_spectral_graph(spectra, cfg)

    chemical_edges = [
        rel for rel in graph.relationships if rel.relationship_type == "chemical"
    ]
    assert len(chemical_edges) == 1
    feature_ids = {feature.feature_id for feature in graph.features}
    assert {
        chemical_edges[0].source_node_id,
        chemical_edges[0].target_node_id,
    } <= feature_ids
    assert graph.fdr_assessments == {}

    restored = MolecularGraph.from_json(graph.to_json())
    assert restored.to_dict() == graph.to_dict()
