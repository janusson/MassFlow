"""
Model construction and validation tests for the MassFlow graph data layer.

Covers Pydantic conventions (``extra="forbid"``, frozen models), float64/missing
semantics, unit-explicit retention time, and the relationship factory methods.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from matchms import Spectrum
from pydantic import ValidationError

from MassFlow.network import (
    AnnotationEvidence,
    ChemicalRelationship,
    Feature,
    FdrAssessment,
    GraphNode,
    IonIdentityRelationship,
    NeutralLoss,
    RelationshipProvenance,
    SpectralRelationship,
    feature_node_id,
    library_node_id,
    spectrum_node_id,
)

pytestmark = pytest.mark.unit


def _spectrum(
    precursor_mz: float = 300.0,
    spec_id: str = "q1",
    **extra: object,
) -> Spectrum:
    metadata: dict[str, object] = {
        "id": spec_id,
        "precursor_mz": precursor_mz,
        "retention_time": 12.5,
        "adduct": "[M+H]+",
        "charge": 1,
    }
    metadata.update(extra)
    return Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([0.5, 1.0], dtype=np.float64),
        metadata=metadata,
    )


def _provenance(algorithm: str = "cosine") -> RelationshipProvenance:
    return RelationshipProvenance(algorithm=algorithm)


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------


def test_graph_node_from_query_spectrum() -> None:
    node = GraphNode.from_query_spectrum(_spectrum())
    assert node.node_kind == "query_spectrum"
    assert node.precursor_mz == 300.0
    assert isinstance(node.precursor_mz, float)
    assert node.retention_time_seconds == 12.5
    assert node.num_peaks == 2
    assert node.node_id == spectrum_node_id(_spectrum())


def test_graph_node_from_library_spectrum_uses_build_id() -> None:
    node = GraphNode.from_library_spectrum(
        _spectrum(spec_id="ref_1"), library_build_id=42
    )
    assert node.node_kind == "library_spectrum"
    assert node.library_build_id == 42
    assert node.node_id == library_node_id(42, "ref_1")


def test_graph_node_missing_retention_time_is_none_not_zero() -> None:
    spectrum = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata={"id": "x", "precursor_mz": 250.0},
    )
    node = GraphNode.from_query_spectrum(spectrum)
    assert node.retention_time_seconds is None


def test_graph_node_rejects_non_positive_precursor() -> None:
    spectrum = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata={"id": "x", "precursor_mz": 0.0},
    )
    with pytest.raises(ValueError):
        GraphNode.from_query_spectrum(spectrum)


def test_graph_node_is_frozen_and_forbids_extra() -> None:
    node = GraphNode.from_query_spectrum(_spectrum())
    with pytest.raises(ValidationError):
        node.precursor_mz = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        GraphNode(  # type: ignore[call-arg]
            node_id=node.node_id,
            node_kind="query_spectrum",
            precursor_mz=300.0,
            unexpected_field=1,
        )


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------


def _feature(**overrides: object) -> Feature:
    node_id = feature_node_id(precursor_mz=195.0, retention_time_seconds=60.0)
    member = spectrum_node_id(_spectrum())
    payload: dict[str, object] = {
        "feature_id": node_id,
        "precursor_mz": 195.0,
        "retention_time_seconds": 60.0,
        "spectrum_node_ids": (member,),
    }
    payload.update(overrides)
    return Feature(**payload)  # type: ignore[arg-type]


def test_feature_accepts_single_member() -> None:
    feature = _feature()
    assert feature.spectrum_node_ids


def test_feature_requires_at_least_one_member() -> None:
    with pytest.raises(ValidationError):
        _feature(spectrum_node_ids=())


def test_feature_members_must_be_unique() -> None:
    member = spectrum_node_id(_spectrum())
    with pytest.raises(ValidationError):
        _feature(spectrum_node_ids=(member, member))


def test_feature_representative_must_be_a_member() -> None:
    member = spectrum_node_id(_spectrum())
    other = spectrum_node_id(_spectrum(precursor_mz=301.0, spec_id="other"))
    assert (
        _feature(
            spectrum_node_ids=(member,), representative_spectrum_node_id=member
        ).representative_spectrum_node_id
        == member
    )
    with pytest.raises(ValidationError):
        _feature(spectrum_node_ids=(member,), representative_spectrum_node_id=other)


# ---------------------------------------------------------------------------
# FdrAssessment / AnnotationEvidence
# ---------------------------------------------------------------------------


def test_fdr_assessment_bounds() -> None:
    assert (
        FdrAssessment(q_value=0.0, calibrated=True, library_size=10).competition_unit
        == "query"
    )
    with pytest.raises(ValidationError):
        FdrAssessment(q_value=1.5, calibrated=True, library_size=10)
    with pytest.raises(ValidationError):
        FdrAssessment(q_value=0.01, calibrated=True, library_size=-1)


def test_annotation_evidence_from_search_result() -> None:
    query = spectrum_node_id(_spectrum())
    reference = library_node_id(1, "ref_1")
    evidence = AnnotationEvidence.from_search_result(
        {
            "score": 0.93,
            "matched_peaks": 12,
            "mass_error_ppm": 1.2,
            "annotation_tier": "Tier 2",
            "score_breakdown": {"cosine": 1.0, "matched": 12.0},
        },
        query_node_id=query,
        reference_node_id=reference,
        algorithm="modified_cosine",
    )
    assert evidence.score == 0.93
    assert evidence.matched_peaks == 12
    assert evidence.score_breakdown == {"cosine": 1.0, "matched": 12.0}
    assert isinstance(evidence.score, float)


def test_annotation_evidence_requires_numeric_score() -> None:
    with pytest.raises(ValueError):
        AnnotationEvidence.from_search_result(
            {"score": None},
            query_node_id=spectrum_node_id(_spectrum()),
            reference_node_id=library_node_id(1, "r"),
            algorithm="cosine",
        )


def test_annotation_evidence_score_is_bounded() -> None:
    with pytest.raises(ValidationError):
        AnnotationEvidence(
            query_node_id=spectrum_node_id(_spectrum()),
            reference_node_id=library_node_id(1, "r"),
            algorithm="cosine",
            score=1.4,
            matched_peaks=1,
        )


def test_annotation_evidence_has_no_fdr_field() -> None:
    assert "q_value" not in AnnotationEvidence.model_fields
    assert "p_value" not in AnnotationEvidence.model_fields
    assert "fdr" not in AnnotationEvidence.model_fields


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


def test_spectral_relationship_create_and_defaults() -> None:
    source = spectrum_node_id(_spectrum())
    target = library_node_id(1, "ref_1")
    rel = SpectralRelationship.create(
        source_node_id=source,
        target_node_id=target,
        score=0.88,
        matched_peaks=9,
        algorithm="cosine",
        provenance=_provenance(),
        ms2_tolerance=0.02,
    )
    assert rel.relationship_type == "spectral"
    assert rel.molecule_relationship == "unknown"
    assert rel.directed is True
    assert rel.library_build_id is None


def test_spectral_relationship_rejects_hand_written_id() -> None:
    with pytest.raises(ValidationError):
        SpectralRelationship(
            relationship_id="rid1:spectral:" + "0" * 32,
            source_node_id=spectrum_node_id(_spectrum()),
            target_node_id=library_node_id(1, "ref_1"),
            score=0.5,
            matched_peaks=3,
            algorithm="cosine",
            provenance=_provenance(),
        )


def test_chemical_relationship_defaults_to_hypothesis() -> None:
    rel = ChemicalRelationship.create(
        source_node_id=spectrum_node_id(_spectrum()),
        target_node_id=spectrum_node_id(_spectrum(precursor_mz=301.0, spec_id="q2")),
        delta_mass_da=-18.010565,
        provenance=_provenance("neutral_loss"),
        neutral_loss=NeutralLoss(label="H2O", formula="H2O", mass_da=18.010565),
    )
    assert rel.interpretation_status == "hypothesis"
    assert rel.molecule_relationship == "related_molecule"
    assert rel.neutral_loss is not None
    assert rel.neutral_loss.label == "H2O"


def test_ion_identity_same_vs_related_molecule() -> None:
    source = spectrum_node_id(_spectrum())
    target = spectrum_node_id(_spectrum(precursor_mz=301.0, spec_id="q2"))
    adduct = IonIdentityRelationship.create(
        source_node_id=source,
        target_node_id=target,
        relationship_kind="adduct",
        delta_mass_da=21.981943,
        provenance=_provenance("adduct_offset"),
        adduct_source="[M+H]+",
        adduct_target="[M+Na]+",
        charge_source=1,
        charge_target=1,
        expected_delta_mass_da=21.981943,
    )
    fragment = IonIdentityRelationship.create(
        source_node_id=source,
        target_node_id=target,
        relationship_kind="in_source_fragment",
        delta_mass_da=-18.010565,
        provenance=_provenance("adduct_offset"),
    )
    assert adduct.molecule_relationship == "same_molecule"
    assert fragment.molecule_relationship == "related_molecule"


def test_relationship_identity_differs_by_type() -> None:
    source = spectrum_node_id(_spectrum())
    target = spectrum_node_id(_spectrum(precursor_mz=301.0, spec_id="q2"))
    spectral = SpectralRelationship.create(
        source_node_id=source,
        target_node_id=target,
        score=0.9,
        matched_peaks=5,
        algorithm="cosine",
        provenance=_provenance(),
    )
    chemical = ChemicalRelationship.create(
        source_node_id=source,
        target_node_id=target,
        delta_mass_da=0.0,
        provenance=_provenance("neutral_loss"),
    )
    assert spectral.relationship_id != chemical.relationship_id


def test_relationship_rejects_non_finite_delta_mass() -> None:
    with pytest.raises(ValidationError):
        ChemicalRelationship.create(
            source_node_id=spectrum_node_id(_spectrum()),
            target_node_id=spectrum_node_id(
                _spectrum(precursor_mz=301.0, spec_id="q2")
            ),
            delta_mass_da=math.nan,
            provenance=_provenance("neutral_loss"),
        )
