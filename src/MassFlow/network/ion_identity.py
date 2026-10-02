"""
Stage 3 ion-identity relationships: **same-molecule** links between LC-MS features.

Two ions are the *same molecule* when they are different adducts of one neutral
mass. This module derives each feature's neutral mass from its precursor m/z and
adduct (:func:`MassFlow.cheminformatics.compute_adduct_offset`) and links
features whose neutral masses agree within a configurable ppm tolerance.

Scope and scientific discipline
-------------------------------
* Only **adduct** relationships are discovered here
  (``relationship_kind="adduct"``, ``molecule_relationship="same_molecule"``).
  Isotope, in-source-fragment and multimer discovery are deferred.
* A feature whose adduct is absent, unknown to the registry, or inconsistent
  with its declared charge is **skipped** (fail closed) — no mass is guessed.
* Relationships are confined to a single ionisation mode; a positive- and a
  negative-mode ion are never linked (conservative).
* A relationship is never a structural or statistical claim: it carries no
  similarity score and no confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from MassFlow.cheminformatics import adduct_charge, compute_adduct_offset
from MassFlow.network.models import IonIdentityRelationship, RelationshipProvenance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MassFlow.config import NetworkConfig

logger = logging.getLogger(__name__)

__all__ = ["IonContext", "resolve_ion_context", "build_ion_identity_relationships"]


@dataclass(frozen=True)
class IonContext:
    """A feature's resolved ion chemistry (neutral mass + adduct accounting)."""

    feature_id: str
    precursor_mz: float
    neutral_mass: float
    offset: float
    charge_signed: int
    charge_abs: int
    ion_mode: Optional[str]
    adduct: str


def resolve_ion_context(feature: object) -> Optional[IonContext]:
    """Resolve a feature's neutral mass, or ``None`` when it cannot be derived."""
    adduct = feature.adduct  # type: ignore[attr-defined]
    if not adduct:
        return None
    offset = compute_adduct_offset(adduct)
    if offset is None:
        return None
    charge_signed = adduct_charge(adduct)
    if charge_signed is None or charge_signed == 0:
        return None

    declared_charge = feature.charge  # type: ignore[attr-defined]
    if declared_charge is not None and int(declared_charge) != charge_signed:
        # Declared charge contradicts the adduct chemistry: fail closed.
        return None

    charge_abs = abs(charge_signed)
    precursor_mz = float(feature.precursor_mz)  # type: ignore[attr-defined]
    neutral_mass = precursor_mz * charge_abs - offset
    if not np.isfinite(neutral_mass) or neutral_mass <= 0.0:
        return None
    return IonContext(
        feature_id=str(feature.feature_id),  # type: ignore[attr-defined]
        precursor_mz=precursor_mz,
        neutral_mass=neutral_mass,
        offset=offset,
        charge_signed=charge_signed,
        charge_abs=charge_abs,
        ion_mode=feature.ion_mode,  # type: ignore[attr-defined]
        adduct=adduct,
    )


def _make_relationship(
    source: IonContext,
    target: IonContext,
    ppm_error: float,
    provenance: RelationshipProvenance,
) -> IonIdentityRelationship:
    """Build one adduct (same-molecule) relationship between two features."""
    expected_target_mz = (source.neutral_mass + target.offset) / target.charge_abs
    return IonIdentityRelationship.create(
        source_node_id=source.feature_id,
        target_node_id=target.feature_id,
        relationship_kind="adduct",
        delta_mass_da=target.precursor_mz - source.precursor_mz,
        provenance=provenance,
        directed=False,
        molecule_relationship="same_molecule",
        adduct_source=source.adduct,
        adduct_target=target.adduct,
        charge_source=source.charge_signed,
        charge_target=target.charge_signed,
        expected_delta_mass_da=expected_target_mz - source.precursor_mz,
        delta_mass_error_ppm=ppm_error,
    )


def build_ion_identity_relationships(
    features: Sequence[object],
    cfg: "NetworkConfig",
    *,
    provenance: RelationshipProvenance,
) -> list[IonIdentityRelationship]:
    """
    Discover adduct (same-molecule) relationships between LC-MS features.

    Parameters
    ----------
    features : sequence of Feature
        Features from :func:`MassFlow.network.features.build_features`.
    cfg : NetworkConfig
        Provides ``build_ion_identity`` (the on/off switch) and
        ``ion_identity_ppm_tolerance`` (neutral-mass agreement, ppm).
    provenance : RelationshipProvenance
        Per-edge provenance recorded on every emitted relationship.

    Returns
    -------
    list of IonIdentityRelationship
        Adduct relationships with ``molecule_relationship="same_molecule"``, in
        canonical ``(source, target)`` order. Deterministic.
    """
    if not cfg.build_ion_identity or len(features) < 2:
        return []

    contexts = [
        context
        for context in (
            resolve_ion_context(feature)
            for feature in sorted(features, key=lambda f: f.feature_id)  # type: ignore[attr-defined]
        )
        if context is not None
    ]
    if len(contexts) < 2:
        return []

    tolerance_ppm = cfg.ion_identity_ppm_tolerance

    # Partition by ionisation mode: same-molecule adduct links never cross modes.
    buckets: dict[Optional[str], list[IonContext]] = {}
    for context in contexts:
        buckets.setdefault(context.ion_mode, []).append(context)

    relationships: list[IonIdentityRelationship] = []
    for group in buckets.values():
        ordered = sorted(group, key=lambda ctx: (ctx.neutral_mass, ctx.feature_id))
        for index, source in enumerate(ordered):
            # The window is monotone in neutral mass because ppm error is measured
            # relative to the source, so the sorted scan may break safely.
            upper_bound = source.neutral_mass * (1.0 + tolerance_ppm * 1e-6)
            for following in range(index + 1, len(ordered)):
                target = ordered[following]
                if target.neutral_mass > upper_bound:
                    break
                if target.adduct == source.adduct:
                    # Same ion (a feature-grouping duplicate), not an adduct link.
                    continue
                ppm_error = (
                    (target.neutral_mass - source.neutral_mass)
                    / source.neutral_mass
                    * 1e6
                )
                if ppm_error > tolerance_ppm:
                    continue
                relationships.append(
                    _make_relationship(source, target, ppm_error, provenance)
                )

    relationships.sort(
        key=lambda rel: (rel.source_node_id, rel.target_node_id, rel.relationship_id)
    )
    logger.debug(
        "Ion-identity networking: %d feature(s), %d adduct relationship(s).",
        len(contexts),
        len(relationships),
    )
    return relationships
