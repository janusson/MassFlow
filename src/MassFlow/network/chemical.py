"""
Stage 4 chemical relationships: precursor Δmass as **explicit hypotheses**.

Two LC-MS features are *related molecules* when their neutral masses differ by a
chemically interpretable neutral fragment (water, ammonia, CO₂, a hexose, …).
This module links such features and records the relation as a **hypothesis** —
never as an automatic structural claim.

Method (classical, deterministic)
---------------------------------
1. Each feature's neutral mass is resolved from its precursor m/z and adduct
   (:func:`MassFlow.network.ion_identity.resolve_ion_context`); features whose
   adduct is unknown or inconsistent are skipped (fail closed).
2. For each feature, every neutral fragment in the curated table
   (:data:`NEUTRAL_LOSSES`) is looked up as a *loss* to a lower-mass feature
   within ``chemical_ppm_tolerance`` (binary search over sorted neutral masses).
3. A match yields a :class:`~MassFlow.network.models.ChemicalRelationship` with
   ``interpretation_status="hypothesis"``, ``molecule_relationship="related_molecule"``
   and an explicit :class:`~MassFlow.network.models.NeutralLoss`.

Fragment masses are computed with ``pyteomics`` (the repository's single source
of truth for exact masses). Relationships are confined to a single ionisation
mode. A relationship carries no similarity score and no statistical confidence.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pyteomics.mass as pmass

from MassFlow.network.ion_identity import IonContext, resolve_ion_context
from MassFlow.network.models import (
    ChemicalRelationship,
    NeutralLoss,
    RelationshipProvenance,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MassFlow.config import NetworkConfig

logger = logging.getLogger(__name__)

__all__ = [
    "NEUTRAL_LOSSES",
    "neutral_loss_mass",
    "build_chemical_relationships",
]

#: Curated common neutral losses / transformations, as ``(label, formula)``.
#: Deliberately a small, explicit table of well-known fragmentations rather than
#: a free-form reaction predictor.
NEUTRAL_LOSSES: tuple[tuple[str, str], ...] = (
    ("H2O", "H2O"),
    ("NH3", "NH3"),
    ("CO", "CO"),
    ("CO2", "CO2"),
    ("CH2O", "CH2O"),
    ("CH2", "CH2"),
    ("CH4", "CH4"),
    ("C2H2", "C2H2"),
    ("C2H4", "C2H4"),
    ("HCN", "HCN"),
    ("H2S", "H2S"),
    ("SO2", "SO2"),
    ("SO3", "SO3"),
    ("O", "O"),
    ("HCl", "HCl"),
    ("C6H10O5", "C6H10O5"),  # anhydrohexose (glycosidic loss)
    ("C6H12O6", "C6H12O6"),  # hexose (glycosylation)
)


@lru_cache(maxsize=None)
def neutral_loss_mass(formula: str) -> float:
    """
    Return the monoisotopic mass (Da) of a neutral fragment formula.

    Parameters
    ----------
    formula : str
        A chemical formula (e.g. ``"H2O"``, ``"C6H12O6"``), parsed by pyteomics.

    Returns
    -------
    float
        The monoisotopic mass in Da.
    """
    return float(pmass.calculate_mass(formula=formula))


@lru_cache(maxsize=1)
def _table() -> tuple[tuple[str, str, float], ...]:
    """Return ``(label, formula, fragment_mass)`` rows for the loss table."""
    return tuple(
        (label, formula, neutral_loss_mass(formula))
        for label, formula in NEUTRAL_LOSSES
    )


def _make_relationship(
    source: IonContext,
    target: IonContext,
    label: str,
    formula: str,
    fragment_mass: float,
    ppm_error: float,
    provenance: RelationshipProvenance,
) -> ChemicalRelationship:
    """Build one hypothesized neutral-loss relationship (larger -> smaller)."""
    observed_loss = source.neutral_mass - target.neutral_mass
    return ChemicalRelationship.create(
        source_node_id=source.feature_id,
        target_node_id=target.feature_id,
        delta_mass_da=target.neutral_mass - source.neutral_mass,
        provenance=provenance,
        directed=False,
        molecule_relationship="related_molecule",
        neutral_loss=NeutralLoss(
            label=label,
            formula=formula,
            mass_da=observed_loss,
            mass_error_ppm=ppm_error,
        ),
        transformation=f"loss:{label}",
        interpretation_status="hypothesis",
    )


def build_chemical_relationships(
    features: Sequence[object],
    cfg: "NetworkConfig",
    *,
    provenance: RelationshipProvenance,
) -> list[ChemicalRelationship]:
    """
    Hypothesize neutral-loss relationships between LC-MS features.

    Parameters
    ----------
    features : sequence of Feature
        Features from :func:`MassFlow.network.features.build_features`.
    cfg : NetworkConfig
        Provides ``build_chemical`` (the on/off switch) and
        ``chemical_ppm_tolerance`` (fragment-mass agreement, ppm).
    provenance : RelationshipProvenance
        Per-edge provenance recorded on every emitted relationship.

    Returns
    -------
    list of ChemicalRelationship
        Hypothesized neutral-loss relationships (``interpretation_status="hypothesis"``),
        in canonical ``(source, target)`` order. Deterministic.
    """
    if not cfg.build_chemical or len(features) < 2:
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

    tolerance_ppm = cfg.chemical_ppm_tolerance
    table = _table()

    # Confine neutral-loss links to a single ionisation mode.
    buckets: dict[Optional[str], list[IonContext]] = {}
    for context in contexts:
        buckets.setdefault(context.ion_mode, []).append(context)

    # Best (lowest-ppm) hypothesis per unordered feature pair.
    best: dict[tuple[str, str], tuple[float, ChemicalRelationship]] = {}

    for group in buckets.values():
        ordered = sorted(group, key=lambda ctx: (ctx.neutral_mass, ctx.feature_id))
        masses = np.array([ctx.neutral_mass for ctx in ordered], dtype=np.float64)
        for source in ordered:
            for label, formula, fragment_mass in table:
                target_mass = source.neutral_mass - fragment_mass
                if target_mass <= 0.0:
                    continue
                # The tolerance is defined against the *fragment* mass (see the
                # exact check below), so the prefilter window must use
                # fragment_mass too. Sizing it on target_mass
                # (= source.neutral_mass - fragment_mass) would narrow the window
                # to (target_mass / fragment_mass) x the tolerance and silently
                # drop real relationships whenever the remaining target is
                # lighter than the loss (e.g. an H2O loss from a 30 Da molecule
                # admits only ~3.3 ppm instead of 5).
                window = fragment_mass * tolerance_ppm * 1e-6
                lower = int(np.searchsorted(masses, target_mass - window, side="left"))
                upper = int(np.searchsorted(masses, target_mass + window, side="right"))
                for candidate_index in range(lower, upper):
                    target = ordered[candidate_index]
                    if target.feature_id == source.feature_id:
                        continue
                    if target.neutral_mass >= source.neutral_mass:
                        continue  # orient the relationship larger -> smaller
                    observed_loss = source.neutral_mass - target.neutral_mass
                    ppm_error = abs(observed_loss - fragment_mass) / fragment_mass * 1e6
                    if ppm_error > tolerance_ppm:
                        continue
                    pair = (source.feature_id, target.feature_id)
                    previous = best.get(pair)
                    if previous is not None and previous[0] <= ppm_error:
                        continue
                    best[pair] = (
                        ppm_error,
                        _make_relationship(
                            source,
                            target,
                            label,
                            formula,
                            fragment_mass,
                            ppm_error,
                            provenance,
                        ),
                    )

    relationships = [relationship for _, relationship in best.values()]
    relationships.sort(
        key=lambda rel: (rel.source_node_id, rel.target_node_id, rel.relationship_id)
    )
    logger.debug(
        "Chemical networking: %d feature(s), %d neutral-loss hypothesis(es).",
        len(contexts),
        len(relationships),
    )
    return relationships
