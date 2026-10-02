"""
MassFlow graph-compatible data layer and (post-1.0) networking package.

The package re-exports the stable **data layer** (see
:mod:`MassFlow.network.models`) so existing imports such as
``from MassFlow.network import MolecularGraph`` continue to work unchanged.

Algorithmic networking modules are imported lazily (via module ``__getattr__``)
so that importing the package — or importing :mod:`MassFlow.config`, which
depends on :class:`~MassFlow.network.config.NetworkConfig` — never pulls in the
similarity engine or any heavy surface. This preserves the invariant that the
stable annotation path has no import-time dependency on networking code.

Layering
--------
* :mod:`MassFlow.network.models` — nodes, features, layered evidence, typed
  relationships, provenance, and the data-only ``MolecularGraph`` container.
* :mod:`MassFlow.network.candidates` — deterministic MS1/RT candidate pairs.
* :mod:`MassFlow.network.features` — LC-MS feature identity (Stage 2).
* :mod:`MassFlow.network.ion_identity` — adduct / same-molecule links (Stage 3).
* :mod:`MassFlow.network.chemical` — neutral-loss hypotheses (Stage 4).
* :mod:`MassFlow.network.families` — deterministic connected components.
* :mod:`MassFlow.network.context` — non-circular contextualization (Stage 5).
* :mod:`MassFlow.network.msmcp` — local, MCP-style read-only interface.
* :mod:`MassFlow.network.spectral` — spectral edge construction (Stage 1).
* :mod:`MassFlow.network.build` — spectra -> ``MolecularGraph``.

The ``NetworkConfig`` model lives with the other configuration sections in
:mod:`MassFlow.config` (disabled by default); this package never imports it at
runtime, only for typing.
"""

from __future__ import annotations

from typing import Any

from MassFlow.network.models import (
    SCHEMA_VERSION,
    AnnotationEvidence,
    ChemicalRelationship,
    Feature,
    FdrAssessment,
    GraphNode,
    GraphProvenance,
    IonIdentityRelationship,
    IonMode,
    MolecularGraph,
    MoleculeRelationship,
    NetworkContext,
    AnnotationInference,
    MolecularFamily,
    NetworkModel,
    NeutralLoss,
    NodeId,
    NodeIdField,
    Relationship,
    RelationshipBase,
    RelationshipId,
    RelationshipIdField,
    RelationshipProvenance,
    SpectralRelationship,
    feature_node_id,
    library_node_id,
    relationship_id,
    spectrum_node_id,
)

__all__ = [
    "SCHEMA_VERSION",
    "IonMode",
    "MoleculeRelationship",
    "NodeId",
    "RelationshipId",
    "NodeIdField",
    "RelationshipIdField",
    "spectrum_node_id",
    "library_node_id",
    "feature_node_id",
    "relationship_id",
    "NetworkModel",
    "GraphNode",
    "Feature",
    "FdrAssessment",
    "AnnotationEvidence",
    "NetworkContext",
    "AnnotationInference",
    "MolecularFamily",
    "NeutralLoss",
    "RelationshipProvenance",
    "RelationshipBase",
    "SpectralRelationship",
    "ChemicalRelationship",
    "IonIdentityRelationship",
    "Relationship",
    "GraphProvenance",
    "MolecularGraph",
    "generate_candidate_pairs",
    "build_features",
    "build_ion_identity_relationships",
    "build_chemical_relationships",
    "build_spectral_relationships",
    "build_spectral_graph",
    "connected_components",
    "detect_families",
    "contextualize",
    "apply_context",
    "seeds_from_results",
    "LocalGraphSource",
]

# Lazily-resolved algorithmic entry points (kept out of the import graph so the
# package and MassFlow.config stay import-light; see the package docstring).
_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "generate_candidate_pairs": (
        "MassFlow.network.candidates",
        "generate_candidate_pairs",
    ),
    "build_spectral_relationships": (
        "MassFlow.network.spectral",
        "build_spectral_relationships",
    ),
    "build_features": ("MassFlow.network.features", "build_features"),
    "build_ion_identity_relationships": (
        "MassFlow.network.ion_identity",
        "build_ion_identity_relationships",
    ),
    "build_chemical_relationships": (
        "MassFlow.network.chemical",
        "build_chemical_relationships",
    ),
    "connected_components": (
        "MassFlow.network.families",
        "connected_components",
    ),
    "detect_families": ("MassFlow.network.families", "detect_families"),
    "contextualize": ("MassFlow.network.context", "contextualize"),
    "apply_context": ("MassFlow.network.context", "apply_context"),
    "seeds_from_results": ("MassFlow.network.context", "seeds_from_results"),
    "LocalGraphSource": ("MassFlow.network.msmcp", "LocalGraphSource"),
    "build_spectral_graph": ("MassFlow.network.build", "build_spectral_graph"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRIBUTES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute = target
    return getattr(importlib.import_module(module_name), attribute)
