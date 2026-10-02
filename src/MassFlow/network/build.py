"""
Orchestration for building a spectral :class:`~MassFlow.network.models.MolecularGraph`.

This is the P1 (post-1.0) entry point: it projects a sequence of experimental
spectra into graph nodes and connects them with
:class:`~MassFlow.network.models.SpectralRelationship` edges. It is a pure
downstream consumer — it does not run the annotation pipeline, does not read or
write q-values, and does not touch the stable path's state.

Nodes are content-addressed (:func:`~MassFlow.network.models.spectrum_node_id`),
so identical spectra collapse to a single node and self-edges are impossible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from MassFlow.network.chemical import build_chemical_relationships
from MassFlow.network.families import detect_families
from MassFlow.network.features import build_features
from MassFlow.network.ion_identity import build_ion_identity_relationships
from MassFlow.network.models import (
    GraphNode,
    GraphProvenance,
    MolecularGraph,
    Relationship,
    RelationshipProvenance,
    spectrum_node_id,
)
from MassFlow.network.spectral import build_spectral_relationships

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MassFlow.config import NetworkConfig

__all__ = ["build_spectral_graph"]


def build_spectral_graph(
    spectra: Sequence[object],
    cfg: NetworkConfig,
    *,
    config_digest_sha256: Optional[str] = None,
    library_build_id: Optional[int] = None,
) -> MolecularGraph:
    """
    Build a spectral molecular graph from experimental spectra.

    Parameters
    ----------
    spectra : sequence of matchms.Spectrum
        Experimental spectra to network (typically the processed query spectra
        of an annotation run). Input spectra are not mutated.
    cfg : NetworkConfig
        Networking parameters (algorithm, tolerances, thresholds, ``top_k``).
    config_digest_sha256 : str or None, optional
        Digest of the effective MassFlow configuration, recorded in provenance.
    library_build_id : int or None, optional
        Reference-library build row to record, when applicable.

    Returns
    -------
    MolecularGraph
        A data-only graph of query nodes and undirected spectral edges. No FDR
        assessments are attached — networking produces similarity, never
        statistical confidence.

    Examples
    --------
    >>> graph = build_spectral_graph(spectra, NetworkConfig(enabled=True))  # doctest: +SKIP
    """
    # De-duplicate by content-addressed node id (deterministic: first wins).
    unique_spectra: list[object] = []
    node_ids: list[str] = []
    seen: set[str] = set()
    for spectrum in spectra:
        node_id = str(spectrum_node_id(spectrum))  # type: ignore[arg-type]
        if node_id in seen:
            continue
        seen.add(node_id)
        unique_spectra.append(spectrum)
        node_ids.append(node_id)

    nodes = [GraphNode.from_query_spectrum(s) for s in unique_spectra]  # type: ignore[arg-type]

    # Stage 2: group spectra into LC-MS features and stamp membership onto the
    # (frozen) nodes. Feature construction is additive and does not affect edges.
    features = build_features(unique_spectra, node_ids, cfg)
    if features:
        feature_by_node: dict[str, str] = {}
        for feature in features:
            for member in feature.spectrum_node_ids:
                feature_by_node[member] = feature.feature_id
        nodes = [
            node.model_copy(update={"feature_id": feature_by_node.get(node.node_id)})
            for node in nodes
        ]

    edge_provenance = RelationshipProvenance(
        algorithm=cfg.algorithm,
        parameters=cfg.model_dump(),
        config_digest_sha256=config_digest_sha256,
        library_build_id=library_build_id,
        source_module="MassFlow.network.spectral",
    )
    relationships: list[Relationship] = []
    relationships.extend(
        build_spectral_relationships(
            unique_spectra,
            node_ids,
            cfg,
            provenance=edge_provenance,
            library_build_id=library_build_id,
        )
    )

    # Stage 3: same-molecule (adduct) relationships between features. These are a
    # separate relationship type and never touch similarity or confidence.
    if features:
        ion_provenance = RelationshipProvenance(
            algorithm="adduct_offset",
            parameters={"ion_identity_ppm_tolerance": cfg.ion_identity_ppm_tolerance},
            config_digest_sha256=config_digest_sha256,
            library_build_id=library_build_id,
            source_module="MassFlow.network.ion_identity",
        )
        relationships.extend(
            build_ion_identity_relationships(features, cfg, provenance=ion_provenance)
        )

        # Stage 4: hypothesized neutral-loss relationships between features. These
        # are explicit, provenance-carrying hypotheses, never structural claims.
        chemical_provenance = RelationshipProvenance(
            algorithm="neutral_loss_table",
            parameters={"chemical_ppm_tolerance": cfg.chemical_ppm_tolerance},
            config_digest_sha256=config_digest_sha256,
            library_build_id=library_build_id,
            source_module="MassFlow.network.chemical",
        )
        relationships.extend(
            build_chemical_relationships(features, cfg, provenance=chemical_provenance)
        )

    relationships.sort(
        key=lambda rel: (rel.source_node_id, rel.target_node_id, rel.relationship_id)
    )

    graph_provenance = GraphProvenance(
        builder="MassFlow.network.build",
        config_digest_sha256=config_digest_sha256,
        library_build_id=library_build_id,
    )
    graph = MolecularGraph(
        nodes=nodes,
        features=features,
        relationships=relationships,
        provenance=graph_provenance,
    )

    # Stage 6: deterministic molecular-family records over the assembled graph.
    if cfg.build_families:
        families = detect_families(graph)
        graph = MolecularGraph(
            schema_version=graph.schema_version,
            nodes=graph.nodes,
            features=graph.features,
            relationships=graph.relationships,
            fdr_assessments=graph.fdr_assessments,
            network_contexts=graph.network_contexts,
            inferences=graph.inferences,
            families=families,
            provenance=graph.provenance,
        )
    return graph
