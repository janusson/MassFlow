# Network (Graph Data Model)

The `MassFlow.network` module defines MassFlow's graph-compatible data layer
(stable identifiers, graph nodes, LC-MS features, layered annotation evidence,
typed relationships, provenance, and the data-only `MolecularGraph` container
with lossless JSON (de)serialization) and the experimental **Stage 1–2
networking** API (candidate generation, `candidates`; LC-MS feature identity,
`features`; ion-identity / same-molecule links, `ion_identity`; neutral-loss
hypotheses, `chemical`; deterministic components and family records, `families`;
non-circular contextualization, `context`; offline read-only interface, `msmcp`;
spectral edges, `spectral`; graph construction, `build`; exposed via
`massflow network build` / `analyse` / `export`, disabled by default).

Community detection, a networked MCP server, `RepositoryGraphSource`, FBMN,
GraphML, and visualization are **not** implemented. See
[Network-Aware MS Annotation — Graph Data Model](../network-data-model.md) for
the architecture, invariants, and the core-vs-deferred split.

::: MassFlow.network

::: MassFlow.network.features

::: MassFlow.network.ion_identity

::: MassFlow.network.chemical

::: MassFlow.network.context

::: MassFlow.network.families

::: MassFlow.network.msmcp
