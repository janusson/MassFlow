# Network (Graph Data Model)

The `MassFlow.network` module defines MassFlow's graph-compatible data layer
(stable identifiers, graph nodes, LC-MS features, layered annotation evidence,
typed relationships, provenance, and the data-only `MolecularGraph` container
with lossless JSON (de)serialization) and the experimental **Stage 1–2
networking** API (candidate generation, `candidates`; LC-MS feature identity,
`features`; ion-identity / same-molecule links, `ion_identity`; neutral-loss
hypotheses, `chemical`; deterministic components and family records, `families`;
non-circular contextualization, `context`; offline read-only interface, `msmcp`;
spectral edges, `spectral`; graph construction, `build`; one-hop annotation
propagation, `propagation`; unified tiered export + graph serialization,
`export`; exposed via
`massflow network build` / `analyse` / `export`, disabled by default).

Community detection, a networked MCP server, and `RepositoryGraphSource` are
**deferred** (post-1.0). FBMN, GraphML, and visualization are **declared
non-goals** — not implemented and not planned. See
[Network-Aware MS Annotation — Graph Data Model](../network-data-model.md) for
the architecture, invariants, and the core-vs-deferred split.

::: MassFlow.network

::: MassFlow.network.features

::: MassFlow.network.ion_identity

::: MassFlow.network.chemical

::: MassFlow.network.context

::: MassFlow.network.families

::: MassFlow.network.msmcp

## Network propagation (experimental, one hop)

`MassFlow.network.propagation` is a small, **DataFrame-oriented** sibling of the
object-graph Stage 5 contextualizer above. It consumes two Polars tables — an
explicit molecular-network **edge list** and the library **search results** — and
returns a new results table enriched with *network-derived* evidence. It is
callable purely through the Python API; it is not wired into the CLI or the
default annotation pipeline.

```python
from MassFlow.network.propagation import (
    NetworkPropagationConfig,
    propagate_annotations,
    validate_network_edges,
)

edges = validate_network_edges(edge_table)          # canonical, validated edges
enriched = propagate_annotations(edges, match_table)  # direct rows + hypotheses
```

### Canonical edge schema

An edge is an undirected, weighted pair of node identifiers sharing one namespace
with the `query_id` column of the results table:

| Column | Type | Meaning |
| --- | --- | --- |
| `source_id` | `Utf8` | node identifier |
| `target_id` | `Utf8` | node identifier |
| `weight` | `Float64` | similarity in `[0, 1]` |
| `edge_id` | `Utf8` | optional; carried for provenance and dedup tie-breaking |

`validate_network_edges` rejects null/empty endpoints and non-finite, negative or
`> 1` weights, drops self-loops, canonicalises direction (undirected), and
de-duplicates edges keeping the maximum weight.

### Neighbour-voting and conflict rules

* Only **direct** hits seed propagation. A row seeds when it is not a decoy,
  `score >= seed_min_score`, and (unless disabled) `q_value <= seed_max_q_value`.
* Each candidate's support is `Σ edge_weight × seed_score` over the node's
  seed neighbours (or `Σ edge_weight` when `weight_seed_by_score` is false). The
  score is used as a similarity **magnitude**, never as a probability.
* Support is **unnormalised and degree-dependent**: it is *not* bounded by `1`, and
  a high-degree hub accumulates more support than a sparsely connected node. The
  thresholds below are therefore relative to local connectivity, and raw support
  values must not be compared across nodes of different degree.
* **Conflict is two distinct rules**, both surfaced through `network_status`:
    * *Ambiguity (network-only nodes)* — a node whose leading candidate holds
      `< conflict_min_support_share` of the qualifying support (or ties within
      `conflict_tie_tolerance`) is flagged `network_conflicted`; every competing
      candidate is preserved and none is promoted.
    * *Disagreement (direct nodes)* — a node that already has a direct match is
      flagged `direct_match_conflict` when a **different** candidate is its leading
      network-supported candidate, regardless of that candidate's share. The direct
      assignment is left untouched and still tiers as a direct hit; only the
      disagreement is surfaced. The asymmetry is deliberate: a direct hit is a real
      identification, the network candidates are hypotheses.
* Propagation is **one hop**: derived candidates are never re-used as seeds.

### Direct identifications vs. network hypotheses

The output preserves every input row and its direct-match fields (`score`,
`q_value`, `p_value`, `is_decoy`, …) **unchanged**. Added columns are:
`evidence_source` (`direct` | `network`), `is_direct_match`, `candidate_id`,
`candidate_inchikey`, `network_support_score`, `supporting_neighbor_count`,
`supporting_seed_ids`, `network_status`, `network_conflict`,
`network_conflicting_candidates`, and `network_rank`.

A network row carries `evidence_source == "network"` and **null** direct-match
fields: it is a hypothesis, not a library identification. A node with a direct
match is never overwritten by neighbour votes; when its neighbours disagree the
direct row is flagged (`network_status == "direct_match_conflict"`). See
[Network-Aware MS Annotation — Graph Data Model](../network-data-model.md) §7 for
the confidence-governance invariants this preserves.

::: MassFlow.network.propagation

## Unified annotation export & graph serialization (experimental)

`MassFlow.network.export` turns propagation results into a single tiered table
and a visualization-ready graph. It is **Python-API only** (no CLI) and writes
files through `MassFlow.io.save_network_graph`.

```python
from MassFlow.network.export import (
    export_network_annotations,
    export_network_graph,
)

table = export_network_annotations(matches, propagated)   # tiered consensus
export_network_graph(edges, table, "network.graphml")      # GraphML (graph extra)
export_network_graph(edges, table, "network.json", format="json")  # dependency-free
```

### Confidence tiers

`confidence_tier` is descriptive and never re-derives a score or q-value:

| Tier | Label | Rule |
| --- | --- | --- |
| 1 | `direct_high_confidence` | non-decoy direct match with `q_value ≤ 0.01` |
| 2 | `network_supported` | unambiguous network consensus (`network_status == "network_supported"`) |
| 3 | `network_conflicted_or_low_support` | conflicted (`network_conflicted` / `direct_match_conflict`) or any other network hypothesis |
| — | `null` | decoys and direct hits that are neither high-confidence nor network-supported |

### Graph attributes

Each node carries `annotation_status`, `reference_id` (the library entry id;
`candidate_id` / `candidate_inchikey` hold the compound identity),
`confidence_tier`, `primary_score` plus `primary_score_kind` (`match_score` — a
similarity in `[0, 1]` — or `network_support` — an unbounded support sum),
`evidence_source`, `network_support_score`, `supporting_neighbor_count`, and the
conflict metadata (`network_conflict`, `conflict_count`, `conflicting_candidates`).
Edge `weight` is preserved. GraphML requires the optional `graph` extra; JSON
node-link requires nothing extra.

::: MassFlow.network.export
