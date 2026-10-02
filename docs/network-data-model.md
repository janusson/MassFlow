# Network-Aware MS Annotation — Graph Data Model

> **Status: data layer (core) + Stage 1–6 networking + local interface (experimental).**
> The graph data layer (`MassFlow.network.models`) is the stable, chain-neutral
> vocabulary. Stages 1–6 — **spectral networking**, **feature identity**,
> **ion-identity (adduct)** relationships, **chemical (neutral-loss)** hypotheses,
> **annotation contextualization** (L5 `NetworkContext` / `AnnotationInference`),
> and **molecular families** (`MolecularFamily`) — plus the offline
> **`LocalGraphSource`** interface shipped experimentally in P1–P6 and are opt-in
> (`network.enabled`, disabled by default). Community detection, a networked MCP
> server, clustering, FBMN, GraphML, and visualization are **not** implemented.

## Purpose

`MassFlow.network` is a strictly downstream, read-only projection of the
existing annotation engine. It exists so a future pipeline —

```
spectra / features → annotation → pairwise relationships → molecular graph
                 → molecular-family analysis → network-aware interpretation
```

— can be built **without** touching the stable annotation path, the meaning of
existing q-values/FDR, or the float64/retention-time/provenance guarantees.

Nothing on the stable path (`workflow`, `cli`, `database`, `io`) imports this
module, and this module imports none of them. The graph layer is a pure
consumer.

## The layered evidence model

The design keeps five concepts structurally separate; the types encode the
separation so it cannot be lost by convention:

| Layer | Meaning | Type |
| --- | --- | --- |
| **L0 Measurement** | raw spectra | `matchms.Spectrum` (never duplicated) |
| **L1 Direct identification evidence** | similarity `score`, `matched_peaks`, `mass_error_ppm`, tier | `AnnotationEvidence` |
| **L2 Primary statistical confidence** | per-query `q_value`/`p_value` | `FdrAssessment` |
| **L3 Spectral similarity** | score-based edges | `SpectralRelationship` |
| **L4 Chemical / ion-identity** | Δmass-based edges | `ChemicalRelationship`, `IonIdentityRelationship` |
| **L5 Network context** | descriptive family labels, network-inferred annotations, and family records | `NetworkContext`, `AnnotationInference`, `MolecularFamily` (never confidence) |

### Invariants (enforced by construction and by tests)

1. **No edge can carry statistical confidence.** No relationship type exposes a
   field named `q_value`, `p_value`, `fdr`, or `confidence`.
2. **FDR is query-scoped and copy-only.** `FdrAssessment` is the *only* carrier
   of FDR. It is keyed by query node in `MolecularGraph.fdr_assessments` and its
   semantics (per-query target-decoy competition) are unchanged. Downstream
   layers must never recompute or adjust it.
3. **Missing is `None`, never `0`.** All numeric fields forbid `NaN`/`inf`
   (`allow_inf_nan=False`); a missing scientific value is `null` on the wire.
4. **`GraphNode` stores no peaks.** It is a metadata projection; float64 m/z and
   intensity arrays stay in the spectral store and are retrieved by node id.
5. **Retention time carries its unit:** the field is `retention_time_seconds`.

## Stable identifiers

Node and relationship identifiers are **content-addressed** using `SHA-256`
over a canonical, length-framed byte encoding (never Python's `hash`), so they
are identical across processes and independent of `PYTHONHASHSEED`.

| Builder | Identifier form | Notes |
| --- | --- | --- |
| `spectrum_node_id(spectrum, *, namespace=None)` | `nid1:query:<hex>` | content hash of precursor, RT, charge, adduct, peaks; `namespace` disambiguates otherwise-identical spectra |
| `library_node_id(library_build_id, original_id)` | `nid1:library:<hex>` | cheap (no library re-hash); links to the `library_builds` row |
| `feature_node_id(...)` | `nid1:feature:<hex>` | LC-MS feature identity |
| `relationship_id(...)` | `rid1:<type>:<hex>` | type + endpoints + direction + key parameters (never provenance) |

## Types

- **`GraphNode`** — query/library spectrum metadata (`precursor_mz` float64,
  `retention_time_seconds`, `charge`, `ion_mode`, `adduct`, optional
  `smiles`/`inchikey`/`formula`, `library_build_id`, `feature_id`,
  `num_peaks`, physical-integrity fields). Built with
  `GraphNode.from_query_spectrum(...)` / `GraphNode.from_library_spectrum(...)`.
- **`Feature`** — an LC-MS feature grouping `spectrum_node_ids`; the
  representative/consensus spectrum field is typed but left unset (post-1.0).
- **`AnnotationEvidence`** — L1 evidence; `from_search_result(row, ...)` projects
  a `SearchResult` into typed evidence (the q-value is deliberately *not* read).
- **`FdrAssessment`** — L2; `competition_unit="query"`, `q_value`, `p_value`,
  `calibrated`, `library_size`.
- **`NeutralLoss`** — value object for Δmass interpretations.
- **`RelationshipProvenance`** — per-edge lineage (algorithm, build id, config
  digest, version, timestamp) mirroring the `library_builds` JSON shape.
- **`SpectralRelationship`**, **`ChemicalRelationship`**,
  **`IonIdentityRelationship`** — the three edge variants, unified as the
  discriminated `Relationship` union. Each exposes a `molecule_relationship`
  flag separating "same molecule" (e.g. adducts/isotopes) from "related
  molecule" (e.g. in-source fragments, transformations). Chemical edges are
  explicit hypotheses (`interpretation_status="hypothesis"` by default), never
  automatic structural claims.
- **`MolecularGraph`** — a **data-only** container (`nodes`, `features`,
  `relationships`, `fdr_assessments`, `provenance`) with lossless JSON
  (de)serialization and JSONL edge streaming. It exposes no analysis methods.

Relationships are constructed with each variant's `create(...)` classmethod,
which derives the deterministic `relationship_id`; construction from an
inconsistent hand-written id is rejected.

## Serialization

- `MolecularGraph.to_dict()` / `from_dict()` and `to_json()` / `from_json()`
  round-trip losslessly (float64 preserved exactly; missing values are `null`).
- `MolecularGraph.to_relationship_jsonl()` streams one edge per line for large
  graphs.
- Documents carry a `schema_version`; an incompatible major version is rejected.
- File writes go through the I/O boundary:
  `MassFlow.io.save_molecular_graph(graph, path)` /
  `MassFlow.io.load_molecular_graph(path)`.

## Spectral networking (P1, experimental)

Stage 1 spectral networking is implemented and strictly opt-in. `massflow
network build` reads the configured experimental input, builds undirected
`SpectralRelationship` edges with the classical similarity engine
(`include_decoys=False` — decoys are an annotation-only FDR device), and writes a
`MolecularGraph` JSON document.

```bash
# Requires network.enabled: true in the YAML configuration
uv run massflow network build --config massflow_config.yaml --output network.json
```

* **Candidacy** — `MassFlow.network.candidates.generate_candidate_pairs` windows
  spectra by precursor m/z (and optionally retention time) before scoring.
* **Edges** — `MassFlow.network.spectral.build_spectral_relationships` scores the
  candidate pairs and keeps those above `min_score`/`min_matched_peaks`, capped
  per node by `top_k_per_node`.
* **Graph** — `MassFlow.network.build.build_spectral_graph` projects spectra to
  content-addressed nodes and returns the data-only container.
* **Config** — `MassFlow.config.NetworkConfig` (the `network:` YAML section),
  **disabled by default**.

Nodes and edges are content-addressed and hash-seed-independent, so repeat runs
on identical input produce identical identifiers. Graphs carry **no** FDR
(`fdr_assessments` is empty): networking produces similarity, never statistical
confidence. Input spectra are never mutated (they are scored via clones).

## Feature identity (P2, experimental)

`MassFlow.network.features.build_features` groups spectra into LC-MS features:

* **Ion channel** — spectra are partitioned by `(ion_mode, charge, adduct)`, so
  an `[M+H]+` and an `[M+Na]+` of the same neutral mass are never merged.
* **Clustering** — within a channel, spectra are connected (single-linkage) when
  their precursor m/z and retention time fall within
  `feature_precursor_tolerance` (Da) and `feature_rt_tolerance` (seconds). A
  missing retention time does not block an m/z-consistent grouping.
* **Representative** — the highest summed-intensity (TIC) member, with a
  deterministic node-id tie-break; consensus MS/MS generation is deferred.

`build_spectral_graph` builds features by default within an enabled network and
stamps each member node's `feature_id`; the `Feature` objects are attached to the
`MolecularGraph`. Membership is currently expressed by
`Feature.spectrum_node_ids` (explicit feature-to-spectrum edges are deferred).

## Ion-identity relationships (P3, experimental)

`MassFlow.network.ion_identity.build_ion_identity_relationships` links features
that are the **same molecule** observed as different adducts. It derives each
feature's neutral mass from its precursor m/z and registered adduct offset
(`cheminformatics.compute_adduct_offset`) and links features whose neutral masses
agree within `ion_identity_ppm_tolerance` (default 5 ppm), confining links to a
single ionisation mode.

* Emitted kind: `relationship_kind="adduct"`, `molecule_relationship="same_molecule"`.
* **Fail closed:** an unregistered adduct, a declared charge that contradicts the
  adduct, or a cross-mode pair yields no link — no mass or structure is guessed.
* Deferred: isotope, in-source-fragment and multimer discovery.

## Chemical relationships (P4, experimental)

`MassFlow.network.chemical.build_chemical_relationships` hypothesizes
neutral-loss relationships between features. It matches the adduct-resolved
neutral-mass difference against a curated table of common fragmentations
(`NEUTRAL_LOSSES`; masses via `pyteomics`) within `chemical_ppm_tolerance`
(default 5 ppm), and orients each relationship larger → smaller.

* Emitted kind: `interpretation_status="hypothesis"`,
  `molecule_relationship="related_molecule"`, with a `NeutralLoss` (observed
  mass + ppm error) and a `transformation` label such as `loss:H2O`.
* **No structural claims.** A relationship is an explicit hypothesis, carrying no
  similarity score and no confidence.
* **Fail closed:** unresolvable adducts and cross-ionisation-mode pairs yield no
  link; an uninterpretable Δmass yields no edge. Multiply-charged adducts are
  handled by comparing neutral masses (not precursor m/z).

## Annotation contextualization (P5, experimental)

`MassFlow.network.context.contextualize` derives **L5** family context and
network-inferred annotations from direct, calibrated seed annotations
(`AnnotationSeed`), and `apply_context` attaches them to a graph
(`MolecularGraph.network_contexts` / `.inferences`).

* A component (`MassFlow.network.families.connected_components`) with at least one
  direct, calibrated seed whose `q_value <= context_seed_q_threshold` gets a
  `NetworkContext` (descriptive `label`, member and seed ids).
* Each unseeded query node in that component gets an `AnnotationInference`
  (`status="network_inferred"`) recording the supporting seeds and its incident
  edges.
* **Anti-circularity (structural):** L5 objects carry no q-value/score/tier;
  inference uses only direct seeds (never other inferences); the graph's
  `FdrAssessment` set is never read, written or re-derived; and `apply_context`
  rebuilds (re-validating) rather than mutating.

## Molecular families, export & MSMCP (P6, experimental)

`MassFlow.network.families.detect_families` turns connected components into
`MolecularFamily` records (content-addressed `family_id`, member ids, internal
edge ids, and — when an L5 `NetworkContext` exists for the component — an
inherited label and its seed ids). `build_spectral_graph` attaches families by
default (`NetworkConfig.build_families`).

* **Export** — `MolecularGraph.to_family_jsonl()` and
  `MassFlow.io.save_families_jsonl` produce machine-readable newline-delimited
  families; `massflow network analyse` attaches families to a graph JSON and
  `massflow network export` writes families JSONL.
* **MSMCP** — `MassFlow.network.msmcp.LocalGraphSource` is a dependency-free,
  offline, read-only MCP-style interface (`list_resources` / `read`) over
  `graph`, `node/{id}` (+ `/neighbors`, `/families`, `/inferences`),
  `family/{id}`, `edge/{id}`, `families`, and `provenance`. Statistical
  confidence is exposed **only** via `annotations()` (from `FdrAssessment`).
  A networked MCP server and community detection are deferred (ML / optional
  extras), so local operation never requires internet-scale data.

## Core v1.0 vs deferred

**Core v1.0 (this module):** identifiers, `GraphNode`, `Feature`,
`AnnotationEvidence`, `FdrAssessment`, `NeutralLoss`,
`RelationshipProvenance`, the three relationship variants, the `Relationship`
union, `MolecularGraph`, and the `io.py` graph reader/writer.

**Implemented (P1–P6, experimental):** Stage 1 spectral networking, Stage 2
feature identity, Stage 3 ion identity (adduct / same-molecule links), Stage 4
chemical relationships (neutral-loss **hypotheses**), Stage 5 annotation
contextualization (L5 `NetworkContext` / `AnnotationInference`), Stage 6
molecular families (`MolecularFamily`), and the offline read-only
`LocalGraphSource` MSMCP interface (`massflow network build` / `analyse` /
`export`). All opt-in, fail-closed on unknown adducts, non-circular, and disabled
by default.

**Deferred (post-1.0 / experimental, not implemented):** community detection,
networked MCP server, wiring of seeds from the annotation run, isotope /
in-source-fragment / multimer ion-identity discovery, explicit feature-membership
edges, `RepositoryGraphSource`, graph persistence in SQLite, GraphML, Cytoscape,
FBMN, visualization, network-aware scoring, and ML. The frozen design for the
whole subsystem is specified in
[Network-Aware MS Annotation — Frozen Design Specification](network-aware-annotation-spec.md).
