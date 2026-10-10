# Spectral Networking (`massflow network`)

!!! warning "EXPERIMENTAL — outside the stable product contract"
    The `massflow network` commands are **post-1.0, experimental**, and
    **disabled by default**. They are outside the stable v0.1 support promise
    (see [`docs/CAPABILITY_MATRIX.md`](../CAPABILITY_MATRIX.md)). Treat their
    output accordingly.

Spectral networking is MassFlow's **local-first, precision-first** framework for
Network-Aware MS Annotation. It builds a typed molecular graph on top of an
annotation run and is a pure downstream consumer: it never changes the stable
annotation path, and it never alters q-values/FDR.

See the frozen design in
[Network-Aware MS Annotation — Frozen Design Specification](../network-aware-annotation-spec.md)
and the data model in
[Network-Aware MS Annotation — Graph Data Model](../network-data-model.md).

## What it builds

```
spectra → features → spectral edges → ion-identity edges → chemical relationships
        → families (+ context) → machine-readable graph
```

| Stage | Edges / records | Notes |
| --- | --- | --- |
| 1 Spectral | `SpectralRelationship` | cosine / modified cosine, candidate-driven scoring |
| 2 Feature | `Feature` | LC-MS feature identity, highest-TIC representative |
| 3 Ion identity | `IonIdentityRelationship` | adduct / **same-molecule** links (fail-closed) |
| 4 Chemical | `ChemicalRelationship` | neutral-loss **hypotheses** (never structural claims) |
| 5 Context | `NetworkContext`, `AnnotationInference` | L5 only; never statistical confidence |
| 6 Families | `MolecularFamily` | deterministic connected components |

## Enabling networking

Networking is off unless `network.enabled: true` is set in the YAML:

```yaml
project:
  output_directory: results
input:
  input_path: data/queries.mgf
  library_path: data/library.msp
network:
  enabled: true
  min_score: 0.7
  min_matched_peaks: 6
  top_k_per_node: 10
```

Every `network` field (all optional) is recorded in graph provenance. Key ones:

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Master switch. |
| `algorithm` | `modified_cosine` | `cosine` or `modified_cosine`. |
| `ms1_tolerance` / `ms2_tolerance` | `0.02` | Precursor window (Da) applied by the similarity engine's MS1 prefilter (also the default candidacy window) / fragment tolerance (Da). |
| `precursor_candidacy_tolerance` / `precursor_candidacy_unit` | `null` / `"Da"` | Optional dedicated Stage-1 candidacy window. `"ppm"` scales with precursor m/z. The effective window is never narrower than `ms1_tolerance`, so widening it can only add scored pairs, never edges. |
| `min_score` / `min_matched_peaks` | `0.7` / `6` | Edge thresholds. |
| `rt_tolerance` | `null` | Optional retention-time window (seconds). |
| `top_k_per_node` | `10` | Max incident edges per node (`null` = unlimited). |
| `build_features` / `feature_precursor_tolerance` / `feature_rt_tolerance` | `true` / `0.01` / `30.0` | Feature identity. |
| `build_ion_identity` / `ion_identity_ppm_tolerance` | `true` / `5.0` | Adduct (same-molecule) links. |
| `build_chemical` / `chemical_ppm_tolerance` | `true` / `5.0` | Neutral-loss hypotheses. |
| `build_context` / `context_seed_q_threshold` | `true` / `0.01` | Context seeding from calibrated hits. |
| `build_families` | `true` | Deterministic family records. |

## Commands

### Build a graph

```bash
uv run massflow network build --config massflow_config.yaml \
    --output results/queries_network.json
```

Loads and processes the configured experimental input and writes a
`MolecularGraph` JSON document (nodes, features, relationships, families, and
provenance). Input spectra are never mutated (they are scored via clones).

### Analyse a graph (families + optional context)

```bash
# Families only
uv run massflow network analyse --input results/queries_network.json \
    --output results/queries_families.json

# Families + L5 context seeded by the annotation run
uv run massflow network analyse --input results/queries_network.json \
    --config massflow_config.yaml --output results/queries_families.json
```

With `--config`, the annotation pipeline is run on the same experimental input;
its **calibrated, high-confidence** library hits (q-value ≤
`context_seed_q_threshold`) seed L5 `NetworkContext` and `AnnotationInference`
records. Inferences are a separate collection — they are **never** placed in the
target-decoy pool, and they never change a q-value.

### Export families

```bash
uv run massflow network export --input results/queries_families.json \
    --output results/queries_families.jsonl
```

Writes one JSON object per family (newline-delimited JSON) for downstream tools.

## Reading a graph locally (MSMCP)

`MassFlow.network.msmcp.LocalGraphSource` is a dependency-free, **offline**,
read-only interface over a graph — no server or internet access required:

```python
from MassFlow.network.msmcp import LocalGraphSource

source = LocalGraphSource.from_file("results/queries_families.json")
source.list_resources()                       # graph, node/{id}, family/{id}, ...
source.read("families")                       # JSON list of families
source.read(f"node/{node_id}/neighbors")      # incident edges
source.families_containing(node_id)
source.annotations(node_id)                   # q-value/FDR — the ONLY confidence surface
```

Statistical confidence is exposed **only** through `annotations()` (sourced from
`FdrAssessment`); edges and inferences never carry confidence.

## Confidence discipline

- No edge, context, inference, or family carries a q-value, score, or tier.
- Network context and inferred annotations can never modify `FdrAssessment` or
  `AnnotationEvidence`.
- Uncalibrated runs (`fdr_uncalibrated`) produce **no** contexts — their q-values
  are the `1/N` rank bound, not an FDR estimate.
- Chemical relationships are explicit hypotheses, never structural claims.

## Not implemented (do not rely on it)

Community detection, a networked MCP server / repository federation, GraphML,
Cytoscape, FBMN export, and visualization.
