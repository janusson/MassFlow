# Yogimass overview

Yogimass is organized into layered modules. This guide summarizes what each layer does, what it depends on, and what it should avoid. Use this as a map when adding features or debugging.

## Dependency flow (simplified)

```text
Raw data (MGF/MSP/MSDial) ──> I/O ──> Similarity/Library ──> Workflow ──> Reporting
                               │            │                 │
                               │            └─> Network ─────┘
                               │
                               └─> Curation/QC (via Workflow) ─────> Reporting

CLI wraps Workflow; configs drive end-to-end runs.
```

## Layers

### I/O (`yogimass.io.*`)

- **Responsibilities**: read/write spectral data (MGF, MSP, MS-DIAL tables), cleaning helpers.
- **Depends on**: external parsers (e.g., `matchms` loaders), standard library.
- **Avoid**: business logic (search, curation, networks); keep pure data access/cleaning to prevent cycles.

### Similarity & libraries (`yogimass.similarity.*`)

- **Responsibilities**: spectrum processing (`SpectrumProcessor`), vectorization/metrics, lightweight library storage (`LocalSpectralLibrary`), search utilities.
- **Depends on**: `matchms` for `Spectrum`, local metrics helpers, minimal logging.
- **Avoid**: knowing about CLI, workflow orchestration, or reporting; do not import `curation` or `networking`.

### Networking (`yogimass.networking.network`)

- **Responsibilities**: build similarity graphs from spectra or libraries, export edge lists/graphs.
- **Depends on**: `similarity.metrics`, `similarity.processing`, optional `networkx`, basic I/O helpers.
- **Avoid**: CLI/config parsing; curation logic; writing reports (delegate to `reporting`).

### Curation/QC (`yogimass.curation`)

- **Responsibilities**: score spectra for quality, detect/merge duplicates, emit structured decisions and summaries.
- **Depends on**: `similarity.library` entries, `similarity.metrics` for cosine comparisons, logging.
- **Avoid**: file I/O beyond metadata in entries; CLI concerns; network building.

### Reporting (`yogimass.reporting`)

- **Responsibilities**: serialize search results, network summaries, and curation/QC reports (JSON/CSV), lightweight summaries.
- **Depends on**: data structures from `networking` and `curation`, standard library.
- **Avoid**: running workflows or mutating data; keep to read/format/write.

### Orchestration (`yogimass.workflow`)

- **Responsibilities**: high-level operations (`load_data`, `build_library`, `search_library`, `build_network`, `curate_library`, `run_from_config`), glueing modules together.
- **Depends on**: `similarity` (processing, library, search), `networking`, `curation`, `reporting`, `matchms` loaders.
- **Avoid**: CLI argument parsing; long-running UI/visualization; keep orchestration thin and composable.

### CLI (`yogimass.cli`)

- **Responsibilities**: user-facing subcommands (`config run`, `library build/search/curate`, `network build`, legacy clean), argument parsing, calling into `workflow` and `reporting`.
- **Depends on**: `workflow`, `reporting`, standard library argparse/logging.
- **Avoid**: implementing business logic; defer to `workflow` and helpers.

## Golden configs

- `examples/simple_workflow.yaml`: end-to-end build/search/network example.
- `examples/curation_workflow.yaml`: library curation/QC example.

These configs are the canonical smoke tests via `python -m yogimass.cli config run --config <file>`.
