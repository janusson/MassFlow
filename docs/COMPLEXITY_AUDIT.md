# Complexity Audit & Subsystem Classification

> **STATUS: authoritative (2026-08-25).** Companion to
> [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md): the capability matrix says
> *what* is in the product; this document says *where the complexity lives*,
> *which tier each subsystem belongs to*, and *how the package boundary keeps
> experimental machinery from silently altering stable scientific behavior*.
>
> Classification legend:
>
> | Tier | Meaning | Change discipline |
> | --- | --- | --- |
> | **Required (stable)** | Part of the v0.1 product contract. Removing or changing it is a product decision | Must not regress; full test gate + coverage |
> | **Optional but supported** | Installed/available by default, contract-tested, but not the default path | Must keep contract tests green |
> | **Experimental** | Implemented and tested, outside the support promise; must be visibly flagged and never silently alter stable behavior | May evolve freely; guards/fallbacks required |
> | **Deprecated** | Kept only for compatibility with existing configs/CLI invocations; warns on use | Remove only at a schema/CLI bump |
> | **Remove** | Dead, duplicated, or superseded; deleted in this pass | — |

---

## 1. Subsystem inventory

### 1.1 `src/MassFlow` — core package

| Module | Lines | Tier | Classification / rationale | Boundary |
| --- | --- | --- | --- | --- |
| `config.py` | 1140 | **Required** (stable schema) + **deprecated** legacy fields | Strict Pydantic schema (`extra="forbid"`, line-numbered errors, relative-path resolution). Contains the deprecated `SimilarityConfig.tolerance` alias, `ProcessingConfig.precursor_mz`/`retention_time` legacy fields, `SolventConfig` (validated but not consumed by the pipeline), `WorkflowConfig` (empty placeholder kept for `workflow:` YAML compat), and `MassFlowConfig.output_directory` legacy property | Deprecated fields warn or are inert; `tolerance` warns and maps onto `ms2_tolerance`; removal deferred to a schema bump |
| `io.py` | 520 | **Required** | File-system boundary: loaders (mzML/mzXML/MGF/MSP), validation layer, CSV/mzTab-M exporters, YAML reports, vendor-format rejection | — |
| `processing.py` | 360 | **Required** | `matchms`-based metadata/peak pipeline | — |
| `similarity.py` | 3135 | **Required** (classical core) + **experimental** (meta/ML engines) | Required: `SimilarityEngine` (cosine/modified_cosine), MS1 prefilter, `generate_decoys`, `calculate_fdr`, `calibrate_query_level_fdr`, `calculate_empirical_p_values`, `spectral_entropy`, `SearchResult`, `get_similarity_engine` factory. Experimental: `ConsensusEngine`, `CascadeEngine`, `Spec2VecEngine`, `MS2DeepScoreEngine`, `MLRouter`, entry-point discovery, remote-endpoint routing | Experimental engines are flagged by `workflow.experimental_surface_flags` at the run boundary; ML engines raise `RuntimeError` when dependencies are missing (no silent swap); consensus/cascade degrade to classical only with explicit warnings + degraded-mode flags |
| `workflow.py` | 1202 | **Required** | Orchestration, per-file failure model (`FileExecutionResult`), provenance, TDC block, worker-owned library backends | New in this pass: `experimental_surface_flags()` + run-start warning |
| `library.py` | 431 | **Required** | Library normalization (`prepare_library`), worker-openable `LibrarySpec`, `RawFileLibraryStore` | — |
| `storage.py` | 382 | **Required** | `SpectralStore` — the single backend interface (metadata query, iteration, batched access, provenance) + factory | — |
| `database.py` | 2370 | **Required** | `SpectralDatabase` (SQLite/hybrid store), schema, migrations. Contains legacy-schema migration helpers (`migrate_legacy_peaks_database`, `_decode_legacy_peaks_payload`, `LegacyDatabaseSchemaError`) | Legacy helpers are **deprecated tooling** for pre-array-schema DBs; new DBs never hit them |
| `zarr_store.py` | 2463 | **Optional but supported** | `ZarrSpectralStore`/`ZarrPeakArrayStore` — supported alternative backend (`storage_backend: zarr|hybrid`), peak-fidelity contract (no silent truncation), storage-contract tests | zarr is a core dependency (no extra); backend selected explicitly in config; default remains `sqlite` |
| `models.py`, `cheminformatics.py` | 386+727 | **Required** | 5 ppm precursor validation, adduct registry, isotopic envelopes | Enforced as a gate in the streaming path; model-layer checks in the classical path (see CAPABILITY_MATRIX C-11/D-8) |
| `protocols.py` | 134 | **Experimental** | `MLEngineProtocol` — contract for the ML satellite boundary | Only consumed by ML engines |
| `ml_client.py` | 573 | **Experimental** | Remote ML engines (REST/gRPC) + circuit breaker | Only active when `similarity.ml_endpoints` is configured (flagged) |
| `hnsw.py` | 585 | **Experimental** | HNSW two-channel candidate index (`[hnsw]` extra) | Only active inside `cascade` with `hnsw_enabled=true` (config-validated); flagged |
| `acceleration.py` | 521 | **Experimental** (as a whole) with a **supported** default-on part | Numba peak/neutral-loss prefilter: default-on for `modified_cosine` but **numerically identical** to the pure-NumPy fallback (identity-tested end-to-end, incl. decoys) — an optimization, not a science change. HNSW spectral binning: experimental | Prefilter falls back to NumPy when numba is absent |
| `convert.py` | 104 | **Experimental** | Thin wrapper around external ProteoWizard `msconvert`; the annotate pipeline rejects vendor formats | CLI help marks it EXPERIMENTAL |
| `log_config.py` | 103 | **Required** | Structured logging setup | — |
| `streaming/` (server 1189, engine 734, queue 542) | 2465 | **Experimental** | gRPC streaming server (loopback default, TLS/auth, bounded queue); `stream-server` CLI command | CLI help marks it EXPERIMENTAL; safe-by-default binds/auth |
| `tui/` (8 modules) | 2114 | **Experimental** | Textual interactive console (`[tui]` extra) | Extra-gated; CLI help marks it EXPERIMENTAL |
| `generated/` + `streaming/generated/` | 439 | **Experimental** | Compiled protobuf/gRPC stubs (ml + streaming) | Excluded from lint; regenerated via `scripts/protoc_gen.sh` |
| `__init__.py` | 90 | **Required** | Public package surface; lazy submodule accessors | Fixed in this pass: removed the dead `try/except PackageNotFoundError` version probe |

### 1.2 CLI commands (`cli.py`)

| Command | Tier | Notes |
| --- | --- | --- |
| `init`, `tutorial`, `annotate` | **Required** | Stable product entry points |
| `db build` / `db inspect` / `db merge` | **Required** | SQLite library workflows |
| `convert` | **Experimental** | External-tool wrapper (help text marked) |
| `stream-server` | **Experimental** | gRPC streaming (help text marked) |
| `serve` | **Deprecated** | Alias for `stream-server`; prints a deprecation notice |
| `watch` | **Experimental** | `[watch]` extra (help text marked) |
| `tui` | **Experimental** | `[tui]` extra (help text marked) |

New in this pass: `annotate` prints a prominent **EXPERIMENTAL SURFACES ACTIVE**
notice when the configuration selects an experimental engine, routing, HNSW,
or remote-ML endpoints — an experimental run is never silent.

### 1.3 Extras (`pyproject.toml`)

| Extra | Tier | Notes |
| --- | --- | --- |
| (none — base) | **Required** | Includes `zarr` (core dependency; the redundant `[zarr]` extra was **removed** in this pass) |
| `chem` | **Optional** | RDKit structural validation |
| `ml` | **Experimental** | torch/gensim/spec2vec/ms2deepscore |
| `hnsw` | **Experimental** | hnswlib |
| `watch` | **Experimental** | watchfiles |
| `tui` | **Experimental** | textual |

### 1.4 Scripts

| Path | Tier | Notes |
| --- | --- | --- |
| `scripts/generate_tutorial_data.py` | **Required** | Backs `massflow tutorial` |
| `scripts/migrations/0001_peaks_to_arrays.py`, `0002_blobs_to_zarr.py` | **Deprecated tooling** | One-shot migration utilities for legacy (pre-array-schema / BLOB-only) databases; new databases never need them |
| `scripts/benchmark_*.py`, `scripts/measure_baseline.py` | **Dev tooling** | Opt-in benchmarks; never run by pytest |
| `scripts/experiments/` | **Research prototypes** | Scratch/exploratory tests; never collected by pytest (`testpaths=["tests"]`) |
| `scripts/protoc_gen.sh` | **Dev tooling** | Regenerates protobuf stubs |
| `scripts/mock_instrument_stream.py`, `scripts/repro_isotope.py`, `scripts/fetch_real_data.py` | **Dev tooling** | Standalone utilities |

### 1.5 Data / examples / docs

| Path | Tier | Notes |
| --- | --- | --- |
| `examples/massflow-ml-satellite/` | **Experimental** | Reference satellite server for the ML boundary |
| `docs/CAPABILITY_MATRIX.md`, `docs/COMPLEXITY_AUDIT.md` | **Required (docs)** | Product contract + this audit |
| `ARCHITECTURE.md` (repo root) | **Remove** | Byte-identical duplicate of `docs/ARCHITECTURE.md` (which is a superset). **Deleted in this pass**; `docs.yml` no longer copies it; references updated |
| `docs/index.md`, `README.md` | **Required (docs)** | Fixed in this pass: experimental surfaces no longer labeled Stable (engines table, watch, FBMN, LSP, 5 ppm scope, `--extra zarr` install line, HNSW-in-default-pipeline diagram) |

---

## 2. Duplicated abstractions — findings

| # | Finding | Verdict |
| --- | --- | --- |
| D-1 | `ARCHITECTURE.md` existed twice (root + `docs/`), kept in sync by a copy step in `docs.yml` | **Resolved in this pass** — root copy deleted, copy step removed, references repointed at `docs/ARCHITECTURE.md` |
| D-2 | Storage abstractions: `SpectralStore` (interface) vs `SpectralDatabase` (SQLite impl) vs `ZarrSpectralStore` | **Not duplication** — the storage-unification audit made `SpectralStore` the one interface; `database.py`/`zarr_store.py` are its implementations; `create_spectral_store` is the factory; the annotation layer consumes only the interface |
| D-3 | DataFrame representations | **Single representation** — Polars only (`io.py`); no pandas anywhere in `src/`; `process_spectra_batch` uses Polars lazyframes internally |
| D-4 | Redundant Pydantic models | `WorkflowConfig` (empty placeholder), `SolventConfig` (unconsumed), legacy `ProcessingConfig.precursor_mz`/`retention_time`, `SimilarityConfig.tolerance`, `MassFlowConfig.output_directory` — all **deprecated compat surface**, kept for existing configs, not dead code in the sense of *duplicated* logic |
| D-5 | Generated protobuf stubs in two directories (`generated/`, `streaming/generated/`) | Distinct contracts (ml vs streaming) — not duplication |
| D-6 | `[zarr]` extra duplicating a core dependency | **Resolved in this pass** — extra removed (zarr is always installed) |
| D-7 | `database.py` legacy-schema decode/migration helpers vs current schema | **Deprecated tooling**, not duplication: they exist to open pre-array-schema DBs and raise `LegacyDatabaseSchemaError` otherwise |
| D-8 | `scripts/experiments/` overlap with `tests/` | Not collected (`testpaths=["tests"]`); documented research prototypes |

---

## 3. Package boundary — how experimental features are kept out of the stable path

### 3.1 Existing gates (verified)

1. **Engine selection**: `algorithm` defaults to `cosine`; every experimental engine requires an explicit config value; ML engines raise `RuntimeError` if dependencies are missing (no silent swap); consensus/cascade degrade to classical **only** with a warning and a `degraded_mode_flags` entry recorded in the report sidecar.
2. **Routing**: `enable_routing` defaults `false`.
3. **HNSW**: `hnsw_enabled` defaults `false`, and config validation rejects `hnsw_enabled` with any engine other than `cascade`.
4. **Storage**: `storage_backend` defaults `sqlite`; `zarr`/`hybrid` are explicit choices with contract tests.
5. **Streaming**: loopback bind by default; TLS/`--admin-token`/`--allow-remote-control` required for remote/control-plane use.
6. **Numba prefilter**: default-on, but numerically identical to the pure-NumPy fallback (identity-tested with decoys); it is an optimization, not a science change.
7. **Pytest**: default selection runs unit + integration + scientific-validation; `benchmark`/`slow`/`optional` are opt-in; `scripts/experiments/` is never collected.

### 3.2 Added in this pass

1. **`workflow.experimental_surface_flags(config)`** — deterministic list of active experimental surfaces (`experimental_engine:*`, `experimental_routing`, `experimental_hnsw`, `experimental_remote_ml`).
2. **`run_annotation_pipeline`** logs a `EXPERIMENTAL SURFACES ACTIVE` warning before any processing when the list is non-empty.
3. **`massflow annotate`** prints the same notice to the console.
4. **CLI help/docstrings** for `convert`, `stream-server`, `watch`, `tui` explicitly say EXPERIMENTAL.
5. **README + docs/index.md** no longer label experimental machinery (consensus, cascade, HNSW, hybrid storage, watch, streaming) as Stable; the "four pillars" framing is now explicitly an experimental/optional layer over a boring stable core.
6. **Redundant `[zarr]` extra removed**; **root `ARCHITECTURE.md` duplicate removed**; **dead version probe in `__init__.py` removed**.

### 3.3 What is deliberately NOT enforced (and why)

- Experimental engines are **not refused** without an opt-in flag: consensus/cascade are implemented, tested (including golden known-answer tests), and documented; the boundary requirement is *visibility* (notice + provenance + warnings), not removal. Refusing them would delete tested functionality, which this pass must not do.
- The deprecated config surface (`tolerance`, `workflow:`, `solvents:`, legacy processing fields) is kept so existing YAML files keep loading; removal is deferred to a v0.2 schema bump.

---

## 4. What changed in this pass

| Change | Files |
| --- | --- |
| `experimental_surface_flags` + run-start warning | `src/MassFlow/workflow.py` |
| CLI experimental notice + EXPERIMENTAL help text (convert, stream-server, watch, tui) | `src/MassFlow/cli.py` |
| Dead version probe removed | `src/MassFlow/__init__.py` |
| Redundant `[zarr]` extra removed | `pyproject.toml` |
| Root `ARCHITECTURE.md` duplicate deleted; `docs.yml` copy step removed; references repointed | `ARCHITECTURE.md` (deleted), `.github/workflows/docs.yml`, `AGENTS.md`, `CONTRIBUTING.md` |
| README: engines table, pillar framing, pipeline-diagram note, install extras | `README.md` |
| `docs/index.md`: Stable/Experimental table corrected | `docs/index.md` |
| Install extras doc corrected | `docs/getting-started/installation.md` |
| Boundary tests | `tests/test_cli.py`, `tests/test_workflow.py` |
| Contradiction log updated | `docs/CAPABILITY_MATRIX.md` |

No scientific functionality was deleted, altered, or added in this pass. The
golden known-answer suite (including consensus/cascade runs) is unchanged.

---

## 5. Remaining complexity decisions (not resolved here)

| ID | Decision | Options |
| --- | --- | --- |
| R-1 | Deprecated config surface removal (`tolerance`, `workflow:`, `solvents:`, legacy `precursor_mz`/`retention_time`) | Remove at a v0.2 schema bump with a migration error message |
| R-2 | Legacy-DB migration helpers + `scripts/migrations/` | Keep as tooling; remove once pre-array-schema DBs are out of circulation |
| R-3 | `-m core` marker discipline | Retired in the pytest-config audit; the default selection + `-m scientific` is the contract gate. Reintroduce only if a stable-only CI job is wanted |
| R-4 | `streaming/engine.py` + `ml_client.py` both speak gRPC | Distinct contracts (instrument stream vs ML scoring); a shared transport helper would be a refactor, not a simplification |
