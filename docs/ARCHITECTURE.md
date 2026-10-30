# MassFlow Architecture

> **This file is a component-level reference only.** The authoritative product
> contract, capability matrix, contradiction log, and core/experimental/future
> research boundaries live in
> [`docs/CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md). Where the two documents
> disagree, `CAPABILITY_MATRIX.md` governs.

## Overview

MassFlow is a local, CLI-first toolkit for tandem mass spectrometry (MS/MS)
annotation workflows.

The core execution path is:

1. load a YAML configuration
2. load an experimental file or directory of files
3. load a reference library
4. process spectra with configurable `matchms` filters
5. run similarity search against the reference library
6. apply score and legacy numerical confidence filtering (uncalibrated)
7. write per-file CSV result tables

The architecture is intentionally modular. Configuration, I/O, processing,
similarity scoring, database access, and workflow orchestration live in
separate modules so the main annotation path stays predictable and testable.

The proposed restricted 1.0 candidate scope centers on:

- `massflow init` and `massflow annotate --config ...`
- SQLite-backed library workflows through `massflow db`
- open-format ingestion for `mzML`, `mzXML`, `MGF`, and `MSP`
- transparent `matchms` processing
- `cosine` and `modified_cosine` candidate ranking
- CSV (stable) and experimental mzTab-shaped output with source-record metadata

Numerical confidence diagnostics are not calibrated FDR or confirmed identities.
Tutorial, conversion, ML, networking, streaming and other experimental utilities
are outside the supported candidate-release path.

---

## Simple Data Flow Workflow

```mermaid
graph LR
    Config[YAML Config] --> CLI{MassFlow CLI}
    Input[Input Files] --> CLI
    Library[Reference Library] --> CLI
    CLI --> Processed[Processed Spectra]
    Processed --> Sim[Similarity Search]
    Sim --> Filter[Candidate and Numeric-Estimate Filtering]
    Filter --> Out[CSV / experimental mzTab-shaped + YAML Report]
```

---

## High-Level Components

### User-facing surfaces

- `src/MassFlow/cli.py`
  - Defines the CLI commands:
    - `init`
    - `annotate`
    - `convert`
    - `db build`
    - `db inspect`
    - `db merge`
    - `stream-server`
    - `serve`
    - `watch`
    - `tui`
- Python API
  - Core modules can also be imported directly for scripting or testing.

### Terminal console

- `src/MassFlow/tui/`
  - A Textual-based interactive console (`massflow tui`, requires the `tui`
    extra) with four tabs: Browser (find/upload/load spectral files),
    Viewer (interactive centroid stick plots), Candidate Review (experimental score/mirror plots), and Diagnostics (problems with
    plain-English fixes plus the quarantine log).
  - All science-facing helpers are pure functions (NumPy/text in, text out)
    so the plot glyph math and file logic are unit-testable headlessly;
    heavy core imports (matchms, pyteomics) are deferred to worker threads.

### Orchestration

- `src/MassFlow/workflow.py`
  - Runs the end-to-end annotation pipeline.
  - Coordinates loading, processing, candidate searching, numerical confidence reporting, and export.
  - Uses multiprocessing to process experimental input files in parallel.
  - Runs strict pre-flight checks (`preflight_annotation_run`) before any
    expensive work: engine/optional-extras availability, reference-library
    and query-input existence/format, vendor-raw rejection — with
    actionable errors and no output artifacts on failure.

### Configuration

- `src/MassFlow/config.py`
  - Defines the Pydantic models used to validate YAML configuration files.
  - Expands `~` for relevant input paths.
  - Stores processing, similarity, workflow, and export settings.

### Data access

- `src/MassFlow/io.py`
  - Loads spectra from supported open formats and SQLite libraries.
  - Rejects vendor raw formats instead of converting them implicitly.
  - Writes match results to CSV (stable) and an experimental mzTab-shaped format, and can export spectra
    as MGF or MSP.

- `src/MassFlow/database.py`
  - Stores and retrieves spectra in SQLite format.
  - Supports database build, inspection, and merge workflows.
  - Supports a hybrid storage mode where SQLite retains metadata plus a
    `zarr_ref`/`zarr_index` reference pair and fragment arrays live in a
    chunked Zarr store (see `src/MassFlow/zarr_store.py`), with a migration
    utility (`scripts/migrations/0002_blobs_to_zarr.py`) for existing BLOB
    libraries.
  - Implements a fast NumPy-based **Triage Bitmask** scan during insertion to flag structurally significant features (e.g., Tyrosine immonium ions at 136.076 Da). These flags are stored as JSON in the `triage_flags` column for future ML routing.
  - Records a lightweight **library-provenance history** (`store_meta` + `library_builds` tables, `PRAGMA user_version` >= 1) for every build/merge event — input files with SHA-256, config hash, processing and similarity (target-decoy) configuration, exact timestamps — surfaced by `db inspect` and linked from annotation result sidecars.

### Processing

- `src/MassFlow/processing.py`
  - Applies metadata cleaning and peak filtering using `matchms`.
  - Handles operations such as retention-time extraction, intensity filtering,
    minimum peak enforcement, m/z range truncation, Top-N peak reduction, and
    normalization.

### Similarity

- `src/MassFlow/similarity.py`
  - Implements similarity engines and result models.
  - Core engines:
    - `cosine`
    - `modified_cosine`
  - Experimental engines:
    - `spec2vec`
    - `ms2deepscore`
    - `consensus` (via the `ConsensusEngine`)
    - `cascade`
- `src/MassFlow/acceleration.py`
  - Numba-accelerated peak/neutral-loss matching prefilter that skips
    query-reference pairs below `min_matched_peaks` before exact
    modified-cosine scoring (pure-NumPy fallback with identical results).
- `src/MassFlow/hnsw.py`
  - hnswlib-backed HNSW (Hierarchical Navigable Small World) index over
    two-channel binned spectral vectors (`[binned exact m/z, binned
    neutral losses]`) for optional approximate candidate retrieval
    libraries. The neutral-loss channel adds a candidate channel for shifted
    analogues under modified cosine. Used by the cascade engine as an
    approximate pre-stage with no general recall guarantee; exact scoring
     follows only for candidates returned by the index.

### Data models and consensus scoring

- `src/MassFlow/models.py`
  - Defines strict Pydantic data contracts (`AnnotationHit`, `ConsensusInput`, `ConsensusResult`, `ConsensusConfig`, `MolecularStructure`).
  - Provides a dependency-free, engine-agnostic language for communication between the lightweight core and heavy ML satellite repositories (e.g., `massflow-ml`).
  - Implements rigorous structural validation (e.g., 5 ppm precursor m/z checks) and automatically calculates theoretical `isotopic_envelope` distributions for valid molecules.
- Consensus *scoring* is provided by the `ConsensusEngine` in `src/MassFlow/similarity.py`, which combines cosine, modified_cosine, and (when the `[ml]` extra is installed) Spec2Vec/MS2DeepScore sub-engines into a weighted consensus score. An earlier standalone `MassFlow.consensus` orchestrator module (with `generate_consensus`) is not part of the current codebase; the tie-breaking and credibility-check logic it implemented was removed alongside it.

---

## Scientific Data Integrity

MassFlow checks declared metadata at ingestion under fixed software rules. Passing these checks does not establish a molecule's identity or universal chemical validity.

### Precursor Validation (5 ppm Tolerance)
The processing gate checks a recognized declared adduct/charge pair for signed-charge consistency even when precursor m/z is missing. When a structure and usable precursor are also declared, the `SpectrumMetadata` contract compares precursor m/z with the theoretical mass under the fixed 5 ppm policy. The theoretical m/z uses the formula-derived exact mass and registered adduct offset. These checks cover declared metadata consistency, not universal instrument tolerances or identity confirmation.

**Enforcement surface** — the verdicts are consumed by three gates:

1. **Classical `annotate` / `db build` metadata gate** (`MassFlow.processing`, run after `matchms` harmonization): every record with a recognized explicit adduct/charge pair is checked for signed-charge consistency; records declaring a formula, SMILES, or InChI also receive structural and 5 ppm checks.
   - query spectra failing the gate are rejected with a human-readable reason, counted in `spectra_rejected`, and recorded in the quarantine log; a query file whose spectra are all rejected is an explicit `failed` result (`<stem>_failed.report.yaml`), never an empty success;
   - a raw reference library containing a failing entry **aborts the annotate run** with `processing.PhysicalIntegrityError` before any file is processed — silently shrinking the reference pool would change candidate searches and legacy numerical estimates. Pre-built MassFlow stores are trusted (they are produced by `db build`, which runs the same gate and quarantines invalid entries);
   - spectra without structural claims skip the structural/5 ppm mass checks; explicitly declared supported adduct/charge pairs are still checked.
2. **Streaming ingestion gate** (`MassFlow.streaming.engine.validate_streaming_spectrum`, experimental `stream-server`) rejects packets failing the Pydantic field constraints before scoring.
3. **Model layer** — direct construction of `SpectrumMetadata`/`MolecularStructure` always computes the verdict.

**Supported Adducts** (22 entries in the `_ADDUCT_SPECS` registry; `docs/user-guide/validation.md` holds the authoritative offset table and the notation rules):
- **Positive Mode:** `[M+H]+`, `[M+NH4]+`, `[M+Na]+`, `[M+K]+`, `[M]+`, `[M+2H]2+`, `[M+3H]3+`, `[M+2Na-H]+`, `[M+H-H2O]+`, `[M+CH3CN+H]+`, `[M+CH3OH+H]+`
- **Negative Mode:** `[M-H]-`, `[M-2H]2-`, `[M+Cl]-`, `[M+Br]-`, `[M+I]-`, `[M+HCOO]-`, `[M+CH3COO]-`, `[M+TFA-H]-`, `[M+HCO3]-`, `[M+NO3]-`, `[M]-`
- **Notation and ionisation mode are read from the chemistry, not the string:** `MassFlow.cheminformatics.normalize_adduct` canonicalises library spellings (`M+H`, `[M+H]1+`, `[m+h]+`, `M+OAc`, legacy `[M+FA-H]-`) onto these keys before lookup, and `MassFlow.cheminformatics.adduct_charge` reports the resolved signed charge, so e.g. `[M+HCOO]-` is negative despite the `+`. Notation that cannot be resolved still fails closed.

### Theoretical Isotopic Envelopes
For advanced structural verification, the `MolecularStructure` model automatically calculates and caches the theoretical isotopic envelope (M, M+1, M+2, etc.) for any parsed SMILES string. By generating abundance-weighted centroid masses normalized to the base peak, the pipeline establishes a ground-truth MS1 signature for every reference candidate. This allows the `ConsensusEngine` and orthogonal ML models to evaluate candidate credibility by checking experimental MS1 isotopic patterns, providing a powerful orthogonal tie-breaking mechanism when MS2 fragmentation scores are ambiguous.

---

## Component Diagram

```mermaid
graph TD
    CLI[CLI & API] --> WORKFLOW[Workflow]
    WORKFLOW --> CONFIG[Configuration]
    WORKFLOW --> IO[I/O Layer]
    IO <--> DB[Database]
    WORKFLOW --> PROCESS[Processing]
    PROCESS --> SIM[Similarity Engine]
    SIM --> WORKFLOW
    SIM --> ML[ML Engines]
```

---

## Actual Annotation Workflow

The main production path begins with:

- `massflow annotate --config path/to/config.yaml`

The CLI loads the config and calls `run_annotation_pipeline()`.

### Pipeline steps

0. **Pre-flight validation (fail fast)**
   - Before any library store is built or any file is searched,
     `preflight_annotation_run()` validates: (a) the configured similarity
     surface is constructible with the installed optional extras (a pure ML
     engine without `massflow-msms[ml]` aborts with the install instruction
     instead of crashing workers later); (b) the reference library is
     configured, exists, and is a loadable open-format/store input (vendor
     raw and unknown formats abort with zero artifacts); (c) the query
     input exists and is dispatchable — vendor files inside a directory are
     announced and then fail explicitly per file (batch robustness).
     Warnings (``optional_extra_warnings``) surface degradations from
     missing ``[ml]``/``[hnsw]`` extras. The CLI prints `fix:` hints
     (``MassFlow.tui.diagnostics.suggest_fix``) for every failure.

1. **Configuration loading**
   - The YAML file is parsed into `MassFlowConfig`.
   - Paths such as `input_path` and `library_path` have `~` expanded.

2. **Library loading**
   - The workflow requires a configured library path.
   - In the preferred config terminology, this is `input.library_path`.
   - The library is loaded first and processed through the configured
     `matchms` pipeline.
   - If no valid spectra are found, the run fails.
   - If the library is small, the workflow logs a warning that FDR estimates
     may be weak.

3. **Experimental input discovery**
   - The workflow reads `input.input_path`, which may point to a single
     spectral file or a directory of files.
   - If a directory is used, files are discovered recursively.

4. **Pre-loading and Multiprocessing**
   - The parent process generates the decoy library ONCE.
   - The fully processed reference library and decoys are loaded into shared memory.
   - Each experimental input file is dispatched to a worker process.

5. **Query spectrum loading and processing**
   - Query spectra are loaded from the experimental file.
   - Each spectrum is passed through metadata and peak processing.
   - Missing query IDs are filled in automatically.

6. **Chunked reference searching (bounded library chunks)**
   - The worker searches its queries against the pre-loaded shared memory library in chunks to preserve RAM, entirely bypassing disk I/O.
   - Results are aggregated.

7. **FDR calculation**
   - Entropy-based decoys are generated from the reference spectra:
     each decoy preserves the precursor m/z, its source's complete peak list,
     and the Shannon entropy of its source's noise-filtered fragment
     intensities while randomizing the fragmentation pathways, avoiding the
     null-distribution bias of naive fragment shuffling.
   - Target and decoy scores are combined to estimate q-values. Decoys are
     scored and gated exactly like targets; because the position randomization
     keeps a decoy from coinciding with its source at scoring tolerance, decoy
     hits are rare at production score/matched-peak gates, and an empty decoy
     null (no decoy competition) is reported explicitly: the engine exposes
     `decoy_diagnostics` + the `decoy_null_empty` flag, the workflow flags
     `fdr_uncalibrated` (q-values are then `1/N` rank diagnostics, not error bounds)
     and the exported rows are labeled `Uncalibrated`.
   - The workflow logs a target-decoy entropy-divergence diagnostic to
     report descriptive target/decoy entropy differences. See
     [scoring_logic.md](user-guide/scoring_logic.md) §5.1 for the empty-null
     contract.
   - Final results are filtered by:
     - score thresholds
     - matched-peak thresholds where applicable
     - configured historical q-value threshold (not calibrated FDR control)

8. **Result Export**
   - The workflow writes one result file per experimental input file.
   - Output filenames follow the pattern:
     - `<input_stem>_results.<ext>`

---

## Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Workflow
    participant IO
    participant Process
    participant Sim

    User->>CLI: massflow annotate
    CLI->>Workflow: run_annotation_pipeline()
    Workflow->>IO: load reference library
    Workflow->>Process: process reference
    loop each file
        Workflow->>IO: load query spectra
        Workflow->>Process: process queries
        Workflow->>Sim: search & score
        Workflow->>Workflow: apply legacy confidence filter
        Workflow->>IO: save results
    end
    Workflow-->>CLI: complete
```

---

## Supported Inputs and Outputs

### Supported direct input formats

MassFlow directly loads:

- `mzML`
- `mzXML`
- `MGF`
- `MSP`
- SQLite libraries:
  - `.db`
  - `.sqlite`

### Unsupported direct input formats

MassFlow intentionally does **not** perform vendor raw conversion internally.

Examples of vendor formats that must be converted before use include:

- `.raw`
- `.d`
- `.wiff`
- `.lcd`
- `.t2d`
- `.baf`

Users should convert these to open formats such as `mzML` or `MGF` before
running the annotation workflow.

### Standard outputs

The stable output format is CSV. An experimental mzTab-shaped table is also
available, but it is not mzTab-M standards-conformant and is outside the stable
export contract.

- CSV (stable) or experimental mzTab-shaped result files written per experimental input

The export format is selected via `export.format` in the config (`csv` is the
stable value; `mztab` is experimental and non-conformant).

---

## Core vs Experimental Features

## Core candidate-retrieval path

These are the features the docs should treat as the main supported path:

- `massflow annotate --config ...`
- YAML configuration via `MassFlowConfig`
- open-format ingestion for `mzML`, `mzXML`, `MGF`, and `MSP`
- explicit SQLite reference-library workflows
- `matchms`-based metadata and peak processing
- `cosine` and `modified_cosine`
- legacy query-level numerical confidence filtering (uncalibrated)
- per-file CSV (stable) and experimental mzTab-shaped result export
- `massflow db build`, `inspect`, and `merge`

## Experimental or less-stable surfaces

These features exist in the repository but should be treated more cautiously:

- `spec2vec`
- `ms2deepscore`
- `consensus`
- `cascade`

See `README.md` for the user-facing guide to what is currently experimental, why it is classified that way, and how to approach those features safely.

---

## Module Responsibilities

### `MassFlow.cli`
Parses command-line arguments, configures logging, and dispatches CLI commands
into the workflow or database layers.

### `MassFlow.workflow`
Implements the annotation pipeline. It loads and processes the reference
library, discovers input files, processes query spectra in parallel, performs
chunked searching, reports uncalibrated query-level confidence diagnostics, and
writes output tables.

### `MassFlow.config`
Defines the nested configuration schema used by the CLI and workflow.
Validation is focused on structural correctness and basic parameter
constraints rather than full runtime guarantees.

### `MassFlow.io`
Provides the file-system boundary for MassFlow. It loads spectra from supported
formats, rejects unsupported vendor raw inputs, and exports result tables.

### `MassFlow.database`
Provides SQLite-backed storage for spectral libraries and helper methods for
build, inspection, merge, and spectrum streaming. In addition to the default
BLOB mode (``mz_array`` / ``intensity_array``), it supports a hybrid mode
where SQLite retains only metadata plus a ``zarr_ref`` / ``zarr_index``
reference pair and fragment arrays are persisted in a chunked Zarr store
(see ``MassFlow.zarr_store``). The ``migrate_blobs_to_zarr`` helper (wrapped
by ``scripts/migrations/0002_blobs_to_zarr.py``) migrates existing BLOB
libraries to the hybrid backend with bitwise verification.

SQLite-backed stores also carry a lightweight **library-provenance schema**
(``store_meta`` + ``library_builds``, ``PRAGMA user_version >= 1``): every
``db build``, ``db merge``, or annotate library-store build records its input
file(s) with content SHA-256, the effective config hash, the processing and
similarity (target-decoy) configuration JSON, and an exact UTC timestamp.
``massflow db inspect`` renders this history (library versions, processing
history, target-decoy configuration), and the annotate provenance sidecars
link every result to the exact build row. The schema is additive and
idempotent — raw SQL ``CREATE TABLE IF NOT EXISTS`` upgrades pre-provenance
databases in place on their next open, and provenance writes never alter
scientific payload bytes.

### `MassFlow.processing`
Applies the configured `matchms` metadata repairs and peak filtering pipeline.

### `MassFlow.similarity`
Creates and runs the scoring engines. It also defines decoy generation and
result structures used during numerical confidence reporting. Modified-cosine full scoring is the default. An experimental Numba peak/neutral-loss prefilter (`MassFlow.acceleration`) is opt-in and has no general candidate-recall guarantee. The experimental cascade can optionally use HNSW (`MassFlow.hnsw`) for approximate candidate retrieval; no universal recall or sub-linear performance claim is made.
External ML engines (Spec2Vec, MS2DeepScore) sit behind the
`MLEngineProtocol` boundary and may run remotely over REST/gRPC
(`MassFlow.ml_client`, `ml_endpoints` in the config); a circuit breaker plus
classical modified_cosine fallbacks in the meta-engines and workflow keep
runs alive when the ML service is unreachable or heavy dependencies are
missing.

### `MassFlow.streaming` (experimental)
Implements the gRPC real-time annotation server (`StreamSpectra` bidirectional
RPC plus `GetStatus` health probe). Incoming packets are validated through a
Pydantic/matchms gate, buffered in a bounded async queue with quality-gated
high-water-mark backpressure (low-quality spectra are shed under overrun and
reported as `spectra_dropped_low_quality`), micro-batched, and routed through
the `ConsensusEngine`; annotations stream back to the client as they complete.

### `MassFlow.tui` (interactive, requires the `tui` extra)
Implements the interactive terminal console (`massflow tui`): a spectral file
browser with workspace upload, an interactive spectrum viewer with
Unicode-glyph stick plots, an experimental candidate-review tab with
mirror plots, and a diagnostics tab that surfaces problems with plain-English fixes
plus the quarantine log. All heavy work runs in Textual worker threads, and
the science-facing helpers (peak downsampling, mirror alignment, plot
rendering, file discovery, error hints) are pure functions covered by
headless unit tests. See `docs/api/tui.md`.

---

## Design Principles

MassFlow follows a few simple design rules:

- **Config first:** reproducible CLI runs are preferred over ad hoc scripts.
- **Open formats only:** vendor conversion happens before MassFlow.
- **Small focused modules:** I/O, processing, scoring, and orchestration stay
  separated.
- **Predictable outputs:** the core workflow should either produce CSV results
  or return actionable errors.
- **Experimental boundaries:** advanced engines and interactive utilities may
  evolve without expanding the core support promise.

---

## Practical Example

A typical config-driven run looks like this:

```yaml
project:
  output_directory: "results/standard_analysis"

input:
  input_path: "data/experiments/experiment.mzML"
  library_path: "data/libraries/library.msp"
  format: "mzml"

similarity:
  algorithm: "cosine"
  ms1_tolerance: 0.02
  ms2_tolerance: 0.02
  tolerance_unit: "Da"
  min_score: 0.6
  min_matched_peaks: 3
  fdr_threshold: 0.05
```

Run it with:

```shell
uv run massflow annotate --config massflow_config.yaml
```

This will process the experimental file, search it against the reference
library, and write a CSV file into the configured output directory.
