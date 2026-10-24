# Result Export & Confidence States

MassFlow writes per-file candidate-search results in CSV or mzTab-M with YAML provenance reports. Scores and confidence states are evidence for review, not confirmed identities or calibrated error bounds.

!!! warning "GNPS FBMN export is not shipped"
    The `fbmn` export format and `consensus_spectra.mgf` are documented in
    places but **not implemented** in this release (`export.format` accepts
    only `csv` and `mztab`). Do not rely on them. See
    `docs/CAPABILITY_MATRIX.md` for the authoritative capability list.

---

## Output Structure

If you process a single experimental file (e.g., `experiment.mzML`), MassFlow will generate standard output files in your configured `project.output_directory` depending on your `export.format`:

### CSV Export (`csv`)
The simplest and most common format.
1.  **`experiment_results.csv`**: The main tabular annotation report.
2.  **`experiment_results.report.yaml`**: The provenance sidecar containing the runtime context.

### mzTab-M Export (`mztab`)
A structured plain-text interchange format. Full mzTab-M standards conformance has not been independently qualified for this release scope.
1.  **`experiment_results.mztab`**: Contains both the experimental metadata (MTD section) and the feature/annotation lists (SML/SME sections) in a tightly controlled schema.
2.  **`experiment_results.report.yaml`**: The provenance sidecar.

### GNPS FBMN Export (`fbmn`) — not shipped

Not implemented in this release. `export.format` accepts only `csv` and
`mztab`; there is no `consensus_spectra.mgf` output. (Planned/aspirational
only — see `docs/CAPABILITY_MATRIX.md` §2.4.)

### The Result Table

The table contains retained candidate matches, including `score`, `matched_peaks`, and a legacy `Annotation_Status` category. A candidate score is not an identification probability.

Each candidate row also contains `reference_metadata`, a compact JSON object that
preserves the source record ID, original accession/identifier, source path and
SHA-256/ordinal locator when available, plus the acquisition metadata carried by
the library record (for example collision energy, instrument, adduct and charge).
This acquisition identity is distinct from chemical identity such as InChIKey.

`Annotation_Status` values:

| Status | Meaning |
| :--- | :--- |
| `Matched` | Legacy score category (score ≥ 0.9); never a confirmed identity. Assessed unvalidated rows use Uncalibrated. |
| `Putative` | Legacy score category (score < 0.9); not a validated confidence level. |
| `Uncalibrated` | The query confidence state is `estimated_uncalibrated`, `unavailable`, or `invalid`. Its legacy `q_value` may remain for compatibility, but neither a non-empty nor empty decoy null establishes FDR control. See [confidence semantics](confidence-semantics.md). |
| `Unknown` | No hit for this query (base row). |

**Example: `experiment_results.csv`**

| query_id | query_precursor_mz | reference_id | reference_name | score | matched_peaks | Annotation_Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query_0 | 195.0 | ref_12 | Caffeine | 0.98 | 5 | Matched |
| query_1 | 304.0 | | | | | Unknown |
| query_2 | 150.0 | ref_8 | Unknown_Metabolite | 0.75 | 3 | Putative |

!!! note "Unmatched Queries"
    The CSV includes matched *and* unmatched query spectra. A query with no retained candidate still has a row with reference-specific columns left blank. Historical q-value filtering may affect candidate retention, but it is not validated FDR control.

### The Provenance Sidecar Report

The sidecar report (`.report.yaml`) acts as a hard link between your CSV results and the run conditions that produced it. It captures:

- When the analysis was run (`report_created_at`).
- Which query file and library file were used (`query_file`, `library_path`).
- The path to the original `massflow_config.yaml`.
- The exact parsed configurations (`processing`, `similarity`, `workflow`) that were applied.
- The normalized configuration representation (`config`) with its SHA-256 digest.
- The number of query spectra processed and the number of retained results.
- A `library` lineage block that links the results to the **exact database that was searched** (see below).

#### Library lineage (`library`)

When the reference library is a MassFlow database (built by `massflow db build`
or prepared from a raw file during `annotate`), the sidecar records where the
results came from and which build produced that database:

```yaml
library:
  configured_library_path: results/master_library.db   # what the config pointed at
  store:
    path: results/master_library.db                    # the store actually searched
    kind: store
    storage_backend: sqlite
    spectrum_count: 15420
    schema_version: 1
    created_at: 2026-09-06T10:00:00+00:00
  build:                                               # the library_builds row
    id: 3
    built_at: 2026-09-06T10:00:00+00:00
    source_path: data/libraries/example_library.msp    # input file used to build it
    category: personal
    spectrum_count: 15420
    config_digest_sha256: 5c7725e3…                     # config hash recorded at build
    processing_fingerprint: 9f4a…
    massflow_version: 0.1.0
```

Comparing the sidecar's own `config.config_digest_sha256` (this run) with
`library.build.config_digest_sha256` (the database build) tells you at a glance
whether the database was built with the exact configuration this run used. When
the store has no recorded history (e.g. a pre-provenance database), `build` is
`null` rather than guessed. The same lineage is queryable per database with
`massflow db inspect`.

This ensures you can always reproduce how a specific CSV was generated months later, meeting the standards of scientific reproducibility.

### Run-Level Provenance (`run_provenance.json`)

Every annotation run writes a first-class run-provenance record into the
output directory (`run_provenance.json`, counter-suffixed when runs share
an output directory). It is written **before** any per-file processing
begins and finalized with the completion summary afterwards. The record
contains:

| Field | Meaning |
| --- | --- |
| `massflow_version` | The MassFlow package version |
| `git_sha` / `git_dirty` | The exact checkout commit (and whether it was dirty) |
| `python_version` / `python_implementation` / `platform` | The interpreter and OS |
| `dependencies` | Resolved versions of every direct dependency |
| `lockfile_digest_sha256` | SHA-256 of the committed `uv.lock` |
| `effective_config` / `config_digest_sha256` | The normalized configuration and its digest |
| `engine` / `processing` | The similarity-engine and processing configuration |
| `backend` | The storage backend (`sqlite` / `zarr` / `hybrid`) |
| `decoy_seed` / `decoy_config` | The decoy-generation seed and parameters |
| `input_file_hashes` | SHA-256 of every experimental input file |
| `reference_library_sha256` / `reference_library_kind` | The reference-library digest |
| `run_started_at` / `completed_at` | Explicitly time-varying timestamps (UTC ISO-8601) |
| `results` | Completion summary: per-status counts, aggregated warnings, degraded-mode flags, failed files |

Provenance is deterministic: two runs from the same checkout and lockfile
over the same inputs produce identical records except for the explicitly
time-varying fields (`run_started_at`, `completed_at`).

### Per-File Execution Status (Failure Model)

Every experimental input file produces exactly one structured execution outcome (`MassFlow.workflow.FileExecutionResult`) with the following contract:

| Field | Meaning |
| --- | --- |
| `status` | `success` \| `degraded` \| `failed` |
| `input_path` | The experimental input file |
| `spectra_loaded` | Spectra that passed I/O validation and entered processing |
| `spectra_rejected` | Spectra dropped by validation or processing (reported explicitly, never silently) |
| `hits_produced` | Annotation hits exported for the file |
| `output_path` | Written results file (`None` for failed files) |
| `warnings` | Non-fatal caveats (e.g. small-library FDR) |
| `fatal_errors` | Non-empty iff the file failed |
| `degraded_mode_flags` | Machine-readable degradation markers (`engine_fallback:<algo>`, `consensus_*`, `cascade_*`, `routing_*`, `decoy_null_empty`, `fdr_uncalibrated`) |

The status is recorded in the provenance sidecar, so a degraded or partially failed run is never indistinguishable from a clean one:

* **`success`** — the file was fully processed with the configured pipeline.
* **`degraded`** — results were produced, but part of the configured pipeline fell back (engine fallback, uncalibrated FDR); the flags explain what changed.
* **`failed`** — the file's data were not processed. **No results CSV is written for a failed file** (an empty CSV would be mistaken for a successful annotation); instead an explicit `<stem>_failed.report.yaml` records the fatal errors.

Batch runs continue across files: one bad file does not stop the others, but the CLI exits **nonzero** when any file failed, and the per-file summary printed by `massflow annotate` accounts for every input. Unsupported vendor formats (`.raw`, `.d`, `.wiff`, ...) are never silently ignored: a vendor raw **library** or a direct single-file/`.d` query input aborts at **pre-flight** (before any store is built, with the conversion hint), while vendor files inside a scanned input directory are announced at run start and reported as explicit per-file failures with a conversion hint — they never silently disappear from a batch. See [the annotation guide](annotation.md) (Pre-flight validation) for the full contract.

---

## False Discovery Rate (FDR)

MassFlow calculates a numerical query-maxima estimate, not empirically demonstrated
FDR control. See [confidence semantics](confidence-semantics.md) for the exact
formula, explicit states, nullable estimate, family, inherited-row limitations and
historical column/filter compatibility. Both populated and empty nulls remain
scientifically uncalibrated; no nominal q threshold proves an observed error rate.

Increasing `fdr_threshold` only changes selection; it does not cure a sparse or
unrepresentative decoy null. Inspect counts, methodology and limitations before
interpreting any confidence column.
## Numerical confidence diagnostics

The implemented target/decoy procedure reports a query-level numerical estimate;
its empirical calibration is pending. Decoys are synthetic altered spectra
intended to provide a null comparison, but exchangeability with incorrect
chemical matches has not been established.

1. Decoys are generated from reference records and searched using the configured
   engine and candidate gates.
2. Each query contributes at most its best target score and best surviving decoy
   score to the numerical competition.
3. The implementation computes a monotone q-like estimate. `estimated_uncalibrated`
   means the calculation had target and decoy observations, not that its error
   rate has been validated. An empty decoy null is `unavailable`; the historical
   `1/N` value is a rank diagnostic, not an error bound.
4. Candidate rows inherit the query-level assessment; it is not candidate-specific.
5. `fdr_threshold` remains a compatibility filter on the numerical value. It does
   not establish controlled FDR or confirm the retained structures.

The exported `p_value` remains a diagnostic only. See
[confidence semantics](confidence-semantics.md) for the method, scope, explicit
states, limitations and migration details.

### Decoy-count warnings

The workflow may warn when a library is below its configured count heuristic.
That warning is not a scientifically validated minimum size, and no decoy count
threshold establishes FDR control. A sparse or unrepresentative decoy null should
be reported with its confidence state and limitations; relaxing `fdr_threshold`
changes which candidates are retained, not the validity of the estimate.
