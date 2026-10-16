# Result Export & FDR

When MassFlow completes an annotation run, it writes reproducible, tabular outputs designed for downstream review and statistical confidence. The stable v0.1 contract generates per-file results (CSV, mzTab-M) accompanied by YAML provenance reports.

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
An industry-standard, plain-text format specifically designed for reporting metabolomics results to public repositories (like MetaboLights).
1.  **`experiment_results.mztab`**: Contains both the experimental metadata (MTD section) and the feature/annotation lists (SML/SME sections) in a tightly controlled schema.
2.  **`experiment_results.report.yaml`**: The provenance sidecar.

### GNPS FBMN Export (`fbmn`) — not shipped

Not implemented in this release. `export.format` accepts only `csv` and
`mztab`; there is no `consensus_spectra.mgf` output. (Planned/aspirational
only — see `docs/CAPABILITY_MATRIX.md` §2.4.)

### The Result Table

The CSV contains the actual annotation hits, including the computed `score`, `matched_peaks`, and an automated `Annotation_Status` tag (`Matched`, `Putative`, `Uncalibrated`, or `Unknown`).

`Annotation_Status` values:

| Status | Meaning |
| :--- | :--- |
| `Matched` | Score ≥ 0.9 and the run was FDR-calibrated (its queries competed against decoys). |
| `Putative` | Score < 0.9 and the run was FDR-calibrated. |
| `Uncalibrated` | The run produced **no decoy competition at all**, so its `q_value` column is the conservative `1/N` rank bound rather than a target-decoy FDR estimate; the file is also flagged `fdr_uncalibrated` + `decoy_null_empty` and carries a warning with the measured best decoy score. The scored hits are still exported — they remain score-filtered candidates — but no row may be read as FDR-controlled. See [Scoring & FDR](scoring_logic.md#5-when-is-fdr-used-what-happens-for-small-libraries). |
| `Unknown` | No hit for this query (base row). |

**Example: `experiment_results.csv`**

| query_id | query_precursor_mz | reference_id | reference_name | score | matched_peaks | Annotation_Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query_0 | 195.0 | ref_12 | Caffeine | 0.98 | 5 | Matched |
| query_1 | 304.0 | | | | | Unknown |
| query_2 | 150.0 | ref_8 | Unknown_Metabolite | 0.75 | 3 | Putative |

!!! note "Unmatched Queries"
    The CSV includes matched *and* unmatched query spectra. If a query spectrum has no retained hit after score and FDR filtering, the row is still exported, but the reference-specific columns are left blank. This allows you to confirm that the input was successfully processed even when no annotation was found.

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

## Zero-Hit Runs (All-`Unknown` Results)

A run that exports rows but retains **no** accepted identification is a valid scientific result, not a failure. MassFlow distinguishes the causes and only attributes a zero-hit outcome to a stage when the pipeline actually has evidence for it:

| # | Situation | Definitive? | Evidence |
| :--- | :--- | :--- | :--- |
| 1 | **Queries rejected during processing** | Yes | `spectra_rejected` / `spectra_loaded` in the `.report.yaml`; rejections in `massflow_quarantine.log`. An all-rejected file is reported as `failed`, not an empty success. |
| 2 | **No compatible library precursors** | Yes for `cosine` (its MS1 gate); a likelihood factor for engines without an MS1 gate (e.g. `modified_cosine`) | `massflow preflight` precursor-compatibility coverage. |
| 3 | **Candidates failing spectral matching** | Where counters permit | `preflight` MS2 peak-count distribution, and the target-side funnel in the run diagnostics. |
| 4 | **Candidates filtered by score / statistical acceptance** | Where counters permit | `fdr_summary` and the `fdr_uncalibrated` / small-library warnings. |

`massflow preflight` reports situations 1–3 before you run; the per-file report and warnings cover the rest. Because a small library is a *hard ceiling* on how many queries can match, `preflight` reports precursor coverage as a fraction and never promises identifications.

---

## False Discovery Rate (FDR)

MassFlow natively calculates a Target-Decoy False Discovery Rate (FDR) to provide statistical confidence in the reported annotations. The full statistical contract — competition unit, tie handling, small-library behavior, and heterogeneous engines — is defined in [Scoring Logic](scoring_logic.md).

### How It Works

1.  **Decoy Generation:** Before searching, MassFlow generates a set of "decoy" spectra from your reference library. These are shuffled, jittered versions of the true targets that are known *not* to exist in your sample. Decoy generation is deterministic and identical across single-file, multi-file, and streaming runs.
2.  **Scoring:** Your experimental queries are scored against both the true target library and the decoy library.
3.  **Per-Query Competition:** The competition unit is the **query spectrum**. Each query's best target hit competes against its best decoy hit exactly once — a query with many hits is never counted more than once.
4.  **q-value Calculation:** By comparing the distribution of per-query best target scores to per-query best decoy scores, MassFlow estimates a *q*-value for each query. A *q*-value of 0.05 means that if all queries with `q ≤ 0.05` are accepted, an estimated 5% of the accepted queries' top annotations are false positives. Every exported row of a query carries that query's q-value.
5.  **Filtering:** Any hit whose query's *q*-value exceeds the configured `fdr_threshold` (e.g., `0.05`) is discarded before the CSV is exported.

An additional `p_value` column is exported as a **diagnostic**: the fraction of decoy competitions that matched or beat the query's best score. It is never used for filtering.

### Small Library Warnings

A smaller library limits the available target/decoy evidence and can reduce statistical precision. MassFlow flags libraries below its current `< 2000`-spectrum size recommendation, but size alone does **not** establish whether a particular run's FDR is calibrated; inspect that file's competition counts and `fdr_uncalibrated` flag. Two *independent* things are reported:

1. **Smallness itself** — always reported, because a small library also caps how many query spectra can match at all. It is a fact about the library and is **not** suppressed by changing `fdr_threshold`.
2. **Strict-threshold caution** — shown only when `fdr_threshold` is strict (`< 0.1`): limited library size may reduce the precision of the decoy estimate, so a strict threshold can exclude plausible candidates. Whether the run actually has decoy calibration is determined by its per-file diagnostics, not library size alone.

You will see a notice of this form:

```text
SMALL REFERENCE LIBRARY: 150 spectra.
This is below the ~2000-spectrum library-size recommendation and caps how
 many query spectra can match. Library size alone does not establish whether
 this run's FDR is calibrated.

fdr_threshold is 0.01 (strict). A smaller library limits the available
 target/decoy evidence and may reduce estimate precision. Check this run's
 per-file decoy-competition counts and calibration flag.

Recommendation:
1. Use a larger, relevant library to support calibrated FDR estimation.
2. Do not relax the FDR threshold just to obtain hits. Retain the configured
   safeguards and treat an empty result as inconclusive when calibration is
   underpowered.
```

The small-library caveat is not a reason to relax the FDR threshold just to obtain hits: doing so removes the intended statistical filter without making the calibration more reliable. Use a larger, relevant reference library for calibrated confidence, or report that the specialized-library result is exploratory and statistically underpowered. Relaxing `fdr_threshold` does **not** remove the smallness notice — the library is still small.
