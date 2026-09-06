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

The CSV contains the actual annotation hits, including the computed `score`, `matched_peaks`, and an automated `Annotation_Status` tag (e.g., `Matched`, `Putative`, or `Unknown`).

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
| `degraded_mode_flags` | Machine-readable degradation markers (`engine_fallback:<algo>`, `consensus_*`, `cascade_*`, `routing_*`, `fdr_uncalibrated`) |

The status is recorded in the provenance sidecar, so a degraded or partially failed run is never indistinguishable from a clean one:

* **`success`** — the file was fully processed with the configured pipeline.
* **`degraded`** — results were produced, but part of the configured pipeline fell back (engine fallback, uncalibrated FDR); the flags explain what changed.
* **`failed`** — the file's data were not processed. **No results CSV is written for a failed file** (an empty CSV would be mistaken for a successful annotation); instead an explicit `<stem>_failed.report.yaml` records the fatal errors.

Batch runs continue across files: one bad file does not stop the others, but the CLI exits **nonzero** when any file failed, and the per-file summary printed by `massflow annotate` accounts for every input. Unsupported vendor formats (`.raw`, `.d`, `.wiff`, ...) are discovered, attempted, and reported as explicit failures with a conversion hint — they never silently disappear from a batch.

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

Statistical FDR requires a sufficiently large null distribution (decoy set) to be accurate. If your reference library is too small (currently defined as `< 2000` spectra), the FDR calculation will be fundamentally under-powered.

If this happens, MassFlow will log a **CRITICAL SCIENTIFIC WARNING**:

```text
CRITICAL SCIENTIFIC WARNING: SMALL LIBRARY DETECTED
The library contains only 150 spectra.
Target-Decoy False Discovery Rate (FDR) statistics are fundamentally invalid on
small sample sizes because the decoy null-distribution will be too sparse.
A strict FDR threshold (currently set to 0.01) will likely eliminate all true and putative matches as false positives.

Recommendation:
1. Use a comprehensive library (e.g., GNPS, MoNA, NIST) for FDR validation.
2. Or, if using a small specialized library, relax the `fdr_threshold`
   (e.g., 0.1 or 1.0) in your config to evaluate raw Cosine scores directly.
```

If you see this warning, it is highly recommended to increase your `fdr_threshold` to `1.0` in your YAML config and rely purely on the absolute `min_score` and `min_matched_peaks` thresholds to filter your results.
