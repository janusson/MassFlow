# Result Export & FDR

When MassFlow completes an annotation run, it writes reproducible, tabular outputs designed for downstream review and statistical confidence. The stable v1.0 contract focuses on generating per-file results (CSV, mzTab-M) accompanied by YAML provenance reports, alongside optional GNPS FBMN paired exports.

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

### GNPS FBMN Export (`fbmn`)
Generates the specific pair of files required to run Feature-Based Molecular Networking (FBMN) on the GNPS web platform.
1.  **`experiment_results.csv`**: The feature quantification table.
2.  **`consensus_spectra.mgf`**: An aggregated MGF file containing the representative MS2 spectra for the annotated features.
3.  **`experiment_results.report.yaml`**: The provenance sidecar.

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
- The number of query spectra processed and the number of retained results.

This ensures you can always reproduce how a specific CSV was generated months later, meeting the standards of scientific reproducibility.

---

## False Discovery Rate (FDR)

MassFlow natively calculates a Target-Decoy False Discovery Rate (FDR) to provide statistical confidence in the reported annotations.

### How It Works

1.  **Decoy Generation:** Before searching, MassFlow generates a set of "decoy" spectra from your reference library. These are mathematically shuffled or shifted versions of the true targets that are known *not* to exist in your sample.
2.  **Scoring:** Your experimental queries are scored against both the true target library and the decoy library.
3.  **q-value Calculation:** By comparing the distribution of target scores to decoy scores, MassFlow estimates a *q*-value for each hit. A *q*-value of 0.05 means there is an estimated 5% chance that the hit is a false positive.
4.  **Filtering:** Any hit with a *q*-value greater than the configured `fdr_threshold` (e.g., `0.05`) is discarded before the CSV is exported.

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
