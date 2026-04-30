# CLI Annotation

The core of MassFlow v1.0 is the config-driven `massflow annotate` command. This workflow is designed to be fully reproducible, predictable, and simple enough to execute in an automated shell script or CI/CD pipeline.

## The Command

The `annotate` subcommand requires a single argument: the path to your validated YAML configuration file.

```shell
uv run massflow annotate --config massflow_config.yaml
```

## How It Works Under the Hood

When you execute this command, MassFlow orchestrates the following pipeline:

1.  **Configuration Parsing (`config.py`):** The YAML file is loaded and strictly validated against Pydantic schemas. If a path doesn't exist, or an invalid tolerance is provided, the pipeline fails fast before consuming heavy resources.
2.  **Library Loading (`io.py`, `database.py`):** MassFlow reads your reference library (e.g., `library.msp` or an optimized SQLite `library.db`).
3.  **Reference Processing (`processing.py`):** Every spectrum in the library is passed through the configured metadata harmonization and peak-filtering pipeline. If the processed library is statistically too small to support robust False Discovery Rate (FDR) calculations, MassFlow will log a critical scientific warning.
4.  **Experimental Input Discovery (`workflow.py`):** The pipeline looks for your experimental data. If a `data_directory` is configured instead of a single `file_path`, MassFlow recursively discovers all supported open formats (`mzML`, `MGF`, etc.).
5.  **Per-File Multiprocessing (`workflow.py`):** Each experimental file is handed off to a separate worker process.
6.  **Query Processing (`processing.py`):** The experimental queries are cleaned and filtered using the exact same configured pipeline as the reference library, ensuring parity for scoring.
7.  **Chunked Searching (`similarity.py`):** To avoid memory exhaustion, the worker searches the processed queries against the reference library in chunks (default: 2000 spectra per chunk) using the configured engine (e.g., `cosine`).
8.  **FDR Calculation (`similarity.py`):** Decoy spectra are generated from the references, scored, and used to calculate *q*-values. Hits failing the `fdr_threshold` or the `min_score` are discarded.
9.  **Reporting (`io.py`):** Finally, MassFlow exports a clean CSV result table for *each* experimental file, accompanied by a YAML sidecar file detailing the exact provenance of that run.

## Supported Input Formats

MassFlow is intentionally conservative at the I/O boundary. The stable v1.0 contract explicitly supports **only** open spectral formats:

*   `.mzML`
*   `.mzXML`
*   `.MGF`
*   `.MSP`
*   `.db` / `.sqlite` (MassFlow native databases)

### Vendor Raw Conversion is Out of Scope

MassFlow intentionally does **not** perform vendor raw conversion internally. Attempting to ingest `.raw` (Thermo), `.d` (Agilent/Bruker), `.wiff` (SCIEX), or other proprietary formats will result in a hard failure:

```text
UnsupportedVendorFormatError: MassFlow requires open data formats. Please convert vendor files to .mzML or .mgf using ProteoWizard or MS-DIAL prior to pipeline ingestion.
```

This ensures MassFlow remains a lightweight, portable Python toolkit without requiring massive, OS-specific binary dependencies.

## Output Structure

If you process a single experimental file (e.g., `data/experiments/COE001.mzML`), MassFlow will generate two files in your configured `output_directory`:

1.  **`COE001_results.csv`**: The main tabular annotation report.
2.  **`COE001_results.report.yaml`**: The provenance sidecar containing the runtime context and the exact configuration parameters that produced the CSV.
