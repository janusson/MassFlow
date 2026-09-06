# CLI Workflows

The core of MassFlow v0.1 is the config-driven CLI. These workflows are designed to be fully reproducible, predictable, and simple enough to execute in an automated shell script or CI/CD pipeline.

## Standard Annotation

The `annotate` subcommand requires a single argument: the path to your validated YAML configuration file.

```shell
uv run massflow annotate --config massflow_config.yaml
```

### How It Works Under the Hood

When you execute this command, MassFlow orchestrates the following pipeline:

1.  **Configuration Parsing (`config.py`):** The YAML file is loaded and strictly validated against Pydantic schemas. If a path doesn't exist, or an invalid tolerance is provided, the pipeline fails fast before consuming heavy resources.
2.  **Library Loading (`io.py`, `database.py`):** MassFlow reads your reference library (e.g., `library.msp` or an optimized SQLite `library.db`).
3.  **Reference Processing (`processing.py`):** Every spectrum in the library is passed through the configured metadata harmonization and peak-filtering pipeline. If the processed library is statistically too small to support robust False Discovery Rate (FDR) calculations, MassFlow will log a critical scientific warning.
4.  **Experimental Input Discovery (`workflow.py`):** The pipeline looks for your experimental data in the configured `input_path`. If it's a directory, MassFlow recursively discovers all supported open formats (`mzML`, `MGF`, etc.).
5.  **Per-File Multiprocessing (`workflow.py`):** Each experimental file is handed off to a separate worker process. MassFlow utilizes a Zero-I/O shared memory architecture, loading the reference library into RAM once, preventing redundant parsing across CPU cores.
6.  **Query Processing (`processing.py`):** The experimental queries are cleaned and filtered using the exact same configured pipeline as the reference library, ensuring parity for scoring.
7.  **Chunked Searching (`similarity.py`):** To avoid memory exhaustion, the worker searches the processed queries against the reference library in chunks using the configured engine (e.g., `cosine`).
8.  **FDR Calculation (`similarity.py`):** Decoy spectra are generated from the references, scored, and used to calculate *q*-values. Hits failing the `fdr_threshold` or the `min_score` are discarded.
9.  **Reporting (`io.py`):** Finally, MassFlow exports a clean result table for *each* experimental file, accompanied by a YAML sidecar file detailing the exact provenance of that run.

---

## Interactive Live Reloading (`watch`)

If you are actively optimizing your data processing thresholds (e.g., tweaking `noise_threshold` or `min_score`), repeatedly running the full `annotate` pipeline can be tedious. MassFlow provides an interactive, live-reloading terminal UI:

```shell
uv run massflow watch --config massflow_config.yaml
```

**Features:**
*   **Real-time Previews:** Displays a live `Rich` table of the top 15 results directly in your terminal.
*   **Hot-Reloading:** The pipeline automatically re-triggers the moment you save changes to your `massflow_config.yaml` or any files in your `input_path`.
*   **Silent Error Handling:** If you make a typo in your config, `watch` mode will catch the error, display it, and wait for your next save without crashing.

---

## Vendor File Conversion (`convert`)

MassFlow is intentionally conservative at the I/O boundary. The core `annotate` pipeline explicitly supports **only** open spectral formats:

*   `.mzML`
*   `.mzXML`
*   `.MGF`
*   `.MSP`
*   `.db` / `.sqlite` (MassFlow native databases)

Attempting to pass proprietary vendor formats directly to the annotator will result in an `UnsupportedVendorFormatError`. To solve this, MassFlow provides a wrapper command around [ProteoWizard's `msconvert`](https://proteowizard.sourceforge.io/):

```shell
uv run massflow convert --input data/raw_files/ --output data/mzml_files/
```

*Note: You must have ProteoWizard installed on your system and available in your `PATH` for this command to work.*

---

## Network Visualization (not shipped)

!!! warning "Not implemented"
    Molecular networking / GraphML export and the `massflow visualize`
    command are **planned, not implemented** in the current release. The
    configuration key and this workflow do not exist yet — do not rely on
    them. See [docs/index.md](../index.md) (Stable vs. Experimental) and
    `docs/CAPABILITY_MATRIX.md` for the authoritative capability list.
