# YAML Configuration

MassFlow operates on a **Config-First** principle. Instead of writing custom Python scripts for every analysis, all operational parameters—from file paths to noise thresholds—are defined in a single YAML file.

This ensures your MS/MS annotation workflows are completely reproducible.

---

## Generating a Template

The easiest way to start is by letting the CLI generate a canonical configuration file for you:

```shell
uv run massflow init --output massflow_config.yaml
```

This creates a `massflow_config.yaml` file with sensible defaults for a classical similarity search.

---

## The Configuration Schema

The YAML configuration is broken down into modular sections. Under the hood, MassFlow uses `Pydantic` to strictly validate these fields before any processing begins.

### `project`
Defines high-level run metadata.

```yaml
project:
  name: "Standard_Annotation_Project"
  output_directory: "results/standard_analysis"
```
*   `name`: A descriptive string for your run.
*   `output_directory`: The folder where the resulting CSVs and YAML sidecar reports will be saved.

### `input`
Defines where MassFlow should look for your experimental data and your reference library.

```yaml
input:
  file_path: "data/experiments/experiment.mzML"
  # data_directory: "data/experiments/" # Use this to process a whole folder
  library_path: "data/libraries/library.msp"
  format: "mzml"
```
*   `file_path` or `data_directory`: Provide exactly one of these. `data_directory` will recursively discover supported spectral files.
*   `library_path`: The file path to your reference library (e.g., an `.msp` file or a MassFlow `.db` SQLite file).
*   `format`: (Optional) An explicit format hint. If omitted, MassFlow infers the format from the file extension.

### `processing`
Controls how spectra are cleaned and filtered *before* they are scored. See the [Processing & Filtering](processing.md) guide for deep dives into these toggles.

```yaml
processing:
  clean_metadata: true
  filter_by_intensity: true
  noise_threshold: 1000.0
  min_intensity: 0.0
  filter_min_peaks: true
  min_peaks: 5
```

### `similarity`
Defines the scoring algorithm and the strict chemical constraints required for a valid match. See the [Classical Similarity](similarity.md) guide.

```yaml
similarity:
  algorithm: "cosine"
  ms1_tolerance: 0.02
  ms2_tolerance: 0.02
  tolerance_unit: "Da"
  min_score: 0.6
  min_matched_peaks: 3
  fdr_threshold: 0.05
```
*   `algorithm`: The core engine. Stable choices are `"cosine"` and `"modified_cosine"`.
*   `ms1_tolerance`: Precursor mass tolerance (in `Da`).
*   `resolution_ppm`: Optional: Precursor mass resolution (in `ppm`). Overrides `ms1_tolerance` if set.
*   `ms2_tolerance`: Fragment mass tolerance (typically in `Da`).
*   `min_score`: The absolute minimum score required to keep a hit.
*   `fdr_threshold`: The target False Discovery Rate (e.g., `0.05` for 5%).

### `export`
Defines the output format.

```yaml
export:
  format: "csv"
```
*   `format`: The only currently stable export format is `"csv"`.

---

## Experimental Sections

You may see other fields generated in the template (such as `workflow.perform_networking` or ML model paths). These are reserved for [Experimental Features](../experimental/ml-engines.md) and are not part of the stable v1.0 pipeline contract.
