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
  input_path: "data/experiments/experiment.mzML"
  # input_path: "data/experiments/" # Use this to process a whole folder
  library_path: "data/libraries/library.msp"
  format: "mzml"
  storage_backend: "sqlite"
```
*   `input_path`: The experimental file **or** directory to process. A directory is recursively scanned for supported spectral files.
*   `library_path`: The file path to your reference library (e.g., an `.msp` file or a MassFlow `.db` SQLite file).
*   `format`: (Optional) An explicit format hint. If omitted, MassFlow infers the format from the file extension.
*   `storage_backend`: (Optional) How the reference library is stored: `"sqlite"` (default, BLOB peak arrays), `"zarr"` (pure Zarr), or `"hybrid"` (SQLite metadata + chunked Zarr peak arrays). The setting has **one unambiguous meaning everywhere**: it selects the backend of the library store built for the run — `massflow db build`, the annotation pipeline (`prepare_library`), and the streaming server all honor it. Pre-existing store inputs (`.db`/`.sqlite`/`.zarr`) are opened directly in their own backend. The annotation layer consumes every backend through the single `SpectralStore` interface, so SQLite, Zarr, and hybrid libraries produce identical results (verified by `tests/test_storage_contract.py`).

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
  min_score: 0.6
  min_matched_peaks: 3
  fdr_threshold: 0.05
```
*   `algorithm`: The core engine. Stable choices are `"cosine"` and `"modified_cosine"`; `"spec2vec"` and `"ms2deepscore"` require the `ml` extra; `"consensus"` and `"cascade"` combine multiple engines (see the [Similarity API](../api/similarity.md)).
*   `ms1_tolerance`: Precursor mass tolerance (in `Da`).
*   `resolution_ppm`: Optional: Precursor mass resolution (in `ppm`). Overrides `ms1_tolerance` if set.
*   `ms2_tolerance`: Fragment mass tolerance (typically in `Da`).
*   `min_score`: The absolute minimum score required to keep a hit.
*   `min_matched_peaks`: Minimum number of matched fragment peaks.
*   `fdr_threshold`: The target False Discovery Rate (e.g., `0.05` for 5%).

Advanced `similarity` keys include `analog_search`, the `consensus_*` / `cascade_*` engine settings, the `hnsw_*` approximate-candidate retrieval knobs (requires the `hnsw` extra), and `ml_endpoints` for routing Spec2Vec/MS2DeepScore scoring to a remote REST/gRPC service with automatic fallback to the classical engines.

### `export`
Defines the output format.

```yaml
export:
  format: "csv"
```
*   `format`: `"csv"` (default) or `"mztab"` — the only formats in the stable v0.1 contract. (Other formats such as `fbmn` are documented in places but not implemented; see the [Results guide](results.md) and `docs/CAPABILITY_MATRIX.md`.)

### `workflow`
Reserved for future pipeline stages (peak picking, retention-time alignment, networking). It currently has **no active fields** — all orchestration is handled directly by the workflow module, and the stable annotation path runs on the `processing` + `similarity` sections alone.

---

## Configuration Validation

MassFlow validates the entire configuration **before any expensive
processing begins**, and rejects invalid configuration with
human-readable errors that include the YAML line number.

### Unknown keys are errors, never silent

Configuration models are strict: any key that is not part of the schema
fails validation immediately. A misspelled key such as

```yaml
similarity:
  ms2_tolerence: 0.02   # typo
```

produces:

```
Configuration validation failed:
Line 3, Key 'similarity -> ms2_tolerence': Unknown configuration key
'ms2_tolerence' under 'similarity'. Did you mean 'ms2_tolerance'?
```

No typo is silently ignored and no default is silently substituted.

### Invalid combinations fail at validation time

- `hnsw_enabled: true` requires `algorithm: "cascade"` (HNSW candidate
  retrieval only exists inside the cascade engine).
- `reduce_to_top_n_peaks: true` requires a positive `n_max` (otherwise the
  toggle would silently do nothing).
- `cascade_stages` must be a non-empty list of leaf engines (`cosine`,
  `modified_cosine`, `spec2vec`, `ms2deepscore`).
- `consensus_weights` keys must be leaf engines with positive weights.
- `fdr_threshold` and `min_score` must lie in `[0, 1]`.
- `ms2_tolerance` / `ms1_tolerance` / `rt_tolerance` must be non-negative.

### Relative paths resolve against the config file

All relative paths (`project.output_directory`, `input.input_path`,
`input.library_path`) are resolved **relative to the YAML file's
directory**, not the caller's current working directory. The same
configuration therefore behaves identically regardless of where the
`massflow` command is invoked from.

Legacy CWD-relative resolution can be restored explicitly with the
compatibility environment variable:

```bash
MASSFLOW_COMPAT_CWD_PATHS=1 massflow annotate --config massflow_config.yaml
```

Programmatically constructed configurations (no YAML file) keep their
paths untouched; relative paths then resolve against the process working
directory at use time.

### Normalized configuration in provenance

Every run writes the normalized configuration representation into
provenance:

- a run-level file `run_provenance.json` (counter-suffixed when a run
  shares an output directory) in the project output directory, written
  before any expensive processing begins;
- a `config:` section in every per-file `*.report.yaml` sidecar.

The representation contains the schema version, the absolute source
config path, the full effective configuration (all paths resolved), and
a `config_digest_sha256` over the canonical JSON, so any result can be
verified to have been produced by exactly the configuration it claims.
