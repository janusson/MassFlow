# SpectralMetricMS Architecture & API

## Module Map

This document outlines the high-level architecture of `SpectralMetricMS` and the responsibilities of each module.

### Core Modules

* **`SpectralMetricMS.cli`**: The entry point for the command-line interface. Responsible for parsing arguments and dispatching commands to the workflow engine or other utilities.
* **`SpectralMetricMS.workflow`**: The orchestration engine. It reads the configuration and executes the defined steps (ingestion, processing, similarity, etc.).
* **`SpectralMetricMS.config`**: handles configuration loading (from YAML), validation, and schema definition. It provides the mechanism for "dotted" config error reporting.

### Submodules

* **`SpectralMetricMS.similarity`**:
  * **`library.py`**: Manages `LocalSpectralLibrary` (JSON/SQLite). Handles storage and retrieval of spectra.
  * **`backends.py`**: Implements search algorithms (Naive, Annoy, etc.).
  * **`processing.py`**: Wrappers around `matchms` filters and processors.
* **`SpectralMetricMS.io`**:
  * Handles reading and writing of spectral data formats (MGF, MSP) and integration with other tools (MS-DIAL).
* **`SpectralMetricMS.filters`**:
  * Custom filtering logic and metadata cleaning.
* **`SpectralMetricMS.networking`**:
  * Logic for constructing similarity networks from search results.
* **`SpectralMetricMS.reporting`**:
  * Generation of reports and QC metrics.

## Public API Surface

The following modules and functions are considered the public API. All others are internal.

### `SpectralMetricMS`

* `__version__`

### `SpectralMetricMS.cli`

* `main()`

### `SpectralMetricMS.config`

* `ConfigError`
* `load_config()` (Planned)

### `SpectralMetricMS.workflow`

* `run_workflow()` (Planned)

### `SpectralMetricMS.similarity`

* `LocalSpectralLibrary` (Planned)
* `SpectrumProcessor` (Planned)

---

**Note:** This architecture document is a living document and will be updated as the implementation evolves in Weeks 2 and 3.
