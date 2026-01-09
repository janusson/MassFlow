# MassFlow Architecture & API

## Module Map

This document outlines the high-level architecture of `MassFlow` and the responsibilities of each module.

### Core Modules

* **`MassFlow.cli`**: The entry point for the command-line interface. Responsible for parsing arguments and dispatching commands to the workflow engine or other utilities.

* **`MassFlow.workflow`**: The orchestration engine. It reads the configuration and executes the defined steps (ingestion, processing, similarity, etc.).
* **`MassFlow.config`**: handles configuration loading (from YAML), validation, and schema definition. It provides the mechanism for "dotted" config error reporting.

### Submodules

* **`MassFlow.similarity`**:
  * **`library.py`**: Manages `LocalSpectralLibrary` (JSON/SQLite). Handles storage and retrieval of spectra.
  * **`backends.py`**: Implements search algorithms (Naive, Annoy, etc.).
  * **`processing.py`**: Wrappers around `matchms` filters and processors.
* **`MassFlow.io`**:
  * Handles reading and writing of spectral data formats (MGF, MSP) and integration with other tools (MS-DIAL).
* **`MassFlow.filters`**:
  * Custom filtering logic and metadata cleaning.
* **`MassFlow.networking`**:
  * Logic for constructing similarity networks from search results.
* **`MassFlow.reporting`**:
  * Generation of reports and QC metrics.

## Public API Surface

The following modules and functions are considered the public API. All others are internal.

### `MassFlow`

* `__version__`

### `MassFlow.cli`

* `main()`

### `MassFlow.config`

* `ConfigError`
* `load_config()` (Planned)

### `MassFlow.workflow`

* `run_workflow()` (Planned)

### `MassFlow.similarity`

* `LocalSpectralLibrary` (Planned)
* `SpectrumProcessor` (Planned)

---

**Note:** This architecture document is a living document and will be updated frequently.
