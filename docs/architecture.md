# Yogimass Architecture & API

## Module Map

This document outlines the high-level architecture of `yogimass` and the responsibilities of each module.

### Core Modules

*   **`yogimass.cli`**: The entry point for the command-line interface. Responsible for parsing arguments and dispatching commands to the workflow engine or other utilities.
*   **`yogimass.workflow`**: The orchestration engine. It reads the configuration and executes the defined steps (ingestion, processing, similarity, etc.).
*   **`yogimass.config`**: handles configuration loading (from YAML), validation, and schema definition. It provides the mechanism for "dotted" config error reporting.

### Submodules

*   **`yogimass.similarity`**:
    *   **`library.py`**: Manages `LocalSpectralLibrary` (JSON/SQLite). Handles storage and retrieval of spectra.
    *   **`backends.py`**: Implements search algorithms (Naive, Annoy, etc.).
    *   **`processing.py`**: Wrappers around `matchms` filters and processors.
*   **`yogimass.io`**:
    *   Handles reading and writing of spectral data formats (MGF, MSP) and integration with other tools (MS-DIAL).
*   **`yogimass.filters`**:
    *   Custom filtering logic and metadata cleaning.
*   **`yogimass.networking`**:
    *   Logic for constructing similarity networks from search results.
*   **`yogimass.reporting`**:
    *   Generation of reports and QC metrics.

## Public API Surface

The following modules and functions are considered the public API. All others are internal.

### `yogimass`

*   `__version__`

### `yogimass.cli`

*   `main()`

### `yogimass.config`

*   `ConfigError`
*   `load_config()` (Planned)

### `yogimass.workflow`

*   `run_workflow()` (Planned)

### `yogimass.similarity`

*   `LocalSpectralLibrary` (Planned)
*   `SpectrumProcessor` (Planned)

---

**Note:** This architecture document is a living document and will be updated as the implementation evolves in Weeks 2 and 3.
