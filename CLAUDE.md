# MassFlow — Project Context

## Project Purpose
MassFlow is a pre-1.0, config-first Python toolkit for local MS/MS annotation workflows. It loads open spectral formats, applies configurable `matchms` processing, runs similarity search against reference libraries, and exports structured outputs through a CLI-first workflow. Optional utilities include SQLite-backed library management and GraphML export.

## Developer Profile
- Domain expert: PhD analytical chemist (LC-MS, metabolomics)
- Programming level: novice — explain unfamiliar patterns before implementing them
- Priority: readable, well-commented code over clever/compact code

## Tech Stack
- Python 3.13+
- Core Libraries: `matchms`, `pydantic`, `PyYAML`, `numpy`, `pandas`, `pyteomics`, and the standard-library `sqlite3` module.
- Optional Analysis and UI Libraries: `ms2deepscore`, `spec2vec`, `plotext`, `networkx`, `matplotlib`.
- Data Formats: Open formats such as mzML, mzXML, MGF, MSP, and MassFlow SQLite libraries, all represented as `matchms.Spectrum` objects in the processing pipeline.
- Testing: `pytest`
- Environment Management: `uv` or `venv`

## Data Persistence
- SQLite via Python's built-in `sqlite3`
- All DB interactions isolated to `src/MassFlow/database.py`
- Schema changes must be documented in a migration note at the top of database.py
- No raw SQL strings outside of database.py

## Project Structure
The core application logic resides in `src/MassFlow/`:
- `cli.py`: Command-line interface definitions and entry points.
- `config.py`: Application configuration and settings, managed with `pydantic`.
- `database.py`: Handles interactions with the `SQLite` database.
- `io.py`: Manages input/output operations, including reading and writing spectral data files.
- `networking.py`: Optional GraphML export utilities for molecular-network views derived from annotation outputs.
- `processing.py`: Contains functions for pre-processing spectral data (e.g., cleaning, normalizing) before similarity calculations.
- `similarity.py`: Implements the core spectral similarity algorithms using `matchms`, `ms2deepscore`, and `spec2vec`.
- `workflow.py`: Orchestrates the overall data processing and similarity workflows.

## Naming & Style Conventions
- Variable names must be explicit: `precursor_mz`, `retention_time_seconds`, `spectrum_peaks`
- No abbreviations unless domain-standard (e.g., `mz`, `rt` are acceptable shorthand in comments only).
- Functions should do one thing — if a docstring needs "and", consider splitting.
- All functions must have a NumPy-style docstring with Parameters, Returns, and a brief example.

## Scientific Constraints
- **m/z values:** `float64` precision required for all mass-to-charge values and peak intensities — never cast to `float32`.
- **Spectral Data Representation:** Primary data structure for spectra should be `matchms.Spectrum` objects.
- **Missing Values:** Use `NaN` (not 0) for absent or undefined signal/metadata.
- **Metadata Integrity:** Critical spectral metadata (e.g., `precursor_mz`, `retention_time`, `adduct`, `compound_name`) must always travel with the `Spectrum` objects and not be stripped.
- **Retention Time Units:** Always explicitly state units (e.g., `retention_time_seconds` or ensure `matchms` metadata is consistently in seconds or minutes).

## Workflow Preferences
- Break problems into small, testable functions before wiring them together.
- Write the test before or alongside the function (not after).
- When proposing file I/O, confirm the target format before writing code.
- Flag any operation that could silently mutate data in-place (e.g., `matchms.Spectrum` objects).

## Current v1.0 Release Stabilization Focus
- [x] Lock the v1.0 Product Contract (Step 1 of V1_0_DEVELOPMENT_PLAN.md).
- [x] Clean the Release Surface and remove packaging/docs drift (Step 2).
- [x] Classify Core vs. Experimental features in CLI and documentation (Step 3).
- [x] Harden the `massflow annotate` workflow and SQLite library paths (Step 4 & 5).
- [x] Reduce documentation debt and finalize the v1.0 test gate (Step 6 & 7).
- [x] Prepare the v1.0.0 release candidate (Step 8).
