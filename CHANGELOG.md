# CHANGELOG

## v 0.6.0

**Summary:**
Architectural refactor of the `MassFlow` core. Transitioned from a script-based workflow to a modular system with Strategy/Facade design patterns, strict type safety, and robust configuration validation.

**Architectural Changes:**

- **Hub-and-Spoke Architecture:** Implemented a centralized `workflow.py` orchestrator that coordinates specialized modules.
- **Strategy Pattern (Similarity)**: Decoupled similarity algorithms from the workflow. `Cosine` and `ModifiedCosine` are now interchangeable strategies in `similarity.py`.
- **Facade Pattern (Processing)**: Encapsulated `matchms` filtering logic behind a simplified, strict interface in `processing.py`.
- **Pydantic Configuration**: Replaced raw dictionary config handling with strict Pydantic models in `config.py`. Mass spec parameters (m/z, RT) are now validated for physical correctness (e.g., non-negative values).

**Code Quality & Standards:**

- **Unified I/O**: Consolidated all loading/saving logic into `io.py`, removing legacy Pandas dependencies and harmonizing file access via `pathlib.Path`.
- **Type Safety**: Enforced Python 3.10+ strict type hinting across all modules.
- **Documentation**: Added `ARCHITECTURE.md` with Mermaid.js diagrams to document system flow.
- **Google-Style Docstrings**: Standardized docstrings for all classes and functions.

**Breaking Changes:**

- **Config Schema**: The YAML configuration structure has changed to align with the new Pydantic models. Old config files will require updates.
- **API**: `processing.clean_msp_library` and other standalone script functions have been removed in favor of the pipeline approach.

## v 0.5.0

- **Project Renaming and Restructuring**:
  - The project was renamed from `yogimass` to `SpectralMetricMS` and finally to `MassFlow`.
  - Core modules were restructured from a generic `config` and `workflow` to more specific `io`, `processing`, and `similarity` modules.
- **CLI Enhancements**:
  - The CLI was enhanced with colors, better feedback, and improved help messages.
  - Removed extensive CLI commands to simplify the core package.
- **Documentation**:
  - Added a comprehensive AI guide and design documentation.
  - Relocated architecture documentation.
- **Features and Fixes**:
  - Added an initial implementation of the MassFlow toolkit.
  - Implemented an MSP verification script.
  - Enhanced type hints and docstrings.
  - Updated copyright year in the LICENSE.
  - Removed unused config files and imports.
  - Added a CLI entry point and initial module structure.
- **CI/CD**:
  - Removed `cli-smoke` and `extras-test` CI jobs and simplified test execution.

## v 0.4.0

- Moved CLI into package: `MassFlow/cli.py` (installable entry point).
- Added module entry (`python -m MassFlow`) via `MassFlow/__main__.py`.
- Exposed console script via `[project.scripts]` in `pyproject.toml` (`MassFlow = "MassFlow.cli:main"`).
- Added smoke CLI tests and a CI job (`.github/workflows/ci.yml`) to run them.
- Deprecated legacy splinter CLI implementation (`splinters/workflows/MassFlow/cli.py`).

## Below v 0.4.0

- Numerous changes and feature additions.
