# Changelog

All notable changes to MassFlow will be documented in this file.

## [Unreleased]
### Added
- **`massflow tutorial` CLI command:** First-class command that generates a self-contained synthetic dataset (reference library, experimental queries, and pre-configured YAML) for instant local evaluation. No external files required.
- **Enhanced tutorial generator:** `scripts/generate_tutorial_data.py` now prints formatted next-steps with the exact `db build` and `annotate` commands to run after generation.

### Changed
- **Onboarding overhaul:** Rewrote `docs/user-guide/usage.md` to use real, runnable tutorial file paths instead of hypothetical placeholders. Added Step 0 (Generate Tutorial Data) so a copy-paste follower can complete the full workflow without `FileNotFoundError`.
- **README and ARCHITECTURE docs:** Added `massflow tutorial` to stable feature lists, CLI command references, and quickstart sections. Fixed a stray `similarity:` key in the README YAML example.

## [0.1.0] - 2026-05-10
### Added
- **Formal Data Model:** Implemented Pydantic-based validation for all spectral metadata.
- **SQLite Backend:** Added `massflow db build` to compile `.msp`/`.mgf` libraries into high-performance relational databases.
- **Flexible Validation:** Added graceful fallbacks for in-house libraries lacking structural (SMILES) data.
- **Small Library Support:** Introduced empirical p-value calculations as an alternative to target-decoy FDR for small datasets.
- **CLI Suite:** Finalized `init`, `db`, and `annotate` command structure.

### Changed
- Refactored `SimilarityEngine` to use a factory pattern for easier algorithm swapping.
- Optimized isotopic envelope generation to use vectorized NumPy operations.
- Updated `matchms` filtering pipeline for stricter precursor mass consistency.

### Fixed
- Resolved memory exhaustion when loading large (>2GB) MSP files.
- Fixed floating-point precision errors in cosine similarity scoring.
