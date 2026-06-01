# Changelog

All notable changes to MassFlow will be documented in this file.

## [1.0.0] - 2026-05-10
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
