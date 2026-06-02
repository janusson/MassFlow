# Changelog

All notable changes to MassFlow will be documented in this file.

## [Unreleased]
### Changed
- **Deduplicated `_ms1_prefilter`:** Unified previously separate Da-tolerance and PPM-resolution code paths into a single search loop, eliminating ~35 lines of duplicated logic. The two modes now produce bit-identical results when configured to equivalent tolerances, which is enforced by a new equivalence-invariant test.

### Added
- **MS1 prefilter test coverage:** Added 6 new unit tests covering PPM boundary edge cases (high/low mass), Da/PPM equivalence invariance, simultaneous missing-precursor handling on both query and reference sides, zero-precursor rejection, and empty-input graceful degradation.

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
