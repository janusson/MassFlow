# Changelog

All notable changes to MassFlow will be documented in this file.

## [Unreleased]

### Added
- Comprehensive MS1 pre-filter test suite (`tests/test_ms1_prefilter.py`) covering Da tolerance,
  ppm tolerance, missing precursor bypass (`np.nan`), and exact boundary conditions.
- Support for the doubly-charged `[M+2H]2+` adduct in `cheminformatics.calculate_theoretical_mass`.
- Regression tests for neutral-loss exact masses (a first-principles cross-check against NIST
  element masses) and for element-count validation in `find_impossible_neutral_losses`.

### Changed
- `COMMON_NEUTRAL_LOSSES` and `ADDUCT_OFFSETS` in `cheminformatics.py` are now derived at module
  load from chemical formulas via `pyteomics`, establishing a single source of truth and removing
  the hand-maintained mass tables. Computed adduct offsets reproduce the previous hardcoded values
  to within 0.001 mDa.
- `find_impossible_neutral_losses` now compares element *counts* (not just presence) against each
  loss's requirement (e.g. CO₂ loss requires ≥2 oxygen atoms). `parse_elements_from_smiles` now
  returns a `collections.Counter` instead of a `set`.

### Removed
- Hardcoded per-element monoisotopic mass constants (`H_MASS`, `C_MASS`, `O_MASS`, etc.) and the
  unused `PROTON_MASS` from `cheminformatics.py`; element masses now come from `pyteomics`.
  (`ELECTRON_MASS` is retained, since `pyteomics` omits the electron from neutral compositions.)

### Fixed
- `_ms1_prefilter` in `similarity.py` now correctly treats `np.nan` precursor m/z values as
  missing. Previously only `None` triggered the bypass; `np.nan` was silently cast to `0.0` and
  incorrectly filtered against real reference masses.
- Corrected the H₂S neutral-loss mass in `COMMON_NEUTRAL_LOSSES` (34.9956 → 33.9877 Da); the old
  value was the monoisotopic mass of H₃S (sulfonium), not H₂S.
- `find_impossible_neutral_losses` no longer produces false negatives for multi-atom losses on
  element-poor candidates (e.g. CO₂/SO₂ loss on a 1-oxygen molecule, or PO₃ loss on a
  fewer-than-3-oxygen molecule), which the previous presence-only check missed.
- `calculate_theoretical_mass` now divides by the absolute adduct charge, fixing the m/z
  calculation for multiply-charged adducts (previously returned the neutral mass).

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
