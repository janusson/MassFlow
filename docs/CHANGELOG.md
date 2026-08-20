# Changelog

All notable changes to MassFlow will be documented in this file.

## [Unreleased]
### Added
- **`massflow tutorial` CLI command:** First-class command that generates a self-contained synthetic dataset (reference library, experimental queries, and pre-configured YAML) for instant local evaluation. No external files required.
- **Enhanced tutorial generator:** `scripts/generate_tutorial_data.py` now prints formatted next-steps with the exact `db build` and `annotate` commands to run after generation.
- **Hybrid SQLite + Zarr storage (Phase 1):** `SpectralDatabase` now supports a hybrid mode (`zarr_path=...` or `--backend hybrid`) where SQLite retains metadata plus a `zarr_ref`/`zarr_index` reference pair and fragment arrays live in a chunked, compressed `ZarrPeakArrayStore` with lock-free concurrent reads for multiprocessing.
- **BLOB → Zarr migration:** `scripts/migrations/0002_blobs_to_zarr.py` (wrapping `MassFlow.database.migrate_blobs_to_zarr`) migrates existing SQLite BLOB libraries to the hybrid backend with per-batch bitwise verification, idempotent re-runs, and orphaned-array recovery.
- **Algorithmic acceleration (Phase 2):** Numba-accelerated peak/neutral-loss prefilter (`MassFlow.acceleration`) that skips query-reference pairs below `min_matched_peaks` before exact modified-cosine scoring, with identical results and a pure-NumPy fallback.
- **HNSW candidate retrieval (Phase 2):** `MassFlow.hnsw` wraps hnswlib for sub-linear approximate candidate generation over binned spectral vectors, integrated into `CascadeEngine` as an optional pre-stage. Construction parameters (`hnsw_m`, `hnsw_ef_construction`, `hnsw_ef_search`, `hnsw_candidates_per_query`, binning) are exposed in `SimilarityConfig` with recall-friendly defaults because spectral similarity is non-metric. Requires the new `[hnsw]` extra.
- **Entropy-based decoy generation (Phase 3):** `generate_decoys` now produces entropy-preserving decoys that keep the precursor m/z and the Shannon entropy (ion information content) of each target while randomizing fragmentation pathways via intensity permutation and fragment-position jitter, replacing naive fragment shuffling. `ProcessingConfig` gains `decoy_min_relative_intensity` (strict baseline noise filtering before entropy computation) and `decoy_mz_shift_da`; the workflow logs a target-decoy entropy-divergence diagnostic for FDR calibration.
- **Real-time streaming hardening (Phase 4):** the gRPC `StreamSpectra` path now routes every packet through the `ConsensusEngine` (weighted multi-engine scoring) and enforces quality-gated backpressure: when the bounded queue reaches a configurable high-water mark, low-quality spectra are shed (reported via the new `ServerStatus.spectra_dropped_low_quality` field) to prevent memory exhaustion and latency collapse under instrument overrun. `massflow serve` gains `--queue-high-water-mark` and `--queue-low-quality-threshold`; `GetStatus` now also reports rolling processing latency.
- **ML API boundary (Phase 5):** external ML scoring is now fully decoupled behind `MLEngineProtocol`: `SimilarityConfig.ml_endpoints` routes Spec2Vec/MS2DeepScore scoring to remote REST (`http(s)://`) or gRPC (`grpc://`, the new `massflow.v1.ml` service) services via a `CircuitBreaker`-protected `RemoteMLEngine` client (no heavy dependencies required in core). `ConsensusEngine`, `CascadeEngine`, `MLRouter`, and the workflow now degrade gracefully to modified_cosine + empirical p-values when the ML service is unreachable or PyTorch/Gensim are missing.
- **Audit fix — HNSW neutral-loss channel:** `spectrum_to_binned_vector` now encodes a 2-channel concatenated vector `[binned exact m/z, binned neutral losses]`, so HNSW candidate retrieval can find precursor-shifted analogues (modified-cosine matches with disjoint exact m/z) that an exact-m/z-only index was blind to. `bin_spectra` and `HNSWSpectralIndex` dimensionality updated accordingly.
- **Audit fix — spectral entropy physics:** `spectral_entropy` and the decoy generator now apply the spectral-entropy `I**0.5` weighting and a hard baseline filter (peaks below `decoy_min_relative_intensity` × the **base peak**, default 1%) before computing the information content, preventing chemical noise from inflating entropy and biasing FDR calibration.

### Changed
- **Onboarding overhaul:** Rewrote `docs/user-guide/usage.md` to use real, runnable tutorial file paths instead of hypothetical placeholders. Added Step 0 (Generate Tutorial Data) so a copy-paste follower can complete the full workflow without `FileNotFoundError`.
- **README and ARCHITECTURE docs:** Added `massflow tutorial` to stable feature lists, CLI command references, and quickstart sections. Fixed a stray `similarity:` key in the README YAML example.
- **Config storage backends:** `input.storage_backend` now accepts `"hybrid"` in addition to `"sqlite"` and `"zarr"`; `massflow db inspect` auto-detects hybrid databases by their sibling `.zarr` store.

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
