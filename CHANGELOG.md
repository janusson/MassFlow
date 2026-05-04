# CHANGELOG

## [1.0.0] - 2026-04-20

### Summary
MassFlow v1.0.0 is the first stable release, locking down a narrow, reliable, config-first Python toolkit for local MS/MS annotation workflows. This release focuses on stability, predictability, and reproducible tabular outputs, clearly separating the core pipeline from experimental features.

### Core Features (Stable Contract)
- **Config-driven CLI workflow** (`massflow annotate --config ...`) and starter template generation (`massflow init`)
- **YAML configuration loading** and validation via `MassFlowConfig`
- **Open-format ingestion** for `mzML`, `mzXML`, `MGF`, and `MSP`
- Strict IO boundaries that explicitly reject vendor raw formats (requiring pre-conversion)
- **SQLite library management** through `massflow db build`, `inspect`, and `merge`
- Configurable `matchms`-based metadata cleaning and peak filtering
- Classical similarity search with `cosine` and `modified_cosine`
- Per-file CSV result export accompanied by YAML provenance sidecar reports
- Automated FDR (False Discovery Rate) estimation with strict target-decoy warnings for under-powered libraries
- **Triage-Aware ML Routing**: Integrated `triage_flags` from the database layer into the workflow, allowing specific spectra (e.g., Tyrosine fragments) to automatically route to high-accuracy ML engines.
- **Zero-I/O Multiprocessing**: High-performance worker pool in `workflow.py` utilizing shared memory to avoid redundant library parsing across CPU cores.

### Experimental Features (Not part of the stable promise)
- **Cascade & Consensus Orchestration**: Advanced algorithmic routing and weighted scoring for multi-engine ensembles.
- **ML Engines**: Support for `spec2vec` and `ms2deepscore` similarity scoring.
- GraphML molecular-network export
- Terminal User Interfaces (TUI)
- Alternative export formats (`pickle`, `json`, `xlsx`, `parquet`)

### Bug Fixes & Edge Cases Resolved
- **Silent Triage Failure**: Extracted the NumPy-based `triage_flags` bitmask generation from the SQLite database insertion loop and integrated it into the central `processing.py` pipeline.
- **Overly Optimistic FDR**: Fixed a statistical edge-case where small datasets with zero decoys yielded an overly optimistic `0.0` False Discovery Rate (q-value). The pipeline now uses a conservative `+1` pseudo-count formula.
- **Missing MS1 Precursor Data Loss**: Fixed a bug where spectra missing a `precursor_mz` in open formats (like `.mgf`) defaulted to `0.0` and were silently dropped during vectorized MS1 pre-filtering. Missing precursors now cleanly bypass the MS1 filter to be evaluated by MS2 fragment matching.

**Migration Notes:**
Legacy databases using the `peaks` column are no longer automatically upgraded. Run the explicit migration script `scripts/migrations/0001_peaks_to_arrays.py` to upgrade your databases safely.

## [0.1.0]

**Summary:**
Initial public release of MassFlow as a lightweight, local-first toolkit for tandem mass spectrometry annotation.

**Included in this release:**
- Config-driven CLI workflow for annotating experimental spectra against a reference library
- Support for open spectral formats including `mzML`, `mzXML`, `MGF`, and `MSP`
- Modular package structure with separate I/O, processing, similarity, and workflow layers
- Configurable metadata cleaning and peak filtering using `matchms`
- Classical spectral similarity search with cosine-based scoring
- CSV export of annotation results
- Early project documentation and installable CLI entry point
