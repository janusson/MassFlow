# CHANGELOG

## [1.0.0] - 2026-04-20

### Summary
MassFlow v1.0.0 is the first stable release, locking down a robust, config-first Python toolkit for tandem mass spectrometry (MS/MS) annotation. This release prioritizes UX, scalability, and scientific integrity, delivering a production-ready CLI for high-throughput spectral analysis.

### Core Features (Stable)
- **Multi-Format Export Pipeline**: Native support for `.xlsx` (Excel), `.parquet`, `.json`, and `.csv` exports, enabling immediate integration into laboratory and data science workflows.
- **Vendor Format Conversion**: Integrated `massflow convert` command that wraps ProteoWizard's `msconvert` to batch-translate proprietary `.raw` and `.d` files into open `.mzML` standards.
- **Unified Configuration Schema**: Simplified `input_path` handling for both single files and directories, and unified scientific tolerances (MS1 in ppm, MS2 in Da).
- **Standardized "init" Template**: New `massflow init` command generates a scientifically sound configuration following the "Sibling Directory" pattern to keep large datasets out of Git repositories.
- **Memory-Optimized Molecular Networking**: Refactored networking engine to utilize double-chunked similarity computation, preventing Out-of-Memory (OOM) errors on large experimental datasets.
- **Triage-Aware ML Routing**: Integrated `triage_flags` from the database layer into the workflow, allowing specific spectra (e.g., Tyrosine fragments) to automatically route to high-accuracy ML engines.
- **Zero-I/O Multiprocessing**: High-performance worker pool in `workflow.py` utilizing shared memory to avoid redundant library parsing across CPU cores.
- **Scientific Safety Checks**: Automated library-size warnings and global Target-Decoy FDR estimation to protect users from statistically under-powered results.

### Experimental Features
- **Cascade & Consensus Orchestration**: Advanced algorithmic routing and weighted scoring for multi-engine ensembles.
- **ML Engines**: Support for `spec2vec` and `ms2deepscore` similarity scoring.

### Bug Fixes & Edge Cases Resolved
- **Silent Triage Failure**: Extracted the NumPy-based `triage_flags` bitmask generation from the SQLite database insertion loop and integrated it into the central `processing.py` pipeline. This ensures standard experimental query files correctly receive triage flags and can be routed to advanced Tier-2 ML engines.
- **Overly Optimistic FDR**: Fixed a statistical edge-case where small datasets with zero decoys yielded an overly optimistic `0.0` False Discovery Rate (q-value). The pipeline now uses a conservative `+1` pseudo-count formula (`FDR = (decoys + 1) / targets`) to properly penalize datasets lacking a robust statistical null-distribution.
- **Missing MS1 Precursor Data Loss**: Fixed a bug where spectra missing a `precursor_mz` in open formats (like `.mgf`) defaulted to `0.0` and were silently dropped during vectorized MS1 pre-filtering. Missing precursors now cleanly bypass the MS1 filter to be evaluated by MS2 fragment matching.

---


**Summary:**
MassFlow v1.0.0 is the first stable release, locking down a narrow, reliable, config-first Python toolkit for local MS/MS annotation workflows. This release focuses on stability, predictability, and reproducible tabular outputs, clearly separating the core pipeline from experimental features.

**Included in this release (Stable Contract):**
- Config-driven CLI workflow (`massflow annotate --config ...`) and starter template generation (`massflow init`)
- YAML configuration loading and validation via `MassFlowConfig`
- Open-format ingestion for `mzML`, `mzXML`, `MGF`, and `MSP`
- Strict IO boundaries that explicitly reject vendor raw formats (requiring pre-conversion)
- SQLite library management through `massflow db build`, `inspect`, and `merge`
- Configurable `matchms`-based metadata cleaning and peak filtering
- Classical similarity search with `cosine` and `modified_cosine`
- Per-file CSV result export accompanied by YAML provenance sidecar reports
- Automated FDR (False Discovery Rate) estimation with strict target-decoy warnings for under-powered libraries

**Experimental (Not part of the stable promise):**
- Advanced ML engines (`spec2vec`, `ms2deepscore`)
- Complex routing and orchestration (`consensus`, `cascade`)
- GraphML molecular-network export
- Terminal User Interfaces (TUI)
- Alternative export formats (`pickle`, `json`, `xlsx`, `parquet`)

**Known constraints:**
- Vendor raw formats must be converted to open formats before use
- The main workflow currently writes CSV outputs directly even though the config schema exposes a broader export surface

## v0.1.0

**Summary:**
Initial public release of MassFlow as a lightweight, local-first toolkit for
tandem mass spectrometry annotation.

**Included in this release:**
- Config-driven CLI workflow for annotating experimental spectra against a
  reference library
- Support for open spectral formats including `mzML`, `mzXML`, `MGF`, and `MSP`
- Modular package structure with separate I/O, processing, similarity, and
  workflow layers
- Configurable metadata cleaning and peak filtering using `matchms`
- Classical spectral similarity search with cosine-based scoring
- CSV export of annotation results
- Early project documentation and installable CLI entry point

## Before v0.1.0

- Early prototype work on spectral processing and project structure
