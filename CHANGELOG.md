# CHANGELOG

## v1.0.0

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

**Architectural Foundations (v1.1 Prep):**
- Established strict Pydantic data contracts (`AnnotationHit`, `ConsensusInput`, `ConsensusResult`) to standardize communication with future ML satellite modules.
- Implemented a pure-Python, dependency-free `ConsensusEngine` for configurable weighted scoring, tie-breaking, and scientific credibility checks.
- Added a fast `numpy`-based triage scan during SQLite `massflow db build` to flag spectra of interest (e.g., Tyrosine immonium ion) without heavy ML compute.

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
