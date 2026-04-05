# CHANGELOG

## Unreleased

**Status:**
MassFlow remains pre-1.0 software and is still being aligned toward a narrower,
more reliable `v1.0` release surface.

**Current focus:**
Current work is centered on making the documented CLI workflow match the actual
code: a config-first local annotation pipeline, predictable CSV outputs, and
SQLite-backed library workflows, while clearly classifying more advanced paths
as experimental.

**Core workflow currently surfaced:**
- `massflow annotate --config ...` for end-to-end annotation runs
- YAML configuration loading and validation via `MassFlowConfig`
- Open-format ingestion for `mzML`, `mzXML`, `MGF`, and `MSP`
- Optional SQLite library usage through `.db` / `.sqlite` inputs
- Configurable `matchms`-based metadata cleaning and peak filtering
- Similarity search with `cosine` and `modified_cosine`
- Per-file CSV result export
- SQLite library management through `massflow db build`, `inspect`, and `merge`

**Experimental or not yet part of the stable promise:**
- Terminal browsing via `massflow browse`
- CAS-driven browsing via `massflow browse-cas`
- GraphML molecular-network export
- Advanced engines and orchestration paths:
  - `spec2vec`
  - `ms2deepscore`
  - `consensus`
  - `cascade`
- Broader config fields that are present in the schema but not yet central to
  the stable annotation contract

**Recent documentation and release-surface cleanup:**
- Simplified the release history to a smaller, more maintainable changelog
- Repositioned MassFlow as a CLI-first local annotation tool rather than a
  broad exploratory platform
- Clarified the distinction between stable core workflows and experimental
  features
- Tightened the docs around open-format ingestion and SQLite-backed libraries
- Reduced drift between top-level docs, package metadata, and the current code
  layout
- Consolidated the experimental terminal interfaces into a single
  `MassFlow.tui` module so the Textual browser and CAS-driven inspector now
  share one import surface
- Added a short experimental-features guide to document non-core interfaces in
  one place
- Renamed the CAS-driven experimental TUI entry point from a generic
  `main()` wrapper to the clearer `browse_cas_main()`

**Known pre-1.0 constraints:**
- Vendor raw formats are not converted inside MassFlow and should be converted
  to open formats before use
- The main workflow currently writes CSV outputs directly even though the config
  schema exposes a broader export surface
- Some configuration fields are placeholders for future or experimental
  workflows and are not fully wired into the main `annotate` path
- Advanced similarity engines exist in the codebase but should not yet be
  treated as equally mature or equally supported

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
