# Changelog

All notable changes to this project will be documented here.

## [0.3.0] - 2025-11-27

### Added
- Typed configuration schema and `ConfigError` with dotted-path messages; CLI now validates configs before running.
- MS-DIAL workflow: TSV fixture, conversion to spectra, example config, and integration test covering build/search/network steps.
- Search backend abstraction with naive default, optional Annoy backend, and embedding hooks; optional extras declared for `annoy`, `faiss`, and future `spec2vec`.
- Centralized logging helpers and import-safety test coverage.
- GitHub Actions CI (pytest + mypy) and lightweight mypy configuration.
- Documentation refresh emphasizing config-first usage, MGF/MS-DIAL examples, and CLI reference.

### Changed
- Default logging is centralized and side-effect-free on import.
- Public docs and README now treat YAML/CLI as the canonical entrypoint.
- Project version bumped to 0.3.0 and dev extras expanded for tooling.

### Removed
- Legacy logging/MS-DIAL stubs and placeholder unit_tests were cleaned up or archived.

## [Unreleased]
- Track future changes here.
