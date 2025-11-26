# Yogimass - Modular Tandem Mass Spectrometry Data Analysis

[![CI](https://github.com/ericjanusson/yogimass/actions/workflows/ci.yml/badge.svg)](https://github.com/ericjanusson/yogimass/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/ericjanusson/yogimass/branch/main/graph/badge.svg)](https://codecov.io/gh/ericjanusson/yogimass)
[![Docs](https://img.shields.io/badge/docs-local-blue)](docs/)

Yogimass is a config-driven LC-MS/MS toolkit for ingesting spectra, building and searching spectral libraries, generating similarity networks, and curating libraries with QC reporting. The recommended way to run Yogimass is through YAML/JSON configs plus `yogimass config run`. See `USAGE.md` for install and smoke-test steps.

## Quick start (config-first)

Drive the full pipeline from a config file:

```bash
yogimass config run --config examples/simple_workflow.yaml
```

The configs in `examples/` are the fastest way to exercise library build/search, network export, and curation/QC. Details live in `USAGE.md`.

## Features

- **Import & Manage**: MGF (GNPS-style) and MSP (NIST-style) support.
- **Cleaning & Processing**: metadata harmonization and peak filtering/normalization.
- **Similarity & Search**: cosine/modified cosine/spec2vec-style vectors for local search.
- **Library Curation & QC**: drop low-quality spectra, merge near-duplicates, emit QC reports.
- **Networks & Reporting**: build similarity networks and write summaries/exports.

## CLI reference

| Command | Description | Key Options |
| --- | --- | --- |
| `yogimass config run --config path/to/config.yaml` | Execute an end-to-end workflow from a YAML/JSON config. | Sections: `input`, `library`, `similarity`, `network`, `outputs`. |
| `yogimass library build --input ... --library out/library.json` | Build/update a local library from MGF/MSP files or folders. | `--format {mgf,msp}`, `--recursive`, `--storage {json,sqlite}` |
| `yogimass library search --queries ... --library out/library.json` | Search query spectra against a stored library. | `--top-n`, `--min-score`, `--output <csv/json>` |
| `yogimass library curate --input raw.json --output curated.json` | QC and de-duplicate a stored library, writing a curated copy and QC report. | `--qc-report`, `--min-peaks`, `--min-tic`, `--max-single-peak-fraction`, `--precursor-tolerance`, `--similarity-threshold` |
| `yogimass network build --input <dir>|--library <lib> --output graph.csv` | Build a similarity network (threshold or k-NN) and export edges/graphs. | `--metric {cosine,modified_cosine,spec2vec}`, `--threshold` or `--knn`, `--summary` |
| `yogimass clean <input_dir> <output_dir>` | (Legacy) Batch-clean every library (MGF/MSP) under `input_dir` and export cleaned copies. | `--type {mgf,msp}` choose input format; `--formats` choose one or more export formats (`mgf`, `msp`, `json`, `pickle`). |

## Documentation

- `USAGE.md`: editable install steps and the two golden-path smoke tests.
- `docs/`: full API documentation and examples.

### Legacy scripts and deprecation

Legacy entrypoints (`yogimass_pipeline.py`, `yogimass_buildDB.py`, `build_library_from_msp.py`, `msdial_fragment_search.py`) remain as thin wrappers that forward to the unified workflow/CLI. Each prints a deprecation warning; prefer config-driven runs via `yogimass config run`.

## Contributing & License

Contributions are welcome—see `CONTRIBUTING.md` for setup and PR guidelines. Yogimass is released under the MIT License (see `LICENSE`).

## Continuous Integration

All pushes and pull requests run `python -m pytest --cov=yogimass` on Python 3.10–3.12 via the GitHub Actions workflow defined in `.github/workflows/ci.yml`, and coverage results are uploaded to Codecov. Make sure the CI and Coverage badges are green before merging or tagging releases.

## Release Plan

1. Ensure CI is green (tests + coverage) and docs/README reflect your changes.
2. Update `pyproject.toml` version and CHANGELOG/`CHANGES_AND_NEXT_STEPS.md`.
3. Tag the release (e.g., `git tag v0.1.0 && git push --tags`) and publish to GitHub.
4. (Optional) Publish to PyPI: `python -m build && twine upload dist/*`.
5. Verify `pip install yogimass==<version>` works in a clean environment and that badges link to the latest build/coverage.
