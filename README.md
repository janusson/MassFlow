# Yogimass - Modular Tandem Mass Spectrometry Data Analysis

[![CI](https://github.com/ericjanusson/yogimass/actions/workflows/ci.yml/badge.svg)](https://github.com/ericjanusson/yogimass/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/ericjanusson/yogimass/branch/main/graph/badge.svg)](https://codecov.io/gh/ericjanusson/yogimass)
[![Docs](https://img.shields.io/badge/docs-local-blue)](docs/)

Yogimass is a modular, Python-based pipeline for the import, cleaning, processing, and comparison of tandem mass spectrometry (MS/MS) data. Built on top of the powerful [matchms](https://matchms.readthedocs.io/en/latest/index.html) library, Yogimass streamlines the workflow for researchers in metabolomics, analytical chemistry, and bioinformatics.

## Features

- **Import & Manage Spectral Libraries**  
  - Supports both MGF (GNPS-style) and MSP (NIST-style) formats.
  - List, filter, and inspect available libraries.

- **Automated Data Cleaning & Harmonization**  
  - Cleans and harmonizes spectrum metadata (names, formulas, adducts, etc.).
  - Processes and normalizes spectral peaks.
  - Modular filters for extensibility, including helpers for both MGF and MSP libraries and directory-wide batch cleaners.

- **Export Processed Data**  
  - Export cleaned spectra to MGF, MSP, JSON, or Python pickle formats.

- **Spectral Similarity & Identification**  
  - Compute cosine and modified cosine similarity scores between spectra.
  - Retrieve top matches and filter by minimum peak matches.
  - Designed for compound identification and spectral library curation.

- **Logging & Debugging**  
  - Integrated logging for reproducibility and troubleshooting.

## Planned & Experimental Features

- **Interactive Visualization**  
  - Integration with Jupyter notebooks and plotting libraries for spectrum visualization.

- **Batch Processing & CLI**  
  - Command-line tools for batch library processing and scoring.

- **Advanced Filtering**  
  - Support for custom, user-defined filters and workflows.

- **Integration with Public Databases**  
  - Automated download and update of public spectral libraries (e.g., GNPS, MassBank).

- **Machine Learning Integration**  
  - Hooks for spectral embedding and clustering (e.g., Spec2Vec, deep learning).

## Why Yogimass?

- **Flexible**: Modular design lets you swap in new filters, formats, or scoring methods.
- **Reproducible**: Logging and configuration management for scientific workflows.
- **Open**: Built on open standards and libraries, easy to extend and contribute.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Example usage (Python)
from yogimass import pipeline

mgf_files = pipeline.list_mgf_libraries('./data')
cleaned = pipeline.clean_mgf_library(mgf_files[0])
pipeline.save_spectra_to_mgf(cleaned, './out', 'cleaned_library')

# Batch clean every MGF file in ./data and export to MGF + JSON
pipeline.batch_clean_mgf_libraries(
    './data',
    './out',
    export_formats=('mgf', 'json'),
)

# Batch clean MSP libraries as well
pipeline.batch_clean_msp_libraries(
    './data',
    './out',
    export_formats=('msp',),
)
```

## Command-line batch cleaning

After installing Yogimass (either from source or via `pip install .`), a `yogimass` CLI becomes available:

```bash
# Clean MGF libraries in ./data and write cleaned MGF + JSON exports to ./out
yogimass clean ./data ./out --type mgf --formats mgf json

# Clean MSP libraries and keep MSP output
yogimass clean ./data ./out --type msp --formats msp
```

Place your raw libraries (MGF or MSP) anywhere under the input directory. Cleaned files follow the `<library>_cleaned.<ext>` naming convention and are written to the specified output directory (created if it does not exist).

## CLI Reference

| Command | Description | Key Options |
| --- | --- | --- |
| `yogimass clean <input_dir> <output_dir>` | Batch-clean every library (MGF/MSP) under `input_dir` and export cleaned copies. | `--type {mgf,msp}` choose input format; `--formats` choose one or more export formats (`mgf`, `msp`, `json`, `pickle`). |

## Documentation

See the `docs/` directory for full API documentation and usage examples.

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
