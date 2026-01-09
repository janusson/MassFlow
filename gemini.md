# Gemini Mission

This document outlines the goals and execution of the Gemini program for this project.

## Introduction

This project is **MassFlow**, a lightweight Python toolkit for processing, cleaning, and analyzing tandem mass spectrometry (MS/MS) data. It leverages the [matchms](https://github.com/matchms/matchms) ecosystem to provide efficient spectral data handling and similarity calculations.

## Getting Started

To get started with the Gemini program, you will need to have Python 3.10+ installed.

### Install from Source

```bash
git clone https://github.com/yourusername/MassFlow.git
cd MassFlow
pip install .
```

### Development Setup

```bash
pip install -r requirements.txt
```

## Usage

MassFlow provides a CLI entry point `massflow` (or `MassFlow` depending on installation).

### Command Line Interface (CLI)

#### 1. Clean and Convert a Library

Process an MGF or MSP file to apply filters (intensity, mz range, metadata cleanup) and save it in a new format.

```bash
# Clean an MSP file and save as Pickle (default)
massflow clean --input data/library.msp --output-dir processed_data/

# Clean an MGF file and save as MSP
massflow clean --input data/query.mgf --output-dir processed_data/ --format msp
```

### Python Library

You can use MassFlow modules directly in your Python scripts.

#### Processing Spectra

```python
from MassFlow import processing, io

# Load and clean a library
spectra = processing.clean_mgf_library("data/test.mgf")
```

#### Similarity Calculations

```python
from MassFlow import similarity

# Calculate Cosine scores
scores = similarity.calculate_cosscores(reference_spectra, query_spectra)
```

## Contributing

Thanks for your interest in improving MassFlow! Please follow these guidelines to keep the project healthy.

### Getting Started

1. Fork the repository and create a feature branch from `main`.

2. Set up a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. For matchms-dependent modules, make sure `matchms` and `numpy` install cleanly before running tests.

### Coding Standards

- Keep imports sorted and avoid introducing unused dependencies.
- Favor small, composable functions with clear logging where side-effects occur.
- Write or update tests in `tests/` for new features and bug fixes.
- Run `python3 -m pytest` before opening a PR.

### Pull Requests

- Describe the motivation and any trade-offs clearly in the PR description.
- Reference relevant issues or TODOs.
- Include notes about manual testing if automated tests don’t cover everything.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
