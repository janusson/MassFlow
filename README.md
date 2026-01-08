# MassFlow

**MassFlow** is a lightweight Python toolkit for processing, cleaning, and analyzing tandem mass spectrometry (MS/MS) data. It leverages the [matchms](https://github.com/matchms/matchms) ecosystem to provide efficient spectral data handling and similarity calculations.

## Features

- **Spectral Cleaning**: Automated metadata repair, peak filtering, and normalization.
- **Format Conversion**: Convert between MGF, MSP, JSON, and Pickle formats.
- **Similarity Search**: Calculate Cosine and Modified Cosine similarity scores between spectra.
- **CLI & Library**: Use as a command-line tool or import as a Python library.

## Installation

### Prerequisites

- Python 3.10+

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

**Options:**

- `--input`: Path to input library (.mgf or .msp).
- `--output-dir`: Directory to save the output.
- `--format`: Output format (`pickle`, `msp`, `mgf`, `json`). Default: `pickle`.

#### 2. Similarity Search

(Functionality available in library, CLI wrapper coming soon)

```bash
massflow search ...
```

### Python Library

You can use MassFlow modules directly in your Python scripts.

#### Processing Spectra

```python
from MassFlow import processing, io

# Load and clean a library
spectra = processing.clean_mgf_library("data/test.mgf")

# The cleaning pipeline automatically applies:
# - Metadata repair (InChI/SMILES)
# - Peak filtering (intensity > 0.01, relative > 0.08)
# - Normalization
```

#### Similarity Calculations

```python
from MassFlow import similarity

# Calculate Cosine scores
scores = similarity.calculate_cosscores(reference_spectra, query_spectra)

# Get top matches
matches = scores.scores_by_query(query_spectra[0], sort=True)
print(matches[:5])
```

## Testing

Run the test suite using `pytest`:

```bash
pytest
```

## Development

- **Style**: Codebase follows PEP8.
- **Structure**:
  - `MassFlow/cli.py`: CLI entry point.
  - `MassFlow/processing.py`: Core cleaning logic using matchms.
  - `MassFlow/io.py`: Input/Output handlers.
  - `MassFlow/similarity.py`: Similarity scoring functions.

## License

MIT License
