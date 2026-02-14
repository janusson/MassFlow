# MassFlow

**MassFlow** is a lightweight Python toolkit for processing, cleaning, and analyzing tandem mass spectrometry (MS/MS) data. It leverages the [matchms](https://github.com/matchms/matchms) ecosystem to provide efficient spectral data handling and similarity calculations.

## Features

- **Pydantic Configuration**: Robust configuration management and validation for complex workflows.
- **Spectral Cleaning**: Automated metadata repair, peak filtering, and normalization using a configurable pipeline.
- **Unified I/O**: Seamless loading and saving of spectra in MGF, MSP, mzML, JSON, and Pickle formats.
- **High-Performance Database**: SQLite-based storage with batch insertion for managing large spectral libraries.
- **Vectorized Similarity Search**: Optimized similarity calculations using matrix operations for high throughput.
- **Modern GUI**: A sleek, dark-themed graphical user interface built with CustomTkinter and Matplotlib.
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

### Installation with uv

If you prefer using [uv](https://github.com/astral-sh/uv) for dependency management:

1.  **Sync dependencies** (using `uv.lock`):
    ```bash
    uv sync
    ```

2.  **Add dependencies manually** (if needed):
    ```bash
    uv add numpy pandas matchms spec2vec ms2deepscore matplotlib customtkinter
    ```

3.  **Run MassFlow**:
    ```bash
    # Run GUI
    uv run python -m MassFlow.gui

    # Run CLI
    uv run python -m MassFlow.cli --help
    ```

## Usage

MassFlow provides a CLI entry point `massflow` (or `python -m MassFlow.cli`).

### Graphical User Interface (GUI)

Launch the modern GUI for viewing spectra and running workflows:

```bash
python -m MassFlow.gui
```

### Command Line Interface (CLI)

#### 1. Run the Full Processing Pipeline

Execute a complete MassFlow pipeline (ingestion, processing, similarity search, result saving) using a YAML configuration file.

```bash
massflow process config.yaml
```

Example `config.yaml`:
```yaml
# MassFlow Configuration Example
project:
  name: MyAnalysis
  output_directory: results

input:
  file_path: data/query_spectra.mgf
  format: mgf
  reference_library: data/reference_library.msp

processing:
  min_peaks: 6
  min_intensity: 0.001
  normalize_intensity: true
  clean_metadata: true

similarity:
  algorithm: cosine # or modified_cosine
  tolerance: 0.01
  tolerance_unit: Da
  min_score: 0.7
  min_matched_peaks: 3

export:
  format: csv
```

#### 2. Clean and Convert a Library

Process an input spectral file to apply default filters and save it in a new format.

```bash
# Clean an MSP file and save as Pickle
massflow clean --input data/library.msp --output-dir processed_data/

# Convert MGF to MSP
massflow clean --input data/query.mgf --output-dir processed_data/ --format msp
```

#### 3. Plot a Spectrum

Visualize a single spectrum from a library file using Matplotlib.

```bash
# List top 20 compound names
massflow plot --input data/library.msp

# Plot a specific spectrum by name
massflow plot --input data/library.msp --name "Caffeine"
```

#### 4. Database Management

Manage SQLite spectral databases.

```bash
# Initialize a new database
massflow database init --db library.db

# Add spectra to database
massflow database add --db library.db --input new_data.msp --category standards

# Export from database
massflow database export --db library.db --output exported.mgf --category standards
```

## Python Library

You can use MassFlow modules directly in your Python scripts.

```python
from pathlib import Path
from MassFlow import io, processing
from MassFlow.config import ProcessingConfig

# Load
spectra = io.load_spectra(Path("data/test.mgf"), "mgf")

# Process
config = ProcessingConfig(min_peaks=5)
cleaned = list(processing.process_spectra(spectra, config))

# Save
io.save_spectra_to_msp(cleaned, Path("output/cleaned.msp"))
```

## Architecture

MassFlow uses a modular architecture:

- **`MassFlow.gui`**: CustomTkinter-based GUI for visualization and workflow management.
- **`MassFlow.cli`**: Command-line interface.
- **`MassFlow.workflow`**: Orchestrates loading, processing, and vectorized similarity search.
- **`MassFlow.database`**: Optimized SQLite storage with batch operations.
- **`MassFlow.processing`**: Facade for `matchms` filtering pipelines.
- **`MassFlow.similarity`**: Strategy pattern for scoring algorithms.

## License

MIT License