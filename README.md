# MassFlow

MassFlow is an open-source, config-first spectral similarity search engine designed for mass spectrometry.

It functions as a lightweight, local-first orchestration layer, optimized for untargeted MS/MS annotation and cheminformatics.  Unlike closed-source databases and web-based molecular networking platforms, MassFlow empowers users to run reproducible, version-controlled pipelines entirely locally. This is accomplished by integrating classical metrics (Cosine) with modern machine learning models (`MS2DeepScore`, `spec2vec`) through a unified YAML configuration.

## Key Features

* **YAML-Driven Orchestration:** Define complex "Ingest -> Match -> Report" workflows entirely in configuration files.
* **Unified Algorithm API:** Switch between classical (`matchms` cosine) and deep learning (`MS2DeepScore`, `spec2vec`) backends without altering ingestion logic.
* **Vectorized Similarity:** High-throughput matrix operations for rapid spectral matching against large custom databases.
* **Robust I/O:** Aggressive metadata sanitization to handle malformed retention times and dirty input files without silent failures.
* **Stateless Processing:** Relies on the standard `matchms.Spectrum` object for seamless interoperability across modules.

## Installation

MassFlow v1.0 requires **Python 3.13+**. Dependency management via `uv` is strictly recommended for environment reproducibility.

```bash
git clone [https://github.com/yourusername/MassFlow.git](https://github.com/yourusername/MassFlow.git)
cd MassFlow
uv python pin 3.13
uv sync

```

## Usage: Config-First Pipeline

MassFlow prioritizes reproducible pipelines over manual scripting. Avoid writing custom Python ingestion scripts unless extending the core algorithmic functionality.

### 1. Define the Workflow (`config.yaml`)

```yaml
pipeline:
  query_data: "data/experiment.mgf"
  reference_library: "data/library.msp"
  output_dir: "results/"
  similarity_metric:
    type: "cosine"
    tolerance: 0.1
    min_score: 0.7
    top_n: 5
```

### 2. Execution via CLI

Pass the configuration file directly to the workflow orchestrator.

```bash
uv run massflow execute --config config.yaml
```

## Python API

For integration into larger automated systems or custom CI/CD data pipelines, import the core modules. Side-effects (I/O) are strictly isolated from processing logic.

```python
from pathlib import Path
from MassFlow import io, workflow
from MassFlow.similarity import SimilarityEngine

# 1. Load Data
query_spectra = list(io.load_spectra(Path("data/experiment.mgf"), "mgf"))
ref_spectra = list(io.load_spectra(Path("data/library.msp"), "msp"))

# 2. Execute Search
engine = SimilarityEngine(tolerance=0.1, min_score=0.7)
results = engine.search(query_spectra, ref_spectra, top_n=5)

# 3. Export
io.save_match_results(results, Path("results.csv"))

```

## Architecture & Constraints

* **Scope:** MassFlow is an infrastructure layer. Proprietary substructure elucidation or molecular networking logic is out of scope.
* **Performance:** Spectral arrays (m/z and intensities) are processed using vectorized `numpy` operations. Standard Python `for` loops are prohibited for peak iteration.
* **Error Handling:** The library is designed to fail fast on malformed data, raising explicit `ValueError` or `TypeError` exceptions rather than issuing warnings.

## License

MIT License
