# MassFlow: Architectural Overview & Project Summary

## 1. Project Description

**MassFlow** is a pre-1.0, config-first Python toolkit for local tandem mass
spectrometry (MS/MS) annotation workflows.

The project is centered on a reproducible CLI pipeline: load open spectral
formats, apply configurable `matchms` processing, run similarity search against
reference libraries, and export structured results. Supporting utilities
include an optional terminal browser, SQLite-backed library storage, and
optional GraphML network export.

## 2. Core Capabilities

### A. Open-format ingestion (`io.py`, `database.py`)
* Supports mzML, mzXML, MGF, MSP, SQLite (`.db` / `.sqlite`), and pickle.
* Rejects vendor-specific raw formats instead of attempting implicit
  conversion.
* Streams spectra from SQLite libraries for larger repeated analyses.

### B. Configurable processing (`processing.py`, `config.py`)
* Uses Pydantic models for YAML-driven configuration.
* Applies optional metadata harmonization, retention-time extraction,
  intensity filtering, m/z truncation, minimum-peak enforcement, Top-N peak
  reduction, and normalization.

### C. Similarity search (`similarity.py`)
* Supports `CosineGreedy`, `ModifiedCosine`, `Spec2Vec`, and `MS2DeepScore`.
* Adds higher-level `ConsensusEngine` and `CascadeEngine` paths for larger or
  more selective search workflows.
* Applies vectorized score processing plus workflow-level filtering and FDR
  handling.

### D. Workflow and export (`workflow.py`, `cli.py`, `networking.py`, `tui.py`)
* Runs end-to-end annotation from validated config files.
* Exports CSV search results and can optionally generate GraphML networks.
* Provides `massflow browse` and `massflow db` as supporting CLI utilities.

## 3. Technology Stack & Constraints

* **Language:** Python 3.13+
* **Core Dependencies:** `matchms`, `numpy`, `pandas`, `pydantic`, `PyYAML`,
  `pyteomics`
* **Optional Interface/Analysis Dependencies:** `textual`, `plotext`,
  `networkx`
* **Optional ML Dependencies:** `spec2vec`, `ms2deepscore`
* **Design Philosophy:** fail fast on malformed inputs, keep workflow behavior
  reproducible, and separate I/O side effects from processing and scoring
  logic

## 4. Current Scope Boundaries

* MassFlow is CLI-first and local-first.
* Terminal browsing is in scope; the legacy desktop GUI is not.
* Open-format ingestion is in scope; vendor raw conversion is not.
* Structured annotation outputs and optional GraphML export are in scope.
* Proprietary structure elucidation workflows are out of scope.

## 5. Typical Usage

MassFlow workflows are expected to be defined in YAML:

```yaml
project:
  output_directory: "results/"
input:
  file_path: "~/MassFlow_Data/experiments/example.mzML"
  reference_library: "~/MassFlow_Data/libraries/library.msp"
  format: "mzml"
similarity:
  algorithm: "cosine"
  tolerance: 0.02
  min_score: 0.7
```

**CLI execution:**

```bash
uv run massflow annotate --config standard_config.yaml
```

**Python API execution:**

```python
from pathlib import Path

from MassFlow import io
from MassFlow.config import MassFlowConfig
from MassFlow.similarity import SimilarityEngine

query_spectra = list(
    io.load_spectra(Path("~/MassFlow_Data/experiments/example.mgf").expanduser(), "mgf")
)
ref_spectra = list(
    io.load_spectra(Path("~/MassFlow_Data/libraries/library.msp").expanduser(), "msp")
)

config = MassFlowConfig.from_yaml("standard_config.yaml")
engine = SimilarityEngine(config.similarity)
results = engine.search(query_spectra, ref_spectra)
```
