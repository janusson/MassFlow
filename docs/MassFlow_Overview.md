# MassFlow: Architectural Overview & Project Summary

## 1. Project Vision & Identity

**MassFlow** is an open-source, config-first Python toolkit and orchestration layer designed for the processing, annotation, and similarity searching of tandem mass spectrometry (MS/MS) data. 

Built as a local-first alternative to web-based molecular networking platforms, MassFlow empowers users to run reproducible, version-controlled pipelines entirely on their own hardware. It combines classical cheminformatics metrics (like Cosine similarity) and modern machine learning models (`MS2DeepScore`, `spec2vec`) through a unified, YAML-driven configuration interface. 

The project is structured to serve as both a lightweight CLI/GUI tool for researchers and a robust, modular Python library for integration into larger CI/CD data pipelines.

## 2. Core Architectural Layers

The MassFlow codebase is strictly modular, separating I/O side-effects from stateless processing logic. The architecture is divided into four primary layers:

### A. Ingestion & Normalization Layer (`io.py`, `database.py`)
**Objective:** Standardize fragmented experimental inputs into a unified computational structure.
* **Capabilities:** Parses diverse formats (mzML, mzXML, MGF, MSP, JSON, Pickle) and automatically triggers `msconvert` (ProteoWizard) for proprietary vendor formats (.raw, .d, .wiff).
* **Sanitization:** Aggressively cleans malformed metadata (e.g., stripping garbage text like "CCS: N/A" from numeric fields) to prevent silent failures in downstream dependencies.
* **Edge Cases:** Importer stability relies heavily on vendor-agnostic file compliance. Poorly formatted files or missing precursor M/Z values are handled via strict Pydantic schemas to avoid silent drops.
* **Storage:** Utilizes a local SQLite database (`database.py`) for efficient, batched caching and retrieval of massive spectral reference libraries.

### B. Pre-Processing Pipeline (`processing.py`, `validation.py`, `config.py`)
**Objective:** Validate schemas, remove noise, and compress data prior to scoring to reduce matrix dimensions and computational overhead.
* **Validation:** Employs Pydantic (`validation.py`, `config.py`) to strictly enforce configuration schemas and metadata types (e.g., ensuring precursor M/Z is a positive float).
* **Filtering:** Acts as a facade for `matchms.filtering`, performing sequence operations such as:
  * Intensity thresholding (removing noise below strict baselines).
  * M/Z range truncation.
  * Max-peak restriction (reducing high-density spectra to the Top-N peaks).

### C. Similarity Computation Engine (`similarity.py`)
**Objective:** Construct an $N \times M$ dense or sparse matrix of similarity scores between $N$ queries and $M$ references.
* **Vectorized Execution:** Uses highly optimized `numpy` arrays to calculate scores, strictly avoiding standard Python `for` loops for peak iteration.
* **Algorithm Matrix:** Provides a unified API to swap between multiple backends:
  * **Cosine (Greedy):** Bipartite peak matching heuristic for direct spectral overlap.
  * **Modified Cosine:** Mass-shift incorporated peak matching for analogue/derivative overlap.
  * **Spec2Vec:** Unsupervised Word2Vec word embeddings for contextual sub-structural similarity.
  * **MS2DeepScore:** Supervised Siamese neural network acting as a Tanimoto structural similarity proxy.

### D. Orchestration & Export Layer (`workflow.py`, `cli.py`, `ui/`)
**Objective:** Manage the high-level execution flow and output structured analytical data.
* **Thresholding:** Filters the computed $N \times M$ score matrix against tunable configuration parameters (`min_score`, `min_matched_peaks`, `top_n`).
* **Export:** Maps indices back to spectrum metadata (SMILES, InChIKey) and exports to a flattened CSV structure using `pandas`.
* **Interfaces:** Executable via a unified Command-Line Interface (`cli.py`), a CustomTkinter Graphical User Interface (`ui/main.py`), or programmatically via the Python API.

---

## 3. Mathematical Implementation: Modified Cosine

To identify structurally related analogues, MassFlow implements a Modified Cosine score that computes the normalized dot product of matched peak intensities while natively incorporating neutral losses.

A peak pair $(A_i, B_j)$ is defined as a valid match if it satisfies either a direct M/Z tolerance or a mass-shifted tolerance:

$$|m/z_{A_i} - m/z_{B_j}| \le \delta \quad \text{OR} \quad |m/z_{A_i} - m/z_{B_j} - \Delta M| \le \delta$$

Where:
* $\delta$ = Defined M/Z error tolerance.
* $\Delta M = \text{precursor } m/z_A - \text{precursor } m/z_B$.

The final score for the matched peaks $k$ is calculated as the cosine of the angle between the intensity vectors:

$$\text{Score} = \frac{\sum (I_{A,k} \cdot I_{B,k})}{\sqrt{\sum I_{A}^2} \sqrt{\sum I_{B}^2}}$$

* **Algorithmic Constraint:** The standard `matchms` implementation utilizes a greedy heuristic to solve the peak assignment problem rather than the exact but computationally expensive Hungarian algorithm. Peak assignment error margins may increase slightly in highly congested spectra with loose mass tolerances ($\delta > 0.1$ Da).

---

## 4. Technology Stack & Constraints

* **Language:** Python 3.10+ (Python 3.13+ recommended; dependency management strictly via `uv`).
* **Core Dependencies:** `matchms` (spectral logic), `numpy` & `pandas` (vectorized matrix math and data manipulation), `pydantic` & `PyYAML` (configuration), `customtkinter` (GUI).
* **Development & Testing:** `pytest` is utilized for test-driven validation of the processing logic.
* **Optional ML Dependencies:** `gensim` (for Spec2Vec), `ms2deepscore`.
* **Design Philosophy (Fail-Fast):** MassFlow operates as an infrastructure layer. It is designed to fail fast on malformed data, raising explicit `ValueError` or `TypeError` exceptions rather than issuing silent warnings, ensuring data integrity across high-throughput runs.

## 5. Usage Paradigms

MassFlow prioritizes reproducible pipelines over manual scripting. Workflows should be defined in a `config.yaml` file:

```yaml
# config.yaml
project:
  output_directory: "results/"
input:
  file_path: "data/experiment.mgf"
  reference_library: "data/library.msp"
similarity:
  algorithm: "cosine"
  tolerance: 0.1
  min_score: 0.7
```

**CLI Execution:**
```bash
uv run massflow annotate --config config.yaml
```

**Python API Execution:**
```python
from pathlib import Path
from MassFlow import io, workflow
from MassFlow.similarity import SimilarityEngine

# 1. Load Data
query_spectra = list(io.load_spectra(Path("data/experiment.mgf"), "mgf"))
ref_spectra = list(io.load_spectra(Path("data/library.msp"), "msp"))

# 2. Execute Search
engine = SimilarityEngine(config_object)
results = engine.search(query_spectra, ref_spectra)

# 3. Export
io.save_match_results(results, Path("results.csv"))
```
