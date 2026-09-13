# MassFlow

[![CI](https://github.com/janusson/MassFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/janusson/MassFlow/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://ericjanusson.github.io/MassFlow/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**MassFlow is a high-precision toolkit for tandem mass spectrometry (MS/MS) annotation.**

It transforms your experimental spectra (`.mzML`, `.mgf`) and reference libraries (`.msp`, SQLite databases) into calibrated structural annotations. Designed for the lab, MassFlow prioritizes scientific rigor, reproducibility, and speed.

---

## 🚀 Quick Start

Get from installation to your first results in three simple steps.

### 1. Installation
Run the setup script to configure your environment and install all necessary dependencies:
\`\`\`shell
git clone https://github.com/janusson/MassFlow && cd MassFlow
chmod +x setup.sh
./setup.sh
\`\`\`

### 2. Configure Your Project
Instead of editing complex text files, use the interactive wizard to set up your analysis parameters:
\`\`\`shell
uv run massflow config-wizard
\`\`\`
*Follow the prompts to define your input files, reference library, and scoring thresholds.*

### 3. Run Annotation
Start the analysis using the configuration file created by the wizard:
\`\`\`shell
uv run massflow annotate --config massflow_config.yaml
\`\`\`
*Your results will be saved as CSV or mzTab files in your output directory, accompanied by a provenance report for every file.*

---

## 🔬 Scientific Integrity

MassFlow is built to ensure your annotations are chemically plausible and statistically sound.

### The Physical Integrity Gate
To prevent "garbage-in, garbage-out" results, MassFlow enforces a strict **5 ppm precursor validation**. If a spectrum's measured mass deviates from the theoretical mass of its claimed molecule by more than 5 ppm, it is rejected before it ever reaches the scoring engine. This ensures that your results are based on real chemistry, not random matches.

### Honest Confidence (FDR Calibration)
We use **entropy-preserving decoys** to calibrate confidence. By creating fake spectra that mimic the information content of your real data, MassFlow provides a reliable **q-value** for every hit. You can trust your annotations because the program knows exactly how often a random match occurs in your specific dataset.

### Full Provenance
Every result is linked back to its source. The generated provenance sidecars record the exact library version, processing parameters, and timestamps used, making your workflow fully auditable for publications.

---

## 🛠️ The Toolset

### Interactive Console (TUI)
Prefer a visual interface over the command line? Launch the MassFlow TUI to explore your data interactively:
\`\`\`shell
uv sync --extra tui
uv run massflow tui
\`\`\`
- **Browser:** Find and upload spectral files.
- **Viewer:** Inspect centroid stick plots with a metadata panel.
- **Identify:** Run target-decoy searches with real-time mirror plots.
- **Diagnostics:** Get plain-English fixes for data problems.

### Library Management
MassFlow uses optimized SQLite and Zarr databases for lightning-fast searches, even with massive libraries.
- **Build:** `uv run massflow db build` converts raw libraries into optimized stores.
- **Inspect:** `uv run massflow db inspect` views the lineage and history of your library.

### Data Conversion
If you have vendor-specific raw files (`.raw`, `.d`), MassFlow provides a wrapper for ProteoWizard's `msconvert` to bring your data into open formats:
\`\`\`shell
uv run massflow convert --input raw_data/ --output converted_data/
\`\`\`

---

## ⚙️ Technical Architecture (For Power Users)

MassFlow is designed for high-performance "on-site" use.

- **Hybrid Storage:** Combines SQLite metadata with compressed Zarr arrays for lock-free, parallel reads.
- **Sub-linear Search:** Uses a two-channel HNSW index (m/z and neutral losses) to find candidates in massive libraries instantly.
- **ML Boundary:** Heavy ML engines (Spec2Vec, MS2DeepScore) run behind a remote REST/gRPC contract with a circuit breaker, ensuring the core remains lightweight and stable.
- **Numba Acceleration:** JIT-compiled prefilters accelerate the classical scoring path.

## Installation Details

MassFlow requires **Python 3.13+**.

\`\`\`shell
# Basic installation (via setup.sh or manually)
uv python pin 3.13 && uv sync

# Optional extras
uv sync --extra tui    # Interactive console
uv sync --extra ml     # ML scoring engines
uv sync --extra hnsw   # Sub-linear search acceleration
\`\`\`

## License
MIT. See [LICENSE](LICENSE).

## Contact
For collaborative development or consulting in cheminformatics, please contact Dr. Eric Janusson at [ericjanusson@outlook.com](mailto:ericjanusson@outlook.com) or visit [ericjanusson.ca/contact](https://ericjanusson.ca/contact/).
