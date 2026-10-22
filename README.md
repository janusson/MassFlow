# MassFlow

[![CI](https://github.com/janusson/MassFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/janusson/MassFlow/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://ericjanusson.github.io/MassFlow/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**MassFlow is a high-precision toolkit for tandem mass spectrometry (MS/MS) annotation.**

It transforms your experimental spectra (`.mzML`, `.mgf`) and reference libraries (`.msp`, SQLite databases) into calibrated structural annotations. Designed for the lab, MassFlow prioritizes scientific rigor, reproducibility, and speed.

---

## 🚀 Quick Start

Get from installation to your first results in three simple steps.

### 1. Installation
For the source checkout, install the committed dependencies with Python 3.13:
```shell
git clone https://github.com/janusson/MassFlow && cd MassFlow
uv python pin 3.13
uv sync --locked
```

### 2. Configure Your Project
Create a configuration, then set input/library/output paths and review its parameters:
```shell
uv run massflow init --output massflow_config.yaml
```
The configuration wizard and TUI are optional experimental conveniences.

### 3. Run Annotation
Start the analysis using the reviewed configuration file:
```shell
uv run massflow annotate --config massflow_config.yaml
```
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

### Interactive Console (TUI, experimental)
Prefer a visual interface over the command line? Launch the MassFlow TUI to explore your data interactively:
```shell
# Install the TUI extra (or use --all-extras for everything)
uv sync --extra tui
uv run massflow tui
```
- **Browser:** Find and upload spectral files.
- **Viewer:** Inspect centroid stick plots with a metadata panel.
- **Review candidates:** Inspect search scores and mirror plots; these do not confirm identity.
- **Diagnostics:** Get plain-English fixes for data problems.

### Library Management
SQLite is the supported library baseline. Zarr/hybrid backends are experimental; performance depends on the library, query set and hardware.
- **Build:** `uv run massflow db build` converts raw libraries into optimized stores.
- **Inspect:** `uv run massflow db inspect` views the lineage and history of your library.

### Input preparation
The supported annotation path consumes open formats. Convert vendor data with an
external tool before annotation. MassFlow's ProteoWizard wrapper is experimental
and excluded from the supported 1.0 launch path.

---

## ⚙️ Technical Architecture (For Power Users)

MassFlow is designed for high-performance "on-site" use.

- **Hybrid Storage:** Combines SQLite metadata with compressed Zarr arrays for lock-free, parallel reads.
- **Approximate Search (experimental):** Optional HNSW candidate indexing requires separate recall measurements and stays off the supported launch path.
- **ML Boundary (experimental):** Optional learned scoring and remote services have separate model/training and confidence limitations.
- **Numba Acceleration (experimental):** Keep disabled until candidate-recall equivalence is qualified on the selected release version.

## Installation Details

The inspected package supports **Python 3.13** (`>=3.13,<3.14`). Python 3.14 is not qualified. The locked matchms dependency also requires RDKit transitively; the project-level chem extra does not imply an RDKit-free core installation.

```shell
# The recommended path is to run `setup.sh`, which automates environment pinning and dependency sync. You can also perform these steps manually:
uv python pin 3.13 && uv sync

# Optional extras
# Install the TUI extra (or use --all-extras for everything)
uv sync --extra tui    # Interactive console
uv sync --extra ml     # ML scoring engines
uv sync --extra hnsw   # Sub-linear search acceleration
```

## Proposed 1.0 scope

See the [release-scope decision and go/no-go gates](docs/release/1.0-readiness.md). The current package remains 0.1.0; no official 1.0 release is approved by this document.

## License
MIT. See [LICENSE](LICENSE).

## Contact
For collaborative development or consulting in cheminformatics, please contact Dr. Eric Janusson at [ericjanusson@outlook.com](mailto:ericjanusson@outlook.com) or visit [ericjanusson.ca/contact](https://ericjanusson.ca/contact/).
