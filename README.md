# MassFlow

[![CI](https://github.com/janusson/MassFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/janusson/MassFlow/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://ericjanusson.github.io/MassFlow/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**MassFlow searches small-molecule MS/MS spectra against local reference libraries.**

It produces spectral similarity scores, ranked candidate annotations, and
query-level numerical target/decoy estimates with provenance. These outputs are
not experimentally confirmed identities. Empirical identification accuracy and
FDR control remain unvalidated for a declared deployment domain.

---

## 🚀 Quick Start

Get from installation to your first results in three simple steps.

### 1. Installation
Run the setup script to configure your environment and install all necessary dependencies:
```shell
git clone https://github.com/janusson/MassFlow && cd MassFlow
chmod +x setup.sh
./setup.sh
```

### 2. Configure Your Project
Instead of editing complex text files, use the interactive wizard to set up your analysis parameters:
```shell
uv run massflow config-wizard
```
*Follow the prompts to define your input files, reference library, and scoring thresholds.*

### 3. Run Annotation
Start the analysis using the configuration file created by the wizard:
```shell
uv run massflow annotate --config massflow_config.yaml
```
*Your results will be saved as CSV or mzTab files in your output directory, accompanied by a provenance report for every file.*

---

## 🔬 Scientific Integrity

MassFlow separates spectral similarity, candidate ranking, chemical plausibility
and statistical confidence. Passing software tests does not validate chemical
identification accuracy.

### Chemical plausibility

Declared formula/structure/adduct metadata can be checked against a fixed 5-ppm
precursor policy. Missing chemical context skips parts of that comparison;
externally supplied persistent libraries are trusted inputs. This is metadata
consistency checking, not proof of identity or a universal instrument specification.

### Confidence availability and limitations

The current estimator pools each query's best qualifying target and decoy scores.
Synthetic decoys preserve precursor metadata and permute/jitter fragments; their
ability to represent incorrect chemical matches has not been empirically validated.
No output is labelled statistically calibrated solely because decoys survive.

Results expose `confidence_state` (`estimated_uncalibrated`, `unavailable`, or
`invalid`), nullable `confidence_q_value`, methodology and limitations. With no
surviving decoys, the statistical estimate is unavailable. Historical `q_value`
and `p_value` columns and their filtering remain for compatibility; their empty-null
values are numerical rank diagnostics, not FDR estimates or error bounds.
A q-value is not the probability that an individual candidate is incorrect.
See [confidence semantics](docs/user-guide/confidence-semantics.md).

### Traceability

Provenance sidecars describe the library input, effective configuration and run
context. Source-spectrum identity must be unique and retained separately from
chemical identity. Legacy stores with missing/duplicate record identifiers need
curation or rebuilding; sidecars cannot reconstruct provenance already lost.

---

## 🛠️ The Toolset

### Interactive Console (TUI)
Prefer a visual interface over the command line? Launch the MassFlow TUI to explore your data interactively:
```shell
# Install the TUI extra (or use --all-extras for everything)
uv sync --extra tui
uv run massflow tui
```
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
```shell
uv run massflow convert --input raw_data/ --output converted_data/
```

---

## ⚙️ Technical Architecture (For Power Users)

MassFlow is designed for high-performance "on-site" use.

- **Hybrid Storage:** Combines SQLite metadata with compressed Zarr arrays for lock-free, parallel reads.
- **Sub-linear Search:** Uses a two-channel HNSW index (m/z and neutral losses) to find candidates in massive libraries instantly.
- **ML Boundary:** Heavy ML engines (Spec2Vec, MS2DeepScore) run behind a remote REST/gRPC contract with a circuit breaker, ensuring the core remains lightweight and stable.
- **Numba Acceleration:** JIT-compiled prefilters accelerate the classical scoring path.

## Installation Details

MassFlow requires **Python 3.13+**.

```shell
# The recommended path is to run `setup.sh`, which automates environment pinning and dependency sync. You can also perform these steps manually:
uv python pin 3.13 && uv sync

# Optional extras
# Install the TUI extra (or use --all-extras for everything)
uv sync --extra tui    # Interactive console
uv sync --extra ml     # ML scoring engines
uv sync --extra hnsw   # Sub-linear search acceleration
```

## License
MIT. See [LICENSE](LICENSE).

## Contact
For collaborative development or consulting in cheminformatics, please contact Dr. Eric Janusson at [ericjanusson@outlook.com](mailto:ericjanusson@outlook.com) or visit [ericjanusson.ca/contact](https://ericjanusson.ca/contact/).
