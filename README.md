# MassFlow

[![CI](https://github.com/janusson/MassFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/janusson/MassFlow/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://ericjanusson.github.io/MassFlow/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**MassFlow is a high-precision toolkit for tandem mass spectrometry (MS/MS) annotation.**

It transforms your experimental spectra (`.mzML`, `.mgf`) and reference libraries (`.msp`, SQLite databases) into calibrated structural annotations. Designed for the lab, MassFlow prioritizes scientific rigor, reproducibility, and speed.

---

## 🚀 Quick Start

Get from installation to your first results in four simple steps.

### 1. Installation
Run the setup script to configure your environment and install all necessary dependencies:
```shell
git clone https://github.com/janusson/MassFlow && cd MassFlow
chmod +x setup.sh
./setup.sh
```

### 2. Create a configuration
Generate a schema-valid starter file, then edit its input and library paths:
```shell
uv run massflow init --output massflow_config.yaml
# Or use the guided wizard:
uv run massflow config-wizard
```
Relative paths resolve against the config file's directory. Leave `input.format` unset (`null`) when the input directory contains mixed formats; MassFlow will infer each file's type from its extension.

### 3. Prepare your reference library
MassFlow searches experimental spectra against a reference library. A raw `.msp`, `.mgf`, or `.mzML` can be used directly **if its entries pass the strict 5 ppm physical-integrity check**. When a raw library has entries whose declared formula/structure contradicts the precursor m/z, curate it into a store first:
```shell
uv run massflow db build \
  --input my_library.msp \
  --output results/my_library.db \
  --config massflow_config.yaml \
  --category library
```
`db build` **quarantines** entries that fail the physical-integrity gate (and applies the configured processing), then stores the rest. Pointing `annotate` directly at a raw library with such integrity failures instead **aborts** — MassFlow will not silently search a shrunken library. Set `input.library_path` to the resulting `.db`. See [Data Requirements](docs/user-guide/data-requirements.md).

### 4. Check compatibility, then annotate
Before a long run, check whether the data and library have plausible precursor coverage:
```shell
uv run massflow preflight --config massflow_config.yaml
uv run massflow annotate --config massflow_config.yaml
```
`preflight` is advisory and does not score spectra, change parameters, or guarantee an identification. Results are saved as CSV or mzTab in the configured output directory, with provenance reports.

---

## 🩺 Why did I get zero hits?

A run that finishes with **0 accepted identifications is a valid scientific outcome, not a crash.** MassFlow distinguishes four situations, and `massflow preflight` plus the per-file report tell you which one you are in:

| Situation | What it means | Where to look |
| :--- | :--- | :--- |
| **Data not usable** | Queries were rejected while loading/validating (empty peaks, missing precursor, failed the 5 ppm gate). | `spectra_loaded` / `spectra_rejected` in the `.report.yaml`; `massflow_quarantine.log` |
| **No library coverage** | For `cosine`, no candidate survives the configured precursor, adduct-mode, and RT gates. Other engines may score outside the precursor window, so preflight labels that coverage as heuristic. | `massflow preflight` compatibility assessment |
| **Candidates rejected** | Precursor-eligible pairs may still fail fragment matching, adduct-mode/RT filters, or score thresholds. Preflight can report sparse query peaks; the annotation report records actual scored-pair funnel counts where available. | `preflight` peak counts and the per-file `.report.yaml` |
| **Filtered statistically** | Hits passed the score gates but were removed by FDR filtering (common on small libraries at a strict `fdr_threshold`). | The `fdr_uncalibrated` / small-library warnings and the `.report.yaml` |

Start with `massflow preflight`; it explains what it can establish versus what remains uncertain, and never changes your parameters for you.

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
