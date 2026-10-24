# MassFlow

[![CI](https://github.com/janusson/MassFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/janusson/MassFlow/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://ericjanusson.github.io/MassFlow/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
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
SQLite is the supported library baseline. Zarr/hybrid backends are experimental.
- **Build:** `uv run massflow db build` prepares raw libraries as inspectable local stores.
- **Inspect:** `uv run massflow db inspect` views the lineage and history of your library.

### Input preparation
The supported annotation path consumes open formats. Convert vendor data with an
external tool before annotation. MassFlow's ProteoWizard wrapper is experimental
and excluded from the supported 1.0 launch path.

---

## ⚙️ Technical Architecture (For Power Users)

MassFlow is intended for transparent local candidate retrieval and review.

- **Hybrid Storage (experimental):** Combines SQLite metadata with compressed Zarr arrays.
- **Approximate Search (experimental):** Optional HNSW candidate indexing requires separate recall measurements and stays off the supported launch path.
- **ML Boundary (experimental):** Optional learned scoring and remote services have separate model/training and confidence limitations.
- **Numba Acceleration (experimental):** Keep disabled until candidate-recall equivalence is qualified on the selected release version.

## Installation Details

MassFlow supports **Python >=3.13,<3.14**. Python 3.14 is not qualified.

### Core installation contract

A core installation is **not** RDKit-free. The locked `matchms` dependency requires `rdkit>=2024.3.5`, so `uv sync` or a package install pulls RDKit transitively even without the `chem` extra. The `chem` extra declares MassFlow’s own direct RDKit requirement for its cheminformatics APIs; it does not control whether RDKit is present.

MassFlow’s core annotation path retains guarded fallbacks when RDKit is unavailable, but that fallback-only environment is not the supported core installation contract. A genuinely RDKit-free core installation is not claimed with the current locked `matchms`.

```shell
# Install the committed environment from the lockfile
uv python pin 3.13 && uv sync --locked

# Optional extras
# Install the TUI extra (or use --all-extras for everything)
uv sync --extra tui    # Interactive console
uv sync --extra ml     # ML scoring engines
uv sync --extra hnsw   # Experimental approximate candidate indexing
uv sync --extra chem   # MassFlow direct chemistry APIs (RDKit is already transitive in core)
```

## Proposed 1.0 scope

See the [release-scope decision and go/no-go gates](docs/release/1.0-readiness.md). The current package remains 0.1.0; no official 1.0 release is approved by this document.

## License
MIT. See [LICENSE](LICENSE).

## Contact
For collaborative development or consulting in cheminformatics, please contact Dr. Eric Janusson at [ericjanusson@outlook.com](mailto:ericjanusson@outlook.com) or visit [ericjanusson.ca/contact](https://ericjanusson.ca/contact/).
