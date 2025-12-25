# Yogimass User Guide

Yogimass is a Python program for mass spectral processing, MSMS library construction, tandem mass spectrometry data similarity searches, and network analysis. This guide covers local installation, core workflows, and how to use the program.

If you are developing, testing, or contributing to Yogimass, you can follow the sections in order.

## How to Install (for development)

Install Yogimass in editable mode with all dependencies. This is recommended for local development, testing, and contributing.

```bash
python -m pip install -e .


After installation, the `yogimass` package and CLI modules will be available to Python.

## Golden-path smoke tests

After making code changes, run the following config-driven workflows to verify core functionality. Each command reads a YAML config and writes outputs under `out/`.

### 1. Base workflow

```bash
python -m yogimass.cli config run --config examples/simple_workflow.yaml


This workflow:

* Loads the sample MGF file at `data/example_library.mgf`
* Builds a spectral library (`out/example_library.json`)
* Searches the library with the same spectra and writes results to `out/example_search.csv`
* Builds a similarity network (`out/example_network.csv`) and summary (`out/example_network_summary.json`)

### 2. Curation workflow

```bash
python -m yogimass.cli config run --config examples/curation_workflow.yaml
```

This workflow:

* Builds a small library from `tests/data/simple.mgf`
* Runs library curation and QC
* Writes a curated library (`out/curation_example_curated.json`)
* Writes a QC report (`out/curation_example_qc.json`)

These two workflows are the recommended smoke tests before committing changes. Together they exercise the config runner, spectrum processing, library build/search, network export, and curation/QC logic.

## Spectrum processor options

The spectrum processor runs before library build, search, and networking. Default settings are:

* No normalization
* `min_relative_intensity = 0.01`
* `min_absolute_intensity = 0.0`
* `max_peaks = 500`
* No m/z deduplication
* Float64 outputs

Processor settings can be overridden via the `similarity.processor` block in YAML or via matching CLI flags.

Example YAML override:

```yaml
similarity:
  processor:
    normalization: basepeak
    min_relative_intensity: 0.02
    mz_dedup_tolerance: 0.01


Example CLI overrides:

```bash
--similarity.processor.normalization basepeak
--similarity.processor.max-peaks 300
```

To disable top-N trimming entirely, pass:

```bash
--similarity.processor.max-peaks None
```

### Local processor benchmark

To time a full workflow and report rough memory usage, run:

```bash
python -m scripts.benchmark_processing --config examples/processing_workflow.yaml


This helper is intended for local performance sanity checks and should not be used in CI.





## Template for real data

To run Yogimass on real datasets:

1. Copy `examples/real_workflow_template.yaml`
2. Fill in your input paths and output locations
3. Run the workflow:

```bash
python -m yogimass.cli config run --config path/to/my_real_config.yaml


For details on the on-disk library schema (JSON/SQLite), see `docs/library_schema.md`.

A quick CSV summary of any library can be exported from Python using:

```python
yogimass.library_io.export_library_summary_csv
```

## End-to-end example: untargeted metabolomics

Yogimass includes a small untargeted metabolomics demo that exercises spectrum processing, library build, search, and networking.

Run the demo with:

```bash
python -m scripts.run_metabolomics_demo


The default config (`examples/metabolomics_workflow.yaml`) uses bundled sample spectra. Outputs (library, search results, and network summaries) are written under `out/` by default. Override paths in the YAML if needed.





## Using Yogimass with MS-DIAL

Yogimass can consume MS-DIAL export folders containing text-based peak tables and MS/MS spectra.

A minimal MS-DIAL pipeline is provided in:

```text
examples/msdial_workflow.yaml


Run it with:

```bash
python -m yogimass.cli config run --config examples/msdial_workflow.yaml
```

Ensure the input directory contains MS-DIAL text exports with MS/MS peak lists. Set:

* `input.format: msdial`
* Optionally `input.msdial_output` to control intermediate outputs

## Interactive quickstart

Generate a starter YAML configuration interactively by running:

```bash
yogimass-quickstart


The tool prompts for a preset, input format and path, and output directory. It writes `yogimass-quickstart.yaml` and prints the command needed to run the workflow.





## Exporting networks to Cytoscape or Gephi

If a similarity network has already been built via a config, Cytoscape-ready node and edge tables can be exported with:

```bash
python -m yogimass.cli network export-cytoscape \
  --config path/to/config.yaml \
  --out-dir out/cytoscape_export


This command writes `nodes.csv` and `edges.csv`, which can be imported directly into Cytoscape or Gephi.





## Contributor quick check

Before committing changes, run the lightweight test harness:

```bash
scripts/check.sh


This compiles Python modules and runs the test suite. Additional pytest arguments may be passed through the script, for example:

```bash
scripts/check.sh -k processing
```
