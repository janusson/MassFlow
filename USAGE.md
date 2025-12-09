# Yogimass usage guide

## Install for development

Install in editable mode with all dependencies (recommended for contributing or local testing):

```bash
python3 -m pip install -e .
```

## Golden-path smoke tests

After any code changes, run these two config-driven workflows to verify core functionality. Each command reads a config file and writes outputs under `out/`.

1. Base workflow

   ```bash
   python -m yogimass.cli config run --config examples/simple_workflow.yaml
   ```

   What it does:

   - Loads the sample MGF at `data/example_library.mgf`
   - Builds a library at `out/example_library.json`
   - Searches the library with the same spectra and writes `out/example_search.csv`
   - Builds a similarity network (`out/example_network.csv`) plus summary (`out/example_network_summary.json`)

2. Curation workflow

   ```bash
   python -m yogimass.cli config run --config examples/curation_workflow.yaml
   ```

   What it does:

   - Builds a small library from `tests/data/simple.mgf`
   - Runs library curation/QC
   - Writes a curated library (`out/curation_example_curated.json`) and QC report (`out/curation_example_qc.json`)

These two runs are the recommended smoke tests before committing changes. They exercise the config runner, library build/search, network export, and curation/QC reporting.

## Template for real data

Copy `examples/real_workflow_template.yaml`, fill in your input paths and outputs, then run:

```bash
python -m yogimass.cli config run --config path/to/my_real_config.yaml
```
