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

## Spectrum processor options

The spectrum processor runs before library build/search/networking with defaults of no normalization, `min_relative_intensity=0.01`, `min_absolute_intensity=0.0`, `max_peaks=500`, no m/z deduplication, and float64 outputs. Override via `similarity.processor` in YAML or the matching CLI flags (e.g., `--similarity.processor.normalization basepeak`, `--similarity.processor.max-peaks 300`; pass `--similarity.processor.max-peaks None` to disable top-N trimming). Example:

```yaml
similarity:
  processor:
    normalization: basepeak
    min_relative_intensity: 0.02
    mz_dedup_tolerance: 0.01
```

### Local processor benchmark

Run `python -m scripts.benchmark_processing --config examples/processing_workflow.yaml` to time a full workflow and report rough memory usage. This helper is intended for local sanity checks, not CI.

## Template for real data

Copy `examples/real_workflow_template.yaml`, fill in your input paths and outputs, then run:

```bash
python -m yogimass.cli config run --config path/to/my_real_config.yaml
```

For details on the on-disk library schema (JSON/SQLite), see `docs/library_schema.md`.
You can export a quick CSV summary of any library via `yogimass.library_io.export_library_summary_csv` from Python.

## End-to-end example: untargeted metabolomics

We ship a small untargeted metabolomics workflow that exercises processing, library build, search, and networking. Run it with:

```bash
python -m scripts.run_metabolomics_demo
```

The default config (`examples/metabolomics_workflow.yaml`) uses the bundled sample spectra. Outputs (library, search results, network summary) are written under `out/` by default; override paths in the YAML if needed.

## Using Yogimass with MS-DIAL

Yogimass can consume MS-DIAL export folders (`*.txt` peak tables). See `examples/msdial_workflow.yaml` for a minimal pipeline that cleans MS-DIAL outputs, builds a library, runs search, and exports a network. Run:

```bash
python -m yogimass.cli config run --config examples/msdial_workflow.yaml
```

Ensure the input directory contains MS-DIAL text exports with MS/MS peak lists; specify `input.format: msdial` and optionally `input.msdial_output` to control intermediate outputs.

## Interactive quickstart

Generate a starter YAML without hand-editing by running:

```bash
yogimass-quickstart
```

Answer the prompts (preset, input format/path, output directory). The tool writes `yogimass-quickstart.yaml` and prints the command to run it via `python -m yogimass.cli config run --config yogimass-quickstart.yaml`.

## Exporting networks to Cytoscape/Gephi

If you already built a network (CSV edges) via a config, you can export Cytoscape-ready node/edge tables with:

```bash
python -m yogimass.cli network export-cytoscape --config path/to/config.yaml --out-dir out/cytoscape_export
```

This writes `nodes.csv` and `edges.csv` that can be imported directly via Cytoscape/Gephi table import.
