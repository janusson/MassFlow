# Yogimass - Config-first LC-MS/MS workflows

[![CI](https://github.com/ericjanusson/yogimass/actions/workflows/ci.yml/badge.svg)](https://github.com/ericjanusson/yogimass/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-local-blue)](docs/)

Yogimass is a config-driven LC-MS/MS toolkit for ingesting spectra, building/searching spectral libraries, generating similarity networks, and curating libraries with QC reporting. The canonical entrypoint is a YAML/JSON config plus the `yogimass` CLI.

## Install

```bash
python -m pip install -e .[dev]        # from a clone
# or, when published:
pip install yogimass
```

Optional extras:

- `yogimass[annoy]` — enable ANN-backed search with Annoy.
- `yogimass[faiss]` — placeholder for FAISS; currently falls back to naive search.
- `yogimass[spec2vec]` — reserved for future model-backed embeddings.

Requires Python 3.10+.

## Quick start (config-first)

Run the full MGF pipeline:

```bash
yogimass config run --config examples/simple_workflow.yaml
```

Run the MS-DIAL pipeline:

```bash
yogimass config run --config examples/msdial_workflow.yaml
```

Configs are validated up front via `yogimass.config.load_config` and raise a `ConfigError` with a dotted path (e.g., `network.threshold`) when something is wrong.

## Minimal configs

MGF (from `examples/simple_workflow.yaml`):

```yaml
input:
  path: data/example_library.mgf
  format: mgf
library:
  path: out/example_library.json
  build: true
similarity:
  search: true
  queries:
    - data/example_library.mgf
network:
  enabled: true
  metric: spec2vec
  knn: 5
  output: out/example_network.csv
outputs:
  search_results: out/example_search.csv
  network_summary: out/example_network_summary.json
```

MS-DIAL (from `examples/msdial_workflow.yaml`):

```yaml
input:
  path: tests/data/msdial_small
  format: msdial
  msdial_output: out/msdial_clean
library:
  path: out/msdial_library.json
  build: true
  input_format: msdial
network:
  enabled: true
  metric: spec2vec
  knn: 2
  output: out/msdial_network.csv
outputs:
  search_results: out/msdial_search.csv
  network_summary: out/msdial_network_summary.json
```

## CLI reference

| Command | Description | Key Options |
| --- | --- | --- |
| `yogimass config run --config <file>` | Execute an end-to-end workflow from a YAML/JSON config. | Sections: `input`, `library`, `similarity`, `network`, `outputs`. |
| `yogimass library build --input ... --library out/library.json` | Build/update a local library from MGF/MSP/MS-DIAL sources. | `--format {mgf,msp}`, `--recursive`, `--storage {json,sqlite}` |
| `yogimass library search --queries ... --library out/library.json` | Search query spectra against a stored library. | `--top-n`, `--min-score`, `--backend {naive,annoy,faiss}`, `--output <csv/json>` |
| `yogimass library curate --input raw.json --output curated.json` | QC and de-duplicate a stored library, writing a curated copy and QC report. | `--qc-report`, `--min-peaks`, `--min-tic`, `--max-single-peak-fraction`, `--precursor-tolerance`, `--similarity-threshold` |
| `yogimass network build --input <dir>|--library <lib> --output graph.csv` | Build a similarity network (threshold or k-NN) and export edges/graphs. | `--metric {cosine,modified_cosine,spec2vec}`, `--threshold` or `--knn`, `--summary` |
| `yogimass clean <input_dir> <output_dir>` | (Legacy) Batch-clean every library (MGF/MSP) under `input_dir`. | `--type {mgf,msp}`, `--formats {mgf,msp,json,pickle}` |

## MS-DIAL expectations

MS-DIAL inputs are tab-delimited exports with columns:

- `Alignment ID` (int), `Average Mz` (float), `Name` (string), `Model ion area` (numeric), `MS/MS spectrum` (space-delimited `mz:intensity` pairs).
- Yogimass cleans per-experiment CSVs under `input.msdial_output`, combines them, and converts spectra for library/search/network steps.

## Development

- Run tests: `python -m pytest`
- Type check (lightweight): `python -m mypy --config-file mypy.ini yogimass`
- Linting is minimal; prefer readability and small, tested changes.

## Legacy scripts

Legacy entrypoints (`yogimass_pipeline.py`, `yogimass_buildDB.py`, `build_library_from_msp.py`, `msdial_fragment_search.py`) remain as thin wrappers that forward to the unified workflow/CLI. Prefer config-driven runs via `yogimass config run`.

## Contributing & License

Contributions are welcome—see `CONTRIBUTING.md` for setup and PR guidelines. Yogimass is released under the MIT License (see `LICENSE`).
