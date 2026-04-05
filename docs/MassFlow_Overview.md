# MassFlow Overview

MassFlow is a pre-1.0, config-first Python toolkit for local tandem mass spectrometry (MS/MS) annotation.

Its main job is simple:

1. load an experimental spectral file
2. load a reference library
3. apply configurable `matchms` processing
4. score query spectra against the reference library
5. write annotation results to CSV

The primary workflow is CLI-first and reproducible:

- `massflow annotate --config standard_config.yaml`

---

## What MassFlow is for

MassFlow is designed for local, scriptable annotation workflows built around open spectral formats and a YAML configuration file.

It is a good fit when you want to:

- annotate one or more experimental files against a reference library
- keep processing settings explicit and reproducible
- reuse processed reference libraries through SQLite
- run a lightweight command-line workflow without a web app or desktop GUI

---

## Core workflow

The stable core of the project is the annotation path:

- `massflow annotate --config ...`

At a high level, the program works like this:

1. parse and validate the YAML config
2. load the reference library
3. process reference spectra with the configured metadata and peak filters
4. load one experimental file or scan a configured input directory
5. process query spectra the same way
6. score processed query spectra against the processed reference library
7. estimate false discovery rate using target-decoy scoring
8. write one CSV results file per experimental input file

For repeated library use, MassFlow also supports SQLite-backed libraries through:

- `massflow db build`
- `massflow db inspect`
- `massflow db merge`

---

## Supported inputs

MassFlow directly supports open spectral formats such as:

- `mzML`
- `mzXML`
- `MGF`
- `MSP`

It also supports SQLite-backed MassFlow libraries:

- `.db`
- `.sqlite`

MassFlow intentionally does **not** perform vendor raw conversion internally. Vendor-specific raw formats should be converted to an open format before use.

---

## Configuration model

MassFlow is driven by YAML. The configuration controls:

- where query and reference data come from
- which processing filters are applied
- which similarity algorithm is used
- where results are written

Typical sections include:

- `project`
- `input`
- `processing`
- `similarity`
- `workflow`
- `export`

In the current core workflow, the most important fields are:

- `project.output_directory`
- `input.file_path` or `input.data_directory`
- `input.library_path`
- `input.format`
- `processing.*`
- `similarity.algorithm`
- `similarity.tolerance`
- `similarity.min_score`
- `similarity.fdr_threshold`

Note that the current annotation workflow writes CSV result tables directly. The broader `export` schema exists in the config model, but CSV is the current documented output path for annotation runs.

---

## Processing behavior

MassFlow uses `matchms` filters to clean and standardize spectra before scoring.

Examples of configurable processing steps include:

- metadata cleaning
- retention-time extraction
- identifier repair
- ion mode and adduct derivation
- intensity filtering
- minimum peak filtering
- m/z range filtering
- Top-N peak reduction
- intensity normalization

The same processing configuration is applied to both query spectra and reference spectra so that scoring is consistent.

---

## Similarity search

MassFlow currently exposes multiple scoring paths, but not all are equally mature.

### Core similarity paths
These are the most stable and best-documented paths for the main annotation workflow:

- `cosine`
- `modified_cosine`

### Experimental similarity paths
These exist in the codebase but should be treated as experimental until further hardened:

- `spec2vec`
- `ms2deepscore`
- `consensus`
- `cascade`

For standard library searching, start with `cosine` or `modified_cosine`.

---

## Outputs

For each processed experimental input file, MassFlow writes a CSV results file to the configured output directory.

The result table contains query information and any retained matches after score and FDR filtering. Queries without retained matches are still represented in the CSV output.

Optional GraphML network export exists in the codebase, but it should be treated as experimental rather than part of the stable release promise.

MassFlow now prefers `input.library_path` as the clearer configuration term for the library used during annotation. Older configs using `input.reference_library` are still accepted as a backward-compatible alias during transition.

Deprecation note:
- prefer `input.library_path` in all new configs and examples
- keep `input.reference_library` only for compatibility with older YAML files
- plan to remove `input.reference_library` only after the post-`v1.0` transition is clearly documented

---

## CLI surface

### Stable core commands
- `massflow annotate --config <config.yaml>`
- `massflow db build --input <library> --output <library.db> --config <config.yaml>`
- `massflow db inspect <library.db>`
- `massflow db merge --inputs <a.db> <b.db> --output <merged.db>`
- `massflow init --output <config.yaml>`

### Experimental utilities
- the consolidated experimental `MassFlow.tui` interface
  - `massflow browse <file>`
  - `massflow browse-cas <library> --cas <identifier>`

The annotation and database commands are the primary documented workflow. The interactive browsing tools are useful, but they are not the core release contract. For a short summary of the current experimental surface, see `docs/EXPERIMENTAL.md`.

---

## Simple example

A minimal config can look like this:

```yaml
project:
  output_directory: "results/example_run"

input:
  file_path: "data/experiments/COE001_16ppm_5uL.mzML"
  library_path: "data/libraries/example_library.msp"
  format: "mzml"

similarity:
  algorithm: "cosine"
  tolerance: 0.02
  min_score: 0.6
  fdr_threshold: 0.05
```

Run it with:

```bash
uv run massflow annotate --config standard_config.yaml
```

Expected result:

- a CSV file in the configured output directory
- typically named after the input file stem, for example `COE001_16ppm_5uL_results.csv`

---

## Python API

MassFlow can also be used from Python.

For the core engines, a simple pattern is:

```python
from pathlib import Path

from MassFlow import io
from MassFlow.config import MassFlowConfig
from MassFlow.similarity import SimilarityEngine

config = MassFlowConfig.from_yaml("standard_config.yaml")

query_spectra = list(io.load_spectra(Path("data/experiments/example.mgf"), "mgf"))
reference_spectra = list(io.load_spectra(Path("data/libraries/example_library.msp"), "msp"))

engine = SimilarityEngine(config.similarity)
results = engine.search(query_spectra, reference_spectra)
```

For advanced or experimental algorithms such as `consensus` and `cascade`, use the higher-level engine factory in the package rather than assuming the base engine class is sufficient.

---

## Scope boundaries

For the current pre-1.0 stabilization effort, MassFlow should be understood as:

### In scope
- CLI-first annotation workflows
- YAML configuration
- open-format spectral ingestion
- configurable `matchms` processing
- cosine and modified cosine similarity
- CSV result export
- SQLite-backed library management

### Experimental
- terminal browsing
- CAS-driven interactive inspection
- GraphML molecular networking
- `spec2vec`
- `ms2deepscore`
- `consensus`
- `cascade`

For a concise guide to these features and their intended status, see `docs/EXPERIMENTAL.md`.

### Out of scope
- desktop GUI workflows
- implicit vendor raw conversion
- cloud orchestration
- treating every advanced feature as production-stable

---

## Design principles

MassFlow is being shaped around a few simple rules:

- keep the main workflow reproducible
- prefer explicit configuration over hidden behavior
- separate I/O, processing, scoring, and orchestration responsibilities
- reject unsupported inputs clearly
- keep the stable support promise smaller than the experimental feature surface

This helps keep the main annotation path understandable, testable, and maintainable as the project moves toward `v1.0.0`.
