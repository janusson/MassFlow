# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose
MassFlow is a config-first Python toolkit for local MS/MS annotation workflows. It loads open spectral formats, applies configurable `matchms` processing, runs similarity search against reference libraries, and exports structured outputs through a CLI-first workflow. Optional utilities include SQLite-backed library management and GraphML export.

## Commands

### Environment Setup
```bash
uv python pin 3.13 && uv sync          # core deps only
uv sync --all-extras                    # CI-like full install (includes ml, chem, network, viz extras)
```

### Run tests
```bash
uv run pytest                                                          # all tests
uv run pytest tests/test_workflow.py                                   # single file
uv run pytest tests/test_cli.py::test_some_case                        # single test
uv run pytest -m core                                                  # v1.0 stable contract tests only
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80  # CI with coverage (80% threshold)
```

### Lint, typecheck, format
```bash
uv run ruff check .          # lint
uv run ruff check . --fix    # lint + autofix
uv run mypy .                # typecheck
pre-commit run --all-files   # run all hooks (ruff + mypy + whitespace/yaml checks)
```

### CLI smoke runs
```bash
uv run massflow annotate --config massflow_config.yaml
uv run massflow init --output my_config.yaml
uv run massflow db build --input <library> --output <out.db> --config <cfg> --category library
uv run massflow db inspect --database <out.db>
uv run massflow db merge --databases <a.db> <b.db> --output merged.db
```

## Developer Profile
- Domain expert: PhD analytical chemist (LC-MS, metabolomics)
- Programming level: novice — explain unfamiliar patterns before implementing them
- Priority: readable, well-commented code over clever/compact code

## Tech Stack
- Python 3.13+
- Core: `matchms`, `pydantic`, `PyYAML`, `numpy`, `polars`, `pyteomics`, `typer`, `rich`
- Optional extras: `ms2deepscore`, `spec2vec` (`ml`), `rdkit` (`chem`), `networkx` (`network`/`viz`)
- Data formats: mzML, mzXML, MGF, MSP, and MassFlow SQLite libraries — all represented as `matchms.Spectrum` objects

## Architecture

The core execution path is: **YAML config → `MassFlowConfig` → `run_annotation_pipeline()`**

```
cli.py → workflow.py → config.py / io.py / processing.py / similarity.py → io.py (export)
                    ↕
                database.py (SQLite reference libraries)
```

### Module responsibilities

| Module | Role |
|---|---|
| `cli.py` | Typer CLI; dispatches `annotate`, `init`, `watch`, `db` subcommands |
| `config.py` | Pydantic models for the YAML config schema; path expansion |
| `workflow.py` | Pipeline orchestration: multiprocessing per input file, chunked reference search, FDR, export |
| `io.py` | File-system boundary; loads mzML/MGF/MSP/SQLite, rejects vendor formats, writes results |
| `processing.py` | Two-stage matchms pipeline: metadata repair then peak filtering |
| `similarity.py` | Scoring engines (`cosine`, `modified_cosine`; experimental: `spec2vec`, `ms2deepscore`, `cascade`, `ConsensusEngine`); decoy generation and FDR calculation |
| `database.py` | All SQLite logic — build/inspect/merge, `SpectralDatabase` class, triage-flag bitmask insertion |
| `models.py` | Pydantic data contracts for the Orchestrator API (`AnnotationHit`, `ConsensusResult`, `MolecularStructure`); engine-agnostic ML boundary |
| `consensus.py` | `ConsensusEngine`: probabilistic score aggregation and tie-breaking across multiple engines |
| `cheminformatics.py` | `ADDUCT_OFFSETS` registry, isotopic envelope calculation, 5 ppm precursor validation |
| `networking.py` | Optional GraphML molecular-network export (experimental) |
| `convert.py` | Format conversion utilities |
| `log_config.py` | Structured logging setup |
| `visualization/` | Optional network visualizations (experimental) |

### Key architectural details

- **Multiprocessing**: the parent process generates decoys and loads the full reference library into shared memory once; each experimental input file is dispatched to a worker process.
- **Chunked search**: workers search queries against the shared-memory library in chunks to bound RAM without disk I/O.
- **FDR**: target-decoy competition; q-values calculated per `similarity.calculate_fdr()`.
- **Output naming**: `<input_stem>_results.<ext>`; FBMN mode also writes `consensus_spectra.mgf`.
- **Stable vs experimental**: `cosine`/`modified_cosine`, CSV/mzTab-M, and SQLite workflows are the v1.0 stable contract. `spec2vec`, `ms2deepscore`, `cascade`, `consensus`, and GraphML networking are experimental.

## Data Persistence
- SQLite via Python's built-in `sqlite3`
- All DB interactions and raw SQL isolated to `src/MassFlow/database.py`
- Schema changes must be documented in a migration note at the top of `database.py`

## Pytest Markers
Tests are tagged with custom markers defined in `pyproject.toml`:
- `@pytest.mark.core` — must pass for the v1.0 stable release contract
- `@pytest.mark.experimental` — post-1.0 or optional features; allowed to be fragile

## Naming & Style Conventions
- Variable names must be explicit: `precursor_mz`, `retention_time_seconds`, `spectrum_peaks`
- No abbreviations unless domain-standard (`mz`, `rt` are acceptable in comments only)
- Functions should do one thing — if a docstring needs "and", consider splitting
- All public functions require a NumPy-style docstring with Parameters, Returns, and a brief example

## Scientific Constraints
- **m/z precision**: `float64` for all mass-to-charge values and peak intensities — never cast to `float32`
- **Spectra**: always `matchms.Spectrum` objects; never strip critical metadata (`precursor_mz`, `retention_time`, `adduct`, `compound_name`)
- **Missing values**: use `NaN`, not `0`, for absent or undefined signal/metadata
- **Retention time**: always state units explicitly (`retention_time_seconds`)
- **Precursor validation**: 5 ppm tolerance enforced in `cheminformatics.py`; deviations beyond this are rejected as physically implausible

## Workflow Preferences
- Break problems into small, testable functions before wiring them together
- Write the test before or alongside the function (not after)
- When proposing file I/O, confirm the target format before writing code
- Flag any operation that could silently mutate data in-place (`matchms.Spectrum` objects are mutable)
