# Contributing to MassFlow

Thanks for your interest in improving MassFlow.

MassFlow is an early-stage, CLI-first Python toolkit for local MS/MS annotation. The
current stabilization priority is the core annotation workflow, SQLite-backed
library workflows, and documentation that matches the code.

## Before You Start

Please keep these project conventions in mind:

- Prefer `uv` for environment setup and dependency management.
- Target **Python 3.13+**.
- Keep functions small, explicit, and easy to read.
- Use `matchms.Spectrum` objects as the primary spectrum data structure.
- Use explicit variable names such as `precursor_mz` and
  `retention_time_seconds`.
- Add or update tests for any behavior change.
- Treat advanced features as experimental unless the docs clearly say
  otherwise.

## Development Setup

Clone the repository and install dependencies with `uv`:

```bash
git clone <your-fork-or-repo-url>
cd MassFlow
uv python pin 3.13
uv sync
```

If you prefer to work in an isolated shell:

```bash
uv shell
```

You can also run commands without activating anything:

```bash
uv run pytest
uv run massflow annotate --help
```

## Recommended Workflow

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make a small, focused change.
4. Add or update tests.
5. Run the relevant test suite locally.
6. Update docs if user-facing behavior changed.
7. Open a pull request with a clear summary and rationale.

Example:

```bash
git checkout -b docs/update-contributing-guide
uv run pytest
git add .
git commit -m "docs: refresh contributing guide"
```

## What to Work On

Good contributions include:

- improving the `massflow annotate --config ...` workflow
- improving `massflow db build`, `inspect`, and `merge`
- fixing documentation drift
- tightening input validation and error messages
- improving tests for core workflows
- clarifying core vs experimental features

Before adding new features, prefer stabilizing or simplifying the existing
workflow.

## Coding Standards

### General
- Keep imports tidy and avoid unused dependencies.
- Prefer small, composable functions over large multi-purpose functions.
- Keep side effects isolated where possible.
- Add descriptive logging around meaningful workflow steps.
- Do not silently change destructive behavior.

### Scientific and data-handling expectations
- Keep m/z values and intensities in `float64`-compatible form.
- Use `NaN` for missing scientific values instead of `0` when appropriate.
- Preserve important spectral metadata such as:
  - `precursor_mz`
  - `retention_time`
  - `adduct`
  - `compound_name`
- Maintain strict theoretical limits: structural validation should enforce a 5 ppm tolerance for precursor m/z calculations.
- Be explicit about retention-time units.
- Support robust theoretical modeling (e.g., fallback to pyteomics for isotopic envelope calculations when RDKit features are missing).

### Docstrings and naming
- Use clear, explicit names.
- Avoid abbreviations unless they are domain-standard.
- Write NumPy-style docstrings for public functions and classes.
- If a function appears to do more than one thing, consider splitting it.

## Tests

Run the full test suite before opening a pull request:

```bash
uv run pytest
```

`pytest` from a clean checkout runs exactly the intended release suite:
**unit + integration + scientific-validation tests** under `tests/`.
Collection is explicit (`testpaths = ["tests"]`); scratch and exploratory
tests under `scripts/experiments/` are never collected, and benchmarks,
slow tests, and optional-dependency tests are excluded unless requested:

```bash
# Benchmark suite (opt-in; never runs by default, never blocks CI)
uv run pytest -m benchmark -s

# Slow / long-running tests (opt-in)
uv run pytest -m slow

# Optional-dependency tests (e.g. hnswlib-backed HNSW; skipped locally
# when the extra is not installed, run in CI with --all-extras)
uv run pytest -m optional

# Scientific-validation group (also part of the default suite; CI runs it
# explicitly)
uv run pytest -m scientific
```

New tests belong to a group:

- default (unmarked) — unit and core integration tests;
- `scientific` — scientific-validation tests (module-level
  `pytestmark = pytest.mark.scientific`);
- `benchmark` — anything in `tests/benchmarks/` (marked automatically);
- `slow` — long-running tests;
- `optional` — tests that require an optional dependency extra.

If you are changing a specific area, run the relevant tests first:

```bash
uv run pytest tests/test_workflow.py
uv run pytest tests/test_cli.py
uv run pytest tests/test_cli_db.py
```

### The golden scientific-validation suite

`tests/test_scientific_validation.py` is a **known-answer** suite: every
expected score, candidate set, q-value, and annotation status is derived
from the published formulas (Watrous 2012 cosine / modified cosine;
Elias & Gygi 2007 target-decoy competition; Li et al. 2021 spectral-entropy
decoys), verified against an independent reference implementation, and
recorded in `tests/scientific_validation/ground_truth_results.json`. The
suite re-runs the pipeline and asserts the recorded ground truth is
reproduced byte-for-byte (see
`docs/user-guide/scientific_validation.md`).

A change that alters any scientific output — scoring, candidate sets,
matched peak counts, FDR calibration, ranking, export — must break these
tests. Do not regenerate the fixtures to "fix" such a failure: regenerate
only when the scientific contract intentionally changes, and review the
diff of `ground_truth_results.json` as a scientific decision:

```bash
uv run python tests/scientific_validation/generate_ground_truth.py
```

The generator refuses to write the manifest if the pipeline diverges from
the reference formulas.

When fixing a bug:

- add a test that reproduces the bug when practical
- confirm the new test fails before the fix
- confirm it passes after the fix

## Documentation

If your change affects how users run the program, update the relevant docs:

- `README.md` for quickstart and user-facing workflow changes
- `docs/ARCHITECTURE.md` for structural or boundary changes
- `CHANGELOG.md` for release-facing summaries
- `docs/post-v0.1-roadmap.md` if the work changes release planning

Keep examples short and realistic. Prefer one clear example over many stale
ones.

## Pull Requests

Please include:

- a short summary of what changed
- why the change was needed
- any trade-offs or limitations
- testing performed
- any doc updates included

Helpful PR titles:

- `docs: clarify annotate quickstart`
- `workflow: improve error handling for missing reference library`
- `db: tighten merge validation`
- `tests: add coverage for malformed mzML input`

## Reporting Issues

When reporting a bug, please include:

- operating system
- Python version
- how you installed dependencies
- command that was run
- expected behavior
- actual behavior
- traceback or log output, if available
- whether the problem involves a specific input file format

## Scope and Stability Notes

For the current pre-0.1 line, the most stable surfaces are:

- `massflow annotate --config ...`
- YAML configuration loading
- open-format ingestion (`mzML`, `mzXML`, `MGF`, `MSP`)
- SQLite library workflows via `massflow db`
- CSV result export

Features such as terminal browsing, GraphML export, and advanced ML-backed
similarity paths should be treated as experimental unless explicitly documented
otherwise.

## Code of Conduct

This repository is for the technical development of MassFlow.

* **Stay on topic:** Keep all discussions focused on code, architecture, and mass spectrometry.
* **Be professional:** Treat others with professional courtesy.
* **No disruptions:** No attacks, spam, or non-technical disputes.

Violations will result in a block. Contact the project maintainers via GitHub if you have any questions or concerns.


## Questions

If something in the docs, tests, or code disagrees with something else, prefer
raising the inconsistency in your issue or pull request rather than guessing.
Keeping the documentation aligned with the real behavior of the code is part of
the work.

Thanks for contributing to MassFlow.
