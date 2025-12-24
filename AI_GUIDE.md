# Yogimass — AI One‑Pager & Design Guide 🧭

## **Short summary**

Yogimass is a config‑first toolkit for working with LC‑MS/MS tandem spectra: ingesting spectra (MGF/MSP/MS‑DIAL), cleaning & filtering them, building and searching local spectral libraries, constructing similarity networks, and performing curation and QC. It focuses on spectral similarity (using matchms/spec2vec), pragmatic tooling for library management/search, and workflows you can drive from a YAML config or the CLI.

---

## Philosophy & Goals ✅

- **Config‑first**: Reproducible workflows are declared as YAML and driven by the `yogimass` CLI (`yogimass config run --config <file>`). Prefer changing configs over ad‑hoc scripts.
- **Composability**: Small, testable components (parsers, processors, filters, backends, exporters) that can be recombined in workflows.
- **Practical ML/AI use**: Use learned representations (e.g., spec2vec via `matchms`) where they provide clear gains (similarity, annotation suggestions), but keep non‑ML fallbacks and simple baselines for reproducibility and debugging.
- **Lightweight, local-first**: Tools to build and search **local** spectral libraries (JSON/SQLite) suitable for iterative development and benchmarking.
- **Test and document everything**: Changes should include tests and, where relevant, example configs under `examples/`.

---

## Core components & where to look 🔎

- CLI / entrypoints: `yogimass/cli.py` — top‑level commands and argument mapping.
- Orchestration: `yogimass/workflow.py` — executes the pipeline defined by config.
- Configuration: `yogimass/config.py` — schema + dotted `ConfigError` validation (use dotted paths in tests).
- Similarity & storage:
  - `yogimass/similarity/library.py` — `LocalSpectralLibrary` (JSON/SQLite storage inference by extension).
  - `yogimass/similarity/backends.py` — search backends (naive, `annoy`, `faiss` placeholders).
  - `yogimass/similarity/processing.py` & `yogimass/scoring/*` — processors and scoring logic.
- IO & filters: `yogimass/io`, `yogimass/filters` — parsers, cleanup and metadata handling.
- Networking & export: `yogimass/networking/*` — building/exporting similarity networks.
- Reporting & curation: `yogimass/curation.py`, `yogimass/reporting.py` (and splintered helpers in `splinters/`).
- Tests & examples: `tests/` and `examples/` provide usage and expected behaviors.

---

## Spectral similarity & matchms ⚙️

- matchms is used for **preprocessing/cleaning** and **spectral similarity**. The repo leverages spec2vec-style embeddings via matchms for robust similarity scoring.
- Keep matchms pipelines (filters/processors) explicit and configurable in the workflow so experiments are reproducible.

---

## Design patterns & conventions 🔧

- **Dotted config validation**: Config validation raises `ConfigError(path, msg)` where `path` is dotted (e.g., `network.threshold`) — tests assert on these.
- **Storage inference**: `LocalSpectralLibrary` infers storage from filename extension (`.db`/`.sqlite` → SQLite; otherwise JSON). Use `--storage` to override.
- **Optional deps & graceful failure**: Optional libs (e.g., `annoy`, `pandas`) are `importorskip`‑style or raise informative errors.
- **Small PRs + tests**: Prefer narrow, well‑tested changes that include example configs where behavior changes.

---

## How to run & test ▶️

- Run CLI locally: `python -m yogimass.cli <command>` or use installed entrypoints.
- Run a workflow: `yogimass config run --config examples/simple_workflow.yaml`.
- Tests: `python -m pytest` (tests use `pytest.importorskip` for optional deps).

---

## Ideas for AI/LLM contributions 💡

- Add new vectorizers or pretrained spectral embeddings and benchmark against spec2vec.
- Implement and test new ANN search backends (FAISS, HNSW) with tests and migration paths.
- Auto‑annotation assistants: propose candidate annotations with confidence scoring and generate curation tasks.
- Automated QC reports and dataset benchmarking scripts (add under `examples/` + tests).
- Create reproducible evaluation datasets & metrics for annotation accuracy & retrieval performance.

---

## Quick tips for contributors ✍️

- Keep behavior reproducible: prefer small changes to config and add example configs.
- Write tests that assert dotted `ConfigError.path` messages and CLI behavior shown in `tests/test_cli.py`.
- Use `yogimass.utils.logging.get_logger` for logging instead of prints.
- Preserve backward compatibility for public CLI flags; if breaking, update `README.md` and `examples/`.

---

## Repository analysis — structure & maintainability 🔍

**Summary:** The codebase has a clear modular layout (CLI, config, workflow, similarity, IO, filters, networking, scoring) and good examples/tests coverage. The `splinters/` tree contains experimental code and variants that should be clarified or consolidated to reduce duplication and maintenance overhead.

### Observations

- **Module boundaries:** Generally coherent and domain-oriented. Watch for duplicated parser/library I/O code across `splinters/` and the main `yogimass/` package, and for backend placeholders (e.g., `annoy`/`faiss`) mixed with runtime code and tests.
- **Naming & organization:** Naming is mostly consistent, but there are small inconsistencies (for example, `similarity` vs `similarity-search` artifacts and `processing` vs `pipeline` terms). Adding short module docstrings and a top-level architecture doc would help new contributors.
- **Tests:** Tests are present and meaningful, but are spread across `tests/`, `splinters/`, and example workflows. Consider reorganizing into `tests/unit/` and `tests/integration/`, centralizing fixture/data under `tests/data/`, and using pytest markers for optional/slow tests and extras.
- **Maintainability:** Improve public API clarity (explicit exports), add incremental type annotations (and a mypy job), and add CI jobs that exercise optional extras so `importorskip` doesn't silently hide regressions. Also document `splinters/` as experimental (or move stable bits into the main package).

### Recommended next steps

1. Add `docs/architecture.md` with a simple module map and public API surface (what's stable vs internal).
2. Rationalize `splinters/` — move stable code into `yogimass/` or mark as experimental with a README and deprecation notes.
3. Reorganize tests into `tests/unit/` and `tests/integration/`, centralize example data in `tests/data/`, and add shared fixtures.
4. Add a `mypy` config and begin incremental type annotations for core modules; add a mypy CI job.
5. Define explicit public exports (e.g., `__all__` in `yogimass/__init__.py`) to clarify the library surface.
6. Add a CI matrix job that installs optional extras (e.g., `[annoy,pandas]`) and runs the full test suite to cover those code paths.

---

Any questions? 🙋‍♂️
