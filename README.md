# SpectralMetricMS — AI One‑Pager & Design Guide 🧭

---

## **Short Summary**

SpectralMetricMS is a config‑first toolkit for working with LC‑MS/MS tandem spectra: ingesting spectra (MGF/MSP/MS‑DIAL), cleaning & filtering them, building and searching local spectral libraries, constructing similarity networks, and performing curation and QC. It focuses on spectral similarity (using matchms/spec2vec), pragmatic tooling for library management/search, and workflows you can drive from a YAML config or the CLI.

---

## Philosophy & Goals ✅

- **Config‑first:** Reproducible workflows are declared as YAML and driven by the `SpectralMetricMS` CLI (`SpectralMetricMS config run --config <file>`). Prefer changing configs over ad‑hoc scripts.
- **Composability:** Small, testable components (parsers, processors, filters, backends, exporters) that can be recombined in workflows.
- **Practical ML/AI use:** Use learned representations (spec2vec via `matchms`) where they provide clear gains, but keep non‑ML fallbacks for reproducibility and debugging.
- **Lightweight, local-first:** Tools to build and search **local** spectral libraries (JSON/SQLite) suitable for iterative development and benchmarking.
- **Test and document everything:** Changes should include tests and, where relevant, example configs under `examples/`.

---

## Core Components & Where to Look 🔎

- **CLI / entrypoints:** `SpectralMetricMS/cli.py` — top‑level commands and argument mapping.
- **Orchestration:** `SpectralMetricMS/workflow.py` — executes the pipeline defined by config.
- **Configuration:** `SpectralMetricMS/config.py` — schema + dotted `ConfigError` validation.
- **Similarity & storage:**
  - `SpectralMetricMS/similarity/library.py` — `LocalSpectralLibrary` (JSON/SQLite storage inference by extension).
  - `SpectralMetricMS/similarity/backends.py` — search backends (naive, `annoy`, `faiss` placeholders).
  - `SpectralMetricMS/similarity/processing.py` & `SpectralMetricMS/scoring/*` — processors and scoring logic.
- **IO & filters:** `SpectralMetricMS/io/`, `SpectralMetricMS/filters/` — parsers, cleanup, metadata handling.
- **Networking & export:** `SpectralMetricMS/networking/*` — building/exporting similarity networks.
- **Reporting & curation:** `SpectralMetricMS/curation.py`, `SpectralMetricMS/reporting.py` (helpers in `splinters/`).
- **Tests & examples:** `tests/` and `examples/` provide usage and expected behaviors.

---

## Workflow Diagram 🌐

```mermaid
flowchart LR
    A[Spectrum Input: MGF/MSP/MS-DIAL] --> B[Preprocessing & Cleaning]
    B --> C[Similarity Computation: spec2vec / matchms]
    C --> D[LocalSpectralLibrary Storage (JSON/SQLite)]
    D --> E[Search & Retrieval: naive / ANN backends]
    E --> F[Network Construction & Export]
    F --> G[Curation & Reporting]
    G --> H[QC & Benchmark Metrics]
```

---

## Spectral Similarity & matchms ⚙️

- matchms is used for **preprocessing/cleaning** and **spectral similarity**.
- Pipelines (filters/processors) remain explicit and configurable for reproducibility.
- Keep non-ML fallbacks for comparison/debugging.

---

## Design Patterns & Conventions 🔧

- **Dotted config validation:** `ConfigError(path, msg)` where `path` is dotted (e.g., `network.threshold`).
- **Storage inference:** `LocalSpectralLibrary` infers from filename extension; override with `--storage`.
- **Optional deps & graceful failure:** Optional libraries (`annoy`, `pandas`) are `importorskip`-style or raise informative errors.
- **Small PRs + tests:** Narrow, well-tested changes with example configs.
- **Public vs internal API:** Explicit `__all__` in `SpectralMetricMS/__init__.py`. Only CLI and main modules are public; helpers remain internal.

---

## How to Run & Test ▶️

- CLI: `python -m SpectralMetricMS.cli <command>`
- Workflow: `SpectralMetricMS config run --config examples/simple_workflow.yaml`
- Tests: `python -m pytest` (use `pytest.importorskip` for optional deps)

---

## License & Attribution 📜

SpectralMetricMS is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## ⚠️ AI Development Rules (Codex, Claude, Gemini, etc.)

- **Source of Truth**: All core logic resides in `original_source/`.
- **No Restructuring**: Do not archived, move, or 'splinter' code without explicit user approval.
- **Simplification**: If asked to simplify, focus on removing unused dependencies or cleaning up imports, NOT deleting functional business logic.
- **No Refactoring**: Do not refactor code without explicit user approval.
- **No Code Generation**: Do not generate code without explicit user approval.

---
