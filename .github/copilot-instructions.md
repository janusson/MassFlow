# GitHub Copilot / Agent instructions for SpectralMetricMS 🚀

Purpose: give a concise, immediately actionable briefing so an AI coding agent can be productive in this repository.

## Big picture (what this project does)
- SpectralMetricMS is a **config-first LC-MS/MS toolkit**: ingest spectra (MGF/MSP/MS-DIAL), build/search local spectral libraries, construct similarity networks, and curate libraries with QC reporting.
- Canonical entrypoint: the `SpectralMetricMS` CLI which dispatches to `SpectralMetricMS.workflow` orchestration code.

## Major components & where to look 🔍
- CLI / entrypoints: `SpectralMetricMS/cli.py` — argument parsing and top-level commands (`config run`, `library build/search/curate`, `network`).
- Orchestration: `SpectralMetricMS/workflow.py` — the high-level pipeline executed by `config run`.
- Configuration: `SpectralMetricMS/config.py` — validation schema, dotted `ConfigError` messages, and how CLI flags map to config fields (e.g., `similarity.processor.*`).
- Library storage & search: `SpectralMetricMS/similarity/library.py` (JSON or SQLite storage) and `SpectralMetricMS/similarity/backends.py` (search backends: `naive`, `annoy`, `faiss` placeholder).
- Networking & exporters: `SpectralMetricMS/networking/*` (build/export networks). See `network.build_network` usage in `workflow.py`.
- IO & processing: `SpectralMetricMS/io`, `SpectralMetricMS/filters`, `SpectralMetricMS/pipeline.py` — concrete file parsers and cleaners.
- Reporting & curation: `SpectralMetricMS/reporting.py` and `SpectralMetricMS/curation.py`.

## Project-specific patterns & conventions ⚙️
- **Config-first**: prefer editing or creating YAML/JSON configs under `examples/` and running `SpectralMetricMS config run --config <yaml>` over writing custom scripts. Examples: `examples/simple_workflow.yaml`, `examples/msdial_workflow.yaml`.
- **Dotted config validation**: invalid config fields raise `ConfigError(path, message)` where `path` is dotted (e.g., `network.threshold`). When fixing config issues, mirror those dotted paths.
- **Storage inference**: `LocalSpectralLibrary` infers storage from the file extension (`.db`, `.sqlite` → SQLite; otherwise JSON). Use `--storage` to override from CLI.
- **Optional backends & deps**: `annoy` (ANN) and `pandas`/MS-DIAL helpers are optional extras. Tests often `importorskip` these libraries—handle missing deps gracefully (raise informative errors).
- **Processor config mapping**: CLI flags like `--similarity.processor.min-relative-intensity` map to `ProcessorConfig` in code. Preserve naming and types when editing.
- **Mutual exclusivity**: For networking, specify either `threshold` or `knn` (not both). `workflow` enforces this.

## Developer workflows & common commands 🧰
- Run the test suite: `python -m pytest` (tests often require optional packages; they use `pytest.importorskip`).
- Run the CLI in-process (for integration tests): `python -m SpectralMetricMS.cli ...` (see `tests/test_cli.py` for examples of building/searching libraries and `config run`).
- Type-checking: `python -m mypy --config-file mypy.ini SpectralMetricMS`.
- Run examples/configs: `SpectralMetricMS config run --config examples/simple_workflow.yaml`.

## Typical PR touchpoints & change guidance ✍️
- Changing CLI flags or config schema: update `SpectralMetricMS/cli.py` and `SpectralMetricMS/config.py` together and add/adjust tests under `tests/` demonstrating the CLI behavior and validation messages (match dotted `ConfigError.path`).
- Adding a new search backend or vectorizer: follow existing `create_index_backend`/`LibrarySearcher` patterns and add tests that demonstrate behavior with/without optional deps.
- Persisted file format changes (library JSON/SQLite): write migration tests and preserve `schema_version` semantics in `similarity/library.py`.

## Examples to copy when making edits 💡
- CLI integration test: `tests/test_cli.py::test_cli_library_build_and_search` — shows how to run `library build` and `library search` via `subprocess`.
- Config validation expectations: look at `SpectralMetricMS/config.py` `WorkflowConfig._validate()` for required combinations and error messages.
- Library storage & index creation: `SpectralMetricMS/similarity/library.py` & `SpectralMetricMS/similarity/backends.py` (see how `annoy` raises a helpful ImportError when missing).

## Safety & style checklist for code edits ✅
- Keep changes small and well-tested (unit + integration where appropriate). The project favors readability and explicit validation.
- Preserve public CLI flags and config schema backwards-compatibly where possible; if breaking, update `README.md` and `examples/`.
- Use existing logging helpers (`SpectralMetricMS.utils.logging.get_logger`) rather than print statements.

## When in doubt — quick references 📚
- User-facing entry: `README.md` (quick start and CLI reference).
- Config shape & error conventions: `SpectralMetricMS/config.py`.
- Orchestrator: `SpectralMetricMS/workflow.py` (implements `run_from_config`).
- Tests showing real usage: `tests/test_cli.py`, `tests/test_quickstart.py`.

---
If you'd like, I can: (1) open a PR adding this file, (2) merge it if you already have a draft, or (3) iterate on phrasing/examples you want included—tell me which direction you prefer.