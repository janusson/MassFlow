# GitHub Copilot / Agent instructions for Yogimass

Purpose: give a concise, immediately actionable briefing so an AI coding agent can be productive in this repository.

## Big picture (what this project does)
- Yogimass is a **config-first LC-MS/MS toolkit**: ingest spectra (MGF/MSP/MS-DIAL), build/search local spectral libraries, construct similarity networks, and curate libraries with QC reporting.
- Canonical entrypoint: the `yogimass` CLI which dispatches to `yogimass.workflow` orchestration code.

## Major components & where to look 🔍
- CLI / entrypoints: `yogimass/cli.py` — argument parsing and top-level commands (`config run`, `library build/search/curate`, `network`).
- Orchestration: `yogimass/workflow.py` — the high-level pipeline executed by `config run`.
- Configuration: `yogimass/config.py` — validation schema, dotted `ConfigError` messages, and how CLI flags map to config fields (e.g., `similarity.processor.*`).
- Library storage & search: `yogimass/similarity/library.py` (JSON or SQLite storage) and `yogimass/similarity/backends.py` (search backends: `naive`, `annoy`, `faiss` placeholder).
- Networking & exporters: `yogimass/networking/*` (build/export networks). See `network.build_network` usage in `workflow.py`.
- IO & processing: `yogimass/io`, `yogimass/filters`, `yogimass/pipeline.py` — concrete file parsers and cleaners.
- Reporting & curation: `yogimass/reporting.py` and `yogimass/curation.py`.

## Project-specific patterns & conventions ⚙️
- **Config-first**: prefer editing or creating YAML/JSON configs under `examples/` and running `yogimass config run --config <yaml>` over writing custom scripts. Examples: `examples/simple_workflow.yaml`, `examples/msdial_workflow.yaml`.
- **Dotted config validation**: invalid config fields raise `ConfigError(path, message)` where `path` is dotted (e.g., `network.threshold`). When fixing config issues, mirror those dotted paths.
- **Storage inference**: `LocalSpectralLibrary` infers storage from the file extension (`.db`, `.sqlite` → SQLite; otherwise JSON). Use `--storage` to override from CLI.
- **Optional backends & deps**: `annoy` (ANN) and `pandas`/MS-DIAL helpers are optional extras. Tests often `importorskip` these libraries—handle missing deps gracefully (raise informative errors).
- **Processor config mapping**: CLI flags like `--similarity.processor.min-relative-intensity` map to `ProcessorConfig` in code. Preserve naming and types when editing.
- **Mutual exclusivity**: For networking, specify either `threshold` or `knn` (not both). `workflow` enforces this.

## Developer workflows & common commands 🧰
- Run the test suite: `python -m pytest` (tests often require optional packages; they use `pytest.importorskip`).
- Run the CLI in-process (for integration tests): `python -m yogimass.cli ...` (see `tests/test_cli.py` for examples of building/searching libraries and `config run`).
- Type-checking: `python -m mypy --config-file mypy.ini yogimass`.
- Run examples/configs: `yogimass config run --config examples/simple_workflow.yaml`.

## Typical PR touchpoints & change guidance ✍️
- Changing CLI flags or config schema: update `yogimass/cli.py` and `yogimass/config.py` together and add/adjust tests under `tests/` demonstrating the CLI behavior and validation messages (match dotted `ConfigError.path`).
- Adding a new search backend or vectorizer: follow existing `create_index_backend`/`LibrarySearcher` patterns and add tests that demonstrate behavior with/without optional deps.
- Persisted file format changes (library JSON/SQLite): write migration tests and preserve `schema_version` semantics in `similarity/library.py`.

## Examples to copy when making edits 💡
- CLI integration test: `tests/test_cli.py::test_cli_library_build_and_search` — shows how to run `library build` and `library search` via `subprocess`.
- Config validation expectations: look at `yogimass/config.py` `WorkflowConfig._validate()` for required combinations and error messages.
- Library storage & index creation: `yogimass/similarity/library.py` & `yogimass/similarity/backends.py` (see how `annoy` raises a helpful ImportError when missing).

## Safety & style checklist for code edits ✅
- Keep changes small and well-tested (unit + integration where appropriate). The project favors readability and explicit validation.
- Preserve public CLI flags and config schema backwards-compatibly where possible; if breaking, update `README.md` and `examples/`.
- Use existing logging helpers (`yogimass.utils.logging.get_logger`) rather than print statements.

## When in doubt — quick references 📚
- User-facing entry: `README.md` (quick start and CLI reference).
- Config shape & error conventions: `yogimass/config.py`.
- Orchestrator: `yogimass/workflow.py` (implements `run_from_config`).
- Tests showing real usage: `tests/test_cli.py`, `tests/test_quickstart.py`.

---
If you'd like, I can: (1) open a PR adding this file, (2) merge it if you already have a draft, or (3) iterate on phrasing/examples you want included—tell me which direction you prefer.