# CHANGES

## Unreleased

- Move CLI into package: `yogimass/cli.py` (installable entry point).
- Add module entry (`python -m yogimass`) via `yogimass/__main__.py`.
- Expose console script via `[project.scripts]` in `pyproject.toml` (`yogimass = "yogimass.cli:main"`).
- Add smoke CLI tests and a CI job (`.github/workflows/ci.yml`) to run them.
- Deprecate legacy splinter CLI implementation (`splinters/workflows/yogimass/cli.py`).
