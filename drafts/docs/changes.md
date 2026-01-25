# CHANGES

## Unreleased

- Move CLI into package: `MassFlow/cli.py` (installable entry point).
- Add module entry (`python -m MassFlow`) via `MassFlow/__main__.py`.
- Expose console script via `[project.scripts]` in `pyproject.toml` (`MassFlow = "MassFlow.cli:main"`).
- Add smoke CLI tests and a CI job (`.github/workflows/ci.yml`) to run them.
- Deprecate legacy splinter CLI implementation (`splinters/workflows/MassFlow/cli.py`).
