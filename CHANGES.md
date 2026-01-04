# CHANGES

## Unreleased

- Move CLI into package: `SpectralMetricMS/cli.py` (installable entry point).
- Add module entry (`python -m SpectralMetricMS`) via `SpectralMetricMS/__main__.py`.
- Expose console script via `[project.scripts]` in `pyproject.toml` (`SpectralMetricMS = "SpectralMetricMS.cli:main"`).
- Add smoke CLI tests and a CI job (`.github/workflows/ci.yml`) to run them.
- Deprecate legacy splinter CLI implementation (`splinters/workflows/SpectralMetricMS/cli.py`).
