# Contributing to Yogimass

Thanks for your interest in improving Yogimass! Please follow these guidelines to keep the project healthy.

## Getting Started

1. Fork the repository and create a feature branch from `main`.
2. Set up a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. For matchms-dependent modules, make sure `matchms`, `numpy`, and `pandas` install cleanly before running tests.

## Coding Standards

- Keep imports sorted and avoid introducing unused dependencies.
- Favor small, composable functions with clear logging where side-effects occur.
- Write or update tests in `tests/` for new features and bug fixes.
- Run `python3 -m pytest` (and `python3 -m pytest --cov=yogimass` if you have `pytest-cov` installed) before opening a PR.

## Pull Requests

- Describe the motivation and any trade-offs clearly in the PR description.
- Reference relevant issues or TODOs.
- Include notes about manual testing (datasets/scripts used) if automated tests don’t cover everything.

## Reporting Issues

- Provide reproduction steps, expected vs actual behavior, and relevant logs/tracebacks.
- Mention the OS, Python version, and dependency versions you used.

Thanks for your help!
