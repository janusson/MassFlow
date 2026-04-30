# Installation

MassFlow requires **Python 3.13+**. It relies heavily on standard scientific Python libraries (like `numpy`, `pandas`, and `matchms`) and is explicitly designed to remain lightweight and dependency-minimal.

---

## Dependency Policy & Recommendation

The MassFlow project uses `pyproject.toml` and `uv.lock` as the single source of truth for packaging, versioning, and dependencies.

**Using `uv` is strictly recommended** to ensure you are running MassFlow in a fully reproducible, isolated virtual environment without encountering complex OS-specific dependency conflicts.

### Installing with `uv`

If you haven't already, install [uv](https://github.com/astral-sh/uv).

Then, clone the repository and sync the environment:

```shell
git clone https://github.com/yourusername/MassFlow.git
cd MassFlow
uv python pin 3.13
uv sync
```

This will automatically create a `.venv` directory, resolve the locked dependencies, and install MassFlow as an editable CLI tool.

You can now run MassFlow commands natively via `uv run`:

```shell
uv run massflow --help
```

---

## Alternative Installation (Standard `pip`)

If you cannot use `uv`, you can install MassFlow directly using standard `pip` within a virtual environment. However, you will not benefit from the strict dependency locking provided by `uv.lock`.

```shell
git clone https://github.com/yourusername/MassFlow.git
cd MassFlow

# Create a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install the package
pip install -e .
```

You can now run MassFlow directly:

```shell
massflow --help
```

---

## Development Installation

If you intend to contribute to MassFlow or run the automated test gate, you need to install the development dependencies (such as `pytest` and `pytest-cov`).

Using `uv`:
```shell
uv sync --all-groups
uv run pytest
```

Using `pip`:
```shell
pip install -e .[dev]
pytest
```
