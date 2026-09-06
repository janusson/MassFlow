# Installation

MassFlow requires **Python 3.13+**. It relies heavily on standard scientific Python libraries (like `numpy`, `polars`, and `matchms`) and is explicitly designed to remain lightweight and dependency-minimal.

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

### Optional Features

MassFlow keeps the core dependency set lightweight. Heavier external dependencies (PyTorch, RDKit, hnswlib, Textual, …) are available as **optional extras** that you opt into explicitly:

```shell
# Machine-learning engines (Spec2Vec, MS2DeepScore, torch, gensim)
uv sync --extra ml

# Chemistry tools (RDKit-based structural validation)
uv sync --extra chem

# HNSW approximate candidate retrieval (hnswlib)
uv sync --extra hnsw

# File-watching mode (`massflow watch`)
uv sync --extra watch

# Interactive terminal console (`massflow tui`)
uv sync --extra tui
```

(Zarr needs no extra: it is always installed as a core dependency.)

Extras can be combined, e.g. `uv sync --extra ml --extra hnsw`. The equivalent `pip` syntax is `pip install -e ".[ml,hnsw]"`.

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

Using `uv` (recommended):
```shell
uv sync --all-groups --all-extras
uv run pytest
```

Using `pip`:
```shell
pip install -e .
pip install -e ".[chem,ml,hnsw,watch,tui]"  # optional extras as needed
```

## Reproducible Environments

MassFlow treats dependency reproducibility as a first-class property:

- **`uv.lock` is committed.** The lockfile pins the exact resolved versions
  of every dependency (including transitive ones) for the supported Python
  version.
- **Supported Python version:** 3.13 (`requires-python = ">=3.13,<3.14"`,
  `.python-version`).
- **CI installs strictly from the lockfile** (`uv sync --frozen`) and fails
  when the lockfile is out of date (`uv lock --check`).

Two people checking out the same commit therefore recreate the same
environment with:

```shell
git checkout <commit>
uv sync --frozen
```

The resolved dependency versions and the committed lockfile's SHA-256 are
recorded in every run's provenance file (`run_provenance.json`), so a given
result can always be traced back to the exact software environment that
produced it.
