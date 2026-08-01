# AGENTS.md — AI Coding Agent Governance for MassFlow

This document governs the behavior of any AI coding assistant (Zed AI, Cursor, GitHub Copilot, Claude Code, etc.) modifying the **MassFlow** repository. It encodes non-negotiable scientific, architectural, and testing standards that **must** be followed by all automated code modifications.

---

## 1. Core Philosophy

### 1.1 Factual & Analytical Correctness Over Fluency

- Every line of generated code must be **mathematically defensible** and **scientifically grounded**. Prefer correctness over conciseness.
- **Never introduce placeholder data, mock algorithms, or approximate heuristics** into production code paths. Production code must use real physical-chemistry logic (isotopic distributions via `pyteomics`, 5-ppm precursor validation, adduct-offset lookups from `ADDUCT_OFFSETS`).
- When implementing similarity scoring, validate exact numerical outputs against hand-computed expected values (see `tests/test_mathematical_proof.py` for the pattern: pristine `matchms.Spectrum` objects with known cosine/modified-cosine scores).
- If a scientific claim requires a reference, cite it in a comment; do not invent numbers.

### 1.2 Mathematical Precision

- **All m/z and intensity values must use `float64` precision**. Never cast to `float32` or any lower-precision type.
- **Use `NaN` for missing scientific values**, never `0`. A missing precursor mass is not a zero-mass precursor.
- Retention times must always carry explicit units in variable names (e.g., `retention_time_seconds`).
- PPM calculations must use the standard formula: `|measured - theoretical| / theoretical × 1e6`.

---

## 2. Testing Enforcement

### 2.1 Mandatory Validation Command

Before any code change is considered complete, **run the full test suite with coverage**:

```bash
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v
```

This must pass with ≥80% coverage. Failure = the change is not ready.

### 2.2 Test Requirements Per Feature

Every new feature, bug fix, or behavior change **must** include:

| Test Type | Description | Pattern to Follow |
|---|---|---|
| **Unit tests** | Validate function-level correctness with hand-crafted, deterministic inputs | `tests/test_mathematical_proof.py` |
| **Boundary/failure-mode tests** | Test the edges of physical limits (5 ppm threshold, zero peaks, empty spectra, missing metadata) | `tests/test_scientific_boundaries.py`, `tests/test_scientific_failures.py` |
| **Physics validation** | Prove that precursor m/z filtering, adduct compatibility, and isotopic checks prevent physically impossible matches | `tests/test_precursor_physics.py` |

### 2.3 What Must Never Be Mocked

- **Do not mock physical chemistry principles.** Use real isotopic calculation logic from `pyteomics` (never fake an isotopic distribution).
- **Do not mock the 5-ppm validation** in `models.py` or `cheminformatics.py`. Tests must exercise the real `SpectrumMetadata.validate_precursor_mass_logic()` and `MolecularStructure.validate_and_compute_mass()` validators.
- Use **standard test molecules** for validation: caffeine (`C8H10N4O2`), aspirin (`C9H8O4`), benzene (`C6H6`), hexachlorobenzene (`C6Cl6`). These have well-known exact masses and isotopic patterns.

### 2.4 RDKit-Optional Test Pattern

When a test requires RDKit, use the established skip pattern (see `tests/test_scientific_boundaries.py`):

```python
try:
    from rdkit import Chem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_something_requiring_rdkit():
    ...
```

Tests that **can** use formula-based fallbacks (via `pyteomics`) should have **two variants**: one using SMILES (RDKit-dependent, skipped when unavailable) and one using formula strings (always runs).

### 2.5 Pytest Markers

- `@pytest.mark.core` — must pass for the v0.1 stable release contract. Use this for tests of `cosine`, `modified_cosine`, CSV/mzTab-M export, and SQLite DB workflows.
- Tests for experimental features (`spec2vec`, `ms2deepscore`, `consensus`, `cascade`, GraphML networking) should NOT be marked `core`.

---

## 3. Dependency Boundaries

### 3.1 Heavy Dependencies Are Always Optional

The following dependencies **must never** become required for the core annotation workflow:

| Extra | Packages | Guard Variable | Fallback Behavior |
|---|---|---|---|
| `chem` | `rdkit` | `_HAS_RDKIT` | Fall back to formula-based mass validation via `pyteomics`; cosine scoring continues uninterrupted |
| `ml` | `torch`, `spec2vec`, `ms2deepscore`, `gensim` | `_HAS_SPEC2VEC`, `_HAS_MS2DEEPSCORE` | Log a warning; classical scoring engines remain available |
| `watch` | `watchfiles` | Guard at import site | Only needed for `massflow watch` subcommand |

### 3.2 Mandatory Import Guard Pattern

Every optional import **must** follow this exact pattern (see `src/MassFlow/cheminformatics.py:32-42` and `src/MassFlow/models.py:22-27`):

```python
try:
    from optional_package import Something
    _HAS_OPTIONAL = True
except ImportError:  # pragma: no cover -- tested via the "no optional" CI variant
    _HAS_OPTIONAL = False
    logger.info(
        "optional_package is not installed. Install the 'extra_name' extra for "
        "feature X. Core functionality remains fully operational."
    )
```

### 3.3 Every Guarded Import Must Have a Functional Fallback

If `_HAS_RDKIT` is `False`, code paths that require RDKit must:
1. Log a `logger.debug(...)` message explaining the skip.
2. Return a safe sentinel (`None`, `[]`, or skip the check entirely).
3. **Never crash with an `ImportError` or `AttributeError`**.

Example from `cheminformatics.py:335-340`:

```python
if not _HAS_RDKIT:
    logger.debug("Tanimoto similarity skipped: RDKit not installed.")
    return None
```

---

## 4. Code Style & Typing

### 4.1 Type Hints Are Mandatory

- **All function signatures** (public and private) must include complete type hints.
- **All Pydantic model fields** must have explicit type annotations.
- Use `from __future__ import annotations` in every module.
- Use `Optional[Type]` (not `Type | None`) for pre-3.10 compatibility, or `Type | None` with the `annotations` future import.

### 4.2 Vectorization Over Loops

- **Never write explicit Python `for` loops over spectral data arrays** in performance-sensitive paths.
- Use **NumPy vectorized operations** for m/z and intensity array computations (see `similarity.py` — `_ms1_prefilter()` and `_ms1_prefilter_arrays()` for the pattern).
- Use **Polars DataFrames/LazyFrames** for multi-dimensional metadata operations (see `processing.py:process_spectra_batch()` and `io.py:_build_results_dataframe()`).
- If a loop is unavoidable (e.g., iterating over a small list of Spectrum objects to extract metadata), document why vectorization isn't applicable.

### 4.3 Naming Conventions

- Variable names must be **explicit and self-documenting**: `precursor_mz`, `retention_time_seconds`, `spectrum_peaks`, `matched_peak_count`.
- Abbreviations are only acceptable when **domain-standard**: `mz`, `rt` (in comments only, never in variable names), `ppm`, `Da`, `FDR`.
- Functions must do **one thing**. If a docstring needs the word "and" to describe the function's purpose, split the function.

### 4.4 Docstrings

- All public functions require **NumPy-style docstrings** with `Parameters`, `Returns`, and a brief `Examples` section.
- Private helpers should have at least a one-line summary docstring.

---

## 5. Scientific & Data Integrity

### 5.1 Spectrum Representation

- The canonical in-memory representation is `matchms.Spectrum`.
- **Never strip** these critical metadata fields from a Spectrum:
  - `precursor_mz`
  - `retention_time` (in seconds)
  - `adduct`
  - `compound_name`
- `matchms.Spectrum` objects are **mutable** — flag any operation that silently mutates data in-place.

### 5.2 Precursor Mass Validation (5 ppm Rule)

The 5-ppm tolerance is **physically mandated** and **not configurable**:

- Enforced in `SpectrumMetadata.validate_precursor_mass_logic()` (see `models.py:178-225`).
- Enforced in `MolecularStructure.validate_and_compute_mass()` (see `models.py:81-150`).
- Supported adducts are registered in `cheminformatics.py:compute_adduct_offset()`.
- Unknown adducts cause `is_physically_valid = False` — this is correct behavior.

### 5.3 Isotopic Envelopes

- Theoretical isotopic distributions must be computed using `pyteomics.mass.isotopologues` (see `cheminformatics.py:get_isotopic_distribution()`).
- For highly halogenated molecules (Cl, Br), the most abundant isotopologue is often **not** M — validate that the base peak shifts correctly.

### 5.4 Input/Output Formats

- **Supported direct inputs**: mzML, mzXML, MGF, MSP, SQLite (`.db`, `.sqlite`).
- **Explicitly rejected**: vendor raw formats (`.raw`, `.d`, `.wiff`, `.lcd`, `.t2d`, `.baf`). Do not add internal conversion logic.
- **Stable outputs**: CSV, mzTab-M, and FBMN consensus_spectra.mgf + CSV pair.

---

## 6. Architecture Boundaries

### 6.1 Stable vs Experimental

| Status | Features | Constraint |
|---|---|---|
| **v0.1 Stable** | `cosine`, `modified_cosine`, `massflow annotate`, `massflow db`, CSV/mzTab-M export, SQLite libraries | Must not regress; all `@pytest.mark.core` tests must pass |
| **Experimental** | `spec2vec`, `ms2deepscore`, `consensus`, `cascade`, GraphML networking, terminal browser | Can evolve freely; must not break stable paths |

### 6.2 Module Boundaries

- **All raw SQL and schema changes** live exclusively in `src/MassFlow/database.py`. Document migration notes at the top of that file.
- **File I/O** lives in `src/MassFlow/io.py`. No other module should open/write files directly.
- **Configuration** is exclusively via Pydantic models in `src/MassFlow/config.py`. No hardcoded paths or magic numbers.
- **Orchestration** lives in `src/MassFlow/workflow.py`. Individual modules should not call each other's internal functions.

### 6.3 Multiprocessing Awareness

- The parent process generates decoys and loads the reference library into shared memory **once**.
- Workers search queries against the shared-memory library in chunks.
- Be careful with module-level state: anything placed at module scope will be copied into every worker process.

---

## 7. Commands Quick Reference

```bash
# Environment
uv python pin 3.13 && uv sync               # core deps
uv sync --all-extras                          # full install (includes chem, ml, watch)

# Testing (mandatory before every change)
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v

# Single test file
uv run pytest tests/test_workflow.py -v

# Core contract tests only
uv run pytest -m core -v

# Linting & Type Checking
uv run ruff check .                          # lint
uv run ruff check . --fix                    # lint + autofix
uv run mypy .                                # typecheck
pre-commit run --all-files                   # all hooks

# CLI smoke tests
uv run massflow annotate --config massflow_config.yaml
uv run massflow init --output test_config.yaml
uv run massflow db build --input <library> --output <out.db> --config <cfg> --category library
uv run massflow db inspect --database <out.db>
uv run massflow db merge --databases <a.db> <b.db> --output merged.db
```

---

## 8. Anti-Patterns (Never Do These)

| Anti-Pattern | Why It's Wrong | Correct Approach |
|---|---|---|
| Casting m/z arrays to `float32` | Loss of precision; violates 5-ppm guarantee | Always use `np.array(..., dtype=np.float64)` |
| Using `0` for missing precursor_mz | Ambiguous: a real precursor could be near zero | Use `NaN` |
| `for peak in spectrum.peaks:` in hot paths | O(n) Python overhead on large spectral arrays | Use NumPy vectorized ops or Polars |
| Importing RDKit at module top-level without a try/except | Crashes the core pipeline when `chem` extra isn't installed | Use `_HAS_RDKIT` guard pattern |
| Mocking `pyteomics.mass.calculate_mass` in tests | Defeats the purpose of physics validation | Use real formula strings (e.g., `"C6H6"`) |
| Hardcoding file paths in source | Breaks reproducibility | Use config-driven paths via `MassFlowConfig` |
| Adding SQL queries outside `database.py` | Schema changes become untraceable | Centralize all SQL in `database.py` |
| Using abbreviated variable names (`rt`, `mz_val`) | Reduces readability and maintainability | Use `retention_time_seconds`, `precursor_mz` |
| Introducing new required dependencies | Breaks users who only need the core workflow | Make it an optional extra with the guard pattern |
| Silent data mutation | `matchms.Spectrum` objects are mutable — side effects are hard to debug | Document mutations explicitly; prefer functional transformations |

---

## 9. When in Doubt

1. **Read the existing tests first.** The test files in `tests/` are the most reliable specification of expected behavior.
2. **Consult `CONTRIBUTING.md`** for development workflow, and `ARCHITECTURE.md` for component responsibilities.
3. **Run the test suite** before and after your change. If coverage drops below 80%, add tests.
4. **Respect stable vs experimental boundaries.** If a change affects the v0.1 stable contract (`cosine`, `modified_cosine`, `massflow annotate`, `massflow db`), it requires extra scrutiny.
5. **Ask before guessing.** If the docs, tests, and code disagree, raise the inconsistency rather than picking one arbitrarily.
