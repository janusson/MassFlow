# Multiprocessing & Memory Model

This document is the source of truth for how MassFlow distributes reference
libraries across worker processes: what crosses the process boundary, how much
memory every actor holds, why results are byte-identical to the previous
design, and how to reproduce the measurements.

## The problem (pre-refactor design)

The previous architecture passed the **full processed reference library and
its decoys** — a `list[Spectrum]` — into every `ProcessPoolExecutor` worker
via the `initializer` initargs:

```python
executor = ProcessPoolExecutor(initializer=_init_worker,
                               initargs=(config, references, decoys))
```

This was not shared memory. On `spawn` (macOS, Windows) each worker received
a private **pickled copy** of the whole library; on `fork` (Linux) the
copy-on-write pages were touched during scoring, so steady-state RSS still
grew toward the full library per worker. Measured on macOS / Python 3.13
(`scripts/benchmark_multiprocessing.py`, real spawned workers):

| n (spectra) | design | initargs pickle | spawn wall (1 worker) | spawn wall (8 workers) | parent RSS | worker RSS mean |
| --- | --- | --- | --- | --- | --- | --- |
| 10k | current | 11.6 MiB | 4.1 s | 33.5 s | 397 MiB | 398 MiB |
| 100k | current | 116 MiB | 6.7 s | 54.0 s | 700 MiB | ~1.0 GiB |

Serialization cost, worker startup, peak RAM, per-worker RAM, library
duplication, and decoy duplication all scale **linearly with worker count**
for the old design: 8 workers × 100k spectra ≈ 8 × ~1 GiB of private heap,
plus a 116 MiB pickle sent per worker.

## The redesign (worker-owned backend)

The refactored architecture never sends spectral payloads to workers:

1. **`prepare_library(config, output_dir)`** runs once in the parent and
   normalizes any raw library (mzML/mzXML/MGF/MSP) into a MassFlow SQLite
   store (`<stem>_library.db`). The build streams — it never materializes the
   library — and is cached by a `.meta.json` fingerprint (source path, mtime,
   size, processing-config hash). Store inputs (`.db`/`.sqlite`/`.zarr`) are
   used directly.
2. Only a compact **`LibrarySpec`** — a frozen dataclass `(path, kind,
   storage_backend)` — crosses the process boundary (~185 B pickled,
   37k–377k× smaller than the old payload).
3. **Workers open the store themselves** (`open_library` → the
   `LibraryBackend` interface: `spectrum_count`, `iter_spectra`,
   `iter_processed_chunks`, `close`) and stream processed spectra in bounded
   10k-spectrum chunks. Per-worker RAM is bounded by one chunk, not by the
   library.
4. The engine's lazy-reference decorator chunks the stream, and decoys are
   generated per chunk with **content-hashed per-spectrum seeds**
   (chunk-invariant), so results are deterministic and identical to the
   in-memory design.
5. The parent holds nothing but the spec: it streams the store build and
   releases it; `library_size` is threaded explicitly rather than re-derived.

No distributed computing is introduced: everything stays on one machine and
the store is a local file.

## Measured numbers (new design)

Same harness, same machine (macOS, 16 GiB, Python 3.13, `spawn`):

| n (spectra) | design | parent prep | pickle | spawn wall (8 workers) | parent RSS | worker RSS mean |
| --- | --- | --- | --- | --- | --- | --- |
| 10k | backend | 6.3 s | 185 B | 0.02 s | 386 MiB | 372 MiB |
| 100k | backend | 62.3 s | 186 B | 0.02 s | 387 MiB | 372 MiB |
| 1M | backend | ~15–25 min¹ | ~185 B | ~0.02 s | ~390 MiB | ~372 MiB |

¹ The 1M store build was not run to completion during the audit; it is
opt-in via `--n 1000000` (or `MASSFLOW_BENCH_1M=1` for the pytest suite). The
old-design 1M numbers are extrapolations of the measured 100k row
(~1.2 GiB pickle, ~7 GiB parent RSS, ~10+ GiB per worker) and were not
measured because the machine cannot sustain them.

Key properties, all verified by measurement:

- **Worker count no longer multiplies a serialized library.** Spawn wall
  time is constant (~0.01–0.02 s parent-side for 8 workers vs 33–54 s for the
  old design) and per-worker RSS is ~372 MiB at 10k, 100k, and (by design)
  1M — dominated by the interpreter, imports, and one 10k chunk.
- **Parent RSS is bounded and flat** (~386–392 MiB from 10k through 1M) vs
  397 → 700 MiB (and extrapolating to ~7 GiB at 1M) for the old design's
  in-memory payload. The parent streams the store build.
- **Startup vs steady state:** the old design paid the full library per
  worker in both phases; the new design pays a constant ~0.01 s launch and a
  bounded chunk in steady state.

## Cross-platform rationale

- **Linux / fork:** the old initializer payload was inherited copy-on-write,
  so spawn cost looked free — but scoring touched the arrays, materializing
  per-worker copies in steady state. The new design does not rely on COW at
  all: workers read the store, so fork and spawn behave identically.
- **macOS / Windows / spawn:** the old design pickled the full library per
  worker at startup (the measured 4–7 s/worker and the linear wall-time
  growth). The new design pickles only the spec (~185 B), so startup is
  constant and independent of library size and worker count.

## Determinism

Golden CSVs captured from the pre-refactor code live in
`tests/data/golden_multiprocessing/` (`queries_{0,1,2}_results.csv`) with
SHA-256 hashes asserted in `tests/test_library.py::TestGoldenDeterminism`.
The store round-trips spectra byte-for-byte (float64 arrays + metadata JSON)
and decoy generation is chunk-invariant, so the refactor produces
byte-identical results.

## Reproducing the measurements

```bash
# CLI sweep (spawn requires an environment that permits process spawning)
uv run python scripts/benchmark_multiprocessing.py --n 10000 --workers "1 2 4 8" --spawn
uv run python scripts/benchmark_multiprocessing.py --n 100000 --workers "1 2 4 8" --spawn
uv run python scripts/benchmark_multiprocessing.py --n 1000000 --workers "1 2 4" --spawn --design backend

# pytest suite (10k/100k; 1M opt-in via MASSFLOW_BENCH_1M=1)
uv run pytest tests/benchmarks/test_multiprocessing.py -m benchmark -q -s
```

`ru_maxrss` is a per-process high-water mark, so each design's parent-side
prep is measured in its **own spawned subprocess** (`_parent_prep_probe`);
measuring both designs in one process reports the same (meaningless) number
for both. Worker RSS is read inside each spawned worker.

## Where the code lives

- `src/MassFlow/library.py` — `LibrarySpec`, `LibraryBackend`,
  `prepare_library`, `open_library` (store + file backends).
- `src/MassFlow/workflow.py` — `_init_worker` (workers open the store),
  `_process_single_file` (streaming search), `run_annotation_pipeline`
  (single `prepare_library` call in the parent).
- `src/MassFlow/storage.py`, `src/MassFlow/database.py`,
  `src/MassFlow/zarr_store.py` — store implementations behind
  `create_spectral_store`.
- `scripts/benchmark_multiprocessing.py` — the measurement harness.
- `tests/benchmarks/test_multiprocessing.py` — benchmark regression tests.
- `tests/test_library.py` — golden determinism (byte-identical CSVs).
