"""
Benchmark the multiprocessing library-distribution architecture.

Compares the current design (full ``list[Spectrum]`` payloads passed into every
``ProcessPoolExecutor`` worker via ``initializer`` initargs) against the
backend design (workers open a SQLite store themselves; only a compact
``LibrarySpec`` crosses the process boundary).

Measurements per design, per library size, per worker count:

* parent-side library preparation time and peak RSS, measured in a FRESH
  spawned subprocess so ``ru_maxrss`` is not contaminated by the sibling
  design's measurements (``_parent_prep_probe``)
* initargs pickle size and pickle time (the spawn serialization cost)
* real worker startup time (ProcessPoolExecutor initializer) when spawning is
  available (blocked inside sandboxes: use ``--spawn`` only when permitted)
* per-worker RSS (ru_maxrss, read inside each spawned worker)
* store size on disk (backend design)

Usage::

    uv run python scripts/benchmark_multiprocessing.py --n 10000 --workers 4
    uv run python scripts/benchmark_multiprocessing.py --n 100000 --workers "1 2 4" --spawn

Sizes 10k/100k/1M are the canonical sweep::

    uv run python scripts/benchmark_multiprocessing.py --all --spawn

``--design current`` measures only the pre-refactor cost model; ``--design
backend`` measures only the worker-owned store. ``--no-probe`` falls back to
inline (same-process) prep numbers for sandboxed environments that cannot
spawn subprocesses.
"""

from __future__ import annotations

import argparse
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def rss_max_kib() -> float:
    """Peak resident set size in KiB (ru_maxrss: KiB on Linux, bytes on macOS)."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if sys.platform == "darwin" else float(value)


def make_spectrum(spec_id: str, precursor_mz: float, rng: np.random.Generator):
    from matchms import Spectrum

    n_peaks = int(rng.integers(8, 40))
    mz = np.sort(rng.uniform(50.0, 1000.0, size=n_peaks))
    intensities = np.exp(rng.normal(0.0, 1.0, size=n_peaks))
    return Spectrum(
        mz=mz,
        intensities=intensities,
        metadata={
            "id": spec_id,
            "precursor_mz": float(precursor_mz),
            "charge": 1,
            "adduct": "[M+H]+",
            "compound_name": spec_id,
            "ionmode": "positive",
        },
    )


def make_library(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Precursor m/z wrapped into [100, 1000] so every spectrum passes the
    # default batch m/z-range filter (mz_max=1000).
    return [make_spectrum(f"ref_{i:06d}", 100.0 + (i % 900), rng) for i in range(n)]


def iter_library(n: int, seed: int = 42):
    """Lazy generator variant of :func:`make_library` (benchmark fixtures
    must not materialize the library in the parent)."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield make_spectrum(f"ref_{i:06d}", 100.0 + (i % 900), rng)


def pickle_cost(payload) -> tuple[int, float]:
    t0 = time.perf_counter()
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - t0
    return len(data), elapsed


# ---------------------------------------------------------------------------
# Parent-side preparation measured in a fresh process
# ---------------------------------------------------------------------------
# ru_maxrss is a per-process high-water mark: measuring both designs in one
# benchmark process makes the second design's number meaningless. Every
# design's parent-side prep (load+decoys, or store build) is therefore run in
# its own spawned subprocess and reported back. Results are cached per
# (n, design) so a worker-count sweep does not rebuild the fixture/store for
# every row.

_PROBE_CACHE: dict[tuple[int, str], dict] = {}


def _parent_prep_probe(n: int, design: str) -> dict:
    """Run one design's parent-side prep in a fresh process; return metrics."""
    key = (n, design)
    if key in _PROBE_CACHE:
        print(f"  [probe] cached prep metrics for n={n} design={design}", flush=True)
        return _PROBE_CACHE[key]

    import multiprocessing as mp

    print(
        f"  [probe] measuring parent-side prep for n={n} design={design} "
        f"in a fresh process (this phase can take minutes for large n)...",
        flush=True,
    )
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    process = ctx.Process(target=_parent_prep_worker, args=(n, design, result_q))
    process.start()
    process.join(timeout=3600)
    if process.exitcode != 0:
        raise RuntimeError(
            f"parent-prep probe failed (design={design!r}, n={n}, "
            f"exitcode={process.exitcode})"
        )
    result = result_q.get(timeout=60)
    _PROBE_CACHE[key] = result
    print(
        f"  [probe] done in {time.perf_counter() - t0:.1f}s: "
        f"prep={result.get('prep_s', result.get('build_s'))}s "
        f"rss={result.get('rss_mib')} MiB",
        flush=True,
    )
    return result


def _parent_prep_worker(n: int, design: str, result_q) -> None:
    """Spawn target: measure one design's parent-side preparation.

    ``current``: process the full library and generate decoys (the payload
    the old initializer would have sent). ``backend``: write the fixture file
    lazily and build the worker-openable SQLite store.
    """
    from MassFlow.config import (
        InputConfig,
        MassFlowConfig,
        ProcessingConfig,
        ProjectConfig,
        SimilarityConfig,
    )

    if design == "current":
        from MassFlow.processing import process_spectra
        from MassFlow.similarity import generate_decoys

        t0 = time.perf_counter()
        references = list(
            process_spectra(iter_library(n), ProcessingConfig(min_peaks=1))
        )
        generate_decoys(references)
        elapsed = time.perf_counter() - t0
        result_q.put(
            {
                "design": "current",
                "prep_s": round(elapsed, 3),
                "rss_mib": round(rss_max_kib() / 1024.0, 1),
            }
        )
        return

    from MassFlow.library import prepare_library

    fixture_path = Path(f"/tmp/bench_lib_{n}.msp")
    t_fixture = time.perf_counter()
    _write_msp(fixture_path, iter_library(n))
    fixture_s = time.perf_counter() - t_fixture

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=Path("/tmp/bench_out")),
        input=InputConfig(input_path=Path("q.mgf"), library_path=fixture_path),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )
    t0 = time.perf_counter()
    spec, count = prepare_library(config, Path("/tmp/bench_out"))
    build_s = time.perf_counter() - t0
    result_q.put(
        {
            "design": "backend",
            "build_s": round(build_s, 3),
            "fixture_write_s": round(fixture_s, 3),
            "rss_mib": round(rss_max_kib() / 1024.0, 1),
            "count": count,
            "spec": spec,
        }
    )


# ---------------------------------------------------------------------------
# Current design: full Spectrum lists in worker initargs
# ---------------------------------------------------------------------------

# The pre-refactor load is materialized once per size and reused across the
# worker-count sweep (the parent-side cost is identical for every row).
_PAYLOAD_CACHE: dict[int, tuple[list, list]] = {}


def _load_payload(n: int) -> tuple[list, list]:
    if n not in _PAYLOAD_CACHE:
        from MassFlow.config import ProcessingConfig
        from MassFlow.processing import process_spectra
        from MassFlow.similarity import generate_decoys

        references = list(
            process_spectra(iter_library(n), ProcessingConfig(min_peaks=1))
        )
        decoys = generate_decoys(references)
        _PAYLOAD_CACHE[n] = (references, decoys)
    return _PAYLOAD_CACHE[n]


def bench_current(n: int, workers: int, spawn: bool, use_probe: bool) -> dict:
    from MassFlow.config import (
        InputConfig,
        MassFlowConfig,
        ProcessingConfig,
        ProjectConfig,
        SimilarityConfig,
    )

    metrics: dict = {"design": "current", "n": n, "workers": workers}

    if use_probe:
        parent = _parent_prep_probe(n, "current")
        metrics["parent_prep_s"] = parent["prep_s"]
        metrics["parent_rss_mib"] = parent["rss_mib"]
    else:
        t0 = time.perf_counter()
        _load_payload(n)
        metrics["parent_prep_s"] = round(time.perf_counter() - t0, 3)
        metrics["parent_rss_mib"] = round(rss_max_kib() / 1024.0, 1)

    # The parent must still hold the payload to measure its pickle cost and
    # to spawn workers carrying it (inherent to the old design).
    print(
        f"  [current] materializing {n}-spectrum payload in the parent...", flush=True
    )
    references, decoys = _load_payload(n)
    print(
        f"  [current] payload ready ({len(references)} refs + "
        f"{len(decoys)} decoys); measuring pickle cost...",
        flush=True,
    )
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=Path("/tmp/bench")),
        input=InputConfig(input_path=Path("q.mgf"), library_path=Path("lib.msp")),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )
    payload = (config, references, decoys)
    size, elapsed = pickle_cost(payload)
    metrics["initargs_pickle_mib"] = round(size / (1024 * 1024), 2)
    metrics["initargs_pickle_s"] = round(elapsed, 3)
    metrics["estimated_spawn_send_s"] = round(workers * elapsed, 3)

    if spawn:
        print(
            f"  [current] spawning {workers} worker(s) with the full payload...",
            flush=True,
        )
        metrics.update(_spawn_worker_metrics(config, references, decoys, workers))
    return metrics


def _current_worker_probe(initargs_q, result_q, config, references, decoys):
    """Spawn target: emulate the pre-refactor worker.

    The old worker received the full payload in its initializer initargs
    (the spawn machinery had already unpickled it into the child), stored it
    in module globals, and constructed its engine. The pre-refactor
    ``_init_worker(config, references, decoys)`` no longer exists, so this
    probe reproduces the same cost model: the payload arrives as unpickled
    arguments and is held alive while the engine is built.
    """
    from MassFlow.similarity import get_similarity_engine

    t0 = time.perf_counter()
    get_similarity_engine(config.similarity)
    result_q.put((time.perf_counter() - t0, rss_max_kib()))
    initargs_q.get()  # hold the payload alive until measured


def _spawn_worker_metrics(config, references, decoys, workers: int) -> dict:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    initargs_q = manager.Queue()
    result_q = manager.Queue()
    processes = []
    t0 = time.perf_counter()
    for _ in range(workers):
        process = ctx.Process(
            target=_current_worker_probe,
            args=(initargs_q, result_q, config, references, decoys),
        )
        process.start()
        processes.append(process)
    startup = time.perf_counter() - t0
    child_rss: list[float] = []
    for _ in range(workers):
        child_rss.append(result_q.get(timeout=120)[1])
    for process in processes:
        initargs_q.put(1)
    for process in processes:
        process.join(timeout=30)
    manager.shutdown()
    return {
        "worker_spawn_wall_s": round(startup, 3),
        "worker_rss_mib": [round(rss / 1024.0, 1) for rss in child_rss],
        "worker_rss_mean_mib": round(float(np.mean(child_rss)) / 1024.0, 1),
    }


# ---------------------------------------------------------------------------
# Backend design: workers open a SQLite store themselves
# ---------------------------------------------------------------------------


def bench_backend(n: int, workers: int, spawn: bool, use_probe: bool) -> dict:
    from MassFlow.config import (
        InputConfig,
        MassFlowConfig,
        ProcessingConfig,
        ProjectConfig,
        SimilarityConfig,
    )
    from MassFlow.library import LibrarySpec, prepare_library

    metrics: dict = {"design": "backend", "n": n, "workers": workers}

    if use_probe:
        parent = _parent_prep_probe(n, "backend")
        spec: LibrarySpec = parent["spec"]
        metrics["store_build_s"] = parent["build_s"]
        metrics["fixture_write_s"] = parent["fixture_write_s"]
        metrics["spectrum_count"] = parent["count"]
        metrics["parent_rss_mib"] = parent["rss_mib"]
    else:
        # Inline fallback (no-probe mode): build here, numbers contaminated
        # by any previously measured design but still indicative.
        library_path = Path(f"/tmp/bench_lib_{n}.msp")
        _write_msp(library_path, iter_library(n))
        config = MassFlowConfig(
            project=ProjectConfig(output_directory=Path("/tmp/bench_out")),
            input=InputConfig(input_path=Path("q.mgf"), library_path=library_path),
            processing=ProcessingConfig(min_peaks=1),
            similarity=SimilarityConfig(),
        )
        t0 = time.perf_counter()
        spec, count = prepare_library(config, Path("/tmp/bench_out"))
        metrics["store_build_s"] = round(time.perf_counter() - t0, 3)
        metrics["spectrum_count"] = count
        metrics["parent_rss_mib"] = round(rss_max_kib() / 1024.0, 1)

    metrics["store_size_mib"] = round(spec.path.stat().st_size / (1024 * 1024), 2)
    metrics["store_size_mib"] = round(spec.path.stat().st_size / (1024 * 1024), 2)
    metrics["spec_pickle_bytes"] = len(pickle.dumps(spec))
    metrics["spec_pickle_s"] = round(pickle_cost(spec)[1], 5)

    config = MassFlowConfig(
        project=ProjectConfig(output_directory=Path("/tmp/bench_out")),
        input=InputConfig(
            input_path=Path("q.mgf"), library_path=Path(f"/tmp/bench_lib_{n}.msp")
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )
    if spawn:
        print(
            f"  [backend] spawning {workers} worker(s) opening the store...", flush=True
        )
        metrics.update(_spawn_backend_metrics(spec, config, workers))
    return metrics


def _write_msp(path: Path, spectra) -> None:
    """Stream an MSP fixture with bounded memory (no giant line list)."""
    with path.open("w") as handle:
        for s in spectra:
            handle.write(f"NAME: {s.get('compound_name')}\n")
            handle.write(f"PRECURSOR_MZ: {s.get('precursor_mz')}\n")
            handle.write("CHARGE: 1\n")
            handle.write(f"NUM PEAKS: {len(s.peaks.mz)}\n")
            for mz, intensity in zip(s.peaks.mz, s.peaks.intensities):
                handle.write(f"{mz}\t{intensity}\n")
            handle.write("\n")


def _backend_worker_probe(result_q, spec, config):
    """Spawn target: open the store backend and read the first chunk."""
    from MassFlow.library import open_library
    from MassFlow.workflow import _init_worker

    t0 = time.perf_counter()
    _init_worker(config, spec)
    backend = open_library(spec, config.processing)
    first_chunk = next(backend.iter_processed_chunks(chunk_size=10_000), None)
    t_open = time.perf_counter() - t0
    result_q.put((t_open, rss_max_kib(), len(first_chunk) if first_chunk else 0))


def _spawn_backend_metrics(spec, config, workers: int) -> dict:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_q = manager.Queue()
    processes = []
    t0 = time.perf_counter()
    for _ in range(workers):
        process = ctx.Process(
            target=_backend_worker_probe, args=(result_q, spec, config)
        )
        process.start()
        processes.append(process)
    startup = time.perf_counter() - t0
    child_metrics = [result_q.get(timeout=120) for _ in range(workers)]
    for process in processes:
        process.join(timeout=30)
    manager.shutdown()
    return {
        "worker_spawn_wall_s": round(startup, 3),
        "worker_open_s": [round(m[0], 3) for m in child_metrics],
        "worker_rss_mib": [round(m[1] / 1024.0, 1) for m in child_metrics],
        "worker_rss_mean_mib": round(
            float(np.mean([m[1] for m in child_metrics])) / 1024.0, 1
        ),
        "first_chunk_size": child_metrics[0][2],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000, help="library size")
    parser.add_argument("--workers", default="4", help="space-separated worker counts")
    parser.add_argument(
        "--all", action="store_true", help="sweep 10k/100k/1M x 1/2/4/8 workers"
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="spawn real worker processes (needs unsandboxed)",
    )
    parser.add_argument(
        "--design", choices=["current", "backend", "both"], default="both"
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="measure parent-side prep inline (same-process, contaminated RSS)",
    )
    args = parser.parse_args()

    sizes = [10_000, 100_000, 1_000_000] if args.all else [args.n]
    worker_counts = (
        [int(w) for w in args.workers.split()] if not args.all else [1, 2, 4, 8]
    )

    header = (
        f"{'design':8s} {'n':>8s} {'workers':>7s} | "
        f"{'prep_s':>8s} {'pickle':>12s} {'pickle_s':>9s} | "
        f"{'spawn_s':>8s} {'parent_rss_mib':>14s} {'worker_rss_mib':>14s}"
    )
    print(header)
    print("-" * len(header))

    for n in sizes:
        for workers in worker_counts:
            stamp = time.strftime("%H:%M:%S")
            if args.design in ("current", "both"):
                print(
                    f"[{stamp}] === current design: n={n} workers={workers} ===",
                    flush=True,
                )
                metrics = bench_current(n, workers, args.spawn, not args.no_probe)
                _print_row(metrics)
            if args.design in ("backend", "both"):
                print(
                    f"[{stamp}] === backend design: n={n} workers={workers} ===",
                    flush=True,
                )
                metrics = bench_backend(n, workers, args.spawn, not args.no_probe)
                _print_row(metrics)


def _print_row(metrics: dict) -> None:
    if "initargs_pickle_mib" in metrics:
        pickle_field = f"{metrics['initargs_pickle_mib']:.2f} MiB"
    else:
        pickle_field = f"{metrics['spec_pickle_bytes']} B"
    prep = metrics.get("parent_prep_s", metrics.get("store_build_s", 0.0))
    pickle_s = metrics.get("initargs_pickle_s", metrics.get("spec_pickle_s", 0.0))
    print(
        f"{metrics['design']:8s} {metrics['n']:8d} {metrics['workers']:7d} | "
        f"{prep:8.2f} {pickle_field:>12s} {pickle_s:9.4f} | "
        f"{metrics.get('worker_spawn_wall_s', 0.0):8.2f} "
        f"{metrics.get('parent_rss_mib', 0.0):14.1f} "
        f"{metrics.get('worker_rss_mean_mib', 0.0):14.1f}"
    )
    extras = []
    if "store_size_mib" in metrics:
        extras.append(f"store {metrics['store_size_mib']} MiB")
    if "spectrum_count" in metrics:
        extras.append(f"{metrics['spectrum_count']} spectra")
    if "fixture_write_s" in metrics:
        extras.append(f"fixture write {metrics['fixture_write_s']}s")
    if "first_chunk_size" in metrics:
        extras.append(f"first chunk {metrics['first_chunk_size']}")
    if extras:
        print(f"{'':8s} {'':>8s} {'':>7s} |   {', '.join(extras)}")


if __name__ == "__main__":
    main()
