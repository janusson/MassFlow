"""
Benchmarks for the multiprocessing library-distribution architecture.

Compares the worker-owned backend design (workers open a SQLite store
themselves) against the previous design's cost model (full ``list[Spectrum]``
payloads pickled into every worker). The "current" measurements reproduce the
pre-refactor costs without needing the removed code path: they measure the
pickle payload and per-worker RSS that the old ``initializer`` initargs would
have produced.

Run with::

    uv run pytest tests/benchmarks/test_multiprocessing.py -m benchmark --benchmark-enable

Sizes: 10k / 100k by default; 1M is opt-in via ``MASSFLOW_BENCH_1M=1``
(generation alone takes minutes and ~1 GiB RSS). Worker-count sweeps cover
1/2/4 workers; real worker-RSS measurements require an environment that
permits process spawning (not inside a restricted sandbox).
"""

import os
import pickle
from pathlib import Path

import pytest

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from scripts.benchmark_multiprocessing import make_library

BENCH_1M = os.environ.get("MASSFLOW_BENCH_1M", "0") == "1"
SIZES = [10_000, 100_000] + ([1_000_000] if BENCH_1M else [])
WORKER_COUNTS = [1, 2, 4]


def _make_config(tmp_path: Path) -> MassFlowConfig:
    return MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=tmp_path / "q.mgf", library_path=tmp_path / "lib.msp"
        ),
        processing=ProcessingConfig(min_peaks=1),
        similarity=SimilarityConfig(),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n_spectra", SIZES)
def test_benchmark_pickle_payload_vs_backend_spec(n_spectra, tmp_path):
    """Serialization cost that crosses the process boundary.

    Current design: the full processed library + decoys are pickled into every
    worker (the old ``initializer`` initargs). Backend design: only the
    compact ``LibrarySpec`` (a path plus two strings) is sent.
    """
    from MassFlow.library import prepare_library
    from MassFlow.processing import process_spectra
    from MassFlow.similarity import generate_decoys

    references = list(
        process_spectra(iter(make_library(n_spectra)), ProcessingConfig(min_peaks=1))
    )
    decoys = generate_decoys(references)
    config = _make_config(tmp_path)
    old_payload = (config, references, decoys)
    old_bytes = len(pickle.dumps(old_payload, protocol=pickle.HIGHEST_PROTOCOL))

    library_path = tmp_path / "lib.msp"
    from scripts.benchmark_multiprocessing import _write_msp

    _write_msp(library_path, references)
    config.input.library_path = library_path
    spec, count = prepare_library(config, tmp_path / "results")
    new_bytes = len(pickle.dumps(spec))

    assert count == n_spectra
    print(
        f"\n  n={n_spectra}: old initargs pickle = {old_bytes / 1e6:.1f} MB, "
        f"new LibrarySpec pickle = {new_bytes} B (ratio {old_bytes / max(new_bytes, 1):.0f}x)"
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n_spectra", SIZES)
@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_benchmark_worker_rss_and_startup(n_spectra, workers, tmp_path):
    """Per-worker RSS and startup for both designs (spawn).

    Old design: the full library travels in the worker initializer payload
    (pickled per worker on spawn). New design: workers open the store
    themselves and stream 10k-spectrum chunks. ``ru_maxrss`` is read inside
    each worker.
    """
    from MassFlow.library import prepare_library
    from MassFlow.processing import process_spectra
    from MassFlow.similarity import generate_decoys
    from scripts.benchmark_multiprocessing import (
        _spawn_backend_metrics,
        _spawn_worker_metrics,
        _write_msp,
        iter_library,
    )

    # Fixture file written lazily (never materialized in the parent).
    library_path = tmp_path / "lib.msp"
    _write_msp(library_path, iter_library(n_spectra))
    config = _make_config(tmp_path)
    config.input.library_path = library_path

    # Old design: full processed library + decoys in every worker.
    references = list(
        process_spectra(iter_library(n_spectra), ProcessingConfig(min_peaks=1))
    )
    decoys = generate_decoys(references)
    old_metrics = _spawn_worker_metrics(config, references, decoys, workers)
    del references, decoys

    # New design: workers open the store; parent streams the build.
    spec, count = prepare_library(config, tmp_path / "results")
    assert count == n_spectra
    new_metrics = _spawn_backend_metrics(spec, config, workers)

    print(
        f"\n  n={n_spectra} workers={workers}: "
        f"old spawn={old_metrics['worker_spawn_wall_s']}s "
        f"rss/worker={old_metrics['worker_rss_mean_mib']} MiB | "
        f"new spawn={new_metrics['worker_spawn_wall_s']}s "
        f"rss/worker={new_metrics['worker_rss_mean_mib']} MiB "
        f"(first chunk {new_metrics['first_chunk_size']})"
    )
    # The whole point of the refactor: per-worker RSS must not scale with the
    # library when workers open the store (bounded by the 10k chunk).
    assert new_metrics["worker_rss_mean_mib"] < old_metrics["worker_rss_mean_mib"]


@pytest.mark.benchmark
def test_benchmark_store_build_vs_parent_load():
    """Parent-side preparation: full in-memory load+decoys (old) vs streaming
    store build (new).

    ``ru_maxrss`` is a per-process high-water mark, so each design's parent
    prep runs in its OWN spawned subprocess (``_parent_prep_probe``).
    Measuring both designs in one process would report the same peak for
    both — the contamination the probes exist to prevent.

    The claims under test:

    * the new design's parent RSS is bounded: it does not scale with the
      library size (the store build streams, it never materializes);
    * the old design's parent RSS grows with the library size;
    * beyond the crossover size the new design's parent is smaller.
    """
    from scripts.benchmark_multiprocessing import _parent_prep_probe

    rows = []
    for n_spectra in SIZES:
        old = _parent_prep_probe(n_spectra, "current")
        new = _parent_prep_probe(n_spectra, "backend")
        rows.append((n_spectra, old, new))
        print(
            f"\n  n={n_spectra}: old parent prep = {old['prep_s']:.1f}s "
            f"(peak rss {old['rss_mib']:.0f} MiB) | "
            f"new store build = {new['build_s']:.1f}s "
            f"(peak rss {new['rss_mib']:.0f} MiB), {new['count']} spectra in store"
        )

    old_rss = [row[1]["rss_mib"] for row in rows]
    new_rss = [row[2]["rss_mib"] for row in rows]

    # New design: bounded parent memory, independent of library size.
    assert max(new_rss) < 800, (
        f"new-design parent RSS {max(new_rss)} MiB is not bounded"
    )
    assert max(new_rss) / min(new_rss) < 1.5, (
        "new-design parent RSS scales with library size"
    )

    # Old design: parent memory grows with the library; the new design's
    # does not.  (The growth is sub-linear at these sizes because fixed
    # interpreter/import overhead dominates at 10k — the ratio comparison
    # is the robust claim, not a linear-scaling one.)
    assert old_rss[-1] > old_rss[0], "old-design parent RSS does not grow with n"
    assert (old_rss[-1] / old_rss[0]) > (new_rss[-1] / new_rss[0]), (
        "old-design parent RSS grows faster than the new design's"
    )

    # Crossover: at the larger sizes the new design's parent is smaller.
    assert new_rss[-1] < old_rss[-1]
