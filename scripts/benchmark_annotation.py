#!/usr/bin/env python3
"""Standalone baseline benchmark for the MassFlow MS/MS annotation pipeline.

The script runs the *unmodified* MassFlow annotation workflow
(:func:`MassFlow.workflow.run_annotation_pipeline`) on one experimental file
against one reference library and reports where the wall clock and the memory
go. Nothing under ``src/MassFlow/`` is touched: every measurement is taken by
temporary wrappers installed around existing call sites with
:mod:`unittest.mock` and removed again via :class:`contextlib.ExitStack`.

Instrumented entry points
-------------------------
============================  ==============================================
``MassFlow.io.load_spectra``  Load wall-clock time and raw input spectra
                              loaded (per role: query / library / internal).
``MassFlow.processing.process_spectra`` and
``MassFlow.processing.process_spectra_batch``
                              Processing wall-clock time, input vs. retained
                              spectra and peaks, and dropped-spectrum counts.
``MassFlow.acceleration.prefilter_candidate_pairs``,
``MassFlow.similarity._ms1_prefilter``,
``MassFlow.similarity._ms1_prefilter_arrays``
                              Candidate prefilter latency and candidates kept
                              vs. prefiltered out.
``MassFlow.similarity.SimilarityEngine.search``
                              Total search window plus the score-matrix time
                              (the engine's ``sparse_array`` scoring call is
                              wrapped per call and restored afterwards).
``MassFlow.similarity.calibrate_query_level_fdr``
                              FDR calibration latency and the target/decoy
                              rows entering the per-query TDC.
``MassFlow.io.save_match_results`` /
``MassFlow.io.save_match_results_to_mztab``
                              Export latency and the retained (post-FDR) row
                              counts for CSV / mzTab-M.
``MassFlow.workflow._emit_entropy_diagnostic``
                              The library-level target/decoy entropy check,
                              an exclusive stage that streams the whole
                              library store before any query is scored.
============================  ==============================================

Only a single query file is benchmarked, which is the one path the workflow
runs in-process (``len(input_files) == 1``); the multi-file
``ProcessPoolExecutor`` path cannot see Python-level wrappers and is not
measured here.

Usage
-----
::

    uv run python scripts/benchmark_annotation.py \\
        --query data/experiment.mzML \\
        --library data/reference_library.msp \\
        --limit-queries 500

Pass ``--config`` to benchmark a specific pipeline configuration (its
processing/similarity/export settings are used as-is; ``--query`` and
``--library`` override its input paths)::

    uv run python scripts/benchmark_annotation.py \\
        --query data/experiment.mzML \\
        --library data/reference_library.msp \\
        --config massflow_config.yaml

A JSON report is written to ``<output-dir>/baseline_<UTC-timestamp>.json`` and
a stage table is printed to stdout. The pipeline's own artifacts (accumulated
results, YAML sidecars, the normalized library store) go to a fresh
``<output-dir>/artifacts/<same-timestamp>/`` directory.

Each invocation therefore builds the library store from scratch. That is
deliberate: MassFlow's stale-store path appends to an existing store instead of
replacing it when the source or the processing fingerprint changed, so reusing
an artifact directory would grow the target pool between runs and make the
baseline drift. Pass a ``.db``/``.sqlite``/``.zarr`` store as ``--library`` to
measure the already-normalized (no-build) path instead.

Exit codes
----------
``0``  the pipeline succeeded for the file and the report was written.
``1``  the pipeline raised, or reported a failed file for the query (degraded
       results still produce a report and exit 0; they are flagged in the
       report's ``warnings`` and in the printed NOTES block).
``2``  the command line or the input files were rejected before the run.
``130`` interrupted by the user (a report is still written, without stages).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional, Sequence
from unittest import mock

# ── MassFlow imports ────────────────────────────────────────────────────────
# Import the *modules* (not just the names) because the instrumentation
# replaces attributes on them at their original call sites.
import MassFlow.acceleration as acceleration_module
import MassFlow.io as io_module
import MassFlow.processing as processing_module
import MassFlow.similarity as similarity_module
import MassFlow.workflow as workflow_module
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)

try:  # POSIX-only; used for the process high-water RSS.
    import resource
except ImportError:  # pragma: no cover - non-POSIX platforms
    resource = None  # type: ignore[assignment]

SCHEMA_VERSION = "massflow.benchmark.baseline.v1"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(".benchmarks")

# Spectrum roles. "query" and "library" are the two user-provided inputs
# (matched by resolved path); "internal" is anything else, which in practice
# means the normalized library store being streamed during the search.
QUERY_ROLE = "query"
LIBRARY_ROLE = "library"
INTERNAL_ROLE = "internal"
ROLES: tuple[str, ...] = (QUERY_ROLE, LIBRARY_ROLE, INTERNAL_ROLE)

_STORE_SUFFIXES = frozenset({".db", ".sqlite", ".zarr"})

_LIMITATIONS: tuple[str, ...] = (
    "A single query file is benchmarked. The workflow runs that case "
    "in-process, which is what makes Python-level instrumentation possible; "
    "the multi-file ProcessPool path (worker throughput and scaling) is not "
    "measured here.",
    "Wrappers run on every call they intercept, so absolute wall times are "
    "slightly inflated. The inflation is per call, not per spectrum, and is "
    "negligible for the stages reported here.",
    "Peak RSS is the process high-water mark since interpreter start: it "
    "includes imports (matchms, polars, numpy, ...) and is monotonic, so it "
    "is a practical signal rather than a per-stage memory profile.",
    "tracemalloc traces Python allocations only. NumPy native buffers are not "
    "attributed, and tracing slows the measured run measurably (allocation "
    "tracing is always on in this script, by design).",
    "Candidate-pair totals are reported per prefilter source. The classical "
    "paths each apply exactly one gate -- the Numba peak/neutral-loss "
    "prefilter for modified_cosine, the MS1 precursor prefilter for cosine "
    "-- so 'possible' and 'kept' pairs are exact for the gate that ran. When "
    "no gate ran (full-matrix fallback), candidate counts are unavailable.",
    "'result_extraction_derived' is the part of the search window not spent "
    "in decoy generation, prefiltering, or sparse scoring (top-N extraction "
    "and row materialization); it is derived by subtraction, not counted.",
    "Decoy generation is attributed to the similarity search only. The "
    "library-level entropy diagnostic also generates decoys; that work is "
    "reported under its own stage and excluded from the search breakdown.",
    "'unaccounted_derived' is wall clock minus the instrumented stages; it "
    "covers preflight validation, provenance writing, and any pipeline work "
    "between the wrapped calls.",
    "Retained hit counts are counted from the payload handed to the exporter, "
    "after the workflow's per-query q-value filter. Decoy rows are dropped by "
    "that filter by construction, so a retained decoy count of zero is "
    "expected and not evidence of a broken calibration.",
    "The library store is always built from scratch (each invocation gets a "
    "fresh artifact directory), so 'library_preparation' is a cold-build "
    "number. A raw --library rebuilds it every run; pass a prepared "
    ".db/.sqlite/.zarr store to measure the no-build path.",
    "The reused store path in MassFlow appends to an existing store when the "
    "source or processing fingerprint changed, so a warm-cache measurement "
    "against a stale store can report more targets than the source library "
    "contains. This harness avoids that by never reusing an artifact "
    "directory.",
)


# ── small helpers ───────────────────────────────────────────────────────────
class _Hook(NamedTuple):
    """One installed instrumentation point."""

    label: str
    target: Any
    attribute: str
    wrapper: Any


def _utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp for report naming."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_resolve(path: Any) -> Optional[Path]:
    """Resolve ``path`` for identity comparison, or None when unusable."""
    if path is None:
        return None
    try:
        return Path(path).expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return None


def _peak_rss_mb() -> float:
    """Return this process's peak RSS in MiB (0.0 when unavailable).

    ``ru_maxrss`` is reported in bytes on macOS and in KiB on Linux.
    """
    if resource is None:  # pragma: no cover - non-POSIX platforms
        return 0.0
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return raw / divisor


def _count_peaks(spectra: Sequence[Any]) -> int:
    """Return the total number of peaks across ``spectra``.

    Missing or non-sized peak containers contribute zero rather than raising:
    the counter must never be the reason a benchmark fails.
    """
    total = 0
    for spectrum in spectra:
        mz: Any = getattr(getattr(spectrum, "peaks", None), "mz", None)
        try:
            total += len(mz)
        except TypeError:
            continue
    return total


def _close_quietly(iterator: Any) -> None:
    """Close an iterator when it supports it, ignoring any error."""
    close = getattr(iterator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # best-effort cleanup only
            pass


def _argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str
) -> Any:
    """Return the argument passed positionally or by keyword."""
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


def _pair_product(first: Any, second: Any) -> Optional[int]:
    """Return ``len(first) * len(second)`` when both are sized, else None."""
    try:
        return len(first) * len(second)
    except TypeError:
        return None


def _sha256_file(path: Path, max_bytes: int = 256 * 1024 * 1024) -> Optional[str]:
    """Return the SHA-256 of ``path``, or None when larger than ``max_bytes``."""
    if path.stat().st_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_version(name: str) -> Optional[str]:
    """Best-effort installed distribution version (None when absent)."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha() -> Optional[str]:
    """Best-effort current repository SHAs (None when unavailable)."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _throughput(numerator: int, seconds: float, unit: str) -> dict[str, Any]:
    """Describe a throughput measurement as a JSON-safe record."""
    value = numerator / seconds if seconds > 0 and numerator > 0 else None
    return {
        "numerator": int(numerator),
        "unit": unit,
        "seconds": round(seconds, 6),
        "value": round(value, 3) if value is not None else None,
    }


# ── instrumentation ─────────────────────────────────────────────────────────
class _Recorder:
    """Temporary stage instrumentation for one annotation run.

    Counters live on the instance and are reset by :meth:`reset` before each
    run. :meth:`install` returns an :class:`ExitStack` that removes every
    wrapper when closed.

    Parameters
    ----------
    query_path : Path
        Experimental input file; used to attribute loading/processing work to
        the ``"query"`` role.
    library_path : Path
        Reference library; used to attribute loading/processing work to the
        ``"library"`` role. Work that matches neither path is attributed to
        ``"internal"`` (the normalized library store being streamed).
    query_limit : int or None
        When set, stop loading query spectra after this many raw spectra.
    """

    def __init__(
        self,
        query_path: Path,
        library_path: Optional[Path],
        query_limit: Optional[int] = None,
    ) -> None:
        self._query_path = _safe_resolve(query_path)
        self._library_path = _safe_resolve(library_path)
        self._query_limit = query_limit
        self._context: list[str] = []
        self.hooks_installed: list[str] = []
        self.hooks_skipped: list[str] = []
        self.reset()

    # ── run state ───────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset every per-run counter."""
        self._context.clear()

        # io.load_spectra
        self.load_seconds: dict[str, float] = {role: 0.0 for role in ROLES}
        self.loaded_spectra: dict[str, int] = {role: 0 for role in ROLES}
        self.load_calls: dict[str, int] = {role: 0 for role in ROLES}

        # processing.process_spectra (outer generator) / process_spectra_batch
        self.process_spectra_seconds: dict[str, float] = {role: 0.0 for role in ROLES}
        self.process_spectra_calls: dict[str, int] = {role: 0 for role in ROLES}
        self.process_seconds: dict[str, float] = {role: 0.0 for role in ROLES}
        self.process_batches: dict[str, int] = {role: 0 for role in ROLES}
        self.process_input_spectra: dict[str, int] = {role: 0 for role in ROLES}
        self.process_retained_spectra: dict[str, int] = {role: 0 for role in ROLES}
        self.process_input_peaks: dict[str, int] = {role: 0 for role in ROLES}
        self.process_retained_peaks: dict[str, int] = {role: 0 for role in ROLES}

        # library normalization
        self.library_preparation_seconds = 0.0
        self.library_size = 0
        self.library_store_path: Optional[str] = None
        self.library_store_kind: Optional[str] = None

        # library-level target/decoy entropy diagnostic
        self.library_entropy_diagnostic_seconds = 0.0
        self._in_entropy_diagnostic = False

        # SimilarityEngine.search and everything below it
        self.search_seconds = 0.0
        self.search_calls = 0
        self.query_spectra_searched = 0
        self.decoy_seconds = 0.0
        self.decoys_generated = 0
        self.prefilter_seconds: dict[str, float] = {}
        self.prefilter_calls: dict[str, int] = {}
        self.prefilter_possible_pairs: dict[str, int] = {}
        self.prefilter_candidate_pairs: dict[str, int] = {}
        self.score_seconds = 0.0
        self.score_calls = 0
        self.full_matrix_seconds = 0.0
        self.full_matrix_calls = 0
        self.full_matrix_pairs = 0
        self.result_rows_pre_fdr = 0
        self.decoy_result_rows = 0

        # FDR calibration
        self.fdr_seconds = 0.0
        self.fdr_calls = 0
        self.fdr_input_target_rows = 0
        self.fdr_input_decoy_rows = 0
        self.fdr_summary: dict[str, int] = {}

        # export
        self.export_seconds = 0.0
        self.export_calls = 0
        self.export_rows = 0
        self.export_target_rows = 0
        self.export_decoy_rows = 0
        self.export_bytes = 0
        self.export_format: Optional[str] = None
        self.export_paths: list[str] = []

    # ── role attribution ────────────────────────────────────────────────────
    def _role(self) -> str:
        """Role of the innermost active context ("query" when idle)."""
        return self._context[-1] if self._context else QUERY_ROLE

    def _role_for_path(self, file_path: Any) -> str:
        """Role of a file being read, decided by resolved path identity."""
        resolved = _safe_resolve(file_path)
        if resolved is not None:
            if resolved == self._query_path:
                return QUERY_ROLE
            if resolved == self._library_path:
                return LIBRARY_ROLE
        return INTERNAL_ROLE

    # ── score-matrix timing ─────────────────────────────────────────────────
    def _install_score_timer(self, engine: Any) -> Optional[Callable[[], None]]:
        """Time the engine's sparse scoring call, returning a restore callback.

        The engine scores its candidate pairs through
        ``similarity_function.sparse_array``; wrapping that instance attribute
        separates exact score-matrix time from candidate generation and row
        extraction. Returns None (and the derived timings are used instead)
        when the attribute is absent or not assignable.
        """
        similarity_function = getattr(engine, "similarity_function", None)
        original = getattr(similarity_function, "sparse_array", None)
        if not callable(original):
            return None

        @functools.wraps(original)
        def timed_sparse_array(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.score_seconds += time.perf_counter() - started
                self.score_calls += 1

        try:
            setattr(similarity_function, "sparse_array", timed_sparse_array)
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            return None

        def restore() -> None:
            try:
                setattr(similarity_function, "sparse_array", original)
            except (AttributeError, TypeError):  # pragma: no cover - defensive
                pass

        return restore

    # ── wrapper factories ───────────────────────────────────────────────────
    def _prefilter_wrapper(
        self,
        original: Callable[..., tuple[Any, Any]],
        source: str,
        first_name: str,
        second_name: str,
    ) -> Callable[..., tuple[Any, Any]]:
        """Wrap a ``(rows, cols)`` candidate gate to time and count it."""

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            possible = _pair_product(
                _argument(args, kwargs, 0, first_name),
                _argument(args, kwargs, 1, second_name),
            )
            started = time.perf_counter()
            rows, cols = original(*args, **kwargs)
            elapsed = time.perf_counter() - started
            self.prefilter_seconds[source] = (
                self.prefilter_seconds.get(source, 0.0) + elapsed
            )
            self.prefilter_calls[source] = self.prefilter_calls.get(source, 0) + 1
            self.prefilter_possible_pairs[source] = self.prefilter_possible_pairs.get(
                source, 0
            ) + (possible or 0)
            self.prefilter_candidate_pairs[source] = self.prefilter_candidate_pairs.get(
                source, 0
            ) + int(len(rows))
            return rows, cols

        return wrapper

    def _build_hooks(self) -> tuple[list[_Hook], list[str]]:
        """Build every wrapper, skipping call sites this version does not have."""
        hooks: list[_Hook] = []
        skipped: list[str] = []

        def add_hook(
            label: str,
            target: Any,
            attribute: str,
            builder: Callable[[Any], Any],
        ) -> None:
            original = getattr(target, attribute, None)
            if not callable(original):
                skipped.append(label)
                return
            hooks.append(_Hook(label, target, attribute, builder(original)))

        # 1. Library normalization into the worker store (once per run, in the
        #    parent process).
        def build_prepare_library(original: Any) -> Any:
            @functools.wraps(original)
            def prepare_library(
                config: MassFlowConfig, output_directory: Path
            ) -> tuple[Any, int]:
                self._context.append(LIBRARY_ROLE)
                started = time.perf_counter()
                try:
                    spec, count = original(config, output_directory)
                finally:
                    self.library_preparation_seconds += time.perf_counter() - started
                    self._context.pop()
                self.library_size = int(count)
                self.library_store_path = str(getattr(spec, "path", "")) or None
                self.library_store_kind = str(getattr(spec, "kind", "")) or None
                return spec, count

            return prepare_library

        add_hook(
            "MassFlow.workflow.prepare_library",
            workflow_module,
            "prepare_library",
            build_prepare_library,
        )

        # 2. Raw spectrum ingestion. `load_spectra` is a generator, so the
        #    wrapper times the producer side (each `next()`), not consumer
        #    work, and enforces --limit-queries on the query file only.
        def build_load_spectra(original: Any) -> Any:
            @functools.wraps(original)
            def load_spectra(
                file_path: Path,
                file_format: Optional[str] = None,
                rejection_reporter: Optional[Callable[[str], None]] = None,
            ) -> Iterator[Any]:
                role = self._role_for_path(file_path)
                limit = self._query_limit if role == QUERY_ROLE else None
                self.load_calls[role] += 1
                loader = original(
                    file_path,
                    file_format=file_format,
                    rejection_reporter=rejection_reporter,
                )
                produced = 0
                while True:
                    started = time.perf_counter()
                    try:
                        spectrum = next(loader)
                    except StopIteration:
                        self.load_seconds[role] += time.perf_counter() - started
                        return
                    self.load_seconds[role] += time.perf_counter() - started
                    if limit is not None and produced >= limit:
                        _close_quietly(loader)
                        return
                    produced += 1
                    self.loaded_spectra[role] += 1
                    yield spectrum

            return load_spectra

        add_hook(
            "MassFlow.io.load_spectra", io_module, "load_spectra", build_load_spectra
        )

        # 3. Metadata + peak processing. The batch call is where all work
        #    happens (the outer generator only chunks), so batch counters are
        #    authoritative and double-counting is impossible even though
        #    process_spectra calls process_spectra_batch internally.
        def build_process_spectra_batch(original: Any) -> Any:
            @functools.wraps(original)
            def process_spectra_batch(
                spectra: list[Any],
                config: ProcessingConfig,
                rejection_reporter: Optional[Callable[[str], None]] = None,
            ) -> list[Any]:
                role = self._role()
                input_spectra = len(spectra)
                input_peaks = _count_peaks(spectra)
                started = time.perf_counter()
                processed = original(spectra, config, rejection_reporter)
                elapsed = time.perf_counter() - started
                self.process_seconds[role] += elapsed
                self.process_batches[role] += 1
                self.process_input_spectra[role] += input_spectra
                self.process_retained_spectra[role] += len(processed)
                self.process_input_peaks[role] += input_peaks
                self.process_retained_peaks[role] += _count_peaks(processed)
                return processed

            return process_spectra_batch

        add_hook(
            "MassFlow.processing.process_spectra_batch",
            processing_module,
            "process_spectra_batch",
            build_process_spectra_batch,
        )

        def build_process_spectra(original: Any) -> Any:
            @functools.wraps(original)
            def process_spectra(
                spectra: Any,
                config: ProcessingConfig,
                rejection_reporter: Optional[Callable[[str], None]] = None,
            ) -> Iterator[Any]:
                role = self._role()
                self.process_spectra_calls[role] += 1
                started = time.perf_counter()
                try:
                    yield from original(spectra, config, rejection_reporter)
                finally:
                    self.process_spectra_seconds[role] += time.perf_counter() - started

            return process_spectra

        add_hook(
            "MassFlow.processing.process_spectra",
            processing_module,
            "process_spectra",
            build_process_spectra,
        )

        # 4. Candidate generation and scoring window. The search wrapper also
        #    pushes the "internal" context: everything the search triggers
        #    (store streaming, per-chunk decoy generation) is internal work,
        #    not query work.
        def build_search(original: Any) -> Any:
            @functools.wraps(original)
            def search(
                engine_self: Any,
                query_spectra: list[Any],
                reference_spectra: Any,
                *args: Any,
                **kwargs: Any,
            ) -> list[Any]:
                n_queries = len(query_spectra)
                self._context.append(INTERNAL_ROLE)
                restore_score_timer = self._install_score_timer(engine_self)
                started = time.perf_counter()
                try:
                    results = original(
                        engine_self, query_spectra, reference_spectra, *args, **kwargs
                    )
                finally:
                    elapsed = time.perf_counter() - started
                    if restore_score_timer is not None:
                        restore_score_timer()
                    self._context.pop()
                    self.search_seconds += elapsed
                    self.search_calls += 1
                    self.query_spectra_searched += n_queries
                self.result_rows_pre_fdr += len(results)
                for row in results:
                    if row.get("is_decoy", False):
                        self.decoy_result_rows += 1
                return results

            return search

        add_hook(
            "MassFlow.similarity.SimilarityEngine.search",
            similarity_module.SimilarityEngine,
            "search",
            build_search,
        )

        add_hook(
            "MassFlow.acceleration.prefilter_candidate_pairs",
            acceleration_module,
            "prefilter_candidate_pairs",
            lambda original: self._prefilter_wrapper(
                original, "acceleration.peaks_neutral_loss", "references", "queries"
            ),
        )

        if not hasattr(acceleration_module, "prefilter_candidate_pairs"):
            skipped.append("MassFlow.acceleration.prefilter_candidate_pairs")

        add_hook(
            "MassFlow.similarity._ms1_prefilter",
            similarity_module,
            "_ms1_prefilter",
            lambda original: self._prefilter_wrapper(
                original, "similarity.ms1_precursor", "all_references", "query_spectra"
            ),
        )

        add_hook(
            "MassFlow.similarity._ms1_prefilter_arrays",
            similarity_module,
            "_ms1_prefilter_arrays",
            lambda original: self._prefilter_wrapper(
                original, "similarity.ms1_precursor_arrays", "ref_mzs", "query_spectra"
            ),
        )

        def build_calculate_scores(original: Any) -> Any:
            @functools.wraps(original)
            def calculate_scores(*args: Any, **kwargs: Any) -> Any:
                pairs = _pair_product(
                    _argument(args, kwargs, 0, "references"),
                    _argument(args, kwargs, 1, "queries"),
                )
                started = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - started
                    self.full_matrix_seconds += elapsed
                    self.full_matrix_calls += 1
                    self.full_matrix_pairs += pairs or 0
                    self.score_seconds += elapsed
                    self.score_calls += 1

            return calculate_scores

        add_hook(
            "MassFlow.similarity.calculate_scores",
            similarity_module,
            "calculate_scores",
            build_calculate_scores,
        )

        def build_generate_decoys(original: Any) -> Any:
            @functools.wraps(original)
            def generate_decoys(*args: Any, **kwargs: Any) -> list[Any]:
                # Decoys generated by the library-level entropy diagnostic are
                # not search decoys: the whole diagnostic is timed as its own
                # stage, so counting them here would break the search breakdown.
                if self._in_entropy_diagnostic:
                    return original(*args, **kwargs)
                started = time.perf_counter()
                try:
                    decoys = original(*args, **kwargs)
                finally:
                    self.decoy_seconds += time.perf_counter() - started
                self.decoys_generated += len(decoys)
                return decoys

            return generate_decoys

        add_hook(
            "MassFlow.similarity.generate_decoys",
            similarity_module,
            "generate_decoys",
            build_generate_decoys,
        )

        # 5. FDR calibration (per-query target-decoy competition). The
        #    workflow imports this function inside its own body, so patching
        #    the module attribute covers that call site too.
        def build_calibrate_query_level_fdr(original: Any) -> Any:
            @functools.wraps(original)
            def calibrate_query_level_fdr(
                results: Sequence[Any],
            ) -> tuple[Any, Any, Any]:
                target_rows = sum(
                    1 for row in results if not row.get("is_decoy", False)
                )
                started = time.perf_counter()
                try:
                    q_by_query, p_by_query, summary = original(results)
                finally:
                    self.fdr_seconds += time.perf_counter() - started
                    self.fdr_calls += 1
                self.fdr_input_target_rows += target_rows
                self.fdr_input_decoy_rows += len(results) - target_rows
                self.fdr_summary = dict(summary)
                return q_by_query, p_by_query, summary

            return calibrate_query_level_fdr

        add_hook(
            "MassFlow.similarity.calibrate_query_level_fdr",
            similarity_module,
            "calibrate_query_level_fdr",
            build_calibrate_query_level_fdr,
        )

        # 6. Result export. The payload is post-FDR by the time it gets here,
        #    so these row counts are the retained (annotated) hits.
        def _record_export(
            results: Sequence[Any], output_path: Any, export_format: str
        ) -> None:
            self.export_calls += 1
            self.export_rows += len(results)
            self.export_target_rows += sum(
                1 for row in results if not row.get("is_decoy", False)
            )
            self.export_decoy_rows += sum(
                1 for row in results if row.get("is_decoy", False)
            )
            self.export_format = export_format
            if output_path is not None:
                path = Path(output_path)
                self.export_paths.append(str(path))
                try:
                    self.export_bytes += path.stat().st_size
                except OSError:
                    pass

        def build_save_match_results(original: Any) -> Any:
            @functools.wraps(original)
            def save_match_results(
                results: list[dict[str, Any]],
                output_path: Path,
                query_spectra: Optional[Any] = None,
            ) -> None:
                started = time.perf_counter()
                try:
                    original(results, output_path, query_spectra)
                finally:
                    self.export_seconds += time.perf_counter() - started
                    _record_export(results, output_path, "csv")

            return save_match_results

        add_hook(
            "MassFlow.io.save_match_results",
            io_module,
            "save_match_results",
            build_save_match_results,
        )

        def build_save_match_results_to_mztab(original: Any) -> Any:
            @functools.wraps(original)
            def save_match_results_to_mztab(
                results: list[dict[str, Any]],
                output_path: Path,
                query_spectra: Optional[Any] = None,
            ) -> None:
                started = time.perf_counter()
                try:
                    original(results, output_path, query_spectra)
                finally:
                    self.export_seconds += time.perf_counter() - started
                    _record_export(results, output_path, "mztab")

            return save_match_results_to_mztab

        add_hook(
            "MassFlow.io.save_match_results_to_mztab",
            io_module,
            "save_match_results_to_mztab",
            build_save_match_results_to_mztab,
        )

        # 7. Library-level target/decoy entropy diagnostic (streams the whole
        #    store once, before any query is scored).
        def build_emit_entropy_diagnostic(original: Any) -> Any:
            @functools.wraps(original)
            def emit_entropy_diagnostic(
                library_spec: Any, config: MassFlowConfig
            ) -> Any:
                self._context.append(INTERNAL_ROLE)
                self._in_entropy_diagnostic = True
                started = time.perf_counter()
                try:
                    return original(library_spec, config)
                finally:
                    self._in_entropy_diagnostic = False
                    self._context.pop()
                    self.library_entropy_diagnostic_seconds += (
                        time.perf_counter() - started
                    )

            return emit_entropy_diagnostic

        add_hook(
            "MassFlow.workflow._emit_entropy_diagnostic",
            workflow_module,
            "_emit_entropy_diagnostic",
            build_emit_entropy_diagnostic,
        )

        return hooks, skipped

    def install(self) -> ExitStack:
        """Patch every instrumented call site; close the stack to restore."""
        hooks, skipped = self._build_hooks()
        self.hooks_installed = [hook.label for hook in hooks]
        self.hooks_skipped = skipped
        stack = ExitStack()
        for hook in hooks:
            stack.enter_context(
                mock.patch.object(hook.target, hook.attribute, new=hook.wrapper)
            )
        return stack

    # ── derived views ───────────────────────────────────────────────────────
    @property
    def prefilter_seconds_total(self) -> float:
        """Total time spent in candidate-pair gates."""
        return sum(self.prefilter_seconds.values())

    @property
    def prefilter_possible_total(self) -> int:
        """Total pairs the candidate gates considered."""
        return sum(self.prefilter_possible_pairs.values())

    @property
    def prefilter_candidate_total(self) -> int:
        """Total pairs the candidate gates kept for scoring."""
        return sum(self.prefilter_candidate_pairs.values())

    @property
    def scored_pairs(self) -> int:
        """Pairs actually scored (sparse candidates plus full-matrix pairs)."""
        return self.full_matrix_pairs + self.prefilter_candidate_total

    @property
    def search_residual_seconds(self) -> float:
        """Search time not spent in decoys, prefilters, or scoring."""
        return max(
            self.search_seconds
            - self.decoy_seconds
            - self.prefilter_seconds_total
            - self.score_seconds,
            0.0,
        )

    @property
    def exclusive_stage_seconds(self) -> float:
        """Sum of the mutually exclusive stages (search counted once)."""
        return (
            self.library_preparation_seconds
            + self.library_entropy_diagnostic_seconds
            + self.load_seconds[QUERY_ROLE]
            + self.process_seconds[QUERY_ROLE]
            + self.search_seconds
            + self.fdr_seconds
            + self.export_seconds
        )

    def library_store_action(self) -> str:
        """Whether the normalized library store was built or reused.

        Derived from instrumentation rather than log text: a raw library file
        is only read by ``load_spectra``/``process_spectra`` when the store had
        to be built, so a nonzero library load count means the run built it.
        """
        if self.loaded_spectra[LIBRARY_ROLE] > 0:
            return "built"
        source_suffix = self._library_path.suffix.lower() if self._library_path else ""
        if source_suffix in _STORE_SUFFIXES:
            return "prebuilt_store"
        return "reused_cached_store"

    # ── report record ───────────────────────────────────────────────────────
    def build_record(
        self,
        wall_seconds: float,
        peak_rss_mb: float,
        traced_heap: dict[str, float],
        file_result: Any,
        status: str,
        failures: Sequence[str],
        warnings: Sequence[str],
    ) -> dict[str, Any]:
        """Assemble the JSON-safe stage/count/memory record for one run."""
        query_input = self.process_input_spectra[QUERY_ROLE]
        query_retained = self.process_retained_spectra[QUERY_ROLE]
        query_loaded = self.loaded_spectra[QUERY_ROLE]
        query_peaks_input = self.process_input_peaks[QUERY_ROLE]
        query_peaks_retained = self.process_retained_peaks[QUERY_ROLE]
        library_retained = (
            self.library_size or self.process_retained_spectra[LIBRARY_ROLE]
        )

        stages: dict[str, float] = {
            "library_preparation": round(self.library_preparation_seconds, 6),
            "library_entropy_diagnostic": round(
                self.library_entropy_diagnostic_seconds, 6
            ),
            "query_load": round(self.load_seconds[QUERY_ROLE], 6),
            "query_processing": round(self.process_seconds[QUERY_ROLE], 6),
            "similarity_search": round(self.search_seconds, 6),
            "search_decoy_generation": round(self.decoy_seconds, 6),
            "search_candidate_prefilter": round(self.prefilter_seconds_total, 6),
            "search_score_matrix": round(self.score_seconds, 6),
            "search_result_extraction_derived": round(self.search_residual_seconds, 6),
            "fdr_calibration": round(self.fdr_seconds, 6),
            "result_export": round(self.export_seconds, 6),
            "pipeline_wall_clock": round(wall_seconds, 6),
            "exclusive_stages_total": round(self.exclusive_stage_seconds, 6),
            "unaccounted_derived": round(
                max(wall_seconds - self.exclusive_stage_seconds, 0.0), 6
            ),
        }

        throughput: dict[str, dict[str, Any]] = {
            "query_load": _throughput(
                query_loaded, self.load_seconds[QUERY_ROLE], "raw spectra/s"
            ),
            "query_processing": _throughput(
                query_input, self.process_seconds[QUERY_ROLE], "input spectra/s"
            ),
            "similarity_search": _throughput(
                self.query_spectra_searched or query_retained,
                self.search_seconds,
                "query spectra/s",
            ),
            "search_candidate_prefilter": _throughput(
                self.prefilter_possible_total,
                self.prefilter_seconds_total,
                "pairs/s",
            ),
            "search_score_matrix": _throughput(
                self.scored_pairs, self.score_seconds, "pairs/s"
            ),
            "fdr_calibration": _throughput(
                self.fdr_input_target_rows + self.fdr_input_decoy_rows,
                self.fdr_seconds,
                "judged rows/s",
            ),
            "result_export": _throughput(
                self.export_rows, self.export_seconds, "rows/s"
            ),
            "end_to_end": _throughput(
                query_retained, wall_seconds, "retained query spectra/s"
            ),
        }

        prefilter_sources: list[dict[str, Any]] = sorted(
            (
                {
                    "source": source,
                    "seconds": round(seconds, 6),
                    "calls": self.prefilter_calls.get(source, 0),
                    "possible_pairs": self.prefilter_possible_pairs.get(source, 0),
                    "candidate_pairs": self.prefilter_candidate_pairs.get(source, 0),
                    "prefiltered_out_pairs": max(
                        self.prefilter_possible_pairs.get(source, 0)
                        - self.prefilter_candidate_pairs.get(source, 0),
                        0,
                    ),
                }
                for source, seconds in self.prefilter_seconds.items()
            ),
            key=lambda item: str(item["source"]),
        )

        record: dict[str, Any] = {
            "status": status,
            "failures": list(failures),
            "warnings": list(warnings),
            "stages_seconds": stages,
            "throughput": throughput,
            "memory": {
                "peak_rss_mb": round(peak_rss_mb, 2),
                "traced_heap_start_mb": round(traced_heap["start_mb"], 2),
                "traced_heap_peak_mb": round(traced_heap["peak_mb"], 2),
                "traced_heap_end_mb": round(traced_heap["end_mb"], 2),
                "tracemalloc_enabled": True,
                "rss_is_process_high_water": True,
            },
            "spectra": {
                QUERY_ROLE: {
                    "loaded_raw": query_loaded,
                    "process_input": query_input,
                    "process_retained": query_retained,
                    "dropped_before_processing_derived": max(
                        query_loaded - query_input, 0
                    ),
                    "dropped_during_processing": max(query_input - query_retained, 0),
                    "load_calls": self.load_calls[QUERY_ROLE],
                    "process_batches": self.process_batches[QUERY_ROLE],
                    "process_spectra_calls": self.process_spectra_calls[QUERY_ROLE],
                    "retention_ratio": (
                        round(query_retained / query_input, 6) if query_input else None
                    ),
                    "query_limit": self._query_limit,
                },
                LIBRARY_ROLE: {
                    "loaded_raw": self.loaded_spectra[LIBRARY_ROLE],
                    "process_input": self.process_input_spectra[LIBRARY_ROLE],
                    "process_retained": self.process_retained_spectra[LIBRARY_ROLE],
                    "size_used_as_targets": library_retained,
                    "load_calls": self.load_calls[LIBRARY_ROLE],
                    "process_batches": self.process_batches[LIBRARY_ROLE],
                    "store_kind": self.library_store_kind,
                    "store_path": self.library_store_path,
                    "store_action": self.library_store_action(),
                },
                INTERNAL_ROLE: {
                    "loaded_raw": self.loaded_spectra[INTERNAL_ROLE],
                    "process_input": self.process_input_spectra[INTERNAL_ROLE],
                    "process_retained": self.process_retained_spectra[INTERNAL_ROLE],
                    "load_calls": self.load_calls[INTERNAL_ROLE],
                    "process_batches": self.process_batches[INTERNAL_ROLE],
                },
            },
            "peaks": {
                QUERY_ROLE: {
                    "process_input": query_peaks_input,
                    "process_retained": query_peaks_retained,
                    "dropped": max(query_peaks_input - query_peaks_retained, 0),
                    "retention_ratio": (
                        round(query_peaks_retained / query_peaks_input, 6)
                        if query_peaks_input
                        else None
                    ),
                },
                LIBRARY_ROLE: {
                    "process_input": self.process_input_peaks[LIBRARY_ROLE],
                    "process_retained": self.process_retained_peaks[LIBRARY_ROLE],
                    "retention_ratio": (
                        round(
                            self.process_retained_peaks[LIBRARY_ROLE]
                            / self.process_input_peaks[LIBRARY_ROLE],
                            6,
                        )
                        if self.process_input_peaks[LIBRARY_ROLE]
                        else None
                    ),
                },
            },
            "candidates": {
                "possible_pairs": self.prefilter_possible_total,
                "candidate_pairs": self.prefilter_candidate_total,
                "prefiltered_out_pairs": max(
                    self.prefilter_possible_total - self.prefilter_candidate_total, 0
                ),
                "prefilter_reduction_ratio": (
                    round(
                        self.prefilter_candidate_total / self.prefilter_possible_total,
                        8,
                    )
                    if self.prefilter_possible_total
                    else None
                ),
                "sources": prefilter_sources,
                "full_matrix_calls": self.full_matrix_calls,
                "full_matrix_pairs": self.full_matrix_pairs,
                "scored_pairs_derived": self.scored_pairs,
                "score_calls": self.score_calls,
                "search_calls": self.search_calls,
                "query_spectra_searched": self.query_spectra_searched,
                "decoys_generated": self.decoys_generated,
                "result_rows_pre_fdr": self.result_rows_pre_fdr,
                "target_rows_pre_fdr": max(
                    self.result_rows_pre_fdr - self.decoy_result_rows, 0
                ),
                "decoy_rows_pre_fdr": self.decoy_result_rows,
            },
            "fdr": {
                "calls": self.fdr_calls,
                "input_target_rows": self.fdr_input_target_rows,
                "input_decoy_rows": self.fdr_input_decoy_rows,
                "summary": dict(self.fdr_summary),
            },
            "export": {
                "calls": self.export_calls,
                "format": self.export_format,
                "rows": self.export_rows,
                "retained_target_rows": self.export_target_rows,
                "retained_decoy_rows": self.export_decoy_rows,
                "bytes": self.export_bytes,
                "paths": list(self.export_paths),
            },
        }

        if file_result is not None:
            record["file_result"] = {
                "status": file_result.status,
                "input_path": str(file_result.input_path),
                "output_path": (
                    str(file_result.output_path)
                    if file_result.output_path is not None
                    else None
                ),
                "spectra_loaded": int(file_result.spectra_loaded),
                "spectra_rejected": int(file_result.spectra_rejected),
                "hits_produced": int(file_result.hits_produced),
                "warnings": list(file_result.warnings),
                "degraded_mode_flags": list(file_result.degraded_mode_flags),
                "fatal_errors": list(file_result.fatal_errors),
            }
        else:
            record["file_result"] = None

        return record


# ── command line ────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="benchmark_annotation.py",
        description=(
            "Measure baseline stage timings, memory, and throughput of the "
            "MassFlow annotation pipeline. Core MassFlow code is not modified: "
            "all instrumentation is installed dynamically and removed again."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=Path,
        required=True,
        help="Experimental query file (.mzML, .mzXML, .mgf, .msp). Single file.",
    )
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Reference library (.msp, .mgf, .mzML, .db, .sqlite, .zarr).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional MassFlow YAML configuration. Its processing/similarity/"
            "export settings are used as-is; --query and --library override "
            "its input paths."
        ),
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Load at most this many query spectra (for quick test runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the JSON report and the pipeline artifacts.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject unusable invocations before any expensive work runs."""
    if args.limit_queries is not None and args.limit_queries < 1:
        parser.error("--limit-queries must be >= 1")
    query_path = Path(args.query).expanduser()
    if not query_path.is_file():
        parser.error(f"--query is not a file: {query_path}")
    if not _is_loadable_query(query_path):
        parser.error(
            "--query must be .mzML, .mzXML, or .mgf "
            f"(got {query_path.suffix or 'no extension'})"
        )
    library_path = Path(args.library).expanduser()
    if not library_path.is_file() and library_path.suffix.lower() != ".zarr":
        parser.error(f"--library is not a file: {library_path}")
    if args.config is not None and not Path(args.config).expanduser().is_file():
        parser.error(f"--config is not a file: {args.config}")


def _is_loadable_query(path: Path) -> bool:
    """Return True when ``path`` has a supported query extension."""
    return path.suffix.lower() in {".mzml", ".mzxml", ".mgf"}


def _build_config(args: argparse.Namespace, artifacts_dir: Path) -> MassFlowConfig:
    """Build the effective config from the optional YAML plus CLI paths."""
    config_path: Optional[Path] = None
    if args.config is not None:
        config_path = Path(args.config).expanduser().resolve()
        config = MassFlowConfig.from_yaml(config_path)
    else:
        config = MassFlowConfig(
            project=ProjectConfig(name="MassFlow_Benchmark"),
            input=InputConfig(
                input_path=Path(args.query),
                library_path=Path(args.library),
            ),
            processing=ProcessingConfig(),
            similarity=SimilarityConfig(),
        )

    # The CLI paths are authoritative in both modes: the report describes
    # exactly the files that a baseline measurement should be reproducible on.
    config.input.input_path = Path(args.query).expanduser().resolve()
    config.input.library_path = Path(args.library).expanduser().resolve()
    config.project.output_directory = artifacts_dir
    if config_path is not None:
        config.config_path = config_path
    return config


def _environment_info(cli_args: Sequence[str]) -> dict[str, Any]:
    """Collect the environment metadata attached to every report."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cli_args": list(cli_args),
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_sha": _git_sha(),
        "packages": {
            "massflow": _distribution_version("massflow"),
            "matchms": _distribution_version("matchms"),
            "numpy": _distribution_version("numpy"),
            "polars": _distribution_version("polars"),
            "numba": _distribution_version("numba"),
        },
        "capabilities": {
            "numba_prefilter": bool(getattr(acceleration_module, "_HAS_NUMBA", False)),
        },
    }


def _dataset_info(config: MassFlowConfig, query_limit: Optional[int]) -> dict[str, Any]:
    """Describe the benchmark inputs (identity, size, hashes)."""
    query_path = Path(config.input.input_path)
    dataset: dict[str, Any] = {
        "query_path": str(query_path),
        "query_size_bytes": query_path.stat().st_size,
        "query_mtime_ns": query_path.stat().st_mtime_ns,
        "query_sha256": _sha256_file(query_path),
        "query_spectra_limit": query_limit,
    }
    if config.input.library_path is not None:
        library_path = Path(config.input.library_path)
        dataset.update(
            {
                "library_path": str(library_path),
                "library_size_bytes": library_path.stat().st_size,
                "library_mtime_ns": library_path.stat().st_mtime_ns,
                "library_sha256": _sha256_file(library_path),
                "library_suffix": library_path.suffix.lower(),
            }
        )
    return dataset


# ── run + report ────────────────────────────────────────────────────────────
def _empty_record(status: str, message: str) -> dict[str, Any]:
    """Minimal record used when the pipeline could not be measured at all."""
    return {
        "status": status,
        "failures": [message],
        "warnings": [],
        "stages_seconds": {},
        "throughput": {},
        "memory": {},
        "spectra": {},
        "peaks": {},
        "candidates": {},
        "fdr": {},
        "export": {},
        "file_result": None,
    }


def _run_pipeline(
    config: MassFlowConfig,
    recorder: _Recorder,
) -> dict[str, Any]:
    """Run the pipeline once under instrumentation and return its record.

    Failures (the run raised, or the workflow reported a failed file) are kept
    separate from warnings (degraded machinery, small-library caveats) so a
    successfully measured run is never labelled as an error.
    """
    recorder.reset()
    failures: list[str] = []
    warnings: list[str] = []
    file_result: Any = None

    tracemalloc.start()
    traced_heap_start = tracemalloc.get_traced_memory()[0]
    started = time.perf_counter()
    try:
        execution_results = workflow_module.run_annotation_pipeline(
            config, config_path=config.config_path
        )
        if execution_results:
            file_result = execution_results[0]
        else:
            failures.append("The pipeline returned no execution results.")
    except Exception as exc:  # benchmark boundary: report, never crash
        failures.append(f"{type(exc).__name__}: {exc}")
    wall_seconds = time.perf_counter() - started
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if file_result is not None:
        if file_result.status == "failed":
            failures.extend(f"{message}" for message in file_result.fatal_errors)
        warnings.extend(str(message) for message in file_result.warnings)
        warnings.extend(f"degraded: {flag}" for flag in file_result.degraded_mode_flags)
        if file_result.output_path is None and file_result.status != "failed":
            warnings.append("no result file was written")

    status = "ok" if not failures else "error"
    return recorder.build_record(
        wall_seconds=wall_seconds,
        peak_rss_mb=_peak_rss_mb(),
        traced_heap={
            "start_mb": traced_heap_start / (1024 * 1024),
            "peak_mb": traced_peak / (1024 * 1024),
            "end_mb": traced_current / (1024 * 1024),
        },
        file_result=file_result,
        status=status,
        failures=failures,
        warnings=warnings,
    )


class _TableRow(NamedTuple):
    """One line of the terminal summary table."""

    label: str
    seconds: float
    throughput: Optional[float]
    unit: str
    substage: bool


def _table_rows(record: dict[str, Any]) -> list[_TableRow]:
    """Build the summary rows from a run record (search sub-stages nested)."""
    stages = record["stages_seconds"]
    throughput = record["throughput"]

    def rate(key: str) -> Optional[float]:
        return throughput.get(key, {}).get("value")

    def unit(key: str) -> str:
        return str(throughput.get(key, {}).get("unit", "—"))

    def row(key: str, label: str, substage: bool = False) -> _TableRow:
        return _TableRow(
            label=label,
            seconds=float(stages[key]),
            throughput=rate(key),
            unit=unit(key),
            substage=substage,
        )

    def plain(key: str) -> _TableRow:
        return _TableRow(key, float(stages[key]), None, "—", False)

    return [
        plain("library_preparation"),
        plain("library_entropy_diagnostic"),
        row("query_load", "query_load"),
        row("query_processing", "query_processing"),
        row("similarity_search", "similarity_search"),
        row("search_decoy_generation", "decoy_generation", substage=True),
        row("search_candidate_prefilter", "candidate_prefilter", substage=True),
        row("search_score_matrix", "score_matrix", substage=True),
        row(
            "search_result_extraction_derived",
            "result_extraction (derived)",
            substage=True,
        ),
        row("fdr_calibration", "fdr_calibration"),
        row("result_export", "result_export"),
        plain("unaccounted_derived"),
    ]


def _print_summary(report: dict[str, Any]) -> None:
    """Print the stage table, the memory block, and the headline counts."""
    record = report["run"]
    if not record.get("stages_seconds"):
        print()
        print(f"MassFlow baseline benchmark — {SCHEMA_VERSION}")
        print("  no measured run: the pipeline did not complete")
        for message in record.get("failures", []):
            print(f"  ! FAILURE: {message}")
        print(f"  report: {report['report_path']}")
        print()
        return
    stages = record["stages_seconds"]
    wall_seconds = float(stages["pipeline_wall_clock"])

    label_width = 30
    print()
    print(f"MassFlow baseline benchmark — {SCHEMA_VERSION}")
    print(f"  query   : {report['dataset']['query_path']}")
    print(f"  library : {report['dataset'].get('library_path', 'n/a')}")
    print(
        "  config  : algorithm={algorithm}, min_score={min_score}, "
        "fdr_threshold={fdr_threshold}, export={fmt}".format(
            algorithm=report["effective_config"]["similarity"]["algorithm"],
            min_score=report["effective_config"]["similarity"]["min_score"],
            fdr_threshold=report["effective_config"]["similarity"]["fdr_threshold"],
            fmt=report["effective_config"]["export"]["format"],
        )
    )
    print()
    separator = "─" * (label_width + 42)
    print(
        f"{'STAGE':<{label_width}}{'SECONDS':>11}{'% WALL':>9}{'THROUGHPUT':>15}   UNIT"
    )
    print(separator)
    rows = _table_rows(record)
    substages = [row for row in rows if row.substage]
    for row in rows:
        if row.substage:
            prefix = "  └─ " if row is substages[-1] else "  ├─ "
        else:
            prefix = ""
        label = f"{prefix}{row.label}"
        share = (row.seconds / wall_seconds) if wall_seconds > 0 else 0.0
        rate = f"{row.throughput:,.1f}" if row.throughput is not None else "—"
        print(
            f"{label:<{label_width}}{row.seconds:>11.4f}{share:>9.1%}"
            f"{rate:>15}   {row.unit}"
        )
    print(separator)
    end_to_end = record["throughput"]["end_to_end"]
    end_to_end_rate = (
        f"{end_to_end['value']:,.1f}" if end_to_end["value"] is not None else "—"
    )
    print(
        f"{'TOTAL (wall clock)':<{label_width}}{wall_seconds:>11.4f}{1.0:>9.1%}"
        f"{end_to_end_rate:>15}   {end_to_end['unit']}"
    )
    search_share = (
        float(stages["similarity_search"]) / wall_seconds if wall_seconds > 0 else 0.0
    )
    print(
        f"  (search sub-stages are a breakdown of similarity_search = "
        f"{search_share:.1%} of wall clock; they are not additive)"
    )

    memory = record["memory"]
    print()
    print("MEMORY")
    print(f"  peak RSS (process high-water)        {memory['peak_rss_mb']:>10,.2f} MB")
    print(
        f"  peak traced Python heap              {memory['traced_heap_peak_mb']:>10,.2f} MB"
    )
    print(
        f"  traced heap start -> end             {memory['traced_heap_start_mb']:>10,.2f} MB -> "
        f"{memory['traced_heap_end_mb']:,.2f} MB"
    )

    query = record["spectra"][QUERY_ROLE]
    library = record["spectra"][LIBRARY_ROLE]
    peaks = record["peaks"][QUERY_ROLE]
    candidates = record["candidates"]
    fdr = record["fdr"]
    export = record["export"]
    print()
    print("COUNTS")
    print(
        f"  query spectra : {query['loaded_raw']} loaded -> "
        f"{query['process_input']} processed -> {query['process_retained']} retained "
        f"({query['dropped_during_processing']} dropped by processing)"
    )
    print(
        f"  query peaks   : {peaks['process_input']} -> {peaks['process_retained']} "
        f"retained"
        + (
            f" ({peaks['retention_ratio']:.1%})"
            if peaks["retention_ratio"] is not None
            else ""
        )
    )
    store_detail = f", kind={library['store_kind']}" if library["store_kind"] else ""
    print(
        f"  library       : {library['size_used_as_targets']} targets "
        f"({library['store_action']}{store_detail})"
    )
    if candidates["possible_pairs"]:
        print(
            f"  candidates    : {candidates['candidate_pairs']:,} kept / "
            f"{candidates['possible_pairs']:,} possible "
            f"({candidates['prefilter_reduction_ratio']:.2%}), "
            f"{candidates['prefiltered_out_pairs']:,} prefiltered out"
        )
    else:
        print(
            f"  candidates    : no candidate gate observed "
            f"({candidates['full_matrix_pairs']:,} full-matrix pairs scored)"
        )
    print(
        f"  hits          : {fdr['input_target_rows']} target / "
        f"{fdr['input_decoy_rows']} decoy rows judged -> "
        f"{export['retained_target_rows']} target rows retained at "
        f"q <= {report['effective_config']['similarity']['fdr_threshold']}"
    )
    file_result = record["file_result"]
    if file_result is not None:
        print(
            f"  file result   : {file_result['status']} "
            f"(loaded={file_result['spectra_loaded']}, "
            f"rejected={file_result['spectra_rejected']}, "
            f"hits={file_result['hits_produced']})"
        )
    if record["failures"] or record["warnings"]:
        print()
        print("NOTES")
        for message in record["failures"]:
            print(f"  ! FAILURE: {message}")
        for message in record["warnings"]:
            print(f"  - {message}")
    print()
    print(f"  report    : {report['report_path']}")
    print(f"  artifacts : {report['results_directory']}")
    for path in export["paths"]:
        print(f"  exported  : {path}")
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: parse arguments, run the benchmark, write the report."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    _validate_args(parser, args)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    timestamp = _utc_timestamp()
    artifacts_dir = output_dir / "artifacts" / timestamp
    try:
        config = _build_config(args, artifacts_dir)
    except Exception as exc:
        print(f"error: unusable configuration: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"baseline_{timestamp}.json"

    recorder = _Recorder(
        query_path=config.input.input_path,
        library_path=config.input.library_path,
        query_limit=args.limit_queries,
    )

    record = _empty_record("error", "the pipeline never ran")
    hooks_removed = False
    exit_code: Optional[int] = None
    try:
        with recorder.install():
            record = _run_pipeline(config, recorder)
        hooks_removed = True
    except KeyboardInterrupt:
        record = _empty_record("error", "interrupted by the user")
        exit_code = 130
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        record = _empty_record("error", f"{type(exc).__name__}: {exc}")
    if exit_code is None:
        exit_code = 0 if record["status"] == "ok" else 1

    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path),
        "environment": _environment_info(cli_args),
        "cli": {
            "query": str(config.input.input_path),
            "library": str(config.input.library_path),
            "config": str(args.config) if args.config is not None else None,
            "limit_queries": args.limit_queries,
            "output_dir": str(output_dir),
        },
        "dataset": _dataset_info(config, args.limit_queries),
        "effective_config": {
            "processing": config.processing.model_dump(mode="json"),
            "similarity": config.similarity.model_dump(mode="json"),
            "export": config.export.model_dump(mode="json"),
        },
        "results_directory": str(config.output_directory),
        "run": record,
        "instrumentation": {
            "method": (
                "unittest.mock.patch.object wrappers with functools.wraps, "
                "installed before the run and removed through an ExitStack; "
                "no file under src/MassFlow/ is modified"
            ),
            "hooks_installed": list(recorder.hooks_installed),
            "hooks_skipped": list(recorder.hooks_skipped),
            "hooks_removed": hooks_removed,
        },
        "limitations": list(_LIMITATIONS),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))

    try:
        _print_summary(report)
    except Exception as exc:  # a broken table must not hide a good report
        print(f"warning: could not print the summary table ({exc})", file=sys.stderr)
        print(f"report: {report_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
