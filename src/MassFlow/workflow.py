"""
High-level orchestration for MassFlow annotation runs.

This module coordinates the end-to-end execution path used by the CLI: loading
and validating the reference library, discovering experimental inputs,
processing spectra, dispatching per-file similarity searches across worker
processes, and exporting result tables. It is the integration layer that turns
the config, I/O, processing, and similarity modules into a reproducible
pipeline.

The workflow is designed to stay memory-aware. Worker processes build their own
similarity engines, reference libraries are accessed through a worker-owned
backend (see :mod:`MassFlow.library`) and searched in chunks rather than as a
single monolithic matrix, and false discovery rate filtering is applied after
aggregating all chunk results for each experimental file. Only a compact
``LibrarySpec`` crosses the process boundary — never the spectral payload.
"""

import hashlib
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

from matchms import Spectrum

from MassFlow import io, processing
from MassFlow.config import MassFlowConfig, SimilarityConfig
from MassFlow.library import LibrarySpec, open_library, prepare_library
from MassFlow.storage import SpectralStore
from MassFlow.similarity import (
    MLRouter,
    SearchResult,
    SimilarityEngine,
    _MLEngineBase,
    get_similarity_engine,
)

from MassFlow.protocols import MLEngineProtocol

logger = logging.getLogger(__name__)


@dataclass
class FileExecutionResult:
    """Structured per-file execution outcome (failure-model contract).

    Every experimental input file processed by the workflow produces exactly
    one of these. A scientific analysis must never silently succeed when
    required data were not processed: a file whose data could not be loaded,
    validated, or scored is reported as ``failed`` (never as an empty
    success), and a file that was processed with degraded machinery is
    reported as ``degraded`` with the degradation made explicit.

    Attributes
    ----------
    status : Literal["success", "degraded", "failed"]
        ``success`` — the file was fully processed with the configured
        pipeline. ``degraded`` — results were produced, but some part of the
        configured pipeline fell back (engine fallback, uncalibrated FDR,
        ...) and the degradation is listed in ``degraded_mode_flags``.
        ``failed`` — the file could not be processed; ``fatal_errors``
        explains why and no results CSV is written.
    input_path : Path
        The experimental input file.
    spectra_loaded : int
        Spectra that passed the I/O validation layer and entered processing.
    spectra_rejected : int
        Spectra dropped by validation or processing (recoverable
        spectrum-level issues, reported explicitly rather than silently).
    hits_produced : int
        Number of annotation hits exported for the file.
    output_path : Path or None
        Path of the written results file (``None`` for failed files).
    warnings : list[str]
        Human-readable non-fatal warnings (e.g. small-library FDR caveat).
    fatal_errors : list[str]
        Human-readable fatal errors; non-empty iff ``status == "failed"``.
    degraded_mode_flags : list[str]
        Machine-readable degradation markers (``engine_fallback:<algo>``,
        ``consensus_*``, ``cascade_*``, ``routing_*``, ``fdr_uncalibrated``).
    """

    status: Literal["success", "degraded", "failed"]
    input_path: Path
    spectra_loaded: int = 0
    spectra_rejected: int = 0
    hits_produced: int = 0
    output_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)
    degraded_mode_flags: list[str] = field(default_factory=list)

    # Internal payload for the exporter (not part of the failure contract).
    query_spectra: List[Spectrum] = field(default_factory=list, repr=False)
    results: List[SearchResult] = field(default_factory=list, repr=False)
    fdr_summary: dict[str, int] | None = field(default=None, repr=False)


_worker_engine: SimilarityEngine | _MLEngineBase | MLEngineProtocol | None = None
_worker_router: MLRouter | None = None

# Worker-owned library backend (see MassFlow.library): opened once per worker
# from the compact LibrarySpec; spectra are streamed in bounded chunks per
# file. The full spectral payload is NEVER passed between processes.
_worker_backend: SpectralStore | None = None
_worker_library_spec: LibrarySpec | None = None

# Classical fallback engine for the ML API boundary: when the configured
# engine (remote ML endpoint or locally-missing heavy dependencies) fails,
# the worker retries with modified_cosine so the run never crashes.
_worker_fallback_engine: SimilarityEngine | None = None


class _RejectionCollector:
    """Collects spectrum-level rejection reasons from the I/O validation layer.

    Recoverable spectrum-level issues (missing precursor, empty peaks, ...)
    must be observable in the per-file execution result instead of existing
    only as quarantine log lines. The collector keeps a bounded list of
    reasons (first occurrences) plus a total count.
    """

    def __init__(self) -> None:
        self.count: int = 0
        self.reasons: list[str] = []

    def __call__(self, reason: str) -> None:
        self.count += 1
        if len(self.reasons) < 10 and reason not in self.reasons:
            self.reasons.append(reason)


class _CountingIterator:
    """Iterator wrapper that counts consumed items.

    Used by the streaming-library path so the true processed reference-
    library size is known when the per-query TDC block runs, even though
    the library is consumed lazily (and never materialized) by the engine.
    """

    def __init__(self, iterator: Any) -> None:
        self._iterator = iter(iterator)
        self.count: int = 0

    def __iter__(self) -> "_CountingIterator":
        return self

    def __next__(self) -> Any:
        item = next(self._iterator)
        self.count += 1
        return item


# Engines outside the stable product contract (docs/CAPABILITY_MATRIX.md).
EXPERIMENTAL_ENGINES = ("spec2vec", "ms2deepscore", "consensus", "cascade")


def experimental_surface_flags(config: MassFlowConfig) -> list[str]:
    """Return the active experimental surfaces of a configuration.

    The stable product contract covers cosine / modified_cosine scoring,
    SQLite libraries, and CSV/mzTab-M exports. Everything else — meta-
    engines (consensus, cascade), ML engines (spec2vec, ms2deepscore), ML
    routing, HNSW candidate retrieval, and remote ML endpoints — is
    experimental and must be visibly flagged at the run boundary so an
    experimental run is never mistaken for a stable-product run.

    Parameters
    ----------
    config : MassFlowConfig
        The validated pipeline configuration.

    Returns
    -------
    list of str
        Stable, order-independent flag strings; empty for a pure stable
        configuration.
    """
    flags: list[str] = []
    algorithm = config.similarity.algorithm
    if algorithm in EXPERIMENTAL_ENGINES:
        flags.append(f"experimental_engine:{algorithm}")
    if config.similarity.enable_routing:
        flags.append("experimental_routing")
    if config.similarity.hnsw_enabled:
        flags.append("experimental_hnsw")
    if config.similarity.ml_endpoints:
        flags.append("experimental_remote_ml")
    return flags


def _emit_small_library_warning(lib_size: int, fdr_threshold: float) -> None:
    """Emit a critical scientific warning about small-library FDR invalidity.

    In an interactive terminal the warning is rendered as a visually distinct
    Rich Panel.  When output is piped or redirected a clean single log line is
    emitted instead.  Only one rendering path is taken — never both.
    """
    msg = (
        f"CRITICAL SCIENTIFIC WARNING: SMALL LIBRARY DETECTED\n"
        f"The library contains only {lib_size} spectra.\n"
        f"Target-Decoy False Discovery Rate (FDR) statistics are fundamentally\n"
        f"invalid on small sample sizes because the decoy null-distribution\n"
        f"will be too sparse. A strict FDR threshold (currently set to\n"
        f"{fdr_threshold}) will likely eliminate all true and putative matches\n"
        f"as false positives.\n"
        f"\n"
        f"Recommendation:\n"
        f"1. Use a comprehensive library (e.g., GNPS, MoNA, NIST) for FDR\n"
        f"   validation.\n"
        f"2. Or, if using a small specialized library, relax the\n"
        f"   `fdr_threshold` (e.g., 0.1 or 1.0) in your config to evaluate\n"
        f"   raw Cosine scores directly."
    )

    if sys.stderr.isatty():
        # Interactive: show a styled Rich Panel only.
        from rich.console import Console
        from rich.panel import Panel

        console = Console(stderr=True)
        console.print(
            Panel(
                msg,
                title="[bold yellow]Warning[/bold yellow]",
                border_style="red",
                highlight=True,
            )
        )
    else:
        # Piped / redirected: emit a clean structured log line.
        logger.warning(msg)


def _init_worker(
    config: MassFlowConfig,
    library_spec: LibrarySpec | None,
) -> None:
    """
    Initialize a worker-local similarity engine and open the library backend.

    Each subprocess instantiates its own engine from the shared configuration
    so large model state is not serialized. The reference library is NOT
    passed as data: the worker receives only the compact :class:`LibrarySpec`
    and opens the SQLite/Zarr store itself (see :mod:`MassFlow.library`), so
    worker count never multiplies a serialized Python library. Spectra are
    streamed from the store in bounded chunks during each file's search.
    """
    from MassFlow.log_config import setup_structured_logging

    setup_structured_logging(level=logging.INFO, force_json=True)

    global _worker_engine, _worker_router, _worker_fallback_engine
    global _worker_backend, _worker_library_spec

    _worker_engine = get_similarity_engine(config.similarity)
    _worker_router = (
        MLRouter(config.similarity) if config.similarity.enable_routing else None
    )
    _worker_fallback_engine = _build_classical_fallback_engine(config.similarity)

    _worker_library_spec = library_spec
    if library_spec is not None:
        _worker_backend = open_library(library_spec, config.processing)
    else:
        _worker_backend = None


def _build_classical_fallback_engine(
    similarity_config: SimilarityConfig,
) -> SimilarityEngine:
    """Build the modified_cosine fallback engine for the ML API boundary.

    Used when the configured engine fails at search time — a remote ML
    endpoint that is unreachable (or whose circuit breaker has opened) or a
    local environment missing the heavy dependencies (PyTorch, Gensim).
    Empirical p-value scoring is applied on top of these results by the
    normal FDR block in ``_process_single_file``.

    Parameters
    ----------
    similarity_config : SimilarityConfig
        The active similarity configuration (tolerances are inherited).

    Returns
    -------
    SimilarityEngine
        A classical modified_cosine engine.
    """
    fallback_config = SimilarityConfig(
        algorithm="modified_cosine",
        ms1_tolerance=similarity_config.ms1_tolerance,
        ms2_tolerance=similarity_config.ms2_tolerance,
        resolution_ppm=similarity_config.resolution_ppm,
        min_score=similarity_config.min_score,
        min_matched_peaks=similarity_config.min_matched_peaks,
        rt_tolerance=similarity_config.rt_tolerance,
    )
    return SimilarityEngine(fallback_config)


def _process_single_file(
    query_file: Path,
    config: MassFlowConfig,
    library_size: int | None = None,
    library_spec: LibrarySpec | None = None,
) -> FileExecutionResult:
    """
    Process one experimental file against the configured reference library.

    The worker loads and processes the query spectra, ensures query IDs are
    stable, streams the reference library from the worker-owned backend in
    bounded chunks (the engine generates deterministic decoys per chunk), runs
    similarity search on each chunk, and then applies FDR filtering across the
    aggregated results for that one file.

    When ``config.similarity.enable_routing`` is True, each query spectrum is
    classified by :class:`MLRouter` as "easy" or "hard" based on its
    :class:`~MassFlow.models.TriageProfile`. Easy queries are scored by a fast
    classical engine (e.g. modified cosine); hard queries are dispatched to an
    ML consensus engine. All results are pooled for a unified per-query
    target-decoy FDR calculation (see :func:`MassFlow.similarity.calculate_fdr`
    for the competition-unit contract).

    If the ML engine fails or times out, the hard batch automatically falls
    back to a classical engine without aborting the file.

    Parameters
    ----------
    query_file : Path
        Experimental spectral file to process.
    config : MassFlowConfig
        Full pipeline configuration used for loading, processing, and scoring.
    library_size : int or None
        The true processed reference-library size (targets only, decoys
        excluded), provided by the parent's :func:`prepare_library` call.
        When None, the size is derived from the backend store count or the
        streamed library iterator.
    library_spec : LibrarySpec or None
        Compact worker-openable library description. When None (direct API
        call without a prepared store), the configured raw library file is
        streamed via a file backend.

    Returns
    -------
    FileExecutionResult
        The structured per-file outcome: status (``success`` / ``degraded`` /
        ``failed``), spectrum counts, hits produced, warnings, fatal errors,
        and degraded-mode flags. A file whose data could not be loaded,
        validated, or scored is reported as ``failed`` with explicit
        ``fatal_errors`` — never as an empty success. The processed query
        spectra and filtered results are carried on the result for export.
    """
    result = FileExecutionResult(status="success", input_path=query_file)
    try:
        # ------------------------------------------------------------------
        # 1. Load, validate, and process query spectra
        # ------------------------------------------------------------------
        # The I/O validation layer drops malformed spectra; each rejection is
        # collected so spectrum-level issues are observable and recorded in
        # provenance instead of vanishing into the quarantine log alone.
        rejection_collector = _RejectionCollector()
        query_gen = io.load_spectra(
            query_file,
            file_format=config.input.format,
            rejection_reporter=rejection_collector,
        )
        validated_queries = _CountingIterator(query_gen)
        query_spectra = list(
            processing.process_spectra(validated_queries, config.processing)
        )

        n_validated = validated_queries.count
        n_processing_drops = max(n_validated - len(query_spectra), 0)
        result.spectra_loaded = n_validated
        result.spectra_rejected = rejection_collector.count + n_processing_drops

        if not query_spectra:
            # The file yielded no analyzable spectra: its data were NOT
            # processed. This is a file-level failure, not a success.
            reasons = "; ".join(rejection_collector.reasons) or "none recorded"
            result.status = "failed"
            result.fatal_errors.append(
                f"No analyzable spectra: {rejection_collector.count} rejected "
                f"by validation ({reasons}), {n_processing_drops} dropped by "
                "processing."
            )
            logger.error("Failed to process %s: %s", query_file, result.fatal_errors[0])
            return result

        # Ensure unique IDs for nodes
        seen_ids = set()
        for i, q in enumerate(query_spectra):
            base_id = q.get("id")
            if base_id is None:
                new_id = f"{query_file.stem}_query_{i}"
            else:
                new_id = str(base_id)
                counter = 1
                while new_id in seen_ids:
                    new_id = f"{base_id}_{counter}"
                    counter += 1

            q.set("id", new_id)
            seen_ids.add(new_id)

        global _worker_engine, _worker_router, _worker_backend

        standard_queries = query_spectra

        # ---- Routing decision --------------------------------------------------
        enable_routing = getattr(config.similarity, "enable_routing", False)

        counted_ref_iterator: _CountingIterator | None = None

        # Decoy-generation parameters come from the processing config; the
        # engine generates deterministic per-chunk decoys (chunk-invariant),
        # identical to a full-library decoy set.
        decoy_min_relative_intensity = config.processing.decoy_min_relative_intensity
        decoy_mz_shift_da = config.processing.decoy_mz_shift_da

        # ---- Library access (unified backend streaming) ------------------------
        # Workers open the SQLite/Zarr store themselves from the compact
        # LibrarySpec: only the spec crosses the process boundary, never the
        # spectral payload. The engine's lazy decorator consumes the backend
        # stream in bounded 10k chunks and generates decoys per chunk, so
        # per-worker memory is bounded by the chunk size, not the library
        # size. Results are identical to a full in-memory library because the
        # store round-trips spectra byte-for-byte and decoys are
        # chunk-invariant.
        backend = _worker_backend
        if backend is None:
            spec = library_spec
            if spec is None:
                # Direct-API / test fallback: stream the configured raw file.
                if config.input.library_path is None:
                    raise ValueError("Library path is not configured.")
                spec = LibrarySpec(path=config.input.library_path, kind="file")
            backend = open_library(spec, config.processing)

        if enable_routing:
            # Routing dispatches queries across engines that each iterate the
            # library, so the (experimental) routing path materializes the
            # library once per file, matching the pre-refactor worker path.
            library_list = list(backend.iter_spectra())
            counted_ref_iterator = _CountingIterator(iter(library_list))
            router = (
                _worker_router
                if _worker_router is not None
                else MLRouter(config.similarity)
            )
            all_results = router.route_and_search(
                standard_queries,
                library_list,
                include_decoys=True,
                decoy_min_relative_intensity=decoy_min_relative_intensity,
                decoy_mz_shift_da=decoy_mz_shift_da,
            )
            result.degraded_mode_flags.extend(
                getattr(router, "degraded_mode_flags", [])
            )
        else:
            counted_ref_iterator = _CountingIterator(backend.iter_spectra())
            engine = (
                _worker_engine
                if _worker_engine is not None
                else get_similarity_engine(config.similarity)
            )
            try:
                all_results = engine.search(
                    standard_queries,
                    counted_ref_iterator,
                    include_decoys=True,
                    decoy_min_relative_intensity=decoy_min_relative_intensity,
                    decoy_mz_shift_da=decoy_mz_shift_da,
                )
            except Exception as exc:
                logger.warning(
                    "Configured engine '%s' failed (%s); falling back "
                    "to modified_cosine scoring.",
                    config.similarity.algorithm,
                    exc,
                )
                result.degraded_mode_flags.append(
                    f"engine_fallback:{config.similarity.algorithm}"
                )
                fallback_engine = (
                    _worker_fallback_engine
                    if _worker_fallback_engine is not None
                    else _build_classical_fallback_engine(config.similarity)
                )
                # Retry with a FRESH library stream: the failed engine may
                # have consumed part of the first iterator.
                all_results = fallback_engine.search(
                    standard_queries,
                    backend.iter_spectra(),
                    include_decoys=True,
                    decoy_min_relative_intensity=decoy_min_relative_intensity,
                    decoy_mz_shift_da=decoy_mz_shift_da,
                )
            result.degraded_mode_flags.extend(
                getattr(engine, "degraded_mode_flags", [])
            )

        # Per-query target-decoy competition (TDC): the competition unit is the
        # query spectrum. Each query contributes its best target hit and its
        # best decoy hit exactly once (see
        # MassFlow.similarity.calculate_fdr for the full contract).
        from MassFlow.similarity import calibrate_query_level_fdr

        q_by_query, p_by_query, fdr_summary = calibrate_query_level_fdr(all_results)

        # True processed library size (targets only): the explicit parameter
        # from the parent's store build is authoritative; direct-API callers
        # fall back to the streamed count (exact after a successful search).
        # The backend's ``spectrum_count`` is deliberately NOT used here for
        # file backends — it would require an extra full parse of the library.
        if library_size is not None:
            lib_size = library_size
        elif counted_ref_iterator is not None:
            lib_size = counted_ref_iterator.count
        else:
            lib_size = 0

        is_small_library = lib_size < 2000

        # When the parent process could not know the library size (direct API
        # invocation without a prepared store), emit the small-library warning
        # here so no execution mode silently loses it.
        if (
            is_small_library
            and library_size is None
            and _worker_library_spec is None
            and getattr(config.similarity, "fdr_threshold", 0.01) < 0.1
        ):
            _emit_small_library_warning(
                lib_size, getattr(config.similarity, "fdr_threshold", 0.01)
            )

        # FDR provenance for the YAML sidecar: the competition-unit summary
        # behind every exported q-value, including the true library size that
        # determined the small-library status.
        result.fdr_summary = {**fdr_summary, "library_size": lib_size}

        if is_small_library:
            result.warnings.append(
                f"Small reference library ({lib_size} spectra): target-decoy FDR "
                "is statistically weak below ~2000 entries; q-values are "
                "conservative."
            )

        if fdr_summary["n_decoy_competitions"] == 0:
            # No decoy evidence at all: q-values are the uncalibrated 1/N
            # bound, not a target-decoy estimate. This must be explicit.
            result.degraded_mode_flags.append("fdr_uncalibrated")
            result.warnings.append(
                "No decoy hits survived scoring; q-values are uncalibrated "
                "(1/N rank bound). FDR claims are not supported for this file."
            )

        fdr_threshold = getattr(config.similarity, "fdr_threshold", 0.01)

        # Single filter, single concept: the per-query TDC q-value. The
        # empirical p-value is exported as a diagnostic column only; it is
        # never compared against fdr_threshold (mixing FWER/Bonferroni
        # semantics with an FDR threshold would silently switch statistical
        # concepts).
        fdr_filtered_results = []
        for res in all_results:
            if res.get("is_decoy", False):
                continue

            query_id = res.get("query_id")
            if query_id is None:
                # A result without a query id cannot be attributed to a
                # competition unit; it is never calibrated or exported.
                continue

            q_val = q_by_query.get(query_id, 1.0)
            p_val = p_by_query.get(query_id, 1.0)

            res["q_value"] = q_val
            res["p_value"] = p_val

            if q_val <= fdr_threshold:
                fdr_filtered_results.append(res)

        # Sort results descending by score
        fdr_filtered_results.sort(key=lambda x: x["score"], reverse=True)

        result.hits_produced = len(fdr_filtered_results)
        result.query_spectra = query_spectra
        result.results = fdr_filtered_results

        # A file processed with any degraded machinery is "degraded", never
        # silently "successful".
        if result.degraded_mode_flags:
            result.status = "degraded"

        return result

    except Exception as e:
        # Last-resort boundary: ANY exception escapes as an explicit file
        # failure. The file must never degrade into an empty successful run.
        result.status = "failed"
        result.fatal_errors.append(f"{type(e).__name__}: {e}")
        logger.error(f"Failed to process {query_file}: {e}", exc_info=True)
        return result


def _handle_file_results(
    result: FileExecutionResult,
    config: MassFlowConfig,
    config_path: Path | str | None = None,
) -> None:
    """Export results (or an explicit failure report) for one processed file.

    Failure model: a ``failed`` file NEVER produces a results CSV — an empty
    CSV would be mistaken for a successful annotation. Instead it produces
    an explicit ``<stem>_failed.report.yaml`` provenance record carrying the
    fatal errors. Successful and degraded files produce the normal
    ``<stem>_results.<ext>`` table plus the provenance sidecar, which records
    the status, spectrum counts, warnings, and degraded-mode flags.
    """
    if result.status == "failed":
        failure_report = (
            config.output_directory / f"{result.input_path.stem}_failed.report.yaml"
        )
        _write_analysis_report(
            report_path=failure_report,
            config=config,
            result=result,
            config_path=config_path,
        )
        logger.error(
            "File FAILED: %s — %s",
            result.input_path,
            "; ".join(result.fatal_errors),
        )
        return

    if not result.query_spectra:
        logger.warning(f"No valid spectra extracted from {result.input_path}.")
        return

    # Save intermediate results for this file (Collision Prevention)
    base_stem = result.input_path.stem
    export_format = config.export.format.lower()
    ext_map = {
        "csv": "csv",
        "mztab": "mztab",
    }
    ext = ext_map.get(export_format, "csv")

    out_file = config.output_directory / f"{base_stem}_results.{ext}"
    counter = 1
    while out_file.exists():
        out_file = config.output_directory / f"{base_stem}_{counter}_results.{ext}"
        counter += 1

    results_dict = cast(List[Dict[str, Any]], result.results)
    # Route to the correct exporter based on config.export.format
    if export_format == "mztab":
        io.save_match_results_to_mztab(
            results_dict, out_file, query_spectra=result.query_spectra
        )
    else:
        io.save_match_results(
            results_dict, out_file, query_spectra=result.query_spectra
        )

    result.output_path = out_file

    report_file = config.output_directory / f"{out_file.stem}.report.yaml"
    _write_analysis_report(
        report_path=report_file,
        config=config,
        result=result,
        config_path=config_path,
    )


def _write_analysis_report(
    report_path: Path,
    config: MassFlowConfig,
    result: FileExecutionResult,
    config_path: Path | str | None = None,
) -> None:
    """
    Generate a concise provenance payload and delegate saving to the I/O layer.

    The sidecar records the per-file execution outcome — status, spectrum
    counts, warnings, fatal errors, and degraded-mode flags — so that a
    degraded or partially failed run is never indistinguishable from a clean
    one.

    The I/O helper :func:`io.save_analysis_report` is used so tests can patch and
    assert calls instead of inspecting on-disk files.
    """
    report_payload = {
        "query_file": str(result.input_path),
        "results_csv": str(result.output_path) if result.output_path else None,
        "library_path": str(config.input.library_path)
        if config.input.library_path
        else None,
        "status": result.status,
        "spectra_loaded": result.spectra_loaded,
        "spectra_rejected": result.spectra_rejected,
        "hits_produced": result.hits_produced,
        "warnings": list(result.warnings),
        "fatal_errors": list(result.fatal_errors),
        "degraded_mode_flags": list(result.degraded_mode_flags),
        "num_queries": len(result.query_spectra),
        "num_matches": len(result.results),
        "processing": config.processing.model_dump(mode="json"),
        "similarity": config.similarity.model_dump(mode="json"),
        "workflow": config.workflow.model_dump(mode="json"),
        # Normalized configuration representation (schema version, source
        # config file, full effective config with resolved paths, digest).
        "config": config.normalized_config(),
    }

    # FDR provenance: the competition-unit summary behind every exported
    # q-value.
    if result.fdr_summary is not None:
        report_payload["fdr"] = {
            "model": "per_query_target_decoy",
            **result.fdr_summary,
        }

    # Delegate to IO layer for file writing so tests can patch io.save_analysis_report
    io.save_analysis_report(report_path, report_payload)


def _write_run_provenance(output_directory: Path, payload: dict[str, Any]) -> Path:
    """Write the run-level provenance file.

    Written BEFORE any per-file processing begins, so every result file can
    be traced back to the exact environment, configuration, and inputs that
    produced it.  Repeated runs sharing an output directory use the same
    counter-suffix convention as result CSVs (``run_provenance.json``,
    ``run_provenance_1.json``, ...) and never overwrite each other.
    """
    provenance_path = output_directory / "run_provenance.json"
    counter = 1
    while provenance_path.exists():
        provenance_path = output_directory / f"run_provenance_{counter}.json"
        counter += 1
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def _repo_root() -> Optional[Path]:
    """Return the repository root (when running from a checkout)."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / ".git").exists() or (candidate / "uv.lock").exists():
        return candidate
    return None


def _git_sha() -> tuple[Optional[str], bool]:
    """Return (commit SHA, dirty flag) for the checkout, or (None, False)."""
    import subprocess

    root = _repo_root()
    if root is None:
        return None, False
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty_proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return (sha or None), bool(dirty_proc.stdout.strip())
    except Exception:
        return None, False


def _dependency_versions() -> dict[str, str]:
    """Resolved versions of the direct dependencies (importlib.metadata)."""
    import importlib.metadata as metadata

    versions: dict[str, str] = {}
    try:
        dist = metadata.distribution("massflow")
        for requirement in dist.requires or []:
            name = requirement.split("[", 1)[0].split(";", 1)[0].strip().split(" ")[0]
            if not name:
                continue
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "<not installed>"
    except metadata.PackageNotFoundError:
        pass
    try:
        versions["massflow"] = metadata.version("massflow")
    except metadata.PackageNotFoundError:
        versions["massflow"] = "<not installed>"
    return versions


def _lockfile_digest() -> Optional[str]:
    """SHA-256 of the committed ``uv.lock`` (None when not a checkout)."""
    root = _repo_root()
    if root is None:
        return None
    lockfile = root / "uv.lock"
    if not lockfile.is_file():
        return None
    return _file_sha256(lockfile)


def _file_sha256(path: Path) -> str:
    """Streaming SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> tuple[str, str]:
    """Content digest of a file, or a manifest digest of a directory.

    Returns ``(kind, sha256)`` where ``kind`` is ``"file"`` or
    ``"directory"``.  Directory digests are computed over the sorted
    relative paths and sizes of every contained file, so identical directory
    contents produce identical digests.
    """
    if path.is_dir():
        digest = hashlib.sha256()
        entries: list[tuple[str, int]] = []
        for child in sorted(path.rglob("*")):
            if child.is_file():
                entries.append((str(child.relative_to(path)), child.stat().st_size))
        for relative, size in entries:
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("utf-8"))
            digest.update(b"\n")
        return "directory", digest.hexdigest()
    return "file", _file_sha256(path)


def _build_run_provenance(
    config: MassFlowConfig,
    input_files: list[Path],
    backend: str,
) -> dict[str, Any]:
    """Assemble the run-level provenance payload (static part).

    Records everything needed to recreate the run's scientific context:
    MassFlow version, git SHA, Python version, resolved dependency
    versions, the committed lockfile digest, the normalized configuration
    (with its own digest), engine/processing configuration, storage
    backend, decoy seed, input file hashes, the reference-library digest,
    and the (explicitly time-varying) run start timestamp.

    The only fields that vary between two identical runs are
    ``run_started_at`` (and ``completed_at``/``results``, added by
    :func:`_finalize_run_provenance` after processing).
    """
    import platform
    from datetime import datetime, timezone

    git_sha, git_dirty = _git_sha()

    library_digest: Optional[str] = None
    library_kind: Optional[str] = None
    if config.input.library_path is not None:
        library_kind, library_digest = _path_digest(config.input.library_path)

    input_hashes: dict[str, str] = {}
    for input_file in input_files:
        kind, digest = _path_digest(input_file)
        input_hashes[str(input_file)] = f"{kind}:{digest}"

    payload: dict[str, Any] = {
        "schema_version": 2,
        "massflow_version": _dependency_versions().get("massflow", "unknown"),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "lockfile_digest_sha256": _lockfile_digest(),
        "config_file": str(config.config_path) if config.config_path else None,
        "effective_config": json.loads(config.model_dump_json()),
        "config_digest_sha256": config.normalized_config()["config_digest_sha256"],
        "engine": json.loads(config.similarity.model_dump_json()),
        "processing": json.loads(config.processing.model_dump_json()),
        "backend": backend,
        "decoy_seed": 42,  # generate_decoys() default; per-spectrum seeds are content-derived
        "decoy_config": {
            "min_relative_intensity": config.processing.decoy_min_relative_intensity,
            "mz_shift_da": config.processing.decoy_mz_shift_da,
        },
        "input_file_hashes": input_hashes,
        "reference_library_path": (
            str(config.input.library_path) if config.input.library_path else None
        ),
        "reference_library_sha256": library_digest,
        "reference_library_kind": library_kind,
        "run_started_at": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def _finalize_run_provenance(
    provenance_path: Path,
    execution_results: list[FileExecutionResult],
) -> None:
    """Append the completion summary to the run provenance file.

    Adds ``completed_at`` (explicitly time-varying) and the aggregated
    ``results`` summary (per-file statuses, warnings, degraded-mode flags,
    failed files) so a completed run's record includes warnings and
    degraded modes.
    """
    from datetime import datetime, timezone

    if not provenance_path.exists():
        return
    payload = json.loads(provenance_path.read_text())
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()

    warnings: list[str] = []
    degraded_flags: list[str] = []
    failed_files: list[dict[str, Any]] = []
    for result in execution_results:
        for warning in result.warnings:
            if warning not in warnings:
                warnings.append(warning)
        for flag in result.degraded_mode_flags:
            if flag not in degraded_flags:
                degraded_flags.append(flag)
        if result.status == "failed":
            failed_files.append(
                {
                    "input_path": str(result.input_path),
                    "fatal_errors": list(result.fatal_errors),
                }
            )

    payload["results"] = {
        "files_total": len(execution_results),
        "files_succeeded": sum(1 for r in execution_results if r.status == "success"),
        "files_degraded": sum(1 for r in execution_results if r.status == "degraded"),
        "files_failed": sum(1 for r in execution_results if r.status == "failed"),
        "spectra_loaded_total": sum(r.spectra_loaded for r in execution_results),
        "spectra_rejected_total": sum(r.spectra_rejected for r in execution_results),
        "hits_produced_total": sum(r.hits_produced for r in execution_results),
        "warnings": warnings,
        "degraded_mode_flags": degraded_flags,
        "failed_files": failed_files,
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _emit_entropy_diagnostic(library_spec: LibrarySpec, config: MassFlowConfig) -> None:
    """Streaming target-decoy entropy diagnostic (bounded memory).

    Replaces the previous full-library in-memory comparison: spectra are
    streamed from the worker-openable store in bounded chunks, decoys are
    generated per chunk (deterministic and chunk-invariant), and the per-pair
    statistics are aggregated exactly (entropy is a per-spectrum quantity, so
    weighted aggregation over chunks equals the full-library comparison).
    """
    from MassFlow.similarity import compare_target_decoy_entropy, generate_decoys

    backend = open_library(library_spec, config.processing)
    try:
        total_pairs = 0
        weighted_mean_target = 0.0
        weighted_mean_decoy = 0.0
        weighted_mean_delta = 0.0
        max_delta = 0.0
        for chunk in backend.iter_processed_chunks(chunk_size=10_000):
            if not chunk:
                continue
            decoys = generate_decoys(
                chunk,
                min_relative_intensity=config.processing.decoy_min_relative_intensity,
                mz_shift_da=config.processing.decoy_mz_shift_da,
            )
            comparison = compare_target_decoy_entropy(
                chunk,
                decoys,
                min_relative_intensity=config.processing.decoy_min_relative_intensity,
            )
            n_pairs = int(comparison["compared_pairs"])
            total_pairs += n_pairs
            if n_pairs == 0:
                continue
            weighted_mean_target += comparison["mean_target_entropy"] * n_pairs
            weighted_mean_decoy += comparison["mean_decoy_entropy"] * n_pairs
            weighted_mean_delta += comparison["mean_abs_entropy_delta"] * n_pairs
            max_delta = max(max_delta, float(comparison["max_abs_entropy_delta"]))
    finally:
        backend.close()

    if total_pairs == 0:
        return
    mean_target = weighted_mean_target / total_pairs
    mean_decoy = weighted_mean_decoy / total_pairs
    mean_abs_delta = weighted_mean_delta / total_pairs
    logger.info(
        "Target-decoy entropy comparison: mean_target=%.4f nats, "
        "mean_decoy=%.4f nats, mean_abs_delta=%.6f, max_abs_delta=%.6f "
        "(%d pairs).",
        mean_target,
        mean_decoy,
        mean_abs_delta,
        max_delta,
        total_pairs,
    )
    if mean_abs_delta > 0.01:
        logger.warning(
            "Target and decoy entropy distributions systematically "
            "diverge (mean |delta| = %.4f nats). FDR calibration may be "
            "biased; check that baseline noise filtering is applied "
            "before decoy generation.",
            mean_abs_delta,
        )


def run_annotation_pipeline(
    config: MassFlowConfig, config_path: Path | str | None = None
) -> List[FileExecutionResult]:
    """
    Execute the full MassFlow annotation workflow.

    The workflow performs these major stages:

    1. Normalize the reference library into a worker-openable store
       (streaming, bounded memory) and emit the FDR-calibration diagnostics.
    2. Discover query inputs from either a single file or a data directory.
    3. Dispatch one task per experimental file to a process pool; workers
       open the library store themselves and never receive spectral data.
    4. Within each worker, search the processed queries against chunked
       reference spectra and apply per-file FDR filtering.
    5. Export a result file (or an explicit failure report) for each
       processed experimental input.

    Parameters
    ----------
    config : MassFlowConfig
        The configuration object containing all settings for input/output paths,
        processing parameters, and similarity search options.
    config_path : Path or str or None, optional
        Original YAML configuration path used to create ``config``. When
        provided, it is written into the per-results provenance report.

    Returns
    -------
    list of FileExecutionResult
        One structured outcome per experimental input file. Files that
        failed are reported with ``status == "failed"`` and non-empty
        ``fatal_errors``; degraded files carry ``degraded_mode_flags``.
        Callers (e.g. the CLI) must treat any ``failed`` entry as a nonzero
        exit condition.

    Raises
    ------
    ValueError
        If the reference library path is missing, no valid reference spectra are found,
        or no supported input files are found.

    Notes
    -----
    Batch robustness is preserved: one problematic experimental file does
    not abort the batch, but its failure is recorded explicitly in the
    returned results and in a ``<stem>_failed.report.yaml`` sidecar — never
    as a silently empty success.
    """
    # 1. Prepare the reference library as a worker-openable store.
    # The full spectral payload never crosses the process boundary: the parent
    # normalizes the library once (streaming, bounded memory) and workers
    # open the store themselves from the compact LibrarySpec.
    config.output_directory.mkdir(parents=True, exist_ok=True)

    # Package boundary: an experimental configuration must be visibly
    # flagged before any processing begins. Experimental features are
    # implemented and tested, but they are outside the stable product
    # contract and must never be mistaken for it (docs/CAPABILITY_MATRIX.md).
    experimental_flags = experimental_surface_flags(config)
    if experimental_flags:
        logger.warning(
            "EXPERIMENTAL SURFACES ACTIVE: %s. This run is NOT part of the "
            "stable MassFlow product contract (see docs/CAPABILITY_MATRIX.md).",
            ", ".join(experimental_flags),
        )

    library_spec, library_size = prepare_library(config, config.output_directory)
    logger.info(
        "Reference library ready: %d spectra via %s (kind=%s).",
        library_size,
        library_spec.path,
        library_spec.kind,
    )

    # FDR-calibration diagnostic: entropy-preserving decoys must not
    # systematically diverge from targets in spectral entropy. A large
    # delta indicates biased decoy generation and unreliable q-values.
    # Streamed in bounded chunks so the parent never holds the library.
    _emit_entropy_diagnostic(library_spec, config)

    # Only warn when both conditions are met:
    # 1. Library is small (< 2000 spectra)
    # 2. User has actually requested strict FDR (threshold < 0.1)
    fdr_threshold = getattr(config.similarity, "fdr_threshold", 0.01)
    if library_size < 2000 and fdr_threshold < 0.1:
        _emit_small_library_warning(library_size, fdr_threshold)

    # 2. Determine Input Files
    input_files = []
    input_path = Path(config.input.input_path)
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        input_files.append(input_path)
    else:
        # Recursively find all supported spectral files
        supported_exts = {
            ".mzml",
            ".mzxml",
            ".mgf",
            ".msp",
            ".raw",
            ".d",
            ".wiff",
            ".lcd",
            ".t2d",
        }
        for f in input_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported_exts:
                input_files.append(f)
            # Handle .d directories (Agilent/Bruker)
            if f.is_dir() and f.suffix.lower() == ".d":
                input_files.append(f)

        if not input_files:
            raise ValueError(f"No supported spectral files found in {input_path}")

    # Run-level provenance: written BEFORE any per-file processing begins so
    # every result file can be traced back to the exact environment,
    # configuration, and inputs that produced it.  The recorded backend is
    # the EFFECTIVE backend of the library store actually used (from the
    # prepared spec), not merely the configured one.
    provenance_payload = _build_run_provenance(
        config, input_files, backend=library_spec.storage_backend or "sqlite"
    )
    provenance_path = _write_run_provenance(config.output_directory, provenance_payload)
    logger.info("Run provenance written to %s", provenance_path)

    # 3. Process Each File
    config.output_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(input_files)} experimental files...")

    execution_results: List[FileExecutionResult] = []

    if len(input_files) == 1:
        # Optimization for single file: avoid ProcessPool overhead and pickling
        qf = input_files[0]
        result = _process_single_file(
            qf, config, library_size=library_size, library_spec=library_spec
        )
        _handle_file_results(result, config, config_path=config_path)
        execution_results.append(result)
    else:
        with ProcessPoolExecutor(
            initializer=_init_worker,
            initargs=(config, library_spec),
        ) as executor:
            futures = {
                executor.submit(
                    _process_single_file, qf, config, library_size, library_spec
                ): qf
                for qf in input_files
            }

            for future in as_completed(futures):
                qf = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    # The worker process died or raised something that escaped
                    # _process_single_file: synthesize an explicit failed
                    # result so the batch continues without pretending the
                    # file succeeded.
                    logger.error(f"Worker crashed for {qf}: {exc}", exc_info=True)
                    result = FileExecutionResult(
                        status="failed",
                        input_path=qf,
                        fatal_errors=[f"worker_crash: {type(exc).__name__}: {exc}"],
                    )
                _handle_file_results(result, config, config_path=config_path)
                execution_results.append(result)

    # Per-file outcome summary: every file is accounted for explicitly.
    for result in sorted(execution_results, key=lambda r: str(r.input_path)):
        if result.status == "failed":
            logger.error(
                "[%s] %s: %s",
                result.status.upper(),
                result.input_path,
                "; ".join(result.fatal_errors),
            )
        else:
            suffix = (
                f" (degraded: {', '.join(result.degraded_mode_flags)})"
                if result.degraded_mode_flags
                else ""
            )
            logger.info(
                "[%s] %s: %d spectra loaded, %d rejected, %d hits%s",
                result.status.upper(),
                result.input_path,
                result.spectra_loaded,
                result.spectra_rejected,
                result.hits_produced,
                suffix,
            )

    n_failed = sum(1 for r in execution_results if r.status == "failed")
    logger.info(
        "Pipeline finished: %d file(s) processed, %d failed.",
        len(execution_results),
        n_failed,
    )

    # Completion summary (warnings, degraded modes, failed files) appended
    # to the run provenance record.
    _finalize_run_provenance(provenance_path, execution_results)

    return execution_results
