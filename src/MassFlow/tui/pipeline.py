"""
Bridge between the terminal console and the core annotation pipeline.

This module is the only place in ``MassFlow.tui`` that imports the heavy core
modules (:mod:`MassFlow.io`, :mod:`MassFlow.processing`,
:mod:`MassFlow.similarity`, :mod:`MassFlow.storage`). Every function here is
synchronous and side-effect-free with respect to the UI, so the Textual app
calls it from a worker thread while the interface stays responsive.

Errors are translated into :class:`MassFlow.tui.diagnostics.TuiError` with a
stage tag (``load-query``, ``load-library``, ``search``) so the diagnostics
tab can show *which* step failed and *why*.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from MassFlow.tui.diagnostics import TuiError
from MassFlow.tui.files import guess_backend
from MassFlow.tui.spectrum_data import (
    DEFAULT_MAX_PEAKS,
    downsample_peaks,
    summarize_spectrum,
)
from MassFlow.tui.state import (
    IdentificationOutcome,
    IdentificationRequest,
    LibraryInfo,
    QueryLoadResult,
    SearchHit,
    SpectrumSummary,
)

logger = logging.getLogger(__name__)

# Library census cap: counting spectra in a text library is O(n); beyond this
# the console reports "> 200000" instead of grinding through a GNPS-sized MSP.
LIBRARY_COUNT_CAP = 200_000

_SMALL_LIBRARY_THRESHOLD = 2000


@contextmanager
def capture_quarantine_records(max_records: int = 200) -> Iterator[list[str]]:
    """Capture quarantine-log messages emitted during a load, in memory.

    The core validation layer (:func:`MassFlow.io._validate_spectra_iterator`)
    logs every rejected spectrum to the ``quarantine`` logger. This context
    manager attaches a lightweight in-memory handler around a load so the
    console can report exactly which spectra were rejected without parsing the
    log file afterwards.

    Parameters
    ----------
    max_records : int
        Maximum number of messages to retain.

    Yields
    ------
    list[str]
        The captured messages (populated as the wrapped block runs).
    """
    captured: list[str] = []
    handler = logging.Handler()
    handler.setLevel(logging.WARNING)
    handler.emit = lambda record: (  # type: ignore[method-assign]
        captured.append(record.getMessage()) if len(captured) < max_records else None
    )
    quarantine_logger = logging.getLogger("quarantine")
    quarantine_logger.addHandler(handler)
    try:
        yield captured
    finally:
        quarantine_logger.removeHandler(handler)


def _preview_processing_config():
    """A pass-through processing config for *viewing* raw spectra.

    The default :class:`MassFlow.config.ProcessingConfig` enables matchms
    metadata repairs and peak filtering that are appropriate for annotation
    but destructive for inspection. Previewing uses a config with every
    filter disabled and ``min_peaks`` lowered so the viewer shows what is
    actually in the file.
    """
    from MassFlow.config import ProcessingConfig

    return ProcessingConfig(
        clean_metadata=False,
        add_retention_time=False,
        repair_inchi_inchikey_smiles=False,
        derive_adduct_from_name=False,
        derive_formula_from_name=False,
        clean_compound_name=False,
        derive_ionmode=False,
        make_charge_int=False,
        filter_by_intensity=False,
        filter_min_peaks=False,
        filter_by_mz=False,
        reduce_to_top_n_peaks=False,
        normalize_intensity=False,
        min_peaks=1,
    )


def _identification_processing_config():
    """A light processing config for identification runs in the console."""
    from MassFlow.config import ProcessingConfig

    return ProcessingConfig(
        clean_metadata=True,
        filter_by_intensity=False,
        filter_min_peaks=False,
        filter_by_mz=False,
        reduce_to_top_n_peaks=False,
        normalize_intensity=True,
        min_peaks=3,
    )


def load_query_preview(path: Path, *, max_spectra: int = 500) -> QueryLoadResult:
    """Load and summarize the first ``max_spectra`` spectra of a query file.

    The load runs through the core validation layer, so invalid spectra are
    quarantined exactly as they would be in a full annotation run — and the
    quarantine messages are returned alongside the valid summaries.

    Parameters
    ----------
    path : Path
        Path to an open-format spectral file (mzML, mzXML, MGF, MSP).
    max_spectra : int
        Hard cap on the number of spectra summarized.

    Returns
    -------
    QueryLoadResult

    Raises
    ------
    TuiError
        If the file cannot be loaded at all (missing path, vendor format,
        unsupported extension, malformed content).
    """
    from MassFlow import io, processing

    path = Path(path)
    if not path.exists():
        raise TuiError.from_exception(
            FileNotFoundError(f"Input path does not exist: {path}"),
            stage="load-query",
        )
    if path.is_dir():
        raise TuiError(
            f"'{path}' is a directory, not a spectral file.",
            stage="load-query",
            hint="Pick a file (.mzml, .mzxml, .mgf, .msp) or build it into a database first.",
        )

    try:
        with capture_quarantine_records() as quarantined:
            raw_spectra = io.load_spectra(path)
            processed = processing.process_spectra(
                raw_spectra, _preview_processing_config()
            )
            summaries: list[SpectrumSummary] = []
            for spectrum in processed:
                summaries.append(summarize_spectrum(spectrum))
                if len(summaries) >= max_spectra:
                    break
    except TuiError:
        raise
    except Exception as exception:
        raise TuiError.from_exception(exception, stage="load-query") from exception

    format_hint = path.suffix.lstrip(".").lower() or "unknown"
    return QueryLoadResult(
        path=path,
        format_hint=format_hint,
        summaries=summaries,
        quarantined_messages=list(quarantined),
    )


def inspect_library(path: Path) -> LibraryInfo:
    """Census a reference library without loading it into the UI.

    Database-backed libraries (SQLite/Zarr/hybrid) are inspected through the
    store API; text libraries (MSP/MGF) are counted by streaming their
    spectra (capped at :data:`LIBRARY_COUNT_CAP`). Errors are *reported*, not
    raised: a broken library becomes a :class:`LibraryInfo` with an
    ``error`` field so the console can explain the problem in place.

    Parameters
    ----------
    path : Path
        Path to a spectral library (text or database).

    Returns
    -------
    LibraryInfo
    """
    from MassFlow import io
    from MassFlow.storage import create_spectral_store

    path = Path(path)
    backend = guess_backend(path)

    if not path.exists():
        return LibraryInfo(
            path=path,
            backend=backend,
            total_spectra=None,
            error=f"Library does not exist: {path}",
        )

    try:
        if backend in {"sqlite", "zarr", "hybrid"}:
            store = create_spectral_store(path, backend=backend)
            try:
                total = store.get_total_spectra_count()
                categories = store.get_category_counts()
                mz_range = store.get_precursor_mz_range()
            finally:
                store.close()
            return LibraryInfo(
                path=path,
                backend=backend,
                total_spectra=total,
                categories=categories,
                precursor_mz_range=mz_range,
            )

        # Text library: stream and count.
        total = 0
        mz_min: Optional[float] = None
        mz_max: Optional[float] = None
        truncated = False
        for spectrum in io.load_spectra(path):
            total += 1
            precursor_mz = spectrum.get("precursor_mz")
            if precursor_mz is not None:
                try:
                    mz = float(precursor_mz)
                except (TypeError, ValueError):
                    mz = np.nan
                if np.isfinite(mz):
                    if mz_min is None or mz < mz_min:
                        mz_min = mz
                    if mz_max is None or mz > mz_max:
                        mz_max = mz
            if total >= LIBRARY_COUNT_CAP:
                truncated = True
                break

        text_mz_range = (
            (mz_min, mz_max) if mz_min is not None and mz_max is not None else None
        )
        return LibraryInfo(
            path=path,
            backend=backend,
            total_spectra=total,
            precursor_mz_range=text_mz_range,
            truncated=truncated,
        )
    except Exception as exception:
        return LibraryInfo(
            path=path,
            backend=backend,
            total_spectra=None,
            error=str(exception) or exception.__class__.__name__,
        )


def run_identification(request: IdentificationRequest) -> IdentificationOutcome:
    """Run a target-decoy similarity search for the console.

    This is the "identify" verb, executed synchronously (the app runs it in a
    worker thread). The flow mirrors the core single-file workflow:

    1. Load and lightly process the query spectra.
    2. Load and lightly process the reference library.
    3. Generate entropy-preserving decoys (seeded, deterministic).
    4. Score queries against targets + decoys with the requested engine.
    5. Compute q-values (target-decoy FDR) and empirical p-values.
    6. Keep the top ``request.top_n`` target hits per query.

    If the requested engine cannot be instantiated (for example an ML
    algorithm without the ``massflow[ml]`` extra), the run falls back to
    modified-cosine scoring and records a warning — the console never dies
    because a heavy engine is unavailable.

    Parameters
    ----------
    request : IdentificationRequest
        Immutable run description.

    Returns
    -------
    IdentificationOutcome

    Raises
    ------
    TuiError
        With stage ``load-query``, ``load-library``, or ``search``.
    """
    from MassFlow import io, processing
    from MassFlow.config import SimilarityConfig
    from MassFlow.similarity import (
        calibrate_query_level_fdr,
        generate_decoys,
        get_similarity_engine,
    )

    started = time.monotonic()
    warnings: list[str] = []

    # --- 1. Load queries ---------------------------------------------------
    try:
        queries = list(
            processing.process_spectra(
                io.load_spectra(request.query_path),
                _identification_processing_config(),
            )
        )
    except Exception as exception:
        raise TuiError.from_exception(exception, stage="load-query") from exception
    if not queries:
        raise TuiError(
            f"No valid spectra found in query file: {request.query_path}",
            stage="load-query",
            hint="Check the quarantine log — every spectrum may have been rejected by the validation layer.",
        )
    queries = queries[: request.max_query_spectra]

    # --- 2. Load library ----------------------------------------------------
    try:
        references = list(
            processing.process_spectra(
                io.load_spectra(request.library_path),
                _identification_processing_config(),
            )
        )
    except Exception as exception:
        raise TuiError.from_exception(exception, stage="load-library") from exception
    if not references:
        raise TuiError(
            f"No valid spectra found in library: {request.library_path}",
            stage="load-library",
            hint="Check that the library path points to an MSP/MGF file or a MassFlow database.",
        )

    # --- 3. Engine selection (with fallback) --------------------------------
    similarity_config = SimilarityConfig(
        algorithm=request.algorithm,
        ms1_tolerance=request.ms1_tolerance,
        ms2_tolerance=request.ms2_tolerance,
        min_score=request.min_score,
        min_matched_peaks=request.min_matched_peaks,
    )
    engine_used = request.algorithm
    try:
        engine = get_similarity_engine(similarity_config)
    except Exception as exception:
        fallback_config = SimilarityConfig(
            algorithm="modified_cosine",
            ms1_tolerance=request.ms1_tolerance,
            ms2_tolerance=request.ms2_tolerance,
            min_score=request.min_score,
            min_matched_peaks=request.min_matched_peaks,
        )
        engine = get_similarity_engine(fallback_config)
        engine_used = f"modified_cosine (fallback from {request.algorithm})"
        warnings.append(
            f"Engine '{request.algorithm}' unavailable ({exception}); "
            "fell back to modified_cosine."
        )

    # --- 4. Decoys + search --------------------------------------------------
    try:
        decoys = generate_decoys(references, random_seed=42)
        all_references = references + decoys
        results = engine.search(
            queries,
            all_references,
            min_score=request.min_score,
            top_n=None,
            include_decoys=False,
        )
    except Exception as exception:
        raise TuiError.from_exception(exception, stage="search") from exception

    if len(references) < _SMALL_LIBRARY_THRESHOLD and request.fdr_threshold < 0.1:
        warnings.append(
            f"Library has only {len(references)} spectra; target-decoy FDR is "
            "statistically weak below ~2000 entries. Relax fdr_threshold or use "
            "a larger library."
        )

    # --- 5. FDR calibration ---------------------------------------------------
    # Per-query target-decoy competition (see
    # MassFlow.similarity.calculate_fdr): the competition unit is the query
    # spectrum, and every hit of a query shares that query's q-value.
    q_by_query, p_by_query, fdr_summary = calibrate_query_level_fdr(results)

    if fdr_summary["n_decoy_competitions"] == 0:
        warnings.append(
            "No decoy hits survived scoring; FDR calibration unavailable "
            "(q-values will be conservative)."
        )

    # --- 6. Build hits ---------------------------------------------------------
    query_peaks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spectrum in queries:
        query_peaks[str(spectrum.get("id"))] = downsample_peaks(
            np.asarray(spectrum.peaks.mz, dtype=np.float64),
            np.asarray(spectrum.peaks.intensities, dtype=np.float64),
            DEFAULT_MAX_PEAKS,
        )

    reference_by_id: dict[str, Any] = {}
    for spectrum in references:
        reference_by_id[str(spectrum.get("id"))] = spectrum

    hits: list[SearchHit] = []
    for result in results:
        if result.get("is_decoy"):
            continue
        query_id = result.get("query_id")
        if query_id is None:
            continue
        result["q_value"] = q_by_query.get(query_id, 1.0)
        result["p_value"] = p_by_query.get(query_id, 1.0)
        hits.append(SearchHit.from_search_result(result))

    # Keep top_n targets per query, ordered by score (desc).
    hits.sort(key=lambda hit: hit.score, reverse=True)
    per_query: dict[str, list[SearchHit]] = {}
    for hit in hits:
        bucket = per_query.setdefault(hit.query_id, [])
        if len(bucket) < request.top_n:
            bucket.append(hit)

    top_hits = [hit for bucket in per_query.values() for hit in bucket]

    hit_reference_peaks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for hit in top_hits:
        if hit.reference_id in hit_reference_peaks:
            continue
        reference = reference_by_id.get(hit.reference_id)
        if reference is None:
            continue
        hit_reference_peaks[hit.reference_id] = downsample_peaks(
            np.asarray(reference.peaks.mz, dtype=np.float64),
            np.asarray(reference.peaks.intensities, dtype=np.float64),
            DEFAULT_MAX_PEAKS,
        )

    return IdentificationOutcome(
        request=request,
        engine_used=engine_used,
        hits=top_hits,
        num_queries=len(queries),
        num_references=len(references),
        duration_seconds=time.monotonic() - started,
        fdr_threshold=request.fdr_threshold,
        warnings=warnings,
        query_peaks=query_peaks,
        hit_reference_peaks=hit_reference_peaks,
    )
