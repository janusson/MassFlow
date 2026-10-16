"""
Feasibility pre-analysis for MassFlow annotation runs.

Before an annotation run spends minutes scoring thousands of spectra against a
library, it is often possible to say — cheaply — whether the query data and the
reference library are *technically compatible*, and which limitations would
make a run produce no accepted identifications. This module answers that
question without scoring anything:

* **Reference library** — how many usable spectra it holds, its precursor m/z
  range, and which ionization modes are reliably available.
* **Query data** — how many spectra loaded and were rejected, their precursor
  coverage, the MS2 peak-count distribution, and the fraction that can meet the
  configured minimum peak requirements.
* **Search compatibility** — how many query spectra have at least one
  potentially compatible library precursor under the *actual* configured
  precursor tolerance and constraints.

Design rules (deliberate, and guarded by tests):

* **Advisory only.** Nothing here changes scientific parameters or blocks a run.
* **Agreement with annotation.** Precursor windows, missing-precursor bypass,
  adduct-mode compatibility, and conditional RT filtering mirror the predicates
  used by `SimilarityEngine.search`. Non-cosine engines are explicitly marked
  heuristic because they do not enforce the cosine MS1 window.
* **Definitive vs heuristic.** A definitive finding is one the pipeline's own
  logic establishes (e.g. no library precursor lies within the MS1 gate that
  `cosine` enforces). Everything else is reported as likelihood, not fact.
* **Lightweight.** Library metadata is read via the store's ``metadata_query``
  (no peak parsing) whenever the library is already a store; raw files are
  streamed once through the same processing pipeline the run uses.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from matchms import Spectrum

from MassFlow import io, processing
from MassFlow.config import MassFlowConfig
from MassFlow.library import LibrarySpec, open_library
from MassFlow.similarity import _adduct_ion_mode_sign, _adduct_modes_compatible

logger = logging.getLogger(__name__)

# Below this many usable reference spectra MassFlow cannot support a calibrated
# target-decoy FDR estimate (mirrors ``workflow.SMALL_LIBRARY_THRESHOLD``).
SMALL_LIBRARY_THRESHOLD = 2000


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@dataclass
class LibraryProfile:
    """What can be established about the reference library without scoring."""

    source_path: Path
    kind: str  # "store" | "file"
    usable_count: int
    rejected_count: int
    io_rejected_count: int
    processing_rejected_count: int
    integrity_rejected_count: int
    rejection_examples: list[str]
    missing_precursor_count: int
    precursor_mzs: np.ndarray  # float64, one per usable spectrum
    retention_times: np.ndarray  # float64, NaN when absent
    adducts: list[Optional[str]]
    ion_modes: dict[str, int]
    ion_mode_reliable: bool

    @property
    def precursor_range(self) -> tuple[float, float] | None:
        finite = self.precursor_mzs[np.isfinite(self.precursor_mzs)]
        finite = finite[finite > 0]
        if finite.size == 0:
            return None
        return float(finite.min()), float(finite.max())


@dataclass
class QueryProfile:
    """What can be established about one query file without scoring."""

    source_path: Path
    loaded_count: int
    rejected_count: int
    io_rejected_count: int
    processing_rejected_count: int
    rejection_examples: list[str]
    peak_counts: np.ndarray  # int, one per retained spectrum
    precursor_mzs: np.ndarray  # float64, one per retained spectrum
    retention_times: np.ndarray  # float64, NaN when absent
    adducts: list[Optional[str]]

    @property
    def n_retained(self) -> int:
        return int(self.peak_counts.size)


@dataclass
class CompatibilityReport:
    """Precursor/ion-mode compatibility between one query file and the library."""

    source_path: Path
    engine: str
    ms1_gate_definitive: bool
    tolerance_da: float
    resolution_ppm: Optional[float]
    rt_tolerance: Optional[float]
    n_queries: int
    n_queries_with_candidate: int
    n_queries_with_precursor_candidate: int
    n_queries_with_compatible_candidate: int
    n_candidate_pairs: int
    n_missing_precursor_pairs: int
    n_candidate_pairs_compatible: int
    min_matched_peaks: int

    @property
    def coverage_fraction(self) -> float:
        if self.n_queries == 0:
            return 0.0
        return self.n_queries_with_compatible_candidate / self.n_queries


@dataclass
class Finding:
    """One diagnostic statement, tagged with how certain it is."""

    text: str
    definitive: bool


@dataclass
class FeasibilityReport:
    """The complete pre-analysis result for one configuration."""

    library: LibraryProfile
    queries: list[QueryProfile]
    compatibility: list[CompatibilityReport]
    findings: list[Finding]
    recommendations: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Library profiling
# ---------------------------------------------------------------------------


class _RejectionCollector:
    """Count every rejection while keeping a bounded list of examples.

    Mirrors ``workflow._RejectionCollector``: the *count* must reflect every
    rejected spectrum (a raw library can have hundreds that fail the gate),
    while only the first few reasons are retained for the report.
    """

    def __init__(self, max_examples: int = 5) -> None:
        self.count = 0
        self.reasons: list[str] = []
        self._max_examples = max_examples

    def __call__(self, reason: str) -> None:
        self.count += 1
        if len(self.reasons) < self._max_examples:
            self.reasons.append(reason)


def _read_store_metadata(store: object, fields: list[str]) -> dict[str, np.ndarray]:
    """Read metadata arrays from a store, falling back to a full iteration.

    ``metadata_query`` is the cheap path (no peak parsing). Backends that do not
    implement a requested field raise ``ValueError``; rather than fail preflight
    we stream the store once and read the same fields from the spectra.
    """
    metadata_query = getattr(store, "metadata_query", None)
    if metadata_query is not None:
        try:
            return dict(metadata_query(fields))
        except (ValueError, NotImplementedError, KeyError):
            logger.debug(
                "metadata_query(%s) unavailable; streaming the store instead.",
                fields,
            )

    collected: dict[str, list] = {f: [] for f in fields}
    for spectrum in store.get_spectra():  # type: ignore[attr-defined]
        for f in fields:
            if f == "extra_metadata":
                collected[f].append(
                    json.dumps({"retention_time": spectrum.get("retention_time")})
                )
            else:
                collected[f].append(spectrum.get(f))
    return {f: np.asarray(vals, dtype=object) for f, vals in collected.items()}


def _ion_mode_summary(
    ion_modes: list[Optional[str]], adducts: list[Optional[str]]
) -> tuple[dict[str, int], bool]:
    """Summarise ionization modes, falling back to the adduct chemistry.

    Returns the mode histogram and whether the modes are *reliably available*
    (i.e. a mode could be determined for a meaningful share of spectra).
    """
    histogram: dict[str, int] = {}
    known = 0
    for mode, adduct in zip(ion_modes, adducts):
        key = str(mode).strip().lower() if mode not in (None, "") else None
        if key not in ("positive", "negative", "neutral"):
            sign = _adduct_ion_mode_sign(adduct)
            key = "positive" if sign == 1 else "negative" if sign == -1 else None
        if key is not None:
            histogram[key] = histogram.get(key, 0) + 1
            known += 1

    reliable = known >= 0.5 * max(len(ion_modes), 1)
    return histogram, reliable


def _retention_time_from_metadata(value: object) -> float:
    """Extract the scalar retention time from stored metadata JSON."""
    if not isinstance(value, str) or not value:
        return float("nan")
    try:
        metadata = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return float("nan")
    return _safe_float(metadata.get("retention_time"))


def profile_library(
    config: MassFlowConfig, library_spec: Optional[LibrarySpec] = None
) -> LibraryProfile:
    """Profile the configured reference library without scoring.

    When ``library_spec`` is provided (the annotation path, where the store has
    already been prepared) or the configured path is a MassFlow store, metadata
    is read via ``metadata_query`` — no peaks are parsed. A raw library file is
    streamed once through the same processing pipeline the run uses, so the
    usable/rejected counts and the exact 5 ppm integrity rejections seen here
    are what `annotate` would see.
    """
    if library_spec is not None and library_spec.kind == "store":
        return _profile_store(
            library_spec.path,
            config,
            storage_backend=library_spec.storage_backend,
        )

    library_path = config.input.library_path
    if library_path is None:
        raise ValueError("Library path not specified in configuration.")
    library_path = Path(library_path)
    if not library_path.exists():
        raise ValueError(f"Library path does not exist: {library_path}")

    if io.is_store_input(library_path):
        return _profile_store(library_path, config, storage_backend=None)

    return _profile_raw_library(library_path, config)


def profile_open_store(
    store: object, store_path: Path, config: MassFlowConfig
) -> LibraryProfile:
    """Profile an already-open store by reading metadata only (no peaks).

    Shared by the standalone preflight (which opens the store itself) and the
    annotation worker (which already holds the library backend), so the two
    never disagree about the library the run is previewing.
    """
    fields = ["precursor_mz", "adduct", "ionmode"]
    if config.similarity.rt_tolerance is not None:
        fields.append("extra_metadata")
    metadata = _read_store_metadata(store, fields)
    count = store.spectrum_count()  # type: ignore[attr-defined]

    mz_raw = metadata.get("precursor_mz", np.asarray([]))
    precursor_mzs = np.asarray([_safe_float(v) for v in mz_raw], dtype=np.float64)
    retention_times = np.asarray(
        [_retention_time_from_metadata(v) for v in metadata.get("extra_metadata", [])],
        dtype=np.float64,
    )
    adducts = [None if v in (None, "") else str(v) for v in metadata.get("adduct", [])]
    ion_modes = [
        None if v in (None, "") else str(v) for v in metadata.get("ionmode", [])
    ]
    histogram, reliable = _ion_mode_summary(ion_modes, adducts)
    return LibraryProfile(
        source_path=store_path,
        kind="store",
        usable_count=int(count),
        rejected_count=0,
        io_rejected_count=0,
        processing_rejected_count=0,
        integrity_rejected_count=0,
        rejection_examples=[],
        missing_precursor_count=int(np.count_nonzero(np.isnan(precursor_mzs))),
        precursor_mzs=precursor_mzs,
        retention_times=retention_times,
        adducts=adducts,
        ion_modes=histogram,
        ion_mode_reliable=reliable,
    )


def _profile_store(
    store_path: Path, config: MassFlowConfig, storage_backend: Optional[str]
) -> LibraryProfile:
    spec = LibrarySpec(path=store_path, kind="store", storage_backend=storage_backend)
    store = open_library(spec, config.processing)
    try:
        return profile_open_store(store, store_path, config)
    finally:
        store.close()


def _profile_raw_library(library_path: Path, config: MassFlowConfig) -> LibraryProfile:
    """Stream a raw library once, capturing integrity rejections."""
    physics = _RejectionCollector()

    io_rejections = _RejectionCollector()
    validated = _CountingIterator(
        io.load_spectra(library_path, rejection_reporter=io_rejections)
    )
    processed = processing.process_spectra(
        validated, config.processing, rejection_reporter=physics
    )

    precursor_mzs: list[float] = []
    retention_times: list[float] = []
    adducts: list[Optional[str]] = []
    ion_modes: list[Optional[str]] = []
    loaded = 0
    for spectrum in processed:
        loaded += 1
        precursor_mzs.append(_safe_float(spectrum.get("precursor_mz")))
        retention_times.append(_safe_float(spectrum.get("retention_time")))
        adduct = spectrum.get("adduct")
        adducts.append(None if adduct in (None, "") else str(adduct))
        mode = spectrum.get("ionmode")
        ion_modes.append(None if mode in (None, "") else str(mode))

    histogram, reliable = _ion_mode_summary(ion_modes, adducts)
    return LibraryProfile(
        source_path=library_path,
        kind="file",
        usable_count=loaded,
        rejected_count=io_rejections.count + max(validated.count - loaded, 0),
        io_rejected_count=io_rejections.count,
        processing_rejected_count=max(validated.count - loaded, 0),
        integrity_rejected_count=physics.count,
        rejection_examples=io_rejections.reasons + physics.reasons,
        missing_precursor_count=int(
            np.count_nonzero(np.isnan(np.asarray(precursor_mzs, dtype=np.float64)))
        ),
        precursor_mzs=np.asarray(precursor_mzs, dtype=np.float64),
        retention_times=np.asarray(retention_times, dtype=np.float64),
        adducts=adducts,
        ion_modes=histogram,
        ion_mode_reliable=reliable,
    )


# ---------------------------------------------------------------------------
# Query profiling
# ---------------------------------------------------------------------------


def query_profile_from_spectra(
    source_path: Path,
    spectra: list[Spectrum],
    *,
    loaded_count: Optional[int] = None,
    rejected_count: int = 0,
    io_rejected_count: int = 0,
    processing_rejected_count: int = 0,
    rejection_examples: Optional[list[str]] = None,
) -> QueryProfile:
    """Build a query profile from already-loaded spectra.

    Shared by file loading and the annotation worker (which already has the
    processed query spectra), so both report the same peak-count and precursor
    statistics.
    """
    peak_counts = np.asarray([len(s.peaks.mz) for s in spectra], dtype=np.int64)
    precursor_mzs = np.asarray(
        [_safe_float(s.get("precursor_mz")) for s in spectra], dtype=np.float64
    )
    retention_times = np.asarray(
        [_safe_float(s.get("retention_time")) for s in spectra], dtype=np.float64
    )
    adducts = [
        None if s.get("adduct") in (None, "") else str(s.get("adduct")) for s in spectra
    ]
    return QueryProfile(
        source_path=source_path,
        loaded_count=len(spectra) if loaded_count is None else loaded_count,
        rejected_count=rejected_count,
        io_rejected_count=io_rejected_count,
        processing_rejected_count=processing_rejected_count,
        rejection_examples=list(rejection_examples or []),
        peak_counts=peak_counts,
        precursor_mzs=precursor_mzs,
        retention_times=retention_times,
        adducts=adducts,
    )


def profile_query_file(path: Path, config: MassFlowConfig) -> QueryProfile:
    """Load and process one query file, capturing counts and rejections.

    The rejected count mirrors the annotation pipeline exactly: I/O validation
    rejections plus processing drops (the length delta).
    """
    io_rejections = _RejectionCollector()
    physics_rejections = _RejectionCollector()

    query_gen = io.load_spectra(
        path, file_format=config.input.format, rejection_reporter=io_rejections
    )
    validated = _CountingIterator(query_gen)
    peak_counts: list[int] = []
    precursor_mzs: list[float] = []
    retention_times: list[float] = []
    adducts: list[Optional[str]] = []
    for spectrum in processing.process_spectra(
        validated, config.processing, rejection_reporter=physics_rejections
    ):
        peak_counts.append(len(spectrum.peaks.mz))
        precursor_mzs.append(_safe_float(spectrum.get("precursor_mz")))
        retention_times.append(_safe_float(spectrum.get("retention_time")))
        adduct = spectrum.get("adduct")
        adducts.append(None if adduct in (None, "") else str(adduct))

    processing_drops = max(validated.count - len(peak_counts), 0)
    return QueryProfile(
        source_path=path,
        loaded_count=validated.count,
        rejected_count=io_rejections.count + processing_drops,
        io_rejected_count=io_rejections.count,
        processing_rejected_count=processing_drops,
        rejection_examples=io_rejections.reasons + physics_rejections.reasons,
        peak_counts=np.asarray(peak_counts, dtype=np.int64),
        precursor_mzs=np.asarray(precursor_mzs, dtype=np.float64),
        retention_times=np.asarray(retention_times, dtype=np.float64),
        adducts=adducts,
    )


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def precursor_compatibility(
    library: LibraryProfile,
    query: QueryProfile,
    config: MassFlowConfig,
) -> CompatibilityReport:
    """Compute precursor, adduct-mode, and RT compatibility for one query.

    The inclusive Da/ppm window, missing-precursor bypass, adduct-mode gate,
    and conditional RT gate mirror `SimilarityEngine.search`. Matching is done
    on sorted precursor arrays one query at a time: no full query-reference
    cross-product or scored-spectrum payload is materialized.
    """
    engine = config.similarity.algorithm
    # Only the `cosine` engine enforces an MS1 precursor gate; other engines
    # (e.g. modified_cosine) score across the whole library, so precursor
    # non-overlap is a likelihood factor there, not a definitive exclusion.
    ms1_gate_definitive = engine == "cosine"
    tolerance = config.similarity.ms1_tolerance

    n_queries = query.n_retained
    # `resolution_ppm` only participates in the cosine MS1 gate. For other
    # engines, retain the configured Da window as a descriptive coverage
    # heuristic rather than claiming the ppm window is enforced.
    resolution_ppm = config.similarity.resolution_ppm if engine == "cosine" else None
    rt_tolerance = config.similarity.rt_tolerance
    ref_mzs = library.precursor_mzs
    # `_ms1_prefilter` treats NaN/malformed metadata as missing; a numeric
    # zero is not missing and only participates when it lies in the window.
    ref_missing = np.isnan(ref_mzs)
    ref_valid_indices = np.flatnonzero(~ref_missing)
    sorted_indices = ref_valid_indices[np.argsort(ref_mzs[ref_valid_indices])].astype(
        np.int64, copy=False
    )
    sorted_mzs = ref_mzs[sorted_indices]
    missing_indices = np.flatnonzero(ref_missing)

    ref_adducts = library.adducts
    ref_rts = library.retention_times
    n_queries_with_candidate = 0
    n_queries_with_precursor_candidate = 0
    n_queries_with_compatible_candidate = 0
    n_candidate_pairs = 0
    n_missing_precursor_pairs = 0
    n_candidate_pairs_compatible = 0

    for query_idx in range(n_queries):
        query_mz = query.precursor_mzs[query_idx]
        candidate_indices: np.ndarray = np.empty(0, dtype=np.int64)
        precursor_indices: np.ndarray = np.empty(0, dtype=np.int64)
        if np.isnan(query_mz):
            # Exact search semantics: a missing query precursor bypasses MS1.
            candidate_indices = np.arange(len(ref_mzs), dtype=np.int64)
        else:
            if query_mz > 0.0:
                half_window = (
                    resolution_ppm * query_mz / 1e6
                    if resolution_ppm is not None
                    else tolerance
                )
                left = int(
                    np.searchsorted(sorted_mzs, query_mz - half_window, side="left")
                )
                right = int(
                    np.searchsorted(sorted_mzs, query_mz + half_window, side="right")
                )
                precursor_indices = sorted_indices[left:right]
            # Missing reference precursors bypass the MS1 gate, matching search.
            candidate_indices = np.concatenate((missing_indices, precursor_indices))

        n_candidate_pairs += int(precursor_indices.size)
        n_missing_precursor_pairs += int(missing_indices.size)
        if precursor_indices.size:
            n_queries_with_precursor_candidate += 1
        if candidate_indices.size == 0:
            continue
        n_queries_with_candidate += 1

        query_adduct = (
            query.adducts[query_idx] if query_idx < len(query.adducts) else None
        )
        query_rt = query.retention_times[query_idx]
        compatible_count = 0
        for ref_idx in candidate_indices:
            ref_adduct = ref_adducts[ref_idx] if ref_idx < len(ref_adducts) else None
            if not _adduct_modes_compatible(ref_adduct, query_adduct):
                continue
            if rt_tolerance is not None and ref_idx < len(ref_rts):
                ref_rt = ref_rts[ref_idx]
                # Search applies RT filtering only when both values are present.
                if np.isfinite(query_rt) and np.isfinite(ref_rt):
                    if abs(float(query_rt) - float(ref_rt)) > rt_tolerance:
                        continue
            compatible_count += 1

        if compatible_count:
            n_queries_with_compatible_candidate += 1
            n_candidate_pairs_compatible += compatible_count

    return CompatibilityReport(
        source_path=query.source_path,
        engine=engine,
        ms1_gate_definitive=ms1_gate_definitive,
        tolerance_da=tolerance,
        resolution_ppm=resolution_ppm,
        rt_tolerance=rt_tolerance,
        n_queries=n_queries,
        n_queries_with_candidate=n_queries_with_candidate,
        n_queries_with_precursor_candidate=n_queries_with_precursor_candidate,
        n_queries_with_compatible_candidate=n_queries_with_compatible_candidate,
        n_candidate_pairs=n_candidate_pairs,
        n_missing_precursor_pairs=n_missing_precursor_pairs,
        n_candidate_pairs_compatible=n_candidate_pairs_compatible,
        min_matched_peaks=config.similarity.min_matched_peaks,
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_feasibility_report(
    config: MassFlowConfig, library_spec: Optional[LibrarySpec] = None
) -> FeasibilityReport:
    """Build a complete feasibility report for a configuration."""
    from MassFlow.workflow import _discover_query_inputs

    library = profile_library(config, library_spec=library_spec)

    input_path = Path(config.input.input_path)
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    query_files = _discover_query_inputs(input_path)
    if not query_files:
        raise ValueError(f"No supported spectral files found in {input_path}")

    queries = [profile_query_file(qf, config) for qf in query_files]
    compatibility = [precursor_compatibility(library, q, config) for q in queries]

    findings, recommendations, summary = _assess(
        library, queries, compatibility, config
    )
    return FeasibilityReport(
        library=library,
        queries=queries,
        compatibility=compatibility,
        findings=findings,
        recommendations=recommendations,
        summary=summary,
    )


def _assess(
    library: LibraryProfile,
    queries: list[QueryProfile],
    compatibility: list[CompatibilityReport],
    config: MassFlowConfig,
) -> tuple[list[Finding], list[str], str]:
    """Turn the profiles into definitive/heuristic findings and advice."""
    findings: list[Finding] = []
    recommendations: list[str] = []

    total_queries = sum(c.n_queries for c in compatibility)
    total_retained = sum(q.n_retained for q in queries)
    total_rejected = sum(q.rejected_count for q in queries)
    covered = sum(c.n_queries_with_compatible_candidate for c in compatibility)
    definitive_gate = any(c.ms1_gate_definitive for c in compatibility)
    coverage = (covered / total_queries) if total_queries else 0.0

    # --- Library integrity / statistics --------------------------------
    if library.usable_count == 0:
        findings.append(
            Finding(
                "The reference library contains no usable spectra; no query can match.",
                definitive=True,
            )
        )
        recommendations.append(
            "Check the library file and the processing configuration."
        )
    if library.integrity_rejected_count and library.kind == "file":
        findings.append(
            Finding(
                f"{library.integrity_rejected_count} reference entr"
                f"{'y' if library.integrity_rejected_count == 1 else 'ies'} failed "
                "the strict 5 ppm physical-integrity validation. `massflow annotate` "
                "will abort on this library; `massflow db build` quarantines them.",
                definitive=True,
            )
        )
        recommendations.append(
            "Curate the library with `massflow db build` and search the "
            "resulting store."
        )
    other_library_rejections = (
        library.io_rejected_count
        + library.processing_rejected_count
        - library.integrity_rejected_count
    )
    if other_library_rejections > 0:
        findings.append(
            Finding(
                f"{other_library_rejections} additional reference entries were "
                "rejected during loading or processing; see the rejection examples.",
                definitive=True,
            )
        )

    if 0 < library.usable_count < SMALL_LIBRARY_THRESHOLD:
        findings.append(
            Finding(
                f"Small reference library: {library.usable_count} usable spectra "
                f"(below the ~{SMALL_LIBRARY_THRESHOLD} recommendation). This is "
                "a factual size limit on the available reference candidates.",
                definitive=True,
            )
        )
        findings.append(
            Finding(
                "Target-decoy FDR calibration may be statistically weak for a "
                "library this small; the exact uncertainty depends on the run's "
                "score distribution and decoy competitions.",
                definitive=False,
            )
        )
        if config.similarity.fdr_threshold < 0.1:
            recommendations.append(
                "For calibrated identifications, use a larger relevant library. "
                "Do not relax the FDR threshold just to obtain hits; treat an "
                "empty result as inconclusive when calibration is underpowered."
            )

    # --- Query data quality --------------------------------------------
    if total_rejected:
        io_rejected = sum(q.io_rejected_count for q in queries)
        processing_rejected = sum(q.processing_rejected_count for q in queries)
        findings.append(
            Finding(
                f"{total_rejected} query spectra were rejected/dropped "
                f"({io_rejected} by input validation; {processing_rejected} by "
                "processing). These spectra were not searched.",
                definitive=True,
            )
        )
    min_peaks = config.similarity.min_matched_peaks
    if total_retained and min_peaks > 0:
        sparse = int(sum(int((q.peak_counts < min_peaks).sum()) for q in queries))
        if sparse:
            frac = sparse / total_retained
            classical_peak_gate = config.similarity.algorithm in (
                "cosine",
                "modified_cosine",
            )
            limitation = (
                "cannot meet the configured matched-peak minimum in the "
                "classical cosine engines."
                if classical_peak_gate
                else "may provide insufficient fragment evidence; the configured "
                "engine's matching semantics determine whether this is a hard limit."
            )
            findings.append(
                Finding(
                    f"{sparse} of {total_retained} retained query spectra "
                    f"({frac:.0%}) contain fewer than min_matched_peaks={min_peaks} "
                    f"raw MS2 peaks and {limitation}",
                    definitive=classical_peak_gate,
                )
            )
            if frac >= 0.25:
                recommendations.append(
                    "Review the MS2 peak counts; check the input acquisition and "
                    "any `filter_min_peaks` / `filter_by_intensity` settings."
                )

    # --- Compatibility --------------------------------------------------
    if total_queries and library.usable_count:
        raw_precursor_candidates = sum(
            c.n_queries_with_precursor_candidate for c in compatibility
        )
        any_ms1_candidates = sum(c.n_queries_with_candidate for c in compatibility)
        if covered == 0:
            if (
                definitive_gate
                and raw_precursor_candidates == 0
                and not any_ms1_candidates
            ):
                text = (
                    "No query precursor falls within the configured cosine MS1 "
                    "window of any usable reference (including the query-dependent "
                    "resolution_ppm window, when configured). No target pair can "
                    "pass the precursor gate."
                )
                findings.append(Finding(text, definitive=True))
            elif (
                definitive_gate and raw_precursor_candidates == 0 and any_ms1_candidates
            ):
                findings.append(
                    Finding(
                        "No numeric query/reference precursor-window matches were "
                        "found, but one or more pairs bypassed the cosine MS1 gate "
                        "because query or reference precursor metadata was missing. "
                        "None survived known adduct-mode / RT constraints.",
                        definitive=True,
                    )
                )
            elif definitive_gate:
                findings.append(
                    Finding(
                        "Query precursor-window candidates exist, but all are "
                        "excluded by the configured adduct-mode or retention-time "
                        "filter; no target pair can pass those eligibility gates.",
                        definitive=True,
                    )
                )
            else:
                findings.append(
                    Finding(
                        "No query has a precursor-window candidate that also "
                        "passes known adduct-mode and retention-time constraints. "
                        "This is not a definitive no-match result for "
                        f"{config.similarity.algorithm}: that engine may score "
                        "references outside the precursor window.",
                        definitive=False,
                    )
                )
            recommendations.append("Check reference-library relevance.")
            recommendations.append(
                "Verify precursor / adduct assumptions and the configured "
                "precursor and retention-time constraints."
            )
        elif coverage < 0.5:
            qualifier = (
                "verified cosine-eligible" if definitive_gate else "precursor-window"
            )
            findings.append(
                Finding(
                    f"Limited reference-library coverage: only {covered} of "
                    f"{total_queries} retained query spectra ({coverage:.1%}) have "
                    f"at least one {qualifier} reference after known adduct-mode "
                    "and retention-time constraints.",
                    definitive=definitive_gate,
                )
            )
            recommendations.append("Consider additional reference libraries.")
            recommendations.append(
                "Verify precursor / adduct assumptions and the ms1_tolerance."
            )
        else:
            findings.append(
                Finding(
                    f"{covered} of {total_queries} retained query spectra "
                    f"({coverage:.1%}) have at least one precursor-window "
                    "candidate surviving known adduct-mode and RT constraints. "
                    "This establishes precursor eligibility only, not an MS2 match.",
                    definitive=definitive_gate,
                )
            )

    summary = _summary_line(
        library,
        total_queries,
        covered,
        total_rejected,
        definitive_gate,
        config.similarity.algorithm,
        sum(q.loaded_count for q in queries),
    )
    # De-duplicate recommendations while preserving order.
    seen: set[str] = set()
    unique_recs: list[str] = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recs.append(rec)
    return findings, unique_recs, summary


def _summary_line(
    library: LibraryProfile,
    total_queries: int,
    covered: int,
    total_rejected: int,
    definitive_gate: bool,
    algorithm: str,
    total_loaded: int,
) -> str:
    if library.usable_count == 0:
        return "No usable reference spectra; annotations are impossible."
    if total_queries == 0:
        if total_loaded or total_rejected:
            return (
                f"No query spectra remained for search ({total_loaded} passed "
                f"input validation; {total_rejected} were rejected/dropped)."
            )
        return "No query spectra loaded from the configured input."
    if covered == 0:
        if definitive_gate:
            return "No query has a pair eligible under the cosine precursor/adduct/RT gates."
        return (
            f"No query has a precursor-window candidate, but {algorithm} may score "
            "references outside that window."
        )
    return (
        f"{covered}/{total_queries} query spectra have a precursor-window "
        "candidate after known adduct/RT constraints."
    )


def diagnose_zero_hits(
    *,
    n_queries: int,
    n_rejected: int,
    compatibility: Optional[dict],
    search_diagnostics: Optional[dict],
    pre_fdr_target_hits: int,
    fdr_uncalibrated: bool,
) -> list[str]:
    """Evidence-based explanations for a file that produced zero accepted hits.

    Only states what the supplied counters support. Returns an empty list when
    the pipeline provides no evidence for a specific cause, so the workflow
    never attributes a zero-hit run to a stage it cannot substantiate.
    """
    reasons: list[str] = []
    if n_queries == 0:
        suffix = f" ({n_rejected} rejected)." if n_rejected else "."
        reasons.append("No analyzable query spectra remained after processing" + suffix)
        return reasons

    compat = compatibility or {}
    n_compatible = compat.get("n_queries_with_compatible_candidate")
    definitive = compat.get("ms1_gate_definitive")
    if n_compatible is not None:
        n_precursor_candidates = compat.get(
            "n_queries_with_precursor_candidate",
            compat.get("n_queries_with_candidate", 0),
        )
        n_any_candidates = compat.get("n_queries_with_candidate", 0)
        engine = compat.get("engine", "unknown")
        if n_compatible == 0 and definitive:
            if n_precursor_candidates == 0 and n_any_candidates == 0:
                window = (
                    f"{compat['resolution_ppm']} ppm (query-dependent)"
                    if compat.get("resolution_ppm") is not None
                    else f"{compat.get('tolerance_da')} Da"
                )
                reasons.append(
                    "No query precursor falls within the configured cosine MS1 "
                    f"window ({window}), and no reference-precursor bypass pair "
                    "remains; no target pair can pass the precursor gate."
                )
            elif n_precursor_candidates == 0 and n_any_candidates:
                reasons.append(
                    "No numeric precursor-window match exists; one or more query/"
                    "reference pairs used the missing-precursor bypass but none "
                    "survived adduct-mode / RT filters."
                )
            else:
                reasons.append(
                    "Precursor-window candidates existed, but none survived the "
                    "configured adduct-mode / retention-time eligibility filters."
                )
        elif n_compatible == 0:
            reasons.append(
                f"No query has a precursor-window candidate surviving known "
                f"adduct-mode / retention-time constraints. This is not a "
                f"definitive no-match result for {engine}, which can score "
                "references outside the precursor window."
            )
        elif search_diagnostics:
            n_gated = search_diagnostics.get("n_target_scores_above_gates")
            n_results = search_diagnostics.get("n_target_results")
            if n_gated == 0:
                scored = search_diagnostics.get("n_target_pairs_scored")
                above_score = search_diagnostics.get("n_target_scores_above_min_score")
                counts = (
                    f" The search scored {scored} target pair(s); {above_score} "
                    "reached min_score."
                    if scored is not None and above_score is not None
                    else ""
                )
                reasons.append(
                    f"{n_compatible} queries had a compatible precursor/mode/RT "
                    "candidate, but no actually scored target pair met both "
                    f"min_score and min_matched_peaks.{counts}"
                )
            elif n_results == 0:
                reasons.append(
                    "Some target pairs passed min_score and min_matched_peaks, "
                    "but none survived the subsequent adduct-mode / RT filters."
                )

    if pre_fdr_target_hits > 0:
        note = (
            " (q-values were uncalibrated: the run had no decoy competition)"
            if fdr_uncalibrated
            else ""
        )
        reasons.append(
            f"{pre_fdr_target_hits} candidate hit(s) passed the score and "
            f"matching gates but were removed by FDR filtering{note}."
        )

    if n_rejected:
        reasons.append(
            f"{n_rejected} query spectra were rejected during loading/processing."
        )
    return reasons


def render_library_summary(library: LibraryProfile) -> str:
    """One-line, advisory summary of the reference library for `annotate`."""
    parts = [f"{library.usable_count:,} usable reference spectra"]
    rng = library.precursor_range
    if rng is not None:
        parts.append(f"precursor m/z {rng[0]:.2f}-{rng[1]:.2f}")
    if library.missing_precursor_count:
        parts.append(
            f"{library.missing_precursor_count} reference spectra lack precursor m/z"
        )
    if library.ion_modes:
        modes = ", ".join(f"{k} {v}" for k, v in sorted(library.ion_modes.items()))
        parts.append(f"ion modes: {modes}")
    else:
        parts.append("ion modes not reliably available")
    if library.kind == "file" and library.rejected_count:
        parts.append(f"{library.rejected_count} rejected entries")
        if library.integrity_rejected_count:
            parts.append(
                f"{library.integrity_rejected_count} failed the 5 ppm integrity gate"
            )
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_BAR = "─" * 45


def render_feasibility_report(report: FeasibilityReport) -> str:
    """Render a human-readable preflight report."""
    lib = report.library
    lines: list[str] = ["", "MassFlow Preflight", _BAR, ""]

    lines.append(f"Reference spectra:      {lib.usable_count:,}")
    rng = lib.precursor_range
    if rng is not None:
        lines.append(f"Precursor m/z range:    {rng[0]:.4f} - {rng[1]:.4f}")
    if lib.missing_precursor_count:
        lines.append(
            f"Missing reference precursor m/z: {lib.missing_precursor_count:,} "
            "(cosine MS1 gate bypasses these entries)"
        )
    if lib.ion_modes:
        modes = ", ".join(f"{k} ({v})" for k, v in sorted(lib.ion_modes.items()))
        qualifier = "" if lib.ion_mode_reliable else " (partial)"
        lines.append(f"Ionization modes:       {modes}{qualifier}")
    else:
        lines.append("Ionization modes:       not reliably available")
    if lib.kind == "file" and lib.rejected_count:
        lines.append(f"Reference entries rejected: {lib.rejected_count:,}")
        if lib.integrity_rejected_count:
            lines.append(
                f"  5 ppm integrity rejections: {lib.integrity_rejected_count:,}"
            )
    lines.append("")

    total_queries = sum(c.n_queries for c in report.compatibility)
    total_retained = sum(q.n_retained for q in report.queries)
    total_rejected = sum(q.rejected_count for q in report.queries)
    total_pairs = sum(c.n_candidate_pairs for c in report.compatibility)
    total_covered = sum(
        c.n_queries_with_compatible_candidate for c in report.compatibility
    )
    coverage = (total_covered / total_queries) if total_queries else 0.0

    total_loaded = sum(q.loaded_count for q in report.queries)
    lines.append(f"Spectra passed input validation: {total_loaded:,}")
    lines.append(f"Retained for search:              {total_retained:,}")
    io_rejected = sum(q.io_rejected_count for q in report.queries)
    processing_rejected = sum(q.processing_rejected_count for q in report.queries)
    lines.append(
        f"Rejected/dropped:                   {total_rejected:,} "
        f"(input validation {io_rejected:,}; processing {processing_rejected:,})"
    )
    if report.compatibility:
        first = report.compatibility[0]
        if first.resolution_ppm is not None:
            lines.append(
                f"Precursor window:                  ±{first.resolution_ppm:g} ppm "
                "(query-dependent)"
            )
        elif first.engine == "cosine":
            lines.append(
                f"Precursor window:                  ±{first.tolerance_da:g} Da"
            )
        else:
            lines.append(
                f"Precursor window (coverage only):  ±{first.tolerance_da:g} Da; "
                f"{first.engine} does not enforce this MS1 gate"
            )
        if first.rt_tolerance is not None:
            lines.append(
                f"Retention-time tolerance:          ±{first.rt_tolerance:g} min"
            )
    if total_retained:
        min_peaks = (
            report.compatibility[0].min_matched_peaks if report.compatibility else 0
        )
        sparse = int(
            sum(int((q.peak_counts < min_peaks).sum()) for q in report.queries)
        )
        if min_peaks > 0:
            lines.append(
                f"Queries with <{min_peaks} MS2 peaks: "
                f"{sparse:,} ({sparse / total_retained:.0%})"
            )
    lines.append(
        f"Queries with eligible precursor/mode/RT candidate: "
        f"{total_covered:,} ({coverage:.1%})"
    )
    if total_queries:
        lines.append(f"Pairs within precursor window: {total_pairs:,}")
        missing_pairs = sum(c.n_missing_precursor_pairs for c in report.compatibility)
        if missing_pairs:
            lines.append(f"Missing-precursor bypass pairs: {missing_pairs:,}")
        compatible_pairs = sum(
            c.n_candidate_pairs_compatible for c in report.compatibility
        )
        lines.append(f"Pairs after adduct-mode / RT filters: {compatible_pairs:,}")
    lines.append("")

    lines.append("Assessment:")
    for finding in report.findings:
        tag = "verified" if finding.definitive else "likely"
        lines.append(f"  ({tag}) {finding.text}")
    if report.summary:
        lines.append(f"  -> {report.summary}")
    lines.append("")

    if report.recommendations:
        lines.append("Recommendations:")
        for rec in report.recommendations:
            lines.append(f"  - {rec}")
        lines.append("")

    lines.append(_BAR)
    lines.append(
        "Advisory only. Parameters are not changed and the run is not blocked."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe_float(value: object) -> float:
    """Best-effort float conversion; missing/non-numeric values become NaN."""
    if value is None:
        return float("nan")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


class _CountingIterator:
    """Iterate a sequence while counting the items yielded (mirrors workflow)."""

    def __init__(self, iterable):
        self._iterable = iterable
        self.count = 0

    def __iter__(self):
        for item in self._iterable:
            self.count += 1
            yield item


__all__ = [
    "FeasibilityReport",
    "LibraryProfile",
    "QueryProfile",
    "CompatibilityReport",
    "Finding",
    "SMALL_LIBRARY_THRESHOLD",
    "build_feasibility_report",
    "diagnose_zero_hits",
    "profile_library",
    "profile_open_store",
    "profile_query_file",
    "query_profile_from_spectra",
    "precursor_compatibility",
    "render_feasibility_report",
    "render_library_summary",
]
