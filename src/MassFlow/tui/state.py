"""
Plain-data containers shared by the MassFlow terminal console.

These dataclasses are deliberately free of Textual, Rich, and matchms imports:
they form the vocabulary that the pure helper modules produce and that the
widget layer renders. Keeping them dependency-free means every transformation
of scientific data can be unit-tested without a terminal.

Conventions
-----------
- Missing scientific values are stored as ``None`` or ``float("nan")`` —
  never as ``0.0``, so "absent" is distinguishable from "zero".
- Peak arrays are always ``numpy.float64``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# How many top hits to keep per query by default when the user has not
# specified a limit.
DEFAULT_TOP_N = 10


@dataclass
class SpectrumSummary:
    """Display-ready summary of a single MS/MS spectrum."""

    spectrum_id: str
    precursor_mz: float | None
    retention_time_seconds: float | None
    num_peaks: int
    charge: int | None
    ionmode: str | None
    adduct: str | None
    compound_name: str | None
    base_peak_mz: float | None
    base_peak_intensity: float | None
    total_ion_current: float | None
    spectral_entropy: float | None
    mz_array: np.ndarray
    intensity_array: np.ndarray


@dataclass
class SearchHit:
    """A single annotation hit, normalized from a core ``SearchResult``."""

    query_id: str
    reference_id: str
    reference_name: str | None
    score: float
    matched_peaks: int
    q_value: float | None
    p_value: float | None
    smiles: str | None
    inchikey: str | None
    mass_error_ppm: float | None
    annotation_tier: str | None
    structural_similarity: float | None
    score_breakdown: dict[str, float] | None
    query_precursor_mz: float | None
    reference_precursor_mz: float | None

    @classmethod
    def from_search_result(cls, result: Mapping[str, Any]) -> "SearchHit":
        """Build a :class:`SearchHit` from a core ``SearchResult`` dict.

        Every field is coerced defensively: a malformed value degrades to
        ``None`` (or ``NaN`` for the score) instead of raising, because a
        single bad hit must never crash the results table.
        """

        def optional(key: str, coerce: Any = None) -> Any:
            value = result.get(key)
            if value is None:
                return None
            if coerce is None:
                return value
            try:
                coerced = coerce(value)
            except (TypeError, ValueError):
                return None
            if isinstance(coerced, float) and math.isnan(coerced):
                return None
            return coerced

        score = optional("score", float)
        if score is None:
            score = math.nan

        breakdown = result.get("score_breakdown")
        if breakdown is not None and not isinstance(breakdown, dict):
            breakdown = None

        return cls(
            query_id=str(result.get("query_id") or ""),
            reference_id=str(result.get("reference_id") or ""),
            reference_name=optional("reference_name"),
            score=float(score),
            matched_peaks=int(optional("matched_peaks", int) or 0),
            q_value=optional("q_value", float),
            p_value=optional("p_value", float),
            smiles=optional("smiles"),
            inchikey=optional("inchikey"),
            mass_error_ppm=optional("mass_error_ppm", float),
            annotation_tier=optional("annotation_tier"),
            structural_similarity=optional("structural_similarity", float),
            score_breakdown=breakdown,
            query_precursor_mz=optional("query_precursor_mz", float),
            reference_precursor_mz=optional("reference_precursor_mz", float),
        )


@dataclass(frozen=True)
class IdentificationRequest:
    """Immutable description of an identification run.

    Pure data: the :mod:`MassFlow.tui.pipeline` module turns it into engine
    configuration and executes the search.
    """

    query_path: Path
    library_path: Path
    algorithm: str = "modified_cosine"
    ms1_tolerance: float = 0.02
    ms2_tolerance: float = 0.02
    min_score: float = 0.6
    min_matched_peaks: int = 3
    fdr_threshold: float = 0.05
    top_n: int = DEFAULT_TOP_N
    max_query_spectra: int = 500


@dataclass
class IdentificationOutcome:
    """Result of a completed identification run."""

    request: IdentificationRequest
    engine_used: str
    hits: list[SearchHit]
    num_queries: int
    num_references: int
    duration_seconds: float
    fdr_threshold: float
    warnings: list[str] = field(default_factory=list)
    # Peak arrays for rendering mirror plots without re-reading the library:
    # query_id -> (mz, intensities) and reference_id -> (mz, intensities).
    query_peaks: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    hit_reference_peaks: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )

    @property
    def num_hits(self) -> int:
        return len(self.hits)

    @property
    def queries_with_hits(self) -> int:
        return len({hit.query_id for hit in self.hits})


@dataclass
class QueryLoadResult:
    """Preview of a loaded experimental file."""

    path: Path
    format_hint: str
    summaries: list[SpectrumSummary]
    quarantined_messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class LibraryInfo:
    """Lightweight census of a spectral library."""

    path: Path
    backend: str
    total_spectra: int | None
    categories: dict[str, int] = field(default_factory=dict)
    precursor_mz_range: tuple[float, float] | None = None
    error: str | None = None
    truncated: bool = False
