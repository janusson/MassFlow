"""Typed, bounded diagnostic records for observable search-stage outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DiagnosticReason(StrEnum):
    """Stable reason codes for observed spectrum and candidate rejection."""

    INPUT_INVALID = "input_invalid"
    PRECURSOR_MISSING = "precursor_missing"
    PRECURSOR_NAN = "precursor_nan"
    PRECURSOR_POSITIVE_INFINITY = "precursor_positive_infinity"
    PRECURSOR_NEGATIVE_INFINITY = "precursor_negative_infinity"
    PRECURSOR_NON_POSITIVE = "precursor_non_positive"
    PRECURSOR_NON_NUMERIC = "precursor_non_numeric"
    PHYSICAL_INTEGRITY_REJECTED = "physical_integrity_rejected"
    INSUFFICIENT_PEAKS = "insufficient_peaks"
    PROCESSING_REJECTED_UNKNOWN = "processing_rejected_unknown"
    NO_PRECURSOR_CANDIDATES = "no_precursor_candidates"
    PREFILTER_REJECTED = "prefilter_rejected"
    SCORE_BELOW_THRESHOLD = "score_below_threshold"
    INSUFFICIENT_FRAGMENT_MATCHES = "insufficient_fragment_matches"
    POLARITY_INCOMPATIBLE = "polarity_incompatible"
    RT_OUTSIDE_TOLERANCE = "rt_outside_tolerance"
    CANDIDATE_RANKED_BELOW_LIMIT = "candidate_ranked_below_limit"
    CONFIDENCE_UNAVAILABLE = "confidence_unavailable"
    CONFIDENCE_FILTER_REJECTED = "confidence_filter_rejected"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"
    CANDIDATE_RETURNED = "candidate_returned"
    UNKNOWN = "unknown"


class DiagnosticStage(StrEnum):
    """Pipeline stage at which an observed rejection occurred."""

    INPUT_VALIDATION = "input_validation"
    METADATA_PROCESSING = "metadata_processing"
    PHYSICAL_INTEGRITY = "physical_integrity"
    PEAK_PROCESSING = "peak_processing"


@dataclass(frozen=True)
class RejectionEvent:
    """One observed spectrum-level rejection, retaining its source detail."""

    stage: DiagnosticStage
    reason: DiagnosticReason
    detail: str
    spectrum_id: str | None = None
    source_path: str | None = None
    source_ordinal: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe event representation."""
        payload: dict[str, Any] = {
            "stage": self.stage.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "spectrum_id": self.spectrum_id,
        }
        # Leave non-ingestion diagnostic payloads backwards compatible.
        if self.source_path is not None:
            payload["source_path"] = self.source_path
        if self.source_ordinal is not None:
            payload["source_ordinal"] = self.source_ordinal
        return payload


@dataclass
class RejectionSummary:
    """Aggregate rejections without retaining an unbounded event trace."""

    sample_limit: int = 20
    reason_counts: dict[str, int] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: RejectionEvent) -> None:
        """Count a rejection and retain only the first bounded sample events."""
        reason = event.reason.value
        stage = event.stage.value
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1
        if len(self.samples) < self.sample_limit:
            self.samples.append(event.as_dict())

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic, JSON-safe aggregate diagnostics."""
        return {
            "total": sum(self.reason_counts.values()),
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "stage_counts": dict(sorted(self.stage_counts.items())),
            "sample_limit": self.sample_limit,
            "samples": list(self.samples),
        }


def raw_rejection_category(event: RejectionEvent) -> str:
    """Return a bounded-cardinality grouping key for raw-ingestion rejections."""
    if event.reason is not DiagnosticReason.INPUT_INVALID:
        return event.reason.value
    # Preserve known legacy labels without retaining arbitrary source values
    # in dictionary keys; record-level detail is kept in the JSONL ledger.
    if event.detail.startswith("Mismatched array lengths ("):
        return "Mismatched array lengths"
    stable_labels = {
        "Duplicate fragment m/z values",
        "Empty peak arrays",
        "M/Z values are not monotonically increasing",
        "Contains non-positive intensity values",
    }
    return event.detail if event.detail in stable_labels else "input_invalid"
