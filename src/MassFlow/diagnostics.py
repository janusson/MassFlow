"""Typed, bounded diagnostic records for observable search-stage outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DiagnosticReason(StrEnum):
    """Stable reason codes for observed spectrum and candidate rejection."""

    INPUT_INVALID = "input_invalid"
    PRECURSOR_MISSING = "precursor_missing"
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

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe event representation."""
        return {
            "stage": self.stage.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "spectrum_id": self.spectrum_id,
        }


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
