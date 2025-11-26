"""
Quality control and curation utilities for spectral libraries.

Responsibilities:
- Score spectra for basic quality, flag/drop low-quality entries.
- Detect and merge near-duplicates using cosine similarity + precursor tolerance.
- Emit structured decisions, summaries, and merged metadata.

Depends on:
- ``yogimass.similarity.library`` for ``LibraryEntry`` representations.
- ``yogimass.similarity.metrics`` for cosine calculations.
- Logging utilities for traceability.

Avoids:
- File I/O (handled by workflow/reporting).
- CLI/config parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

from yogimass.similarity.library import LibraryEntry
from yogimass.similarity.metrics import cosine_from_vectors
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityThresholds:
    """
    Thresholds controlling when spectra are considered low quality.
    """

    min_peaks: int = 3
    min_total_ion_current: float = 0.05
    max_single_peak_fraction: float = 0.9
    require_precursor_mz: bool = True


@dataclass
class QualityResult:
    """
    Per-spectrum quality assessment.
    """

    identifier: str
    num_peaks: int
    total_ion_current: float
    max_intensity: float
    single_peak_fraction: float
    has_precursor: bool
    issues: list[str]
    quality_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "num_peaks": self.num_peaks,
            "total_ion_current": self.total_ion_current,
            "max_intensity": self.max_intensity,
            "single_peak_fraction": self.single_peak_fraction,
            "has_precursor": self.has_precursor,
            "issues": list(self.issues),
            "quality_score": self.quality_score,
        }


@dataclass
class CurationDecision:
    """
    Action taken on a spectrum during curation.
    """

    identifier: str
    action: Literal["keep", "drop", "merge"]
    reason: str | None = None
    representative: str | None = None
    merged_ids: list[str] | None = None
    quality_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "action": self.action,
            "reason": self.reason,
            "representative": self.representative,
            "merged_ids": list(self.merged_ids) if self.merged_ids else [],
            "quality_score": self.quality_score,
        }


@dataclass
class DuplicateGroup:
    representative: str
    members: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"representative": self.representative, "members": list(self.members)}


@dataclass
class CurationResult:
    curated_entries: list[LibraryEntry]
    decisions: list[CurationDecision]
    quality: dict[str, QualityResult]
    duplicate_groups: list[DuplicateGroup]
    summary: dict[str, Any]


def score_quality(entry: LibraryEntry, thresholds: QualityThresholds) -> QualityResult:
    """
    Score a spectrum using simple heuristics and flag quality issues.
    """
    vector = entry.vector or {}
    num_peaks = len(vector)
    total_ion_current = float(sum(vector.values())) if vector else 0.0
    max_intensity = float(max(vector.values())) if vector else 0.0
    single_peak_fraction = (max_intensity / total_ion_current) if total_ion_current > 0 else 0.0
    has_precursor = entry.precursor_mz is not None

    issues: list[str] = []
    if thresholds.require_precursor_mz and not has_precursor:
        issues.append("missing_precursor_mz")
    if num_peaks < thresholds.min_peaks:
        issues.append("too_few_peaks")
    if total_ion_current < thresholds.min_total_ion_current:
        issues.append("low_total_ion_current")
    if single_peak_fraction > thresholds.max_single_peak_fraction:
        issues.append("single_peak_dominance")

    quality_score = (total_ion_current + 1.0) * (1.0 - single_peak_fraction) * (
        1.0 + num_peaks / max(thresholds.min_peaks, 1)
    )
    if not has_precursor:
        quality_score *= 0.5

    return QualityResult(
        identifier=entry.identifier,
        num_peaks=num_peaks,
        total_ion_current=total_ion_current,
        max_intensity=max_intensity,
        single_peak_fraction=single_peak_fraction,
        has_precursor=has_precursor,
        issues=issues,
        quality_score=quality_score,
    )


def detect_duplicate_groups(
    entries: Sequence[LibraryEntry],
    *,
    precursor_tolerance: float,
    similarity_threshold: float,
) -> list[list[str]]:
    """
    Group near-duplicate spectra based on precursor proximity and cosine similarity.
    """
    adjacency: dict[str, set[str]] = {entry.identifier: set() for entry in entries}
    for idx, left in enumerate(entries):
        if left.precursor_mz is None or not left.vector:
            continue
        for jdx in range(idx + 1, len(entries)):
            right = entries[jdx]
            if right.precursor_mz is None or not right.vector:
                continue
            if abs(left.precursor_mz - right.precursor_mz) > precursor_tolerance:
                continue
            similarity = cosine_from_vectors(left.vector, right.vector)
            if similarity >= similarity_threshold:
                adjacency[left.identifier].add(right.identifier)
                adjacency[right.identifier].add(left.identifier)

    groups: list[list[str]] = []
    visited: set[str] = set()
    for node, neighbors in adjacency.items():
        if node in visited or not neighbors:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(adjacency[current] - visited)
        if len(component) > 1:
            groups.append(sorted(component))
    return groups


def curate_entries(
    entries: Sequence[LibraryEntry],
    *,
    config: Mapping[str, Any] | None = None,
) -> CurationResult:
    """
    Apply quality filtering and de-duplication to a collection of library entries.
    """
    config = config or {}
    thresholds = QualityThresholds(
        min_peaks=int(config.get("min_peaks", config.get("min_peaks_count", 3))),
        min_total_ion_current=float(
            config.get("min_total_ion_current", config.get("min_tic", 0.05))
        ),
        max_single_peak_fraction=float(config.get("max_single_peak_fraction", 0.9)),
        require_precursor_mz=bool(config.get("require_precursor_mz", True)),
    )
    precursor_tolerance = float(config.get("precursor_tolerance", 0.01))
    similarity_threshold = float(config.get("similarity_threshold", 0.95))

    quality_results: dict[str, QualityResult] = {
        entry.identifier: score_quality(entry, thresholds) for entry in entries
    }
    drop_ids: set[str] = set()
    drop_reasons: dict[str, int] = {}
    for result in quality_results.values():
        if result.issues:
            drop_ids.add(result.identifier)
            for issue in result.issues:
                drop_reasons[issue] = drop_reasons.get(issue, 0) + 1

    candidates = [entry for entry in entries if entry.identifier not in drop_ids]
    duplicate_groups_ids = detect_duplicate_groups(
        candidates,
        precursor_tolerance=precursor_tolerance,
        similarity_threshold=similarity_threshold,
    )

    id_to_entry = {entry.identifier: entry for entry in entries}
    curated_entries: list[LibraryEntry] = []
    decisions: list[CurationDecision] = []
    duplicate_groups: list[DuplicateGroup] = []
    handled_ids: set[str] = set()

    for group_ids in duplicate_groups_ids:
        group_entries = [id_to_entry[group_id] for group_id in group_ids if group_id in id_to_entry]
        representative = _choose_representative(group_entries, quality_results)
        merged_ids = [entry.identifier for entry in group_entries if entry.identifier != representative.identifier]
        metadata = dict(representative.metadata)
        if merged_ids:
            metadata["merged_ids"] = sorted([representative.identifier, *merged_ids])
        curated_entries.append(replace(representative, metadata=metadata))
        decisions.append(
            CurationDecision(
                identifier=representative.identifier,
                action="keep",
                representative=representative.identifier,
                merged_ids=sorted(merged_ids),
                quality_score=quality_results[representative.identifier].quality_score,
                reason="representative_duplicate" if merged_ids else None,
            )
        )
        for merged_id in merged_ids:
            decisions.append(
                CurationDecision(
                    identifier=merged_id,
                    action="merge",
                    representative=representative.identifier,
                    reason="duplicate",
                    quality_score=quality_results[merged_id].quality_score,
                )
            )
        duplicate_groups.append(
            DuplicateGroup(representative=representative.identifier, members=sorted(group_ids))
        )
        handled_ids.update(group_ids)

    for entry in entries:
        if entry.identifier in handled_ids or entry.identifier in drop_ids:
            continue
        curated_entries.append(entry)
        decisions.append(
            CurationDecision(
                identifier=entry.identifier,
                action="keep",
                quality_score=quality_results[entry.identifier].quality_score,
            )
        )
        handled_ids.add(entry.identifier)

    for dropped in drop_ids:
        decisions.append(
            CurationDecision(
                identifier=dropped,
                action="drop",
                reason=";".join(quality_results[dropped].issues),
                quality_score=quality_results[dropped].quality_score,
            )
        )

    summary = {
        "total": len(entries),
        "kept": len(curated_entries),
        "dropped": len(drop_ids),
        "merged_groups": len(duplicate_groups),
        "merged_entries": sum(len(group.members) - 1 for group in duplicate_groups),
        "drop_reasons": drop_reasons,
        "precursor_tolerance": precursor_tolerance,
        "similarity_threshold": similarity_threshold,
    }
    logger.info(
        "Curation summary: %d total, %d kept, %d dropped, %d merged groups.",
        summary["total"],
        summary["kept"],
        summary["dropped"],
        summary["merged_groups"],
    )
    return CurationResult(
        curated_entries=curated_entries,
        decisions=decisions,
        quality=quality_results,
        duplicate_groups=duplicate_groups,
        summary=summary,
    )


def _choose_representative(entries: Sequence[LibraryEntry], quality: Mapping[str, QualityResult]) -> LibraryEntry:
    """
    Select the highest-quality entry from a duplicate group.
    """
    if not entries:
        raise ValueError("Cannot choose a representative from an empty group.")
    return max(
        entries,
        key=lambda item: (
            quality[item.identifier].quality_score,
            quality[item.identifier].total_ion_current,
            quality[item.identifier].num_peaks,
            item.identifier,
        ),
    )


__all__ = [
    "CurationDecision",
    "CurationResult",
    "DuplicateGroup",
    "QualityResult",
    "QualityThresholds",
    "curate_entries",
    "detect_duplicate_groups",
    "score_quality",
]
