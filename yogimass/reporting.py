"""
Lightweight reporting helpers for Yogimass workflows.

Responsibilities:
- Serialize search results, network summaries, and curation/QC reports to JSON/CSV/text.
- Provide simple data classes for search matches.
- Build coarse summaries (e.g., quality histograms) without plotting dependencies.

Depends on:
- ``yogimass.networking.network`` for node/edge structures.
- ``yogimass.curation`` for QC results.
- Standard library JSON/CSV and logging.

Avoids:
- Running workflows or modifying core data; focuses on read/format/write.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from yogimass.curation import CurationResult, QualityResult
from yogimass.networking.network import SimilarityEdge, SpectrumNode
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchMatch:
    """
    Flattened representation of a library search hit for reporting.
    """

    query_id: str
    hit_id: str
    score: float
    precursor_mz: float | None
    metadata: dict[str, Any]


def write_search_results(
    results: Sequence[SearchMatch], output_path: str | Path
) -> Path:
    """
    Write search results to CSV or JSON. Metadata is JSON-encoded for CSV output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    rows = [
        {
            "query_id": result.query_id,
            "hit_id": result.hit_id,
            "score": float(result.score),
            "precursor_mz": result.precursor_mz,
            "metadata": result.metadata,
        }
        for result in results
    ]
    if suffix in {".json", ".jsn"}:
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "query_id",
                "hit_id",
                "score",
                "precursor_mz",
                "metadata_json",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "query_id": row["query_id"],
                        "hit_id": row["hit_id"],
                        "score": f"{row['score']:.6f}",
                        "precursor_mz": row["precursor_mz"]
                        if row["precursor_mz"] is not None
                        else "",
                        "metadata_json": json.dumps(row["metadata"]),
                    }
                )
    logger.info("Wrote %d search hits to %s", len(results), output_path)
    return output_path


def summarize_network(
    nodes: Sequence[SpectrumNode], edges: Sequence[SimilarityEdge]
) -> dict[str, Any]:
    """
    Basic network summary: counts and simple degree statistics.
    """
    degree: dict[str, int] = {node.identifier: 0 for node in nodes}
    for edge in edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1

    degrees = list(degree.values()) if degree else []
    summary = {
        "nodes": len(nodes),
        "edges": len(edges),
    }
    if degrees:
        summary["degree"] = {
            "min": min(degrees),
            "max": max(degrees),
            "mean": sum(degrees) / len(degrees),
        }
    return summary


def write_network_summary(summary: Mapping[str, Any], output_path: str | Path) -> Path:
    """
    Persist a network summary to JSON or plain text.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".json", ".jsn"}:
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        lines = []
        for key, value in summary.items():
            if isinstance(value, Mapping):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            else:
                lines.append(f"{key}: {value}")
        output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote network summary to %s", output_path)
    return output_path


def summarize_quality_distributions(
    quality: Mapping[str, QualityResult],
) -> dict[str, Mapping[str, int]]:
    """
    Build coarse histograms for quality metrics to avoid plotting dependencies.
    """
    num_peaks = [result.num_peaks for result in quality.values()]
    total_ion_current = [result.total_ion_current for result in quality.values()]
    single_peak_fraction = [result.single_peak_fraction for result in quality.values()]
    return {
        "num_peaks": _bucketize(num_peaks, [0, 2, 5, 10]),
        "total_ion_current": _bucketize(
            total_ion_current, [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        ),
        "single_peak_fraction": _bucketize(
            single_peak_fraction, [0.25, 0.5, 0.75, 0.9]
        ),
    }


def write_curation_report(result: CurationResult, output_path: str | Path) -> Path:
    """
    Persist a curation/QC report as JSON or CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality_distributions = summarize_quality_distributions(result.quality)
    summary = dict(result.summary)
    summary["quality"] = quality_distributions

    if output_path.suffix.lower() in {".json", ".jsn"}:
        payload = {
            "summary": summary,
            "actions": [decision.to_dict() for decision in result.decisions],
            "merged": [group.to_dict() for group in result.duplicate_groups],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        fieldnames = [
            "identifier",
            "action",
            "reason",
            "representative",
            "merged_ids",
            "num_peaks",
            "total_ion_current",
            "single_peak_fraction",
            "quality_score",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for decision in result.decisions:
                quality = result.quality.get(decision.identifier)
                writer.writerow(
                    {
                        "identifier": decision.identifier,
                        "action": decision.action,
                        "reason": decision.reason or "",
                        "representative": decision.representative or "",
                        "merged_ids": ",".join(decision.merged_ids or []),
                        "num_peaks": quality.num_peaks if quality else "",
                        "total_ion_current": f"{quality.total_ion_current:.4f}"
                        if quality
                        else "",
                        "single_peak_fraction": f"{quality.single_peak_fraction:.4f}"
                        if quality
                        else "",
                        "quality_score": f"{quality.quality_score:.4f}"
                        if quality
                        else "",
                    }
                )
    logger.info("Wrote curation report to %s", output_path)
    return output_path


def _bucketize(values: Sequence[float], edges: Sequence[float]) -> dict[str, int]:
    """
    Count values into buckets defined by increasing edges.
    """
    buckets: dict[str, int] = {}
    sorted_edges = list(edges)
    labels: list[str] = []
    for idx, edge in enumerate(sorted_edges):
        low = sorted_edges[idx - 1] if idx > 0 else None
        label = f"<= {edge}" if low is None else f"{low} - {edge}"
        labels.append(label)
        buckets[label] = 0
    labels.append(f"> {sorted_edges[-1]}")
    buckets[labels[-1]] = 0

    for value in values:
        placed = False
        for edge, label in zip(sorted_edges, labels):
            if value <= edge:
                buckets[label] += 1
                placed = True
                break
        if not placed:
            buckets[labels[-1]] += 1
    return buckets


__all__ = [
    "SearchMatch",
    "summarize_network",
    "summarize_quality_distributions",
    "write_curation_report",
    "write_network_summary",
    "write_search_results",
]
