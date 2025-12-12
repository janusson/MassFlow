"""
Helpers to export networks for downstream tools (e.g., Cytoscape/Gephi).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from yogimass.networking.network import SimilarityEdge, SpectrumNode


def export_network_for_cytoscape(
    nodes: Sequence[SpectrumNode],
    edges: Sequence[SimilarityEdge],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Write Cytoscape-friendly node and edge tables (CSV).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    node_path = out_dir / "nodes.csv"
    edge_path = out_dir / "edges.csv"

    # Collect metadata keys to emit stable columns.
    metadata_keys: set[str] = set()
    for node in nodes:
        metadata_keys.update(k for k, v in node.metadata.items() if v is not None)

    node_fieldnames = ["id", "label", "precursor_mz"] + [f"meta_{k}" for k in sorted(metadata_keys)]
    with node_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=node_fieldnames)
        writer.writeheader()
        for node in nodes:
            row = {
                "id": node.identifier,
                "label": node.metadata.get("name") or node.identifier,
                "precursor_mz": node.precursor_mz if node.precursor_mz is not None else "",
            }
            for key in metadata_keys:
                value = node.metadata.get(key)
                if value is not None:
                    row[f"meta_{key}"] = value
            writer.writerow(row)

    with edge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "target", "similarity", "metric"])
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "similarity": edge.similarity,
                    "metric": edge.metric,
                }
            )

    return node_path, edge_path
