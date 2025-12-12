import csv
from pathlib import Path

import pytest

pytest.importorskip("matchms", reason="Network export tests require matchms.")

from matchms import Spectrum

from yogimass.cli import main as cli_main
from yogimass.networking.exporters import export_network_for_cytoscape
from yogimass.networking.network import SimilarityEdge, SpectrumNode
from yogimass.similarity.library import LocalSpectralLibrary


def _spectrum(mz, intensities, **metadata):
    import numpy as np

    return Spectrum(
        mz=np.asarray(mz, dtype="float64"),
        intensities=np.asarray(intensities, dtype="float64"),
        metadata=metadata,
    )


def test_export_network_for_cytoscape_writes_tables(tmp_path):
    nodes = [
        SpectrumNode("n1", 100.0, {"name": "a"}, spectrum=None, vector=None),
        SpectrumNode("n2", 200.0, {}, spectrum=None, vector=None),
    ]
    edges = [SimilarityEdge("n1", "n2", 0.9, "cosine")]
    node_path, edge_path = export_network_for_cytoscape(nodes, edges, tmp_path)

    with node_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with edge_path.open("r", encoding="utf-8") as handle:
        edge_rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert {"id", "label", "precursor_mz"}.issubset(rows[0].keys())
    assert len(edge_rows) == 1
    assert edge_rows[0]["source"] == "n1"


def test_network_export_cli_with_library_metadata(tmp_path):
    lib_path = tmp_path / "lib.json"
    lib = LocalSpectralLibrary(lib_path)
    lib.add_spectrum(_spectrum([100.0], [1.0], name="A"), identifier="s1", overwrite=True)
    lib.add_spectrum(_spectrum([200.0], [2.0], name="B"), identifier="s2", overwrite=True)

    edges_path = tmp_path / "network.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "target", "similarity", "metric"])
        writer.writeheader()
        writer.writerow({"source": "s1", "target": "s2", "similarity": 0.8, "metric": "cosine"})

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
input:
  path: tests/data/simple.mgf
  format: mgf
library:
  path: {lib_path}
  build: false
network:
  enabled: true
  output: {edges_path}
  knn: 1
        """,
        encoding="utf-8",
    )

    out_dir = tmp_path / "cyto"
    rc = cli_main(
        [
            "network",
            "export-cytoscape",
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    node_csv = out_dir / "nodes.csv"
    edge_csv = out_dir / "edges.csv"
    assert node_csv.exists()
    assert edge_csv.exists()
    nodes = list(csv.DictReader(node_csv.open("r", encoding="utf-8")))
    assert len(nodes) == 2
    assert {"id", "label", "precursor_mz"}.issubset(nodes[0].keys())
