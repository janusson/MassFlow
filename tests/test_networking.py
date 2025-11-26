import numpy as np
import pytest

pytest.importorskip("matchms", reason="Networking tests depend on matchms.")

from matchms import Spectrum

from yogimass.networking.network import (
    build_network_from_spectra,
    export_network,
)


def _spectrum(peaks, intensities, **metadata):
    return Spectrum(
        mz=np.asarray(peaks, dtype="float32"),
        intensities=np.asarray(intensities, dtype="float32"),
        metadata=metadata,
    )


def test_threshold_network_builds_expected_edges():
    spectra = [
        _spectrum([50, 100], [0.7, 0.3], name="a"),
        _spectrum([50, 100], [0.7, 0.3], name="b"),
        _spectrum([200, 300], [1.0, 0.5], name="c"),
    ]

    nodes, edges = build_network_from_spectra(spectra, metric="cosine", threshold=0.8)

    assert len(nodes) == 3
    assert len(edges) == 1
    edge = edges[0]
    assert {edge.source, edge.target} == {"a", "b"}
    assert edge.similarity > 0.8


def test_knn_network_avoids_duplicates():
    spectra = [
        _spectrum([50], [1.0], name="a"),
        _spectrum([50], [0.9], name="b"),
        _spectrum([60], [1.0], name="c"),
    ]

    nodes, edges = build_network_from_spectra(spectra, metric="cosine", knn=1)

    assert len(nodes) == 3
    # Undirected kNN should not double-count edges
    assert len(edges) <= 3
    pairs = {tuple(sorted((edge.source, edge.target))) for edge in edges}
    assert len(pairs) == len(edges)


def test_exports_csv_and_graphml(tmp_path):
    spectra = [
        _spectrum([50], [1.0], name="a"),
        _spectrum([50], [1.0], name="b"),
    ]
    nodes, edges = build_network_from_spectra(spectra, metric="cosine", threshold=0.1)

    csv_path = tmp_path / "graph.csv"
    export_network(nodes, edges, csv_path)
    assert csv_path.exists()
    csv_contents = csv_path.read_text()
    assert "source,target,similarity,metric" in csv_contents

    try:
        import networkx as nx  # noqa: F401
    except ImportError:
        pytest.skip("networkx not installed; skipping GraphML export test.")

    graphml_path = tmp_path / "graph.graphml"
    export_network(nodes, edges, graphml_path)
    assert graphml_path.exists()
