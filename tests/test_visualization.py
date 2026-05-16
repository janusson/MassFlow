import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from MassFlow.visualization import visualize_graphml


def test_visualize_graphml_creates_file(tmp_path):
    """Verify that visualize_graphml generates an HTML file from a GraphML."""
    mock_pyvis = MagicMock()
    mock_network_class = MagicMock()

    def mock_save_graph(path):
        Path(path).write_text("<html>vis-network</html>")

    mock_net_instance = MagicMock()
    mock_net_instance.save_graph.side_effect = mock_save_graph
    mock_network_class.return_value = mock_net_instance
    mock_pyvis.network.Network = mock_network_class

    with patch.dict(
        sys.modules, {"pyvis": mock_pyvis, "pyvis.network": mock_pyvis.network}
    ):
        # 1. Create a minimal mock GraphML file
        graph_path = tmp_path / "test_network.graphml"
        G = nx.Graph()
        G.add_node("A", label="Node A", mz=100.0)
        G.add_node("B", label="Node B", mz=200.0)
        G.add_edge("A", "B", weight=0.8)
        nx.write_graphml(G, graph_path)

        output_html = tmp_path / "viz.html"

        # 2. Run visualization
        # We pass notebook=False to ensure it saves to disk
        visualize_graphml(graphml_path=graph_path, output_html=output_html)

        # 3. Check if file exists and has content
        assert output_html.exists()
        content = output_html.read_text()
        assert "html" in content.lower()
        assert "vis-network" in content.lower()


def test_visualize_graphml_missing_file():
    """Verify that visualize_graphml raises FileNotFoundError if input is missing."""
    mock_pyvis = MagicMock()
    with patch.dict(
        sys.modules, {"pyvis": mock_pyvis, "pyvis.network": mock_pyvis.network}
    ):
        with pytest.raises(FileNotFoundError):
            visualize_graphml("non_existent_file.graphml")
