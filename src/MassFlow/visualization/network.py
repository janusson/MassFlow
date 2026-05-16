import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def visualize_graphml(
    graphml_path: str | Path,
    output_html: str | Path = "network.html",
    notebook: bool = False,
) -> None:
    """
    Visualize a GraphML molecular network interactively.

    This function reads a .graphml file using NetworkX and generates an
    interactive HTML visualization using PyVis. It includes UI controls
    for physics customization and data exploration.

    Args:
        graphml_path: Path to the input .graphml file.
        output_html: Path to save the output interactive HTML file.
        notebook: Set to True if running within a Jupyter Notebook.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.error(
            "The 'pyvis' package is required for network visualization. "
            "Please install it with: uv pip install pyvis networkx"
        )
        raise ImportError("pyvis is not installed.")

    graphml_path = Path(graphml_path)
    output_html = Path(output_html)

    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

    logger.info(f"Loading GraphML network from {graphml_path}")

    # Read the GraphML file
    # NetworkX automatically parses node/edge attributes stored in GraphML
    G = nx.read_graphml(graphml_path)

    logger.info(
        f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    # Initialize PyVis network
    # select_menu and filter_menu add UI panels for exploring data interactively
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        select_menu=True,
        filter_menu=True,
        notebook=notebook,
    )

    # Convert NetworkX graph to PyVis graph
    net.from_nx(G)

    # Enable physics for automatic layout, and expose physics settings
    # to the user in the HTML output for customization.
    net.toggle_physics(True)
    net.show_buttons(filter_=["physics", "nodes", "edges"])

    # Save the interactive network visualization
    logger.info(f"Saving interactive network visualization to {output_html}")

    if notebook:
        return net.show(str(output_html))
    else:
        net.save_graph(str(output_html))
        logger.info(f"Successfully generated: {output_html}")
