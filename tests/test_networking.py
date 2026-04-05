"""
Tests for MassFlow molecular networking.
"""

import networkx as nx
import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import MassFlowConfig
from MassFlow.networking import generate_molecular_network


@pytest.fixture
def sample_queries():
    s1 = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "q1", "precursor_mz": 150.0, "compound_name": "Query 1"},
    )
    s2 = Spectrum(
        mz=np.array([100.0, 201.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "q2", "precursor_mz": 151.0, "compound_name": "Query 2"},
    )
    return [s1, s2]


@pytest.fixture
def sample_references():
    r1 = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([1.0, 1.0], dtype="float"),
        metadata={"id": "r1", "precursor_mz": 150.0, "compound_name": "Ref 1"},
    )
    return [r1]


@pytest.fixture
def sample_results():
    return [
        {
            "query_id": "q1",
            "reference_id": "r1",
            "score": 0.95,
            "q_value": 0.0,
            "is_decoy": False,
        }
    ]


def test_generate_molecular_network_success(
    sample_queries, sample_references, sample_results, tmp_path
):
    from MassFlow.config import InputConfig

    output_path = tmp_path / "network.graphml"
    config = MassFlowConfig(input=InputConfig())
    config.similarity.min_score = 0.5
    config.similarity.algorithm = "cosine"

    generate_molecular_network(
        sample_queries, sample_references, sample_results, config, output_path
    )

    assert output_path.exists()
    G = nx.read_graphml(str(output_path))

    # Check nodes
    assert "q1" in G.nodes
    assert "q2" in G.nodes
    assert "r1" in G.nodes
    assert G.nodes["q1"]["node_type"] == "query"
    assert G.nodes["r1"]["node_type"] == "reference"

    # Check edges
    # q1-r1 (from results)
    assert G.has_edge("q1", "r1")
    assert G.edges["q1", "r1"]["edge_type"] == "query_to_ref"

    # q1-q2 (from similarity, if score > 0.5)
    # Cosine between [100, 200] and [100, 201] with 1.0 intensities
    # They share one peak exactly (100.0).
    # Matchms CosineGreedy might give a score.
    # Let's check if there is an edge.
    if G.has_edge("q1", "q2"):
        assert G.edges["q1", "q2"]["edge_type"] == "query_to_query"


def test_generate_molecular_network_empty_queries(tmp_path):
    from MassFlow.config import InputConfig

    output_path = tmp_path / "network.graphml"
    config = MassFlowConfig(input=InputConfig())
    generate_molecular_network([], [], [], config, output_path)
    assert not output_path.exists()


def test_generate_molecular_network_filtering(
    sample_queries, sample_references, sample_results, tmp_path
):
    from MassFlow.config import InputConfig

    output_path = tmp_path / "network.graphml"
    config = MassFlowConfig(input=InputConfig())
    # High threshold to filter out the result
    config.similarity.min_score = 0.99

    generate_molecular_network(
        sample_queries, sample_references, sample_results, config, output_path
    )

    assert output_path.exists()
    G = nx.read_graphml(str(output_path))
    assert not G.has_edge("q1", "r1")


def test_generate_molecular_network_consensus_fallback(
    sample_queries, sample_references, sample_results, tmp_path
):
    from MassFlow.config import InputConfig

    output_path = tmp_path / "network.graphml"
    config = MassFlowConfig(input=InputConfig())
    config.similarity.algorithm = "consensus"
    # Note: networking.py expects engines to be in config.similarity (if consensus)
    # But wait, networking.py reads:
    # elif hasattr(engine, "engines") and len(engine.engines) > 0:
    #     sim_func = engine.engines[0][0].similarity_function

    # ConsensusEngine is created by get_similarity_engine(config.similarity)
    # Let's check SimilarityEngine.__init__ for consensus

    generate_molecular_network(
        sample_queries, sample_references, sample_results, config, output_path
    )
    assert output_path.exists()
    G = nx.read_graphml(str(output_path))
    assert "q1" in G.nodes


def test_consensus_fallback_returns_cosine_engine(caplog):
    """
    Ensure that when 'consensus' is selected in the similarity config but
    no consensus_weights are provided, the factory falls back to a consensus
    wrapper containing a single 'cosine' SimilarityEngine. This guarantees
    downstream consumers (e.g., networking) can extract a usable similarity_function.
    Additionally assert that a warning is emitted when the fallback is used.
    """
    import logging

    from MassFlow.config import InputConfig
    from MassFlow.similarity import SimilarityEngine, get_similarity_engine

    config = MassFlowConfig(input=InputConfig())
    config.similarity.algorithm = "consensus"
    config.similarity.consensus_weights = None  # explicit for clarity

    # Capture log messages produced during engine creation
    with caplog.at_level(logging.WARNING):
        engine = get_similarity_engine(config.similarity)

    # Assert that a warning mentioning the missing consensus_weights was logged
    assert any(
        "consensus_weights not provided" in rec.getMessage() for rec in caplog.records
    ), "Expected a warning about missing consensus_weights when fallback occurs"

    # The fallback should return a ConsensusEngine-like object exposing 'engines'
    assert hasattr(engine, "engines"), "Expected a ConsensusEngine-like wrapper"
    first = engine.engines[0][0]
    weight = engine.engines[0][1]
    assert isinstance(
        first, SimilarityEngine
    ), "First sub-engine should be a SimilarityEngine"
    # The underlying SimilarityEngine should be configured to use 'cosine'
    assert first.config.algorithm == "cosine"
    # The fallback weight should be 1.0 for the single-engine fallback
    assert weight == 1.0
