"""Benchmark: CSR neighbor-voting propagation on a mock molecular network.

Opt-in via ``-m benchmark`` (see :mod:`tests.benchmarks.conftest`). This exercises
the same custom-CSR one-hop label voting used by the shipped
``scripts/benchmark_network_propagation.py`` on the dataset size the frozen spec
targets (thousands to tens of thousands of nodes), and asserts the vote is
deterministic and preserves direct seed labels so the recorded timings are
meaningful.

Voting is deliberately bounded to direct, fixed seed labels; predictions are never
reused as seeds (this is label *propagation*, not re-ranking — no confidence is
altered).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark_network_propagation import (
    _csr_components,
    _csr_vote,
    _make_csr_adjacency,
    _mock_edges,
    _mock_seed_labels,
)

_NODE_COUNT = 5000
_LABEL_COUNT = 32


@pytest.fixture(scope="module")
def mock_network():
    """A deterministic 5,000-node mock network and its fixed seed labels."""
    source, target, weights = _mock_edges(
        _NODE_COUNT, block_size=100, extra_edges_per_node=4, seed=23
    )
    seed_labels = _mock_seed_labels(
        _NODE_COUNT, label_count=_LABEL_COUNT, seed_fraction=0.15, seed=24
    )
    adjacency = _make_csr_adjacency(_NODE_COUNT, source, target, weights)
    return adjacency, seed_labels


@pytest.mark.benchmark(group="network_traversal")
def test_csr_component_traversal_5000_nodes(benchmark, mock_network) -> None:
    """Time connected-component traversal of the mock network."""
    adjacency, _ = mock_network

    component_sizes = benchmark(_csr_components, adjacency)

    assert sum(component_sizes) == _NODE_COUNT
    assert component_sizes == tuple(sorted(component_sizes, reverse=True))


@pytest.mark.benchmark(group="network_propagation")
def test_csr_neighbor_vote_5000_nodes(benchmark, mock_network) -> None:
    """Time one-hop neighbor-vote propagation and check it is well-behaved."""
    adjacency, seed_labels = mock_network

    predictions = benchmark(_csr_vote, adjacency, seed_labels, _LABEL_COUNT)

    # Determinism: an untimed repeat reproduces the predictions exactly.
    assert np.array_equal(predictions, _csr_vote(adjacency, seed_labels, _LABEL_COUNT))

    assert predictions.shape == (_NODE_COUNT,)
    # Direct seed labels are never overwritten by propagation.
    seeded = seed_labels >= 0
    assert np.array_equal(predictions[seeded], seed_labels[seeded])
    # Propagation actually fired: some previously-unlabelled node was labelled.
    assert np.any(predictions[~seeded] >= 0)
