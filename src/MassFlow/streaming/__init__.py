"""
MassFlow Streaming – Real-time gRPC spectral annotation.

This sub-package provides a high-performance gRPC streaming service for
live MS2 data ingestion and immediate similarity-based structural
annotation.

Quick start
-----------
::

    # 1. Compile protobuf stubs
    uv run scripts/protoc_gen.sh

    # 2. Start the server
    uv run python -m MassFlow.streaming.server \\
        --config massflow_config.yaml

    # 3. Stream from a mock instrument
    uv run python scripts/mock_instrument_stream.py \\
        --input experiment.mzML --rate 20

Exports
-------
* ``serve`` / ``run_server`` – Bootstrap the gRPC server.
* ``MassFlowStreamingServicer`` – The service implementation.
* ``BoundedQueue`` – Capacity-limited async queue with backpressure.
* ``StreamingEngine`` – Wraps ``SimilarityEngine`` for single-spectrum scoring.
* ``QueuedPacket`` – Internal representation of an ingested spectrum.
* ``QueueStats`` – Live throughput / latency metrics.
"""

from MassFlow.streaming.engine import StreamingEngine, load_reference_library
from MassFlow.streaming.queue import BoundedQueue, QueueStats, QueuedPacket
from MassFlow.streaming.server import MassFlowStreamingServicer, serve, run_server

__all__ = [
    "BoundedQueue",
    "MassFlowStreamingServicer",
    "QueueStats",
    "QueuedPacket",
    "StreamingEngine",
    "load_reference_library",
    "run_server",
    "serve",
]
