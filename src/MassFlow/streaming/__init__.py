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
* ``OverflowPolicy`` – Backpressure policy enum (BLOCK / DROP_OLDEST).
* ``MicroBatcher`` – Time/batch-size accumulator for amortised scoring.
* ``StreamingEngine`` – Wraps ``SimilarityEngine`` for real-time scoring.
* ``StreamingValidationError`` – Raised when a spectrum fails the validation gate.
* ``validate_streaming_spectrum`` – Pre-scoring validation + peak filtering gate.
* ``QueuedPacket`` – Internal representation of an ingested spectrum.
* ``QueueStats`` – Live throughput / latency metrics.
"""

from MassFlow.streaming.engine import (
    MicroBatcher,
    StreamingEngine,
    StreamingValidationError,
    load_reference_library,
    validate_streaming_spectrum,
)
from MassFlow.streaming.queue import (
    BoundedQueue,
    OverflowPolicy,
    QueueStats,
    QueuedPacket,
)
from MassFlow.streaming.server import (
    MassFlowStreamingServicer,
    run_server,
    serve,
)

__all__ = [
    "BoundedQueue",
    "MassFlowStreamingServicer",
    "MicroBatcher",
    "OverflowPolicy",
    "QueueStats",
    "QueuedPacket",
    "StreamingEngine",
    "StreamingValidationError",
    "load_reference_library",
    "run_server",
    "serve",
    "validate_streaming_spectrum",
]
