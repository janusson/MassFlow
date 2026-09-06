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

Security
--------
The server defaults to a loopback-only bind (``127.0.0.1``), no TLS on
loopback, a disabled control plane (no admin token configured), and remote
config mutation blocked. Remote binds require TLS (or an explicit
``--allow-insecure-remote`` override, which logs a prominent warning).
Control operations require ``--admin-token``; config/library mutation
additionally requires ``--allow-remote-control``. Every administrative
action is audit-logged to ``massflow.streaming.security``. See
``MassFlow.streaming.server`` for the full security model and
``tests/test_streaming_security.py`` for the regression suite.
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
