"""
Asynchronous gRPC server for real-time mass spectrometry streaming.

This module implements ``MassFlowStreamingServicer``, the server-side
handler for the ``massflow.v1.streaming`` service.  It uses
``grpc.aio`` (built on asyncio) to concurrently ingest spectral
packets from instrument clients and emit annotation responses as soon
as they are computed.

Architecture
------------
::

    gRPC Client ──(bidi stream)──> StreamSpectra handler
                                       │
                                       ▼
                                 StreamRequest ──> BoundedQueue
                                                       │
                                                       ▼
                                              asyncio Worker Task
                                                       │
                                               StreamingEngine.annotate()
                                                       │
                                                       ▼
                                              AnnotationResponse
                                                       │
                                                       ▼
                                                response_queue ──> gRPC Client

Key design decisions
--------------------
* A single consumer task drains the bounded queue and feeds spectra to
  the ``StreamingEngine`` one at a time.  This avoids thread-safety
  issues with matchms and keeps latency predictable.
* Responses are written back to the client on a separate ``asyncio``
  coroutine, so the ingestion path (writer) is never blocked by a
  slow client.
* The server supports in-stream configuration reloads via the
  ``ControlMessage`` envelope, allowing the operator to hot-swap
  reference libraries without restarting.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import grpc
from google.protobuf.empty_pb2 import Empty  # type: ignore[import-untyped]

from MassFlow.config import MassFlowConfig
from MassFlow.streaming.engine import (
    StreamingEngine,
    load_reference_library,
)
from MassFlow.streaming.queue import BoundedQueue, QueuedPacket

logger = logging.getLogger(__name__)

# ── Import generated stubs (created by protoc_gen.sh) ──────────────────
try:
    # Compiled protobuf + gRPC stubs.
    from MassFlow.streaming.generated.massflow.v1 import (
        streaming_pb2 as pb,  # type: ignore[import-untyped]
    )
    from MassFlow.streaming.generated.massflow.v1 import (
        streaming_pb2_grpc as pb_grpc,  # type: ignore[import-untyped]
    )
except ImportError:
    logger.warning(
        "gRPC stubs not found. Run 'scripts/protoc_gen.sh' to generate them."
    )
    raise


class MassFlowStreamingServicer(pb_grpc.MassFlowStreamingServicer):
    """Async gRPC service implementation for live spectral annotation.

    Parameters
    ----------
    config : MassFlowConfig
        The validated MassFlow configuration used to load the reference
        library and configure the similarity engine.
    queue_capacity : int
        Maximum number of spectra to buffer before applying backpressure.
    queue_drop_on_full : bool
        If ``True``, packets are dropped with an error status when the
        queue is full.  If ``False``, the server blocks the gRPC stream
        until space is available.
    top_n : int
        Number of top annotation hits to return per spectrum.
    """

    def __init__(
        self,
        config: MassFlowConfig,
        queue_capacity: int = 2048,
        queue_drop_on_full: bool = False,
        top_n: int = 5,
    ) -> None:
        self._config = config
        self._top_n = top_n
        self._queue = BoundedQueue(
            capacity=queue_capacity, drop_on_full=queue_drop_on_full
        )
        self._engine: Optional[StreamingEngine] = None
        self._start_time_ns = time.time_ns()
        self._active = asyncio.Event()
        self._active.set()  # Accept spectra by default.

        # Build the engine immediately.
        self._init_engine()

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _init_engine(self) -> None:
        """(Re)initialise the similarity engine from the current config."""
        refs = load_reference_library(self._config)
        self._engine = StreamingEngine(
            config=self._config,
            reference_spectra=refs,
            top_n=self._top_n,
        )

    # ── Service RPCs ───────────────────────────────────────────────────

    async def StreamSpectra(
        self,
        request_iterator: AsyncIterator[pb.StreamRequest],  # type: ignore[name-defined, attr-defined]
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.AnnotationResponse]:  # type: ignore[name-defined, attr-defined]
        """Bidirectional streaming RPC.

        Reads ``StreamRequest`` messages from the client, enqueues them
        for processing, and writes ``AnnotationResponse`` messages back
        as they complete.
        """
        # A local queue for responses destined for *this* client connection.
        response_queue: asyncio.Queue[pb.AnnotationResponse | None] = asyncio.Queue(  # type: ignore[name-defined, attr-defined]
            maxsize=1024
        )

        async def _consumer() -> None:
            """Drain the bounded queue and push results to response_queue."""
            while True:
                packet = await self._queue.get()
                if packet is None:
                    break

                if not self._active.is_set():
                    self._queue.task_done()
                    continue

                t0 = time.perf_counter_ns()
                try:
                    if self._engine is None:
                        self._init_engine()

                    result = self._engine.annotate(packet)  # type: ignore[union-attr]
                except Exception:
                    logger.exception(
                        "Consumer worker failed for %s.", packet.spectrum_id
                    )
                    result = {
                        "spectrum_id": packet.spectrum_id,
                        "status": "error",
                        "error_message": "Engine unavailable.",
                        "top_hits": [],
                        "fdr_q_value": float("nan"),
                    }

                latency_us = (time.perf_counter_ns() - t0) / 1e3
                self._queue.task_done(latency_us)

                response = self._build_response(result)
                response.processing_latency_us = int(latency_us)

                try:
                    response_queue.put_nowait(response)
                except asyncio.QueueFull:
                    logger.debug(
                        "Response queue full; dropping response for %s.",
                        packet.spectrum_id,
                    )

        async def _request_handler() -> None:
            """Read requests from the client stream."""
            async for request in request_iterator:
                if request.HasField("control"):
                    await self._handle_control(request.control, response_queue)
                elif request.HasField("spectrum"):
                    await self._ingest_spectrum(request.spectrum)
                else:
                    logger.warning("Received empty StreamRequest; ignoring.")

        # Start consumer.
        consumer_task = asyncio.create_task(_consumer())

        try:
            request_task = asyncio.create_task(_request_handler())

            # Yield responses as they arrive.  When the client stops sending
            # and the request handler exits, wait for the consumer to drain
            # the remaining queue items, then signal completion.
            while True:
                # Use a short timeout so we can check if the request handler
                # has finished and the queue is empty.
                try:
                    response = await asyncio.wait_for(response_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if request_task.done():
                        # Client stopped sending.  Check whether the bounded
                        # queue and response queue are both drained.
                        if self._queue.stats.current_depth == 0:
                            break
                    continue

                if response is None:
                    break
                yield response
                response_queue.task_done()

            await request_task

        except asyncio.CancelledError:
            logger.info("StreamSpectra cancelled; draining.")
        finally:
            if consumer_task and not consumer_task.done():
                consumer_task.cancel()
                try:
                    await consumer_task
                except asyncio.CancelledError:
                    pass

    async def GetStatus(
        self, _request: Empty, _context: grpc.aio.ServicerContext
    ) -> pb.ServerStatus:  # type: ignore[name-defined, attr-defined]
        """Return live health and throughput statistics."""
        stats = self._queue.stats
        elapsed_s = max((time.time_ns() - self._start_time_ns) / 1e9, 1e-9)
        throughput = stats.total_completed / elapsed_s

        return pb.ServerStatus(  # type: ignore[name-defined, attr-defined]
            queue_depth=stats.current_depth,
            spectra_ingested=stats.total_ingested,
            spectra_annotated=stats.total_completed,
            spectra_dropped=stats.total_dropped,
            avg_latency_us=stats.avg_latency_us,
            throughput_hz=throughput,
            is_active=self._active.is_set(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ingest_spectrum(self, packet: pb.SpectrumPacket) -> None:  # type: ignore[name-defined, attr-defined]
        """Convert a protobuf SpectrumPacket and enqueue it."""
        qp = QueuedPacket(
            spectrum_id=packet.spectrum_id,
            mz_array=list(packet.mz_array),
            intensity_array=list(packet.intensity_array),
            precursor_mz=packet.precursor_mz,
            retention_time_seconds=packet.retention_time_seconds,
            charge=packet.charge,
            ion_mode=packet.ion_mode,
            adduct=packet.adduct,
            collision_energy=packet.collision_energy,
            acquisition_timestamp_ns=packet.acquisition_timestamp_ns,
        )

        try:
            await self._queue.put(qp)
        except Exception:
            logger.exception("Failed to enqueue spectrum %s.", packet.spectrum_id)

    async def _handle_control(
        self,
        ctrl: pb.ControlMessage,  # type: ignore[name-defined, attr-defined]
        response_queue: asyncio.Queue[pb.AnnotationResponse | None],  # type: ignore[name-defined, attr-defined]
    ) -> None:
        """Process an in-stream control message."""
        cmd = ctrl.command

        if cmd == pb.ControlMessage.COMMAND_SET_CONFIG:  # type: ignore[name-defined, attr-defined]
            if ctrl.config_yaml:
                logger.info("Reloading configuration from in-stream YAML.")
                try:
                    import yaml

                    data = yaml.safe_load(ctrl.config_yaml)
                    if data is None:
                        data = {}
                    self._config = MassFlowConfig(**data)
                    self._init_engine()
                except Exception:
                    logger.exception("Config reload failed.")

        elif cmd == pb.ControlMessage.COMMAND_LOAD_LIBRARY:  # type: ignore[name-defined, attr-defined]
            if ctrl.library_path:
                logger.info("Reloading reference library from %s.", ctrl.library_path)
                self._config.input.library_path = Path(ctrl.library_path)
                try:
                    self._init_engine()
                except Exception:
                    logger.exception("Library reload failed.")

        elif cmd == pb.ControlMessage.COMMAND_START:  # type: ignore[name-defined, attr-defined]
            self._active.set()
            logger.info("Server resumed; accepting spectra.")

        elif cmd == pb.ControlMessage.COMMAND_STOP:  # type: ignore[name-defined, attr-defined]
            self._active.clear()
            logger.info("Server paused; spectra will be drained but not annotated.")

        else:
            logger.warning("Unknown control command: %s", cmd)

    def _build_response(self, result: dict) -> pb.AnnotationResponse:  # type: ignore[name-defined, attr-defined]
        """Build a protobuf ``AnnotationResponse`` from an engine result dict."""
        hits = []
        for h in result.get("top_hits", []):
            hits.append(
                pb.AnnotationHit(  # type: ignore[name-defined, attr-defined]
                    reference_id=h["reference_id"],
                    reference_name=h.get("reference_name", ""),
                    smiles=h.get("smiles", ""),
                    inchikey=h.get("inchikey", ""),
                    score=h["score"],
                    matched_peaks=h["matched_peaks"],
                    precursor_mz_error_ppm=h.get("precursor_mz_error_ppm", 0.0),
                    annotation_tier=h.get("annotation_tier", ""),
                    structural_similarity=h.get("structural_similarity", 0.0),
                )
            )

        return pb.AnnotationResponse(  # type: ignore[name-defined, attr-defined]
            spectrum_id=result["spectrum_id"],
            status=result["status"],
            error_message=result.get("error_message", ""),
            fdr_q_value=result.get("fdr_q_value", 0.0),
            top_hits=hits,
        )


# -----------------------------------------------------------------------
# Server bootstrap
# -----------------------------------------------------------------------


async def serve(
    config: MassFlowConfig,
    host: str = "[::]",
    port: int = 50051,
    queue_capacity: int = 2048,
    queue_drop_on_full: bool = False,
    top_n: int = 5,
    max_workers: int = 10,
) -> grpc.aio.Server:
    """Start the async gRPC streaming server.

    Parameters
    ----------
    config : MassFlowConfig
        Validated configuration (reference library path must be set).
    host : str
        Bind address.  Default ``[::]`` listens on all IPv4+IPv6 interfaces.
    port : int
        TCP port.  Default 50051.
    queue_capacity : int
        Maximum buffered spectra before backpressure.
    queue_drop_on_full : bool
        Drop packets instead of blocking when the queue is full.
    top_n : int
        Number of top annotation hits per spectrum.
    max_workers : int
        gRPC thread-pool size for non-async work.

    Returns
    -------
    grpc.aio.Server
        The running server instance (call ``await server.wait_for_termination()``).
    """
    server = grpc.aio.server(
        options=[
            ("grpc.max_concurrent_streams", "100"),
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),  # 16 MiB
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.max_pings_without_data", 0),
        ],
    )

    servicer = MassFlowStreamingServicer(
        config=config,
        queue_capacity=queue_capacity,
        queue_drop_on_full=queue_drop_on_full,
        top_n=top_n,
    )
    pb_grpc.add_MassFlowStreamingServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("gRPC streaming server listening on %s", listen_addr)

    await server.start()
    return server


async def run_server(
    config_path: str,
    host: str = "[::]",
    port: int = 50051,
    queue_capacity: int = 2048,
    queue_drop_on_full: bool = False,
    top_n: int = 5,
) -> None:
    """Convenience entry-point: load config, start server, wait forever."""
    config = MassFlowConfig.from_yaml(config_path)
    server = await serve(
        config,
        host=host,
        port=port,
        queue_capacity=queue_capacity,
        queue_drop_on_full=queue_drop_on_full,
        top_n=top_n,
    )

    # Graceful shutdown on SIGTERM / SIGINT.
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Windows does not support add_signal_handler for SIGTERM.
            pass

    await stop_event.wait()
    logger.info("Shutting down gRPC server...")
    await server.stop(grace=5.0)
