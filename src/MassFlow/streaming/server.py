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
                              Streaming Validation Gate
                              (Pydantic + matchms filters)
                                       │
                                       ▼
                              BoundedQueue (backpressure)
                                       │
                                       ▼
                              MicroBatcher (time/batch-size)
                                       │
                                       ▼
                              StreamingEngine.annotate_batch()
                                       │
                                       ▼
                              AnnotationResponse
                                       │
                                       ▼
                              response_queue ──> gRPC Client

Key design decisions
--------------------
* A single consumer task drains the bounded queue, validates each
  spectrum, feeds them through the micro-batcher, and dispatches
  batches to ``StreamingEngine``.  This avoids thread-safety issues
  with matchms and keeps latency predictable.
* **Quality-gated backpressure**: when the queue depth reaches a
  configurable high-water mark, incoming low-quality spectra (see
  ``BoundedQueue.compute_packet_quality``) are shed and counted in
  ``ServerStatus.spectra_dropped_low_quality`` so an instrument that
  acquires faster than the engines can score cannot exhaust memory or
  inflate latency.
* The micro-batcher accumulates spectra over a configurable time window
  (default 50 ms) or batch size (default 64) before dispatching,
  amortising the per-call overhead of the scoring machinery.
* Every batch is routed through the ``ConsensusEngine``: the configured
  sub-engines (cosine, modified_cosine, and ML engines when installed)
  are combined into a weighted consensus score for higher-confidence
  real-time annotations.
* Responses are written back to the client on a separate ``asyncio``
  coroutine, so the ingestion path (writer) is never blocked by a
  slow client.
* The server supports in-stream configuration reloads via the
  ``ControlMessage`` envelope, allowing the operator to hot-swap
  reference libraries without restarting.
* Graceful shutdown drains all pending items in the bounded queue
  before terminating the gRPC server, ensuring no spectra are lost
  during controlled restarts.
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
    MicroBatcher,
    StreamingEngine,
    StreamingValidationError,
    load_reference_library,
    validate_streaming_spectrum,
)
from MassFlow.streaming.queue import (
    BoundedQueue,
    OverflowPolicy,
    QueuedPacket,
)

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
    queue_overflow : OverflowPolicy
        Backpressure policy: ``BLOCK`` (suspend producer) or
        ``DROP_OLDEST`` (evict oldest packet).
    queue_put_timeout : float or None
        Maximum seconds to wait for queue space when backpressure is
        active (``overflow=BLOCK``).  ``None`` means block indefinitely.
    high_water_mark : float
        Fraction of queue capacity at which the quality gate engages:
        once reached, incoming low-quality spectra (quality score below
        ``low_quality_threshold``) are dropped to protect buffer space
        for high-quality acquisitions.
    low_quality_threshold : float
        Minimum packet quality required to enqueue above the high-water
        mark.
    top_n : int
        Number of top annotation hits to return per spectrum.
    batch_max_size : int
        Maximum spectra per micro-batch before dispatch.
    batch_timeout_seconds : float
        Maximum wait time (seconds) before dispatching a partial batch.
    """

    def __init__(
        self,
        config: MassFlowConfig,
        queue_capacity: int = 2048,
        queue_overflow: OverflowPolicy = OverflowPolicy.BLOCK,
        queue_put_timeout: float | None = 5.0,
        high_water_mark: float = 0.8,
        low_quality_threshold: float = 0.5,
        top_n: int = 5,
        batch_max_size: int = 64,
        batch_timeout_seconds: float = 0.050,
    ) -> None:
        self._config = config
        self._top_n = top_n
        self._queue_put_timeout = queue_put_timeout
        self._queue = BoundedQueue(
            capacity=queue_capacity,
            overflow=queue_overflow,
            high_water_mark=high_water_mark,
            low_quality_threshold=low_quality_threshold,
        )
        self._batcher = MicroBatcher(
            batch_max_size=batch_max_size,
            batch_timeout_seconds=batch_timeout_seconds,
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
            """Drain the bounded queue, micro-batch, and push results."""
            while True:
                packet = await self._queue.get()
                if packet is None:
                    # Poison pill — flush any remaining micro-batch.
                    batch = await self._batcher.flush()
                    if batch is not None:
                        await _dispatch_batch(batch[0], batch[1])
                    break

                if not self._active.is_set():
                    self._queue.task_done()
                    continue

                # ── Streaming validation gate ──────────────────────────
                try:
                    spectrum = validate_streaming_spectrum(
                        packet, self._config.processing
                    )
                except StreamingValidationError as exc:
                    logger.warning(
                        "Validation rejected spectrum %s: %s",
                        packet.spectrum_id,
                        exc,
                    )
                    self._queue.task_done()
                    # Return a structured error to the client.
                    response = _build_error_response(
                        packet.spectrum_id,
                        str(exc),
                    )
                    try:
                        response_queue.put_nowait(response)
                    except asyncio.QueueFull:
                        pass
                    continue

                # ── Micro-batch accumulation ───────────────────────────
                batch = await self._batcher.add(packet, spectrum)
                self._queue.task_done()

                if batch is not None:
                    await _dispatch_batch(batch[0], batch[1])

        async def _dispatch_batch(
            packets: list[QueuedPacket],
            spectra: list,
        ) -> None:
            """Run a micro-batch through the engine and enqueue responses."""
            if self._engine is None:
                self._init_engine()

            t0 = time.perf_counter_ns()
            try:
                results = self._engine.annotate_batch(packets, spectra)  # type: ignore[union-attr]
            except Exception:
                logger.exception("Batch dispatch failed (%d spectra).", len(spectra))
                batch_latency_us = (time.perf_counter_ns() - t0) / 1e3
                self._queue.record_latency(
                    batch_latency_us / max(len(spectra), 1), count=len(spectra)
                )
                for pkt in packets:
                    response = _build_error_response(
                        pkt.spectrum_id, "Engine unavailable."
                    )
                    response.processing_latency_us = int(batch_latency_us)
                    try:
                        response_queue.put_nowait(response)
                    except asyncio.QueueFull:
                        pass
                return

            batch_latency_us = (time.perf_counter_ns() - t0) / 1e3
            # Feed the rolling latency window so GetStatus reports live
            # processing latency even though task_done() was signalled at
            # dequeue time.
            self._queue.record_latency(
                batch_latency_us / max(len(spectra), 1), count=len(spectra)
            )
            for result in results:
                response = self._build_response(result)
                response.processing_latency_us = int(batch_latency_us)
                try:
                    response_queue.put_nowait(response)
                except asyncio.QueueFull:
                    logger.debug(
                        "Response queue full; dropping response for %s.",
                        result.get("spectrum_id", "?"),
                    )

        async def _request_handler() -> None:
            """Read requests from the client stream."""
            async for request in request_iterator:
                if request.HasField("control"):
                    await self._handle_control(request.control, response_queue)
                elif request.HasField("spectrum"):
                    await self._ingest_spectrum(request.spectrum, response_queue)
                else:
                    logger.warning("Received empty StreamRequest; ignoring.")

        # Start consumer.
        consumer_task = asyncio.create_task(_consumer())

        try:
            request_task = asyncio.create_task(_request_handler())

            # Yield responses as they arrive.  When the client stops sending
            # and the request handler exits, flush the micro-batcher to
            # score any accumulated spectra, then drain responses.
            _handler_done_flushed = False
            while True:
                try:
                    response = await asyncio.wait_for(response_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if request_task.done():
                        if not _handler_done_flushed:
                            # Client stopped sending.  Flush the micro-batcher
                            # so any pending spectra are scored.
                            pending = await self._batcher.flush()
                            if pending is not None:
                                await _dispatch_batch(pending[0], pending[1])
                            _handler_done_flushed = True
                            continue  # Drain responses from the flush.

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
            spectra_dropped_low_quality=stats.total_dropped_low_quality,
            avg_latency_us=stats.avg_latency_us,
            throughput_hz=throughput,
            is_active=self._active.is_set(),
        )

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    async def shutdown(self, drain_timeout: float = 30.0) -> None:
        """Gracefully stop accepting spectra and drain pending items.

        After this call:
        * New ``put()`` calls are rejected.
        * The bounded queue is drained (existing items are processed).
        * A poison pill is sent to unblock consumers.

        Parameters
        ----------
        drain_timeout : float
            Maximum seconds to wait for the queue to drain.
        """
        logger.info(
            "Servicer shutdown requested; draining queue (timeout=%.1f s).",
            drain_timeout,
        )
        self._active.clear()
        remaining = await self._queue.drain(timeout=drain_timeout)
        if remaining > 0:
            logger.warning(
                "Servicer shutdown: %d items could not be drained.", remaining
            )
        else:
            logger.info("Servicer shutdown: all pending items drained.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ingest_spectrum(
        self,
        packet: pb.SpectrumPacket,  # type: ignore[name-defined, attr-defined]
        response_queue: asyncio.Queue[pb.AnnotationResponse | None],  # type: ignore[name-defined, attr-defined]
    ) -> None:
        """Convert a protobuf SpectrumPacket, validate, and enqueue it.

        Basic pre-enqueue checks (empty arrays, NaN precursor_mz) are
        performed here to avoid clogging the bounded queue with
        obviously invalid packets.  Deeper structural validation
        (centroiding, peak count, 5-ppm gate) happens in the consumer.
        """
        # ── Pre-enqueue sanity checks ──────────────────────────────
        if not packet.mz_array or not packet.intensity_array:
            logger.warning(
                "Rejecting spectrum %s: empty peak arrays.", packet.spectrum_id
            )
            response = _build_error_response(
                packet.spectrum_id,
                "Empty m/z or intensity array.",
            )
            try:
                response_queue.put_nowait(response)
            except asyncio.QueueFull:
                pass
            return

        if len(packet.mz_array) != len(packet.intensity_array):
            logger.warning(
                "Rejecting spectrum %s: mismatched array lengths (%d vs %d).",
                packet.spectrum_id,
                len(packet.mz_array),
                len(packet.intensity_array),
            )
            response = _build_error_response(
                packet.spectrum_id,
                "Mismatched m/z and intensity array lengths.",
            )
            try:
                response_queue.put_nowait(response)
            except asyncio.QueueFull:
                pass
            return

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
            await self._queue.put(qp, timeout=self._queue_put_timeout)
        except Exception:
            logger.exception("Failed to enqueue spectrum %s.", packet.spectrum_id)
            # If the queue rejected the packet (e.g. shut down), send an
            # error response so the client is not left hanging.
            response = _build_error_response(
                packet.spectrum_id,
                "Server queue unavailable; spectrum rejected.",
            )
            try:
                response_queue.put_nowait(response)
            except asyncio.QueueFull:
                pass

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
# Helpers
# -----------------------------------------------------------------------


def _build_error_response(
    spectrum_id: str,
    error_message: str,
) -> pb.AnnotationResponse:  # type: ignore[name-defined, attr-defined]
    """Build a protobuf ``AnnotationResponse`` with an error status."""
    return pb.AnnotationResponse(  # type: ignore[name-defined, attr-defined]
        spectrum_id=spectrum_id,
        status="error",
        error_message=error_message,
        fdr_q_value=float("nan"),
        top_hits=[],
    )


# -----------------------------------------------------------------------
# Server bootstrap
# -----------------------------------------------------------------------


async def serve(
    config: MassFlowConfig,
    host: str = "[::]",
    port: int = 50051,
    queue_capacity: int = 2048,
    queue_overflow: OverflowPolicy = OverflowPolicy.BLOCK,
    queue_put_timeout: float | None = 5.0,
    high_water_mark: float = 0.8,
    low_quality_threshold: float = 0.5,
    top_n: int = 5,
    batch_max_size: int = 64,
    batch_timeout_seconds: float = 0.050,
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
    queue_overflow : OverflowPolicy
        Backpressure policy: ``BLOCK`` or ``DROP_OLDEST``.
    queue_put_timeout : float or None
        Maximum seconds to wait for queue space when ``overflow=BLOCK``.
    high_water_mark : float
        Fraction of queue capacity at which low-quality spectra start
        being shed under backpressure.
    low_quality_threshold : float
        Minimum packet quality required to enqueue above the HWM.
    top_n : int
        Number of top annotation hits per spectrum.
    batch_max_size : int
        Maximum spectra per micro-batch.
    batch_timeout_seconds : float
        Max wait before dispatching partial batch.
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
        queue_overflow=queue_overflow,
        queue_put_timeout=queue_put_timeout,
        high_water_mark=high_water_mark,
        low_quality_threshold=low_quality_threshold,
        top_n=top_n,
        batch_max_size=batch_max_size,
        batch_timeout_seconds=batch_timeout_seconds,
    )
    pb_grpc.add_MassFlowStreamingServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("gRPC streaming server listening on %s", listen_addr)

    await server.start()
    # Attach the servicer to the server so the shutdown handler can
    # access it for graceful draining.
    server._massflow_servicer = servicer  # type: ignore[attr-defined]
    return server


async def run_server(
    config_path: str,
    host: str = "[::]",
    port: int = 50051,
    queue_capacity: int = 2048,
    queue_overflow: OverflowPolicy = OverflowPolicy.BLOCK,
    queue_put_timeout: float | None = 5.0,
    high_water_mark: float = 0.8,
    low_quality_threshold: float = 0.5,
    top_n: int = 5,
    batch_max_size: int = 64,
    batch_timeout_seconds: float = 0.050,
    drain_timeout: float = 30.0,
) -> None:
    """Convenience entry-point: load config, start server, wait forever.

    On SIGINT/SIGTERM the bounded queue is drained before the gRPC
    server is stopped, ensuring no pending spectra are lost during a
    controlled shutdown.
    """
    config = MassFlowConfig.from_yaml(config_path)
    server = await serve(
        config,
        host=host,
        port=port,
        queue_capacity=queue_capacity,
        queue_overflow=queue_overflow,
        queue_put_timeout=queue_put_timeout,
        high_water_mark=high_water_mark,
        low_quality_threshold=low_quality_threshold,
        top_n=top_n,
        batch_max_size=batch_max_size,
        batch_timeout_seconds=batch_timeout_seconds,
    )

    # ── Graceful shutdown on SIGTERM / SIGINT ──────────────────────
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    async def _graceful_stop() -> None:
        """Drain the bounded queue, then stop the gRPC server."""
        logger.info("Shutting down gRPC server gracefully...")
        servicer: MassFlowStreamingServicer | None = getattr(
            server, "_massflow_servicer", None
        )
        if servicer is not None:
            await servicer.shutdown(drain_timeout=drain_timeout)
        await server.stop(grace=5.0)
        logger.info("gRPC server stopped.")

    def _signal_handler() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows does not support add_signal_handler for SIGTERM.
            pass

    await stop_event.wait()
    await _graceful_stop()
