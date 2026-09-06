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

Security model
--------------
The streaming server is a **security-sensitive subsystem**: it exposes
an instrument-facing network service that can load reference libraries
and replace the runtime configuration.  The following invariants hold:

* **Bind address**: the default is loopback-only (``127.0.0.1``).  A
  non-loopback bind is an explicit opt-in (``--host 0.0.0.0`` etc.).
* **Transport**: remote binds require TLS (``--tls-cert`` +
  ``--tls-key``).  Insecure remote deployment is refused unless
  ``--allow-insecure-remote`` is passed, which emits a prominent
  warning and an audit log entry.
* **Control plane**: every ``ControlMessage`` requires a valid admin
  token (``--admin-token`` / ``MASSFLOW_ADMIN_TOKEN``).  If no token
  is configured, the control plane is **disabled entirely** (data
  plane unaffected).  Config/library *mutation* commands
  (``SET_CONFIG``, ``LOAD_LIBRARY``) additionally require
  ``--allow-remote-control``.
* **Data plane**: spectrum ingestion is open to any client that can
  reach the port (the server binds loopback by default).  Responses
  are routed strictly back to the connection that sent the packet —
  annotations never cross clients.
* **Audit**: every administrative action (attempted or successful) is
  logged to ``massflow.streaming.security`` with the peer address,
  command, and outcome.
* **Limits**: configurable maximum gRPC message size, a 1 MiB cap on
  control payloads, and a per-packet peak-count cap.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import signal
import socket
import time
import uuid
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
# Dedicated audit logger for administrative actions (control-plane
# operations and security-relevant startup decisions).
audit_logger = logging.getLogger("massflow.streaming.security")

# Default cap on the size of a single control payload (config YAML or
# library path).  Control messages are tiny; anything larger is abuse.
_DEFAULT_MAX_CONTROL_MESSAGE_BYTES: int = 1_048_576  # 1 MiB

# Default cap on fragment peaks per spectrum packet.  A packet above this
# is rejected before it can occupy queue memory.
_DEFAULT_MAX_SPECTRUM_PEAKS: int = 1_000_000

# spectrum_id used in control-plane acknowledgements.
_CONTROL_SPECTRUM_ID = "__control__"


class SecurityConfigurationError(ValueError):
    """Raised when the requested server configuration is unsafe.

    Currently raised for remote (non-loopback) binds without TLS when
    ``allow_insecure_remote`` is not set, and for incomplete TLS
    configurations.
    """


def _is_loopback_host(host: str) -> bool:
    """Return ``True`` when *host* binds to a loopback interface only.

    Handles the conventional names (``localhost``, ``127.0.0.1``,
    ``::1``) plus any hostname that resolves exclusively to loopback
    addresses.  The wildcard addresses ``0.0.0.0`` / ``[::]`` / ``::``
    are never loopback.
    """
    normalized = host.strip().strip("[]")
    if normalized in ("localhost", "127.0.0.1", "::1"):
        return True
    if normalized in ("0.0.0.0", "::", ""):
        return False
    try:
        infos = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    return all(info[4][0] in ("127.0.0.1", "::1") for info in infos)


def _validate_bind_config(
    host: str,
    tls_cert_path: Optional[Path],
    tls_key_path: Optional[Path],
    allow_insecure_remote: bool,
) -> bool:
    """Validate the bind/transport configuration; return whether TLS is on.

    Raises
    ------
    SecurityConfigurationError
        * Remote (non-loopback) bind without TLS when
          ``allow_insecure_remote`` is ``False`` (the safe default).
        * TLS certificate without key (or vice versa).
        * TLS files that do not exist.
    """
    loopback = _is_loopback_host(host)
    tls_configured = tls_cert_path is not None or tls_key_path is not None

    if tls_cert_path is None or tls_key_path is None:
        if tls_configured:
            raise SecurityConfigurationError(
                "TLS requires both --tls-cert and --tls-key."
            )
    else:
        if not tls_cert_path.is_file():
            raise SecurityConfigurationError(
                f"TLS certificate not found: {tls_cert_path}"
            )
        if not tls_key_path.is_file():
            raise SecurityConfigurationError(
                f"TLS private key not found: {tls_key_path}"
            )

    if not loopback and not tls_configured and not allow_insecure_remote:
        raise SecurityConfigurationError(
            f"Refusing to bind {host!r}: remote (non-loopback) binds require "
            "TLS (--tls-cert/--tls-key). To explicitly accept an INSECURE "
            "plaintext remote deployment, pass --allow-insecure-remote."
        )

    return tls_configured


def _extract_bearer_token(context: grpc.aio.ServicerContext) -> Optional[str]:
    """Return the bearer token from the RPC's ``authorization`` metadata."""
    try:
        metadata = context.invocation_metadata()
    except Exception:
        return None
    for key, value in metadata:
        if key.lower() == "authorization":
            token = value.strip()
            if token.lower().startswith("bearer "):
                return token[7:].strip()
            return token
    return None


def _token_matches(presented: Optional[str], expected: str) -> bool:
    """Constant-time token comparison (no timing side channel)."""
    if presented is None:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


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
        admin_token: str | None = None,
        allow_remote_control: bool = False,
        max_control_message_bytes: int = _DEFAULT_MAX_CONTROL_MESSAGE_BYTES,
        max_spectrum_peaks: int = _DEFAULT_MAX_SPECTRUM_PEAKS,
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

        # ── Security configuration ─────────────────────────────────────
        self._admin_token = admin_token
        self._allow_remote_control = allow_remote_control
        self._max_control_message_bytes = max_control_message_bytes
        self._max_spectrum_peaks = max_spectrum_peaks

        # ── Response routing (one shared consumer, per-connection sinks) ──
        # The bounded queue is shared by all connections, but annotations
        # must be delivered to the connection that sent the packet.  Each
        # connection registers a response sink here; packets carry their
        # owning connection id and responses are routed accordingly.
        self._response_sinks: dict[str, asyncio.Queue] = {}
        self._in_flight: dict[str, int] = {}
        self._consumer_task: Optional[asyncio.Task] = None

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

    # ------------------------------------------------------------------
    # Per-connection response routing
    # ------------------------------------------------------------------

    def _register_connection(self) -> tuple[str, asyncio.Queue]:
        """Register a new client connection; return (id, response sink)."""
        connection_id = uuid.uuid4().hex
        self._response_sinks[connection_id] = asyncio.Queue(maxsize=1024)
        self._in_flight[connection_id] = 0
        return connection_id, self._response_sinks[connection_id]

    def _unregister_connection(self, connection_id: str) -> None:
        """Drop a closed connection's sink and in-flight accounting."""
        self._response_sinks.pop(connection_id, None)
        self._in_flight.pop(connection_id, None)

    def _route_response(self, packet: QueuedPacket, response) -> None:
        """Deliver a response to the connection that owns *packet*.

        Responses must never cross connections: if the owning connection is
        gone, the response is dropped with a warning instead of being
        delivered to another client.
        """
        sink = self._response_sinks.get(packet.connection_id)
        if sink is None:
            logger.warning(
                "Dropping response for %s: owning connection %s is gone.",
                packet.spectrum_id,
                packet.connection_id,
            )
            return
        try:
            sink.put_nowait(response)
        except asyncio.QueueFull:
            logger.debug(
                "Response queue full; dropping response for %s.",
                packet.spectrum_id,
            )

    def _decrement_in_flight(self, connection_id: str) -> None:
        if connection_id in self._in_flight:
            self._in_flight[connection_id] = max(self._in_flight[connection_id] - 1, 0)

    # ------------------------------------------------------------------
    # Shared consumer (single task draining the bounded queue)
    # ------------------------------------------------------------------

    def _ensure_consumer(self) -> None:
        """Start the shared consumer task exactly once."""
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._consumer_loop())

    async def _consumer_loop(self) -> None:
        """Drain the shared bounded queue, micro-batch, and route responses
        back to the connection that owns each packet."""
        while True:
            packet = await self._queue.get()
            if packet is None:
                # Poison pill — flush any remaining micro-batch.
                batch = await self._batcher.flush()
                if batch is not None:
                    await self._dispatch_batch(batch[0], batch[1])
                break

            if not self._active.is_set():
                self._queue.task_done()
                self._decrement_in_flight(packet.connection_id)
                continue

            # ── Streaming validation gate ──────────────────────────
            try:
                spectrum = validate_streaming_spectrum(packet, self._config.processing)
            except StreamingValidationError as exc:
                logger.warning(
                    "Validation rejected spectrum %s: %s",
                    packet.spectrum_id,
                    exc,
                )
                self._queue.task_done()
                self._decrement_in_flight(packet.connection_id)
                # Return a structured error to the owning client.
                self._route_response(
                    packet, _build_error_response(packet.spectrum_id, str(exc))
                )
                continue

            # ── Micro-batch accumulation ────────────────────────────
            batch = await self._batcher.add(packet, spectrum)
            self._queue.task_done()

            if batch is not None:
                await self._dispatch_batch(batch[0], batch[1])

    async def _dispatch_batch(
        self,
        packets: list[QueuedPacket],
        spectra: list,
    ) -> None:
        """Run a micro-batch through the engine and route responses."""
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
                self._decrement_in_flight(pkt.connection_id)
                response = _build_error_response(pkt.spectrum_id, "Engine unavailable.")
                response.processing_latency_us = int(batch_latency_us)
                self._route_response(pkt, response)
            return

        batch_latency_us = (time.perf_counter_ns() - t0) / 1e3
        # Feed the rolling latency window so GetStatus reports live
        # processing latency even though task_done() was signalled at
        # dequeue time.
        self._queue.record_latency(
            batch_latency_us / max(len(spectra), 1), count=len(spectra)
        )
        for pkt, result in zip(packets, results):
            self._decrement_in_flight(pkt.connection_id)
            response = self._build_response(result)
            response.processing_latency_us = int(batch_latency_us)
            self._route_response(pkt, response)

    # ── Service RPCs ───────────────────────────────────────────────────

    async def StreamSpectra(
        self,
        request_iterator: AsyncIterator[pb.StreamRequest],  # type: ignore[name-defined, attr-defined]
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.AnnotationResponse]:  # type: ignore[name-defined, attr-defined]
        """Bidirectional streaming RPC.

        Reads ``StreamRequest`` messages from the client, enqueues them
        for processing, and writes ``AnnotationResponse`` messages back
        as they complete.  Responses are routed strictly to the
        connection that sent the packet (see :meth:`_route_response`).
        """
        connection_id, response_queue = self._register_connection()
        self._ensure_consumer()

        async def _request_handler() -> None:
            """Read requests from the client stream."""
            async for request in request_iterator:
                if request.HasField("control"):
                    await self._handle_control(request.control, response_queue, context)
                elif request.HasField("spectrum"):
                    await self._ingest_spectrum(
                        request.spectrum, connection_id, response_queue
                    )
                else:
                    logger.warning("Received empty StreamRequest; ignoring.")

        try:
            request_task = asyncio.create_task(_request_handler())

            # Yield responses as they arrive.  When the client stops sending
            # and the request handler exits, flush the micro-batcher to
            # score any accumulated spectra, then drain this connection's
            # responses.
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
                                await self._dispatch_batch(pending[0], pending[1])
                            _handler_done_flushed = True
                            continue  # Drain responses from the flush.

                        # Exit once this connection's own work is finished:
                        # no more requests, no responses left, and no
                        # in-flight packets owned by this connection.
                        if (
                            response_queue.empty()
                            and self._in_flight.get(connection_id, 0) == 0
                        ):
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
            self._unregister_connection(connection_id)

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
        # Stop the shared consumer once the queue is drained.
        if self._consumer_task is not None and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ingest_spectrum(
        self,
        packet: pb.SpectrumPacket,  # type: ignore[name-defined, attr-defined]
        connection_id: str,
        response_queue: asyncio.Queue[pb.AnnotationResponse | None],  # type: ignore[name-defined, attr-defined]
    ) -> None:
        """Convert a protobuf SpectrumPacket, validate, and enqueue it.

        Basic pre-enqueue checks (empty arrays, mismatched lengths, peak
        count cap) are performed here to avoid clogging the bounded queue
        with obviously invalid or abusive packets.  Deeper structural
        validation (centroiding, peak count, 5-ppm gate) happens in the
        consumer.
        """

        # ── Pre-enqueue sanity checks ──────────────────────────────
        def _reject(reason: str) -> None:
            logger.warning("Rejecting spectrum %s: %s", packet.spectrum_id, reason)
            response = _build_error_response(packet.spectrum_id, reason)
            try:
                response_queue.put_nowait(response)
            except asyncio.QueueFull:
                pass

        if not packet.mz_array or not packet.intensity_array:
            _reject("Empty m/z or intensity array.")
            return

        if len(packet.mz_array) != len(packet.intensity_array):
            _reject(
                "Mismatched m/z and intensity array lengths "
                f"({len(packet.mz_array)} vs {len(packet.intensity_array)})."
            )
            return

        if len(packet.mz_array) > self._max_spectrum_peaks:
            _reject(
                f"Spectrum exceeds the peak-count limit "
                f"({len(packet.mz_array)} > {self._max_spectrum_peaks})."
            )
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
            connection_id=connection_id,
        )

        try:
            enqueued = await self._queue.put(qp, timeout=self._queue_put_timeout)
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
            return

        if enqueued:
            # Track the packet until the shared consumer finishes it, so a
            # closing connection can wait for its own responses.
            self._in_flight[connection_id] = self._in_flight.get(connection_id, 0) + 1
        else:
            # The queue rejected the packet (backpressure timeout / quality
            # gate / shutdown): the client must never be left with a silent
            # drop — an explicit error response is required.
            response = _build_error_response(
                packet.spectrum_id,
                "Server queue full or shutting down; spectrum rejected.",
            )
            try:
                response_queue.put_nowait(response)
            except asyncio.QueueFull:
                pass

    async def _handle_control(
        self,
        ctrl: pb.ControlMessage,  # type: ignore[name-defined, attr-defined]
        response_queue: asyncio.Queue[pb.AnnotationResponse | None],  # type: ignore[name-defined, attr-defined]
        context: grpc.aio.ServicerContext,
    ) -> None:
        """Process an in-stream control message under the security gates.

        Authorization model:

        1. If the server has no admin token configured, the control plane
           is disabled entirely (data plane unaffected).
        2. Every control message must present the configured admin token
           (``authorization: Bearer <token>`` metadata).
        3. Config/library *mutation* commands (``SET_CONFIG``,
           ``LOAD_LIBRARY``) additionally require the server to have been
           started with ``allow_remote_control=True``.
        4. Mutations are applied transactionally: the new config/library
           is fully validated and the engine rebuilt BEFORE the running
           state is swapped.  A failed reload leaves the old state intact.

        Every attempt (accepted or rejected) is recorded in the audit log.
        """
        cmd = ctrl.command
        peer = _peer_of(context)
        token = _extract_bearer_token(context)
        command_name = pb.ControlMessage.Command.Name(cmd)  # type: ignore[name-defined, attr-defined]

        def _reject(reason: str) -> None:
            audit_logger.warning(
                "control rejected peer=%s command=%s reason=%s",
                peer,
                command_name,
                reason,
            )
            response = _build_error_response(_CONTROL_SPECTRUM_ID, reason)
            try:
                response_queue.put_nowait(response)
            except asyncio.QueueFull:
                pass

        def _accept(detail: str = "") -> None:
            audit_logger.info(
                "control accepted peer=%s command=%s detail=%s",
                peer,
                command_name,
                detail,
            )
            response = pb.AnnotationResponse(  # type: ignore[name-defined, attr-defined]
                spectrum_id=_CONTROL_SPECTRUM_ID,
                status="control",
                error_message=detail,
                fdr_q_value=float("nan"),
            )
            try:
                response_queue.put_nowait(response)
            except asyncio.QueueFull:
                pass

        # Gate 1: control plane requires a configured admin token.
        if self._admin_token is None:
            _reject(
                "Control plane disabled: no admin token is configured on "
                "the server (restart with --admin-token)."
            )
            return

        # Gate 2: every control message must present a valid token.
        if not _token_matches(token, self._admin_token):
            _reject(
                "Unauthorized: a valid admin token is required for control operations."
            )
            return

        # Gate 3: config/library mutation requires explicit authorization.
        if (
            cmd
            in (
                pb.ControlMessage.COMMAND_SET_CONFIG,  # type: ignore[name-defined, attr-defined]
                pb.ControlMessage.COMMAND_LOAD_LIBRARY,  # type: ignore[name-defined, attr-defined]
            )
            and not self._allow_remote_control
        ):
            _reject(
                "Remote config/library mutation is disabled on this server "
                "(restart with --allow-remote-control)."
            )
            return

        if cmd == pb.ControlMessage.COMMAND_SET_CONFIG:  # type: ignore[name-defined, attr-defined]
            if not ctrl.config_yaml:
                _reject("SET_CONFIG requires a non-empty config_yaml payload.")
                return
            if len(ctrl.config_yaml) > self._max_control_message_bytes:
                _reject(
                    f"SET_CONFIG payload exceeds the control message limit "
                    f"({len(ctrl.config_yaml)} > "
                    f"{self._max_control_message_bytes} bytes)."
                )
                return
            try:
                new_config = _parse_config_yaml(ctrl.config_yaml)
                # Build the engine from the NEW config BEFORE committing:
                # a failed reload must leave the running state untouched.
                new_engine = _build_engine(new_config, self._top_n)
            except Exception as exc:
                logger.exception("Config reload rejected.")
                _reject(f"SET_CONFIG rejected: {exc}")
                return
            self._config = new_config
            self._engine = new_engine
            _accept("configuration reloaded")

        elif cmd == pb.ControlMessage.COMMAND_LOAD_LIBRARY:  # type: ignore[name-defined, attr-defined]
            library_path = (ctrl.library_path or "").strip()
            if not library_path:
                _reject("LOAD_LIBRARY requires a non-empty library_path.")
                return
            if len(library_path) > self._max_control_message_bytes:
                _reject("LOAD_LIBRARY path exceeds the control message limit.")
                return
            library_file = Path(library_path)
            if not library_file.is_file():
                _reject(f"LOAD_LIBRARY rejected: library not found: {library_path}")
                return
            try:
                new_config = self._config.model_copy(deep=True)
                new_config.input.library_path = library_file
                new_engine = _build_engine(new_config, self._top_n)
            except Exception as exc:
                logger.exception("Library reload rejected.")
                _reject(f"LOAD_LIBRARY rejected: {exc}")
                return
            self._config = new_config
            self._engine = new_engine
            _accept(f"library reloaded from {library_path}")

        elif cmd == pb.ControlMessage.COMMAND_START:  # type: ignore[name-defined, attr-defined]
            self._active.set()
            _accept("server resumed; accepting spectra")

        elif cmd == pb.ControlMessage.COMMAND_STOP:  # type: ignore[name-defined, attr-defined]
            self._active.clear()
            _accept("server paused; spectra will be drained but not annotated")

        else:
            _reject(f"Unknown control command: {command_name} ({cmd})")

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


def _peer_of(context: grpc.aio.ServicerContext) -> str:
    """Best-effort peer address for audit logging."""
    try:
        return str(context.peer())
    except Exception:
        return "unknown"


def _parse_config_yaml(config_yaml: bytes) -> MassFlowConfig:
    """Parse and validate a ``SET_CONFIG`` YAML payload."""
    import yaml

    data = yaml.safe_load(config_yaml)
    if data is None:
        data = {}
    return MassFlowConfig(**data)


def _build_engine(config: MassFlowConfig, top_n: int) -> StreamingEngine:
    """Load the reference library and build a fresh streaming engine.

    Raises if the library cannot be loaded — the caller must treat a
    failed build as a rejected control mutation (transactional apply).
    """
    refs = load_reference_library(config)
    return StreamingEngine(
        config=config,
        reference_spectra=refs,
        top_n=top_n,
    )


# -----------------------------------------------------------------------
# Server bootstrap
# -----------------------------------------------------------------------


async def serve(
    config: MassFlowConfig,
    host: str = "127.0.0.1",
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
    tls_cert_path: Optional[Path] = None,
    tls_key_path: Optional[Path] = None,
    admin_token: Optional[str] = None,
    allow_remote_control: bool = False,
    allow_insecure_remote: bool = False,
    max_message_size_mb: int = 16,
    max_control_message_bytes: int = _DEFAULT_MAX_CONTROL_MESSAGE_BYTES,
    max_spectrum_peaks: int = _DEFAULT_MAX_SPECTRUM_PEAKS,
) -> grpc.aio.Server:
    """Start the async gRPC streaming server.

    Parameters
    ----------
    config : MassFlowConfig
        Validated configuration (reference library path must be set).
    host : str
        Bind address.  Default ``127.0.0.1`` (loopback only).  A
        non-loopback bind is an explicit opt-in and requires TLS (or an
        explicit ``allow_insecure_remote`` override).
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
    tls_cert_path : Path or None
        PEM certificate chain for TLS.  Both ``tls_cert_path`` and
        ``tls_key_path`` are required together.
    tls_key_path : Path or None
        PEM private key for TLS.
    admin_token : str or None
        Bearer token required for ALL control-plane operations.  When
        ``None`` (default) the control plane is disabled entirely.
    allow_remote_control : bool
        Permit config/library *mutation* commands (``SET_CONFIG``,
        ``LOAD_LIBRARY``) from clients that present a valid admin token.
        Default ``False``: even authenticated clients cannot mutate the
        running configuration.
    allow_insecure_remote : bool
        Explicitly allow plaintext (no TLS) non-loopback binds.  Refused
        by default; emits a prominent warning when set.
    max_message_size_mb : int
        Maximum gRPC message size in MiB (receive and send).
    max_control_message_bytes : int
        Maximum size of a control payload (config YAML / library path).
    max_spectrum_peaks : int
        Maximum fragment peaks per spectrum packet.

    Returns
    -------
    grpc.aio.Server
        The running server instance (call ``await server.wait_for_termination()``).

    Raises
    ------
    SecurityConfigurationError
        For unsafe bind/transport combinations (remote bind without TLS,
        incomplete TLS configuration).
    """
    secure = _validate_bind_config(
        host, tls_cert_path, tls_key_path, allow_insecure_remote
    )

    loopback = _is_loopback_host(host)
    if not loopback:
        if secure:
            audit_logger.info(
                "server startup bind=%s:%s transport=TLS remote_bind=yes",
                host,
                port,
            )
        else:
            audit_logger.warning(
                "server startup bind=%s:%s transport=INSECURE remote_bind=yes "
                "(allow_insecure_remote override active) — any host on the "
                "network can connect and stream spectra",
                host,
                port,
            )
            logger.warning(
                "INSECURE REMOTE DEPLOYMENT: gRPC streaming server bound to %s "
                "without TLS. Do not expose this port to untrusted networks.",
                host,
            )
    else:
        audit_logger.info(
            "server startup bind=%s:%s transport=%s remote_bind=no",
            host,
            port,
            "TLS" if secure else "plaintext(loopback)",
        )

    control_state = (
        "disabled (no admin token)"
        if admin_token is None
        else "enabled"
        + (" (mutation allowed)" if allow_remote_control else " (mutation blocked)")
    )
    logger.info(
        "Control plane %s; data plane open on %s:%s.",
        control_state,
        host,
        port,
    )

    max_message_bytes = max_message_size_mb * 1024 * 1024
    server = grpc.aio.server(
        options=[
            ("grpc.max_concurrent_streams", "100"),
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", max_message_bytes),
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
        admin_token=admin_token,
        allow_remote_control=allow_remote_control,
        max_control_message_bytes=max_control_message_bytes,
        max_spectrum_peaks=max_spectrum_peaks,
    )
    pb_grpc.add_MassFlowStreamingServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    if secure:
        cert_bytes = tls_cert_path.read_bytes()  # type: ignore[union-attr]
        key_bytes = tls_key_path.read_bytes()  # type: ignore[union-attr]
        credentials = grpc.ssl_server_credentials([(key_bytes, cert_bytes)])
        server.add_secure_port(listen_addr, credentials)
        logger.info("gRPC streaming server (TLS) listening on %s", listen_addr)
    else:
        server.add_insecure_port(listen_addr)
        logger.info("gRPC streaming server listening on %s", listen_addr)

    await server.start()
    # Attach the servicer to the server so the shutdown handler can
    # access it for graceful draining.
    server._massflow_servicer = servicer  # type: ignore[attr-defined]
    return server


async def run_server(
    config_path: str,
    host: str = "127.0.0.1",
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
    tls_cert_path: Optional[Path] = None,
    tls_key_path: Optional[Path] = None,
    admin_token: Optional[str] = None,
    allow_remote_control: bool = False,
    allow_insecure_remote: bool = False,
    max_message_size_mb: int = 16,
    max_control_message_bytes: int = _DEFAULT_MAX_CONTROL_MESSAGE_BYTES,
    max_spectrum_peaks: int = _DEFAULT_MAX_SPECTRUM_PEAKS,
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
        tls_cert_path=tls_cert_path,
        tls_key_path=tls_key_path,
        admin_token=admin_token,
        allow_remote_control=allow_remote_control,
        allow_insecure_remote=allow_insecure_remote,
        max_message_size_mb=max_message_size_mb,
        max_control_message_bytes=max_control_message_bytes,
        max_spectrum_peaks=max_spectrum_peaks,
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
