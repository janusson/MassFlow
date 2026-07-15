"""
Bounded queue with backpressure for real-time spectral ingestion.

This module provides a thread-safe, capacity-limited ``asyncio.Queue``
wrapper that implements controlled backpressure for gRPC spectral
streaming.  When the queue is full and backpressure is enabled,
``put()`` blocks until space is available, preventing memory
exhaustion during acquisition bursts.  Alternatively, a ``put_nowait``
variant raises ``QueueFull`` so the server can choose to drop packets
with a client-visible warning.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class QueueFull(Exception):
    """Raised when the bounded queue is at capacity and drop-on-full is set."""


@dataclass
class QueueStats:
    """Live statistics for a ``BoundedQueue`` instance."""

    current_depth: int = 0
    total_ingested: int = 0
    total_dropped: int = 0
    total_completed: int = 0
    _latency_samples: list[float] = field(default_factory=list)

    @property
    def avg_latency_us(self) -> float:
        """Rolling average processing latency in microseconds."""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    def record_latency(self, latency_us: float) -> None:
        """Append a latency sample, keeping a sliding window of 1000."""
        self._latency_samples.append(latency_us)
        if len(self._latency_samples) > 1000:
            self._latency_samples.pop(0)


@dataclass
class QueuedPacket:
    """A spectrum packet together with its arrival timestamp."""

    spectrum_id: str
    mz_array: list[float]
    intensity_array: list[float]
    precursor_mz: float
    retention_time_seconds: float
    charge: int
    ion_mode: str
    adduct: str
    collision_energy: float
    acquisition_timestamp_ns: int
    enqueue_time_ns: int = field(default_factory=time.time_ns)


class BoundedQueue:
    """Capacity-limited async queue with backpressure and drop semantics.

    Parameters
    ----------
    capacity : int
        Maximum number of queued spectra.  When exceeded the behaviour
        depends on ``drop_on_full``.
    drop_on_full : bool
        If ``True``, ``put_nowait`` raises ``QueueFull`` when the queue
        is at capacity.  If ``False``, ``put`` blocks until space is
        available (backpressure).
    """

    def __init__(self, capacity: int = 2048, drop_on_full: bool = False) -> None:
        if capacity <= 0:
            raise ValueError("Queue capacity must be positive.")
        self._capacity = capacity
        self._drop_on_full = drop_on_full
        self._queue: asyncio.Queue[QueuedPacket | None] = asyncio.Queue(
            maxsize=capacity
        )
        self._stats = QueueStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stats(self) -> QueueStats:
        """Read-only snapshot of queue statistics."""
        return self._stats

    @property
    def is_full(self) -> bool:
        return self._queue.qsize() >= self._capacity

    async def put(self, packet: QueuedPacket) -> None:
        """Enqueue a packet, blocking if the queue is full (backpressure).

        This is the recommended method for typical streaming where
        dropping packets is unacceptable.  The caller coroutine is
        suspended until a consumer drains enough items.

        A ``None`` packet acts as a poison-pill for the consumer.
        """
        if self.is_full and self._drop_on_full:
            self._stats.total_dropped += 1
            logger.warning(
                "Queue at capacity (%d/%d); dropping spectrum %s.",
                self._stats.current_depth,
                self._capacity,
                packet.spectrum_id,
            )
            raise QueueFull(
                f"Queue full ({self._capacity}); spectrum {packet.spectrum_id} dropped."
            )

        await self._queue.put(packet)
        self._stats.total_ingested += 1
        self._stats.current_depth = self._queue.qsize()

    def try_put_nowait(self, packet: QueuedPacket) -> None:
        """Enqueue a packet without blocking.  Raises ``QueueFull`` if full.

        Suitable for callers that want to retry or degrade gracefully.
        """
        if self.is_full:
            self._stats.total_dropped += 1
            raise QueueFull(
                f"Queue full ({self._capacity}); spectrum {packet.spectrum_id} dropped."
            )
        self._queue.put_nowait(packet)
        self._stats.total_ingested += 1
        self._stats.current_depth = self._queue.qsize()

    async def get(self) -> QueuedPacket | None:
        """Dequeue a packet, blocking until one is available.

        Returns ``None`` if a poison-pill has been received (consumer
        should exit).
        """
        packet = await self._queue.get()
        if packet is not None:
            self._stats.current_depth = self._queue.qsize()
        return packet

    def task_done(self, latency_us: float = 0.0) -> None:
        """Signal that a dequeued packet has been fully processed."""
        self._stats.total_completed += 1
        if latency_us > 0:
            self._stats.record_latency(latency_us)
        self._queue.task_done()

    async def join(self) -> None:
        """Block until all enqueued packets have been processed."""
        await self._queue.join()

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Send poison-pill and drain remaining items.

        After this call no new packets can be enqueued without raising.
        """
        logger.info(
            "Shutting down bounded queue; draining %d items.", self._queue.qsize()
        )
        # Send poison to unblock consumers.
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        try:
            await asyncio.wait_for(self.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Queue drain timed out after %.1f s; %d items may be lost.",
                timeout,
                self._queue.qsize(),
            )
