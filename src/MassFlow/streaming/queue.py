"""
Bounded queue with backpressure for real-time spectral ingestion.

This module provides a thread-safe, capacity-limited ``asyncio.Queue``
wrapper that implements controlled backpressure for gRPC spectral
streaming.  When the queue is full the behaviour is governed by the
configured ``OverflowPolicy``:

* ``BLOCK`` – ``put()`` suspends the caller until space is available
  (classic backpressure).  An optional timeout drops the packet when
  exceeded.
* ``DROP_OLDEST`` – the oldest unprocessed packet is discarded to
  make room for the newest arrival.  This is the preferred policy for
  real-time instrument feeds where freshness outweighs completeness.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class OverflowPolicy(str, enum.Enum):
    """Backpressure policy when the bounded queue reaches capacity."""

    BLOCK = "block"
    """Suspend the producer until a slot opens (classic backpressure)."""

    DROP_OLDEST = "drop_oldest"
    """Discard the oldest enqueued spectrum to admit the newest one."""


class QueueFull(Exception):
    """Raised when the bounded queue is at capacity and the overflow policy
    rejects the enqueue operation."""


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
    """Capacity-limited async queue with backpressure and overflow semantics.

    Parameters
    ----------
    capacity : int
        Maximum number of queued spectra.  Must be positive.
    overflow : OverflowPolicy
        * ``BLOCK`` – ``put()`` blocks until space is available.
        * ``DROP_OLDEST`` – the oldest packet is evicted to make room.
    drop_on_full : bool, optional
        **Deprecated.**  If ``True`` the ``put()`` method raises
        ``QueueFull`` when the queue is full (equivalent to a
        "drop newest" strategy).  Prefer ``OverflowPolicy`` in new code.
    """

    def __init__(
        self,
        capacity: int = 2048,
        overflow: OverflowPolicy = OverflowPolicy.BLOCK,
        drop_on_full: Optional[bool] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Queue capacity must be positive.")

        # Resolve overflow policy, honouring the legacy ``drop_on_full`` flag.
        if drop_on_full is not None:
            if drop_on_full:
                # Legacy drop-on-full maps to a "reject newest" semantic.
                self._overflow = OverflowPolicy.DROP_OLDEST
                self._reject_on_full = True
            else:
                self._overflow = OverflowPolicy.BLOCK
                self._reject_on_full = False
        else:
            self._overflow = overflow
            self._reject_on_full = False

        self._capacity = capacity
        self._queue: asyncio.Queue[QueuedPacket | None] = asyncio.Queue(
            maxsize=capacity
        )
        self._stats = QueueStats()
        self._shutdown_requested = False

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

    @property
    def is_empty(self) -> bool:
        return self._queue.qsize() == 0

    @property
    def overflow(self) -> OverflowPolicy:
        """The active overflow policy."""
        return self._overflow

    @property
    def current_depth(self) -> int:
        """Number of packets currently in the queue."""
        return self._queue.qsize()

    async def put(
        self,
        packet: QueuedPacket,
        timeout: float | None = None,
    ) -> None:
        """Enqueue a packet, governed by the active overflow policy.

        Parameters
        ----------
        packet : QueuedPacket
            The spectral packet to enqueue.  A ``None`` packet acts as a
            poison-pill for the consumer.
        timeout : float or None
            Maximum seconds to wait for queue space when the overflow
            policy is ``BLOCK``.  If the timeout expires, the packet is
            discarded, ``total_dropped`` is incremented, and a critical
            warning is logged.  ``None`` means block indefinitely.

        Raises
        ------
        QueueFull
            When the ``reject_on_full`` legacy flag is set and the queue
            is at capacity.
        """
        if self._shutdown_requested:
            self._stats.total_dropped += 1
            logger.warning(
                "Queue is shut down; dropping spectrum %s.", packet.spectrum_id
            )
            return

        if self.is_full:
            if self._reject_on_full:
                self._stats.total_dropped += 1
                logger.warning(
                    "Queue at capacity (%d/%d); dropping spectrum %s.",
                    self._stats.current_depth,
                    self._capacity,
                    packet.spectrum_id,
                )
                raise QueueFull(
                    f"Queue full ({self._capacity}); "
                    f"spectrum {packet.spectrum_id} dropped."
                )

            if self._overflow == OverflowPolicy.DROP_OLDEST:
                # --- Evict the oldest packet ---
                try:
                    discarded = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass  # race: another consumer drained the slot
                else:
                    self._stats.total_dropped += 1
                    # Mark the evicted item as "done" to keep the
                    # ``_unfinished_tasks`` counter in sync.
                    self._queue.task_done()
                    logger.debug(
                        "Queue full; evicted oldest spectrum %s to admit %s.",
                        discarded.spectrum_id if discarded else "<poison>",
                        packet.spectrum_id,
                    )
                # Now there is room; proceed to put.
                self._queue.put_nowait(packet)
                self._stats.total_ingested += 1
                self._stats.current_depth = self._queue.qsize()
                return

            # --- BLOCK policy with optional timeout ---
            if timeout is not None:
                try:
                    await asyncio.wait_for(self._queue.put(packet), timeout=timeout)
                except asyncio.TimeoutError:
                    self._stats.total_dropped += 1
                    logger.critical(
                        (
                            "Queue backpressure timeout (%.1f s) at depth "
                            "%d/%d; discarding spectrum %s. Consumer may be "
                            "stalled."
                        ),
                        timeout,
                        self._stats.current_depth,
                        self._capacity,
                        packet.spectrum_id,
                    )
                    return
            else:
                await self._queue.put(packet)
        else:
            self._queue.put_nowait(packet)

        self._stats.total_ingested += 1
        self._stats.current_depth = self._queue.qsize()

    def try_put_nowait(self, packet: QueuedPacket) -> None:
        """Enqueue a packet without blocking.

        Raises ``QueueFull`` if the queue is full and the overflow policy
        is ``BLOCK``.  Under ``DROP_OLDEST`` the oldest packet is evicted
        silently and the call always succeeds (unless the queue has been
        shut down).
        """
        if self._shutdown_requested:
            self._stats.total_dropped += 1
            raise QueueFull("Queue is shut down; cannot enqueue.")

        if self.is_full:
            if self._overflow == OverflowPolicy.DROP_OLDEST:
                try:
                    discarded = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    self._stats.total_dropped += 1
                    self._queue.task_done()
                    logger.debug(
                        "Queue full; evicted oldest spectrum %s to admit %s.",
                        discarded.spectrum_id if discarded else "<poison>",
                        packet.spectrum_id,
                    )
                self._queue.put_nowait(packet)
                self._stats.total_ingested += 1
                self._stats.current_depth = self._queue.qsize()
                return

            # BLOCK policy: reject.
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

    async def drain(self, timeout: float = 30.0) -> int:
        """Drain all pending items without accepting new ones.

        Sets the ``_shutdown_requested`` flag so that subsequent ``put``
        calls are rejected, then waits for all currently-enqueued packets
        to be consumed.  Does **not** send a poison pill (the caller
        should do that separately if needed).

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for the queue to drain.

        Returns
        -------
        int
            Number of items remaining in the queue after the drain
            attempt (0 on success).
        """
        self._shutdown_requested = True
        remaining = self._queue.qsize()
        logger.info(
            "Draining bounded queue (%d items, timeout=%.1f s).",
            remaining,
            timeout,
        )
        try:
            await asyncio.wait_for(self.join(), timeout=timeout)
            remaining = 0
        except asyncio.TimeoutError:
            remaining = self._queue.qsize()
            logger.warning(
                "Queue drain timed out after %.1f s; %d items remain.",
                timeout,
                remaining,
            )
        return remaining

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Send poison-pill, drain remaining items, and block new enqueues.

        After this call no new packets can be enqueued without raising.
        """
        logger.info(
            "Shutting down bounded queue; draining %d items.",
            self._queue.qsize(),
        )
        self._shutdown_requested = True

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
