#!/usr/bin/env python3
"""
Mock instrument streaming client for MassFlow gRPC service.

Replays spectra from a ``.mzML`` (or other supported) file over gRPC at
a configurable acquisition rate, simulating a live instrument feed.
Useful for integration testing, latency benchmarking, and stress-testing
the streaming server without a physical mass spectrometer.

Usage
-----
::

    uv run python scripts/mock_instrument_stream.py \\
        --input data/experiment.mzML \\
        --rate 20 \\
        --host localhost \\
        --port 50051 \\
        --duration 60

The ``--rate`` parameter controls the target spectra-per-second emission
rate.  Use ``--burst`` to simulate fast DDA peak elution (short periods
of very high throughput).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import grpc
import numpy as np

# ── Ensure the MassFlow package is importable ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from MassFlow.io import load_spectra  # noqa: E402
from MassFlow.streaming.generated.massflow.v1 import (  # type: ignore[import-untyped]  # noqa: E402
    streaming_pb2 as pb,
    streaming_pb2_grpc as pb_grpc,
)
from google.protobuf import empty_pb2  # noqa: E402

logger = logging.getLogger("mock_instrument")


def _spectrum_to_packet(spectrum, index: int) -> pb.SpectrumPacket:  # type: ignore[name-defined, attr-defined]
    """Convert a ``matchms.Spectrum`` into a ``SpectrumPacket`` protobuf."""
    mz = spectrum.peaks.mz.astype(np.float64)
    intensities = spectrum.peaks.intensities.astype(np.float64)

    meta = spectrum.metadata
    precursor_mz = float(meta.get("precursor_mz", 0.0))
    rt = float(meta.get("retention_time", 0.0))
    charge = int(meta.get("charge", 1))
    ion_mode = str(meta.get("ionmode", "positive"))
    adduct = str(meta.get("adduct", ""))
    ce = float(meta.get("collision_energy", 0.0))

    spec_id = str(meta.get("spectrum_id", f"scan_{index:06d}"))
    ts_ns = int(time.time_ns())

    return pb.SpectrumPacket(  # type: ignore[name-defined, attr-defined]
        spectrum_id=spec_id,
        mz_array=mz.tolist(),
        intensity_array=intensities.tolist(),
        precursor_mz=precursor_mz,
        retention_time_seconds=rt,
        charge=charge,
        ion_mode=ion_mode,
        adduct=adduct,
        collision_energy=ce,
        acquisition_timestamp_ns=ts_ns,
    )


async def _request_generator(
    spectra: list,
    rate_hz: float,
    burst: bool,
    burst_size: int,
    burst_interval_s: float,
) -> AsyncIterator[pb.StreamRequest]:  # type: ignore[name-defined, attr-defined]
    """Async generator yielding ``StreamRequest`` messages at the target rate."""
    interval_s = 1.0 / rate_hz if rate_hz > 0 else 0.0
    count = 0
    burst_phase = False
    burst_sent = 0
    last_burst_end = time.monotonic()

    for spec in spectra:
        loop = asyncio.get_running_loop()
        now = loop.time()

        if burst:
            # Burst phase: send burst_size packets as fast as possible,
            # then wait burst_interval_s before the next burst.
            if not burst_phase and (now - last_burst_end) >= burst_interval_s:
                burst_phase = True
                burst_sent = 0

            if burst_phase:
                yield pb.StreamRequest(spectrum=_spectrum_to_packet(spec, count))  # type: ignore[name-defined, attr-defined]
                burst_sent += 1
                count += 1
                if burst_sent >= burst_size:
                    burst_phase = False
                    last_burst_end = now
                    logger.debug(
                        "Burst of %d spectra complete; pausing %.1f s.",
                        burst_size,
                        burst_interval_s,
                    )
                    await asyncio.sleep(burst_interval_s)
                continue

        # Normal pacing.
        t_target = last_burst_end + interval_s * (count + 1)
        delay = max(t_target - now, 0.0)
        if delay > 0:
            await asyncio.sleep(delay)

        yield pb.StreamRequest(spectrum=_spectrum_to_packet(spec, count))  # type: ignore[name-defined, attr-defined]
        count += 1

    logger.info("Sent %d spectra; stream exhausted.", count)


async def _response_printer(
    response_iter: AsyncIterator[pb.AnnotationResponse],  # type: ignore[name-defined, attr-defined]
    verbose: bool,
) -> None:
    """Print each annotation response to stdout."""
    annotated = 0
    no_match = 0
    errors = 0
    t_start = time.monotonic()

    async for resp in response_iter:
        if resp.status == "annotated":
            annotated += 1
            if verbose and resp.top_hits:
                top = resp.top_hits[0]
                print(
                    f"[{resp.spectrum_id}] "
                    f"{top.reference_name:30s} "
                    f"score={top.score:.3f} "
                    f"peaks={top.matched_peaks} "
                    f"q={resp.fdr_q_value:.4f} "
                    f"lat={resp.processing_latency_us / 1e3:.1f}ms"
                )
        elif resp.status == "no_match":
            no_match += 1
        elif resp.status == "error":
            errors += 1
            print(f"[{resp.spectrum_id}] ERROR: {resp.error_message}")

    elapsed = time.monotonic() - t_start
    total = annotated + no_match + errors
    print(f"\n── Stream Summary ({elapsed:.1f} s) ──")
    print(f"  Total responses : {total}")
    print(f"  Annotated       : {annotated}")
    print(f"  No match        : {no_match}")
    print(f"  Errors          : {errors}")
    if elapsed > 0:
        print(f"  Throughput      : {total / elapsed:.1f} Hz")


async def run(
    input_path: str,
    host: str,
    port: int,
    rate_hz: float,
    burst: bool,
    burst_size: int,
    burst_interval_s: float,
    verbose: bool,
    timeout_s: Optional[float],
    admin_token: Optional[str] = None,
) -> None:
    """Main async entry point for the mock client."""

    # ── Load spectra ───────────────────────────────────────────────────
    print(f"Loading spectra from {input_path} ...")
    spectra = list(load_spectra(Path(input_path)))
    if not spectra:
        print("ERROR: No spectra loaded from input file.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(spectra)} spectra.  Target rate: {rate_hz} Hz.")

    # ── Connect to server ──────────────────────────────────────────────
    target = f"{host}:{port}"
    print(f"Connecting to gRPC server at {target} ...")

    metadata = None
    if admin_token:
        metadata = (("authorization", f"Bearer {admin_token}"),)

    async with grpc.aio.insecure_channel(target) as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)

        # Check server status first.
        try:
            status = await stub.GetStatus(empty_pb2.Empty(), metadata=metadata)
            print(
                f"Server status: active={status.is_active}, queue_depth={status.queue_depth}"
            )
        except grpc.RpcError as exc:
            print(f"WARNING: Could not reach server: {exc.code()} – {exc.details()}")
            print("Continuing with stream anyway...")

        # ── Bidirectional stream ────────────────────────────────────
        print(f"\nStarting stream at {rate_hz} Hz (burst={burst}) ...\n")

        try:
            call = stub.StreamSpectra(
                _request_generator(
                    spectra, rate_hz, burst, burst_size, burst_interval_s
                ),
                timeout=timeout_s,
                metadata=metadata,
            )
            await _response_printer(call, verbose)
        except grpc.aio.AioRpcError as exc:
            print(f"\nStream error: {exc.code()} – {exc.details()}", file=sys.stderr)
        except asyncio.TimeoutError:
            print(f"\nStream timed out after {timeout_s} s.")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mock instrument – stream spectra to MassFlow gRPC server.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to .mzML (or supported) file to replay.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="gRPC server host (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="gRPC server port (default: 50051).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="Target spectra-per-second emission rate (default: 20).",
    )
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Simulate fast DDA bursts instead of steady pacing.",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=50,
        help="Number of spectra per burst (default: 50).",
    )
    parser.add_argument(
        "--burst-interval",
        type=float,
        default=2.0,
        help="Pause between bursts in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-spectrum annotation details.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Stream timeout in seconds.",
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help=(
            "Bearer token for control-plane operations "
            "(required by servers started with --admin-token)."
        ),
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    asyncio.run(
        run(
            input_path=args.input,
            host=args.host,
            port=args.port,
            rate_hz=args.rate,
            burst=args.burst,
            burst_size=args.burst_size,
            burst_interval_s=args.burst_interval,
            verbose=args.verbose,
            timeout_s=args.timeout,
            admin_token=args.admin_token,
        )
    )


if __name__ == "__main__":
    main()
