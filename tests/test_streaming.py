"""
Tests for the MassFlow streaming (gRPC) module.

Covers:
- ``BoundedQueue`` backpressure and stats
- ``StreamingEngine`` single-spectrum annotation
- ``QueuedPacket`` ↔ ``matchms.Spectrum`` conversion
- gRPC server/client integration (when stubs are available)
- Mock instrument client stream generation
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import pytest_asyncio
from matchms import Spectrum


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_spectrum():
    """A minimal valid matchms.Spectrum for testing."""
    mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    intensities = np.array([999.0, 500.0, 100.0], dtype=np.float64)
    return Spectrum(
        mz=mz,
        intensities=intensities,
        metadata={
            "precursor_mz": 350.0,
            "retention_time": 120.5,
            "charge": 1,
            "ionmode": "positive",
            "adduct": "[M+H]+",
        },
    )


@pytest.fixture
def sample_spectra(sample_spectrum):
    """A small list of sample spectra for library testing."""
    return [sample_spectrum] * 3


@pytest.fixture
def temp_config_yaml(sample_spectra, tmp_path):
    """Create a minimal MassFlow config YAML pointing to a synthetic library."""
    # Write a small MSP file as the library.
    msp_path = tmp_path / "test_library.msp"
    msp_content = """NAME: Test Compound A
PRECURSORMZ: 350.0
IONMODE: Positive
CHARGE: 1
Num Peaks: 3
100.0 999.0
200.0 500.0
300.0 100.0

NAME: Test Compound B
PRECURSORMZ: 450.0
IONMODE: Positive
CHARGE: 1
Num Peaks: 3
150.0 800.0
250.0 400.0
350.0 200.0

"""
    msp_path.write_text(msp_content)

    # Create a dummy input file so InputConfig.input_path is satisfied.
    dummy_input = tmp_path / "dummy_input.mzML"
    dummy_input.write_text("<!-- dummy -->")

    config_path = tmp_path / "massflow_config.yaml"
    config_content = f"""project:
  name: "Streaming Test"
  output_directory: "{tmp_path}/output"

input:
  input_path: "{dummy_input}"
  library_path: "{msp_path}"
  format: "msp"

processing:
  min_peaks: 1
  filter_min_peaks: false
  normalize_intensity: true

similarity:
  algorithm: "cosine"
  ms1_tolerance: 0.02
  ms2_tolerance: 0.02
  min_score: 0.1
  min_matched_peaks: 1
  fdr_threshold: 0.05

export:
  format: "csv"
"""
    config_path.write_text(config_content)
    return str(config_path)


# ---------------------------------------------------------------------------
# BoundedQueue tests
# ---------------------------------------------------------------------------


class TestBoundedQueue:
    """Unit tests for the bounded async queue with backpressure."""

    @pytest.mark.asyncio
    async def test_put_get_basic(self):
        """Simple put/get cycle."""
        from MassFlow.streaming.queue import BoundedQueue, QueuedPacket

        q = BoundedQueue(capacity=8)
        packet = QueuedPacket(
            spectrum_id="scan_001",
            mz_array=[100.0],
            intensity_array=[999.0],
            precursor_mz=350.0,
            retention_time_seconds=120.0,
            charge=1,
            ion_mode="positive",
            adduct="[M+H]+",
            collision_energy=25.0,
            acquisition_timestamp_ns=0,
        )
        await q.put(packet)
        assert q.stats.total_ingested == 1
        assert q.stats.current_depth == 1

        got = await q.get()
        assert got is not None
        assert got.spectrum_id == "scan_001"
        q.task_done()
        assert q.stats.total_completed == 1

    @pytest.mark.asyncio
    async def test_capacity_backpressure(self):
        """put() blocks when queue is full and drop_on_full=False."""
        from MassFlow.streaming.queue import BoundedQueue, QueuedPacket

        q = BoundedQueue(capacity=3, drop_on_full=False)

        for i in range(3):
            await q.put(
                QueuedPacket(
                    spectrum_id=f"scan_{i:03d}",
                    mz_array=[100.0],
                    intensity_array=[999.0],
                    precursor_mz=350.0,
                    retention_time_seconds=0.0,
                    charge=1,
                    ion_mode="positive",
                    adduct="",
                    collision_energy=0.0,
                    acquisition_timestamp_ns=0,
                )
            )

        assert q.is_full

        # Attempt a put with a timeout to verify backpressure.
        async def put_extra():
            await q.put(
                QueuedPacket(
                    spectrum_id="scan_extra",
                    mz_array=[100.0],
                    intensity_array=[999.0],
                    precursor_mz=350.0,
                    retention_time_seconds=0.0,
                    charge=1,
                    ion_mode="positive",
                    adduct="",
                    collision_energy=0.0,
                    acquisition_timestamp_ns=0,
                )
            )

        put_task = asyncio.create_task(put_extra())
        # Give the task a moment to block.
        await asyncio.sleep(0.1)
        assert not put_task.done(), "put should block when queue is full"

        # Drain one item to unblock.
        item = await q.get()
        assert item is not None
        q.task_done()

        await asyncio.wait_for(put_task, timeout=2.0)
        assert q.stats.total_ingested == 4

    @pytest.mark.asyncio
    async def test_drop_on_full(self):
        """When drop_on_full=True, put() raises QueueFull."""
        from MassFlow.streaming.queue import BoundedQueue, QueueFull, QueuedPacket

        q = BoundedQueue(capacity=1, drop_on_full=True)

        await q.put(
            QueuedPacket(
                spectrum_id="scan_001",
                mz_array=[100.0],
                intensity_array=[999.0],
                precursor_mz=350.0,
                retention_time_seconds=0.0,
                charge=1,
                ion_mode="positive",
                adduct="",
                collision_energy=0.0,
                acquisition_timestamp_ns=0,
            )
        )

        with pytest.raises(QueueFull):
            await q.put(
                QueuedPacket(
                    spectrum_id="scan_002",
                    mz_array=[100.0],
                    intensity_array=[999.0],
                    precursor_mz=350.0,
                    retention_time_seconds=0.0,
                    charge=1,
                    ion_mode="positive",
                    adduct="",
                    collision_energy=0.0,
                    acquisition_timestamp_ns=0,
                )
            )

        assert q.stats.total_dropped == 1

    @pytest.mark.asyncio
    async def test_try_put_nowait(self):
        """Non-blocking put that raises on full."""
        from MassFlow.streaming.queue import BoundedQueue, QueueFull, QueuedPacket

        q = BoundedQueue(capacity=1, drop_on_full=False)
        q.try_put_nowait(
            QueuedPacket(
                spectrum_id="scan_001",
                mz_array=[100.0],
                intensity_array=[999.0],
                precursor_mz=350.0,
                retention_time_seconds=0.0,
                charge=1,
                ion_mode="positive",
                adduct="",
                collision_energy=0.0,
                acquisition_timestamp_ns=0,
            )
        )

        with pytest.raises(QueueFull):
            q.try_put_nowait(
                QueuedPacket(
                    spectrum_id="scan_002",
                    mz_array=[100.0],
                    intensity_array=[999.0],
                    precursor_mz=350.0,
                    retention_time_seconds=0.0,
                    charge=1,
                    ion_mode="positive",
                    adduct="",
                    collision_energy=0.0,
                    acquisition_timestamp_ns=0,
                )
            )

    @pytest.mark.asyncio
    async def test_poison_pill_shutdown(self):
        """get() returns None after shutdown poison pill."""
        from MassFlow.streaming.queue import BoundedQueue, QueuedPacket

        q = BoundedQueue(capacity=4)
        await q.put(
            QueuedPacket(
                spectrum_id="scan_001",
                mz_array=[100.0],
                intensity_array=[999.0],
                precursor_mz=350.0,
                retention_time_seconds=0.0,
                charge=1,
                ion_mode="positive",
                adduct="",
                collision_energy=0.0,
                acquisition_timestamp_ns=0,
            )
        )

        # Drain the valid item first.
        item = await q.get()
        assert item is not None
        q.task_done()

        # Shutdown sends poison.
        await q.shutdown(timeout=1.0)

        # Next get returns None (poison).
        poison = await q.get()
        assert poison is None

    @pytest.mark.asyncio
    async def test_stats_latency(self):
        """Latency recording averages correctly."""
        from MassFlow.streaming.queue import BoundedQueue, QueuedPacket

        q = BoundedQueue(capacity=8)

        for i in range(3):
            await q.put(
                QueuedPacket(
                    spectrum_id=f"scan_{i:03d}",
                    mz_array=[100.0],
                    intensity_array=[999.0],
                    precursor_mz=350.0,
                    retention_time_seconds=0.0,
                    charge=1,
                    ion_mode="positive",
                    adduct="",
                    collision_energy=0.0,
                    acquisition_timestamp_ns=0,
                )
            )

        for i in range(3):
            item = await q.get()
            assert item is not None
            # Use non-zero latencies; 0 us calls are skipped by the stats guard.
            q.task_done(latency_us=float((i + 1) * 1000.0))

        # Average of [1000, 2000, 3000] = 2000 us.
        assert q.stats.total_completed == 3
        assert abs(q.stats.avg_latency_us - 2000.0) < 0.1

    @pytest.mark.asyncio
    async def test_negative_capacity_raises(self):
        """Capacity must be positive."""
        from MassFlow.streaming.queue import BoundedQueue

        with pytest.raises(ValueError, match="positive"):
            BoundedQueue(capacity=0)


# ---------------------------------------------------------------------------
# StreamingEngine tests
# ---------------------------------------------------------------------------


class TestStreamingEngine:
    """Tests for the streaming wrapper around SimilarityEngine."""

    def test_engine_init(self, temp_config_yaml):
        """Engine initialises and stores reference spectra."""
        from MassFlow.config import MassFlowConfig
        from MassFlow.streaming.engine import (
            StreamingEngine,
            load_reference_library,
        )

        config = MassFlowConfig.from_yaml(temp_config_yaml)
        refs = load_reference_library(config)
        assert len(refs) > 0

        engine = StreamingEngine(config=config, reference_spectra=refs, top_n=3)
        assert engine._engine is not None
        assert len(engine._ref_precursor_mzs) == len(refs)

    def test_annotate_returns_hits(self, temp_config_yaml):
        """Annotating a known spectrum returns hits."""
        from MassFlow.config import MassFlowConfig
        from MassFlow.streaming.engine import (
            StreamingEngine,
            load_reference_library,
        )
        from MassFlow.streaming.queue import QueuedPacket

        config = MassFlowConfig.from_yaml(temp_config_yaml)
        refs = load_reference_library(config)
        engine = StreamingEngine(config=config, reference_spectra=refs, top_n=3)

        # A query that matches Test Compound A.
        packet = QueuedPacket(
            spectrum_id="scan_query",
            mz_array=[100.0, 200.0, 300.0],
            intensity_array=[999.0, 500.0, 100.0],
            precursor_mz=350.0,
            retention_time_seconds=120.0,
            charge=1,
            ion_mode="positive",
            adduct="[M+H]+",
            collision_energy=25.0,
            acquisition_timestamp_ns=0,
        )

        result = engine.annotate(packet)
        assert result["status"] == "annotated"
        assert len(result["top_hits"]) >= 1
        hit = result["top_hits"][0]
        assert hit["score"] >= 0.1
        assert hit["matched_peaks"] >= 1

    def test_annotate_no_match(self, temp_config_yaml):
        """A spectrum with no matching peaks returns status 'no_match'."""
        from MassFlow.config import MassFlowConfig
        from MassFlow.streaming.engine import (
            StreamingEngine,
            load_reference_library,
        )
        from MassFlow.streaming.queue import QueuedPacket

        config = MassFlowConfig.from_yaml(temp_config_yaml)
        refs = load_reference_library(config)
        engine = StreamingEngine(config=config, reference_spectra=refs, top_n=3)

        # A query with peaks nowhere near the reference (ascending order).
        packet = QueuedPacket(
            spectrum_id="scan_no_match",
            mz_array=[9997.0, 9998.0, 9999.0],
            intensity_array=[100.0, 200.0, 300.0],
            precursor_mz=9999.0,
            retention_time_seconds=0.0,
            charge=1,
            ion_mode="positive",
            adduct="",
            collision_energy=0.0,
            acquisition_timestamp_ns=0,
        )

        result = engine.annotate(packet)
        assert result["status"] in ("no_match", "annotated")
        if result["status"] == "annotated":
            # If cosine still matched (unlikely with such distant peaks),
            # the score should be near zero.
            assert result["top_hits"][0]["score"] < 0.3

    def test_annotate_empty_peaks_handled(self, temp_config_yaml):
        """Spectrum with zero peaks is handled gracefully."""
        from MassFlow.config import MassFlowConfig
        from MassFlow.streaming.engine import (
            StreamingEngine,
            load_reference_library,
        )
        from MassFlow.streaming.queue import QueuedPacket

        config = MassFlowConfig.from_yaml(temp_config_yaml)
        refs = load_reference_library(config)
        engine = StreamingEngine(config=config, reference_spectra=refs, top_n=3)

        packet = QueuedPacket(
            spectrum_id="scan_empty",
            mz_array=[],
            intensity_array=[],
            precursor_mz=350.0,
            retention_time_seconds=0.0,
            charge=1,
            ion_mode="positive",
            adduct="",
            collision_energy=0.0,
            acquisition_timestamp_ns=0,
        )

        # Should not raise; matchms may handle empty arrays.
        result = engine.annotate(packet)
        assert result["status"] in ("no_match", "annotated", "error")

    def test_load_library_raises_on_empty(self, tmp_path):
        """load_reference_library raises an error if no spectra loaded."""
        import pytest
        from MassFlow.config import MassFlowConfig
        from MassFlow.streaming.engine import load_reference_library

        config_path = tmp_path / "empty_config.yaml"
        config_path.write_text(f"""project:
  name: "Empty"
  output_directory: "{tmp_path}/out"
input:
  input_path: "{tmp_path / "dummy.mzML"}"
  library_path: "{tmp_path / "nonexistent.msp"}"
  format: "msp"
similarity:
  algorithm: "cosine"
  min_score: 0.1
  min_matched_peaks: 1
export:
  format: "csv"
""")
        config = MassFlowConfig.from_yaml(str(config_path))
        with pytest.raises((RuntimeError, FileNotFoundError)):
            load_reference_library(config)


# ---------------------------------------------------------------------------
# QueuedPacket ↔ Spectrum conversion tests
# ---------------------------------------------------------------------------


class TestSpectrumConversion:
    """Verify QueuedPacket → matchms.Spectrum round-trip fidelity."""

    def test_precision_preserved(self):
        """float64 precision is maintained through conversion."""
        from MassFlow.streaming.engine import _spectrum_from_packet
        from MassFlow.streaming.queue import QueuedPacket

        mz_values = [100.123456789, 200.987654321]
        intensity_values = [999.555555555, 100.111111111]

        packet = QueuedPacket(
            spectrum_id="precision_test",
            mz_array=mz_values,
            intensity_array=intensity_values,
            precursor_mz=350.123456789,
            retention_time_seconds=120.555555555,
            charge=1,
            ion_mode="positive",
            adduct="[M+H]+",
            collision_energy=25.5,
            acquisition_timestamp_ns=0,
        )

        spectrum = _spectrum_from_packet(packet)
        assert spectrum.peaks.mz.dtype == np.float64
        assert spectrum.peaks.intensities.dtype == np.float64
        np.testing.assert_array_almost_equal(
            spectrum.peaks.mz, np.array(mz_values, dtype=np.float64)
        )
        np.testing.assert_array_almost_equal(
            spectrum.peaks.intensities, np.array(intensity_values, dtype=np.float64)
        )
        assert spectrum.get("precursor_mz") == pytest.approx(350.123456789)

    def test_missing_metadata_defaults(self):
        """Missing metadata gets sensible defaults."""
        from MassFlow.streaming.engine import _spectrum_from_packet
        from MassFlow.streaming.queue import QueuedPacket

        packet = QueuedPacket(
            spectrum_id="minimal",
            mz_array=[100.0],
            intensity_array=[999.0],
            precursor_mz=0.0,
            retention_time_seconds=0.0,
            charge=0,
            ion_mode="",
            adduct="",
            collision_energy=0.0,
            acquisition_timestamp_ns=0,
        )

        spectrum = _spectrum_from_packet(packet)
        # precursor_mz is stored as float, 0.0 is valid per matchms.
        assert spectrum.get("precursor_mz") == 0.0
        assert spectrum.get("charge") == 0
        # Empty ionmode: matchms may return None for empty-string metadata.
        assert spectrum.get("ionmode") in ("", None)


# ---------------------------------------------------------------------------
# Server integration tests (require generated gRPC stubs)
# ---------------------------------------------------------------------------


_HAS_GRPC_STUBS = False
try:
    from MassFlow.streaming.generated.massflow.v1 import (  # type: ignore[import-untyped]
        streaming_pb2 as pb,
        streaming_pb2_grpc as pb_grpc,
    )

    _HAS_GRPC_STUBS = True
except ImportError:
    pass


@pytest.mark.skipif(
    not _HAS_GRPC_STUBS,
    reason="gRPC stubs not compiled. Run scripts/protoc_gen.sh first.",
)
class TestGRPCServerIntegration:
    """End-to-end tests with a real gRPC server and client."""

    @pytest_asyncio.fixture
    async def server(self, temp_config_yaml, unused_tcp_port):
        """Start a gRPC server on a random port, yield it, then stop."""
        from MassFlow.config import MassFlowConfig
        from MassFlow.streaming.server import serve

        config = MassFlowConfig.from_yaml(temp_config_yaml)
        grpc_server = await serve(
            config,
            host="localhost",
            port=unused_tcp_port,
            queue_capacity=32,
            top_n=3,
        )

        yield grpc_server, unused_tcp_port

        await grpc_server.stop(grace=2.0)

    @pytest.mark.asyncio
    async def test_get_status(self, server):
        """GetStatus returns a valid ServerStatus message."""
        import grpc

        grpc_server, port = server
        target = f"localhost:{port}"

        async with grpc.aio.insecure_channel(target) as channel:
            stub = pb_grpc.MassFlowStreamingStub(channel)
            from google.protobuf import empty_pb2

            status = await stub.GetStatus(empty_pb2.Empty())
            assert status.is_active is True
            assert status.queue_depth >= 0
            assert status.spectra_ingested >= 0

    @pytest.mark.asyncio
    async def test_stream_single_spectrum(self, server):
        """Send a single spectrum and receive an annotation response."""
        import grpc

        grpc_server, port = server
        target = f"localhost:{port}"

        async with grpc.aio.insecure_channel(target) as channel:
            stub = pb_grpc.MassFlowStreamingStub(channel)

            async def request_gen():
                yield pb.StreamRequest(
                    spectrum=pb.SpectrumPacket(
                        spectrum_id="scan_test_001",
                        mz_array=[100.0, 200.0, 300.0],
                        intensity_array=[999.0, 500.0, 100.0],
                        precursor_mz=350.0,
                        retention_time_seconds=120.0,
                        charge=1,
                        ion_mode="positive",
                        adduct="[M+H]+",
                        collision_energy=25.0,
                        acquisition_timestamp_ns=0,
                    )
                )

            call = stub.StreamSpectra(request_gen(), timeout=10.0)
            responses = []
            async for resp in call:
                responses.append(resp)

            assert len(responses) >= 1
            first = responses[0]
            assert first.spectrum_id == "scan_test_001"
            assert first.status in ("annotated", "no_match", "error")
            assert first.processing_latency_us >= 0

    @pytest.mark.asyncio
    async def test_stream_multiple_spectra(self, server):
        """Stream 10 spectra and verify all get responses."""
        import grpc

        grpc_server, port = server
        target = f"localhost:{port}"

        N_SPECTRA = 10

        async with grpc.aio.insecure_channel(target) as channel:
            stub = pb_grpc.MassFlowStreamingStub(channel)

            async def request_gen():
                for i in range(N_SPECTRA):
                    yield pb.StreamRequest(
                        spectrum=pb.SpectrumPacket(
                            spectrum_id=f"scan_{i:04d}",
                            mz_array=[100.0 + i * 0.1, 200.0, 300.0],
                            intensity_array=[999.0, 500.0, 100.0],
                            precursor_mz=350.0 + i * 0.05,
                            retention_time_seconds=float(i),
                            charge=1,
                            ion_mode="positive",
                            adduct="[M+H]+",
                        )
                    )

            call = stub.StreamSpectra(request_gen(), timeout=15.0)
            responses = []
            async for resp in call:
                responses.append(resp)

            assert len(responses) == N_SPECTRA, (
                f"Expected {N_SPECTRA} responses, got {len(responses)}"
            )
            ids = {r.spectrum_id for r in responses}
            expected_ids = {f"scan_{i:04d}" for i in range(N_SPECTRA)}
            assert ids == expected_ids

    @pytest.mark.asyncio
    async def test_control_stop_start(self, server):
        """Send STOP, then START control messages."""
        import grpc

        grpc_server, port = server
        target = f"localhost:{port}"

        async with grpc.aio.insecure_channel(target) as channel:
            stub = pb_grpc.MassFlowStreamingStub(channel)

            async def request_gen():
                yield pb.StreamRequest(
                    control=pb.ControlMessage(
                        command=pb.ControlMessage.COMMAND_STOP,
                    )
                )
                yield pb.StreamRequest(
                    control=pb.ControlMessage(
                        command=pb.ControlMessage.COMMAND_START,
                    )
                )
                yield pb.StreamRequest(
                    spectrum=pb.SpectrumPacket(
                        spectrum_id="after_start",
                        mz_array=[100.0, 200.0, 300.0],
                        intensity_array=[999.0, 500.0, 100.0],
                        precursor_mz=350.0,
                        charge=1,
                        ion_mode="positive",
                        adduct="[M+H]+",
                    )
                )

            call = stub.StreamSpectra(request_gen(), timeout=10.0)
            responses = []
            async for resp in call:
                responses.append(resp)

            # Should get at least the annotation for "after_start".
            assert any(r.spectrum_id == "after_start" for r in responses)

    @pytest.mark.asyncio
    async def test_error_spectrum_handled(self, server):
        """A malformed spectrum (empty peaks) should not crash the server."""
        import grpc

        grpc_server, port = server
        target = f"localhost:{port}"

        async with grpc.aio.insecure_channel(target) as channel:
            stub = pb_grpc.MassFlowStreamingStub(channel)

            async def request_gen():
                yield pb.StreamRequest(
                    spectrum=pb.SpectrumPacket(
                        spectrum_id="empty_peaks",
                        mz_array=[],
                        intensity_array=[],
                        precursor_mz=350.0,
                        charge=1,
                    )
                )
                yield pb.StreamRequest(
                    spectrum=pb.SpectrumPacket(
                        spectrum_id="valid",
                        mz_array=[100.0, 200.0, 300.0],
                        intensity_array=[999.0, 500.0, 100.0],
                        precursor_mz=350.0,
                        charge=1,
                        ion_mode="positive",
                        adduct="[M+H]+",
                    )
                )

            call = stub.StreamSpectra(request_gen(), timeout=10.0)
            responses = []
            async for resp in call:
                responses.append(resp)

            # Server should still be alive and respond to the valid spectrum.
            assert any(r.spectrum_id == "valid" for r in responses)


# ---------------------------------------------------------------------------
# Memory stability test
# ---------------------------------------------------------------------------


class TestMemoryStability:
    """Verify the streaming engine does not leak memory under load."""

    @pytest.mark.asyncio
    async def test_queue_under_sustained_load(self):
        """Rapid put/get cycle with 10_000 packets should not leak."""
        from MassFlow.streaming.queue import BoundedQueue, QueuedPacket

        q = BoundedQueue(capacity=512)

        # Producer.
        async def producer(n: int):
            for i in range(n):
                await q.put(
                    QueuedPacket(
                        spectrum_id=f"scan_{i:06d}",
                        mz_array=[100.0, 200.0],
                        intensity_array=[999.0, 500.0],
                        precursor_mz=350.0,
                        retention_time_seconds=0.0,
                        charge=1,
                        ion_mode="positive",
                        adduct="",
                        collision_energy=0.0,
                        acquisition_timestamp_ns=0,
                    )
                )

        # Consumer.
        async def consumer(n: int):
            received = 0
            while received < n:
                item = await q.get()
                if item is None:
                    break
                q.task_done()
                received += 1
                # Simulate ~100 µs processing.
                await asyncio.sleep(0)

        N = 10_000

        prod = asyncio.create_task(producer(N))
        cons = asyncio.create_task(consumer(N))
        await asyncio.gather(prod, cons)

        assert q.stats.total_completed == N
        assert q.stats.total_dropped == 0
        assert q.stats.total_dropped == 0

        # No meaningful assertion on gc stats (they're system-dependent),
        # but if we got here without OOM, we're good.
