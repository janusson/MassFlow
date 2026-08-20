"""
End-to-end smoke demo for the massflow-ml satellite transports.

Boots the REST (uvicorn) and gRPC transports in-process, then drives both
through the MassFlow core's own ``RemoteMLEngine`` client to prove the
``massflow.v1.ml`` contract is fulfilled end-to-end.  Also demonstrates the
fail-safe circuit breaker: a dead endpoint trips after the configured
failure threshold, which is the signal orchestrators use to fall back to
classical scoring.

Run from the repository root:

    uv run python examples/massflow-ml-satellite/client_smoke.py
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MassFlow.similarity import SearchResult

import grpc
import httpx
import numpy as np
import uvicorn
from matchms import Spectrum

# Make the satellite directory importable regardless of the current cwd.
SATELLITE_DIR = Path(__file__).resolve().parent
if str(SATELLITE_DIR) not in sys.path:
    sys.path.insert(0, str(SATELLITE_DIR))

from MassFlow.generated.massflow.v1 import ml_pb2_grpc  # noqa: E402
from MassFlow.ml_client import CircuitOpenError, RemoteMLEngine  # noqa: E402
from dummy_model import DummyBinnedCosineModel  # noqa: E402
from grpc_server import MLEngineService  # noqa: E402
from rest_server import app  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an ephemeral free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_spectrum(
    spectrum_id: str,
    precursor_mz: float,
    peaks: list[tuple[float, float]],
    compound_name: str | None = None,
    smiles: str | None = None,
    inchikey: str | None = None,
) -> Spectrum:
    """Build a float64 matchms.Spectrum with wire-friendly metadata."""
    mz_values, intensity_values = zip(*peaks)
    metadata: dict[str, object] = {"precursor_mz": precursor_mz}
    if compound_name:
        metadata["compound_name"] = compound_name
    if smiles:
        metadata["smiles"] = smiles
    if inchikey:
        metadata["inchikey"] = inchikey
    spectrum = Spectrum(
        mz=np.asarray(mz_values, dtype=np.float64),
        intensities=np.asarray(intensity_values, dtype=np.float64),
        metadata=metadata,
    )
    spectrum.set("id", spectrum_id)
    return spectrum


def _make_synthetic_data() -> tuple[list[Spectrum], list[Spectrum]]:
    """Three references and three noisy copies used as queries."""
    reference_peaks = {
        "ref_1": [
            (100.2, 1.0),
            (150.7, 0.8),
            (220.4, 0.6),
            (311.9, 0.9),
        ],
        "ref_2": [
            (83.1, 0.9),
            (129.6, 1.0),
            (197.3, 0.5),
            (260.8, 0.7),
        ],
        "ref_3": [
            (71.2, 0.7),
            (145.4, 1.0),
            (230.9, 0.8),
            (350.6, 0.5),
        ],
    }
    reference_spectra = [
        _make_spectrum(
            spectrum_id,
            precursor_mz=300.0 + index,
            peaks=peaks,
            compound_name=f"reference_{index + 1}",
            smiles=f"C{index + 1}",
            inchikey=f"INCHI-{index + 1}",
        )
        for index, (spectrum_id, peaks) in enumerate(reference_peaks.items())
    ]
    query_spectra = [
        _make_spectrum(
            f"query_{index + 1}",
            precursor_mz=300.0 + index,
            peaks=[
                (mz + 0.03, intensity)
                for mz, intensity in reference_peaks[f"ref_{index + 1}"]
            ]
            + [(400.0 + index, 0.05)],
        )
        for index in range(len(reference_spectra))
    ]
    return query_spectra, reference_spectra


class _RestServerThread(threading.Thread):
    """Run a uvicorn server for the REST transport in a daemon thread."""

    def __init__(self, port: int) -> None:
        super().__init__(daemon=True, name="massflow-ml-rest")
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )

    def run(self) -> None:
        self._server.run()

    def stop(self) -> None:
        self._server.should_exit = True


class _GrpcServerThread(threading.Thread):
    """Run the grpc.aio server in a daemon thread with its own event loop.

    ``grpc.aio`` servers require coroutine handlers, so the server owns a
    dedicated ``asyncio`` event loop in this thread.  Shutdown is scheduled
    onto that loop with ``run_coroutine_threadsafe``.
    """

    def __init__(self, model: DummyBinnedCosineModel, port: int) -> None:
        super().__init__(daemon=True, name="massflow-ml-grpc")
        self._model = model
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: grpc.aio.Server | None = None

    def run(self) -> None:
        async def _serve() -> None:
            self._loop = asyncio.get_running_loop()
            server = grpc.aio.server()
            self._server = server
            ml_pb2_grpc.add_MLEngineServiceServicer_to_server(
                MLEngineService(self._model), server
            )
            server.add_insecure_port(f"127.0.0.1:{self._port}")
            await server.start()
            await server.wait_for_termination()

        asyncio.run(_serve())

    def stop(self) -> None:
        if self._server is not None and self._loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._server.stop(grace=0), self._loop
            ).result(timeout=5.0)


def _wait_for_rest(port: int, timeout_seconds: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError("REST server did not become healthy in time.")


def _wait_for_grpc(port: int, timeout_seconds: float = 15.0) -> None:
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_seconds)
    finally:
        channel.close()


def _check_search_results(
    transport: str,
    results: list["SearchResult"],
    expected_top_hits: list[str],
    min_score: float,
) -> None:
    """Assert basic contract invariants on a flattened SearchResult list."""
    assert len(results) == len(expected_top_hits), (
        f"[{transport}] expected {len(expected_top_hits)} results, got {len(results)}"
    )
    for result, expected_id in zip(results, expected_top_hits):
        score = float(result["score"])
        assert 0.0 <= score <= 1.0, f"[{transport}] score {score} out of range"
        assert score >= min_score, f"[{transport}] score below threshold"
        assert result["reference_id"] == expected_id, (
            f"[{transport}] top hit {result['reference_id']!r} != "
            f"expected {expected_id!r} (score={score:.4f})"
        )
        assert result["reference_name"], "[{transport}] missing reference_name"
        assert result["matched_peaks"] >= 1, "[{transport}] matched_peaks is 0"
        assert result["q_value"] == 1.0, "[{transport}] unexpected q_value"
    print(
        f"  [{transport}] search OK: {len(results)} top hits retrieved, "
        f"scores in [{min_score}, 1.0]"
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the end-to-end smoke checks across both transports."""
    print("massflow-ml satellite smoke test")
    print("--------------------------------")

    query_spectra, reference_spectra = _make_synthetic_data()
    expected_top_hits = ["ref_1", "ref_2", "ref_3"]

    rest_port = _free_port()
    grpc_port = _free_port()

    # --- Boot transports -------------------------------------------------
    rest_thread = _RestServerThread(rest_port)
    rest_thread.start()
    model = DummyBinnedCosineModel.load(SATELLITE_DIR)
    grpc_thread = _GrpcServerThread(model, grpc_port)
    grpc_thread.start()

    try:
        _wait_for_rest(rest_port)
        _wait_for_grpc(grpc_port)
        print(f"  REST transport up on http://127.0.0.1:{rest_port}")
        print(f"  gRPC transport up on grpc://127.0.0.1:{grpc_port}")

        # --- REST transport (algorithm routed by path) ---------------------
        rest_engine = RemoteMLEngine(
            "spec2vec",
            f"http://127.0.0.1:{rest_port}/spec2vec",
            timeout_seconds=15.0,
        )
        results = rest_engine.search(
            query_spectra,
            reference_spectra,
            min_score=0.5,
            top_n=2,
            include_decoys=False,
        )
        _check_search_results("REST", results, expected_top_hits, 0.5)

        rest_scores = rest_engine.batch_score(query_spectra, reference_spectra)
        assert rest_scores.shape == (3,), f"REST scores shape {rest_scores.shape}"
        assert rest_scores.dtype == np.float64, "REST scores not float64"
        assert np.all(rest_scores >= 0.9), f"REST pair scores too low: {rest_scores}"
        print(f"  [REST] batch_score OK: {rest_scores}")

        # --- gRPC transport (top_n truncation exercised) ------------------
        grpc_engine = RemoteMLEngine(
            "ms2deepscore",
            f"grpc://127.0.0.1:{grpc_port}",
            timeout_seconds=15.0,
        )
        results = grpc_engine.search(
            query_spectra,
            reference_spectra,
            min_score=0.0,
            top_n=1,
            include_decoys=False,
        )
        _check_search_results("gRPC", results, expected_top_hits, 0.0)
        for result in results:
            assert result["score"] >= 0.99, (
                f"[gRPC] top hit score {result['score']:.4f} unexpectedly low"
            )
            structural_similarity = result["structural_similarity"]
            assert structural_similarity is not None, (
                "[gRPC] structural_similarity should be present (NaN when unset)"
            )
            assert np.isnan(structural_similarity), (
                "[gRPC] structural_similarity should map to NaN when unset"
            )
        print("  [gRPC] top_n=1 truncation and NaN structural_similarity OK")

        grpc_scores = grpc_engine.batch_score(query_spectra, reference_spectra)
        assert grpc_scores.shape == (3,), f"gRPC scores shape {grpc_scores.shape}"
        assert grpc_scores.dtype == np.float64, "gRPC scores not float64"
        assert np.all(grpc_scores >= 0.9), f"gRPC pair scores too low: {grpc_scores}"
        print(f"  [gRPC] batch_score OK: {grpc_scores}")

        # --- Fail-safe: circuit breaker trips on a dead endpoint ----------
        dead_engine = RemoteMLEngine(
            "spec2vec",
            f"http://127.0.0.1:{_free_port()}/spec2vec",
            timeout_seconds=2.0,
            circuit_failure_threshold=2,
            circuit_cooldown_seconds=60.0,
        )
        failures = 0
        for _ in range(2):
            try:
                dead_engine.search(query_spectra, reference_spectra)
            except CircuitOpenError:
                raise AssertionError(
                    "breaker opened before reaching the failure threshold"
                ) from None
            except Exception:
                failures += 1
        assert failures == 2, f"expected 2 recorded failures, got {failures}"
        try:
            dead_engine.search(query_spectra, reference_spectra)
            raise AssertionError("expected CircuitOpenError from open breaker")
        except CircuitOpenError:
            print(
                "  [fail-safe] circuit breaker opened after 2 failures; "
                "classical fallback would engage here"
            )

        print("--------------------------------")
        print("All satellite smoke checks passed.")
    finally:
        grpc_thread.stop()
        grpc_thread.join(timeout=5.0)
        rest_thread.stop()
        rest_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
