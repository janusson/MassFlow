"""
Tests for the Phase 5 ML API boundary (massflow-ml satellite architecture).

Covers:
- ``CircuitBreaker``: open/close/half-open state machine, fast-fail
  rejection, cooldown recovery, stats.
- ``RemoteMLEngine``: REST transport against a local stub HTTP server,
  payload fidelity, and failure→breaker-open behaviour.
- Config: ``ml_endpoints`` scheme validation.
- Factory: remote endpoints take priority over local engines.
- Graceful fallback in ``ConsensusEngine`` and ``CascadeEngine`` when the
  ML service is unreachable or heavy dependencies are missing, and the
  workflow-level fallback for direct ML configs.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum
from pydantic import ValidationError

from MassFlow.config import MassFlowConfig, SimilarityConfig
from MassFlow.ml_client import CircuitBreaker, CircuitOpenError, RemoteMLEngine
from MassFlow.similarity import (
    CascadeEngine,
    ConsensusEngine,
    SimilarityEngine,
    get_similarity_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spectrum(
    spectrum_id: str,
    mz: list[float],
    precursor_mz: float,
    intensities: list[float] | None = None,
) -> Spectrum:
    """Build a simple spectrum with a precursor and id."""
    mz_array = np.asarray(sorted(mz), dtype=np.float64)
    if intensities is None:
        intensities_array: np.ndarray = np.ones(mz_array.size, dtype=np.float64)
    else:
        intensities_array = np.asarray(intensities, dtype=np.float64)
    return Spectrum(
        mz=mz_array,
        intensities=intensities_array,
        metadata={"id": spectrum_id, "precursor_mz": precursor_mz},
    )


class _StubMLHandler(BaseHTTPRequestHandler):
    """Local REST stub implementing the massflow.v1.ml JSON contract."""

    requests: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(payload)
        if self.path.endswith("/search"):
            results = []
            for query in payload["query_spectra"]:
                hits = [
                    {
                        "reference_id": ref["spectrum_id"],
                        "reference_name": ref["metadata"].get("compound_name", ""),
                        "score": 0.9,
                        "matched_peaks": 3,
                        "q_value": 0.05,
                    }
                    for ref in payload["reference_spectra"][:1]
                ]
                results.append({"query_id": query["spectrum_id"], "hits": hits})
            body = json.dumps({"results": results}).encode("utf-8")
        else:
            body = json.dumps({"scores": [0.8, 0.6]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # silence test output
        pass


@pytest.fixture
def stub_ml_server() -> Iterator[str]:
    """Run a stub REST ML service and return its base URL."""
    _StubMLHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _StubMLHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def sample_pair() -> tuple[Spectrum, Spectrum]:
    """A query/reference pair with identical peaks."""
    query = make_spectrum("q1", [100.0, 200.0, 300.0], precursor_mz=400.0)
    reference = make_spectrum("r1", [100.0, 200.0, 300.0], precursor_mz=400.0)
    return query, reference


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Circuit-breaker state machine."""

    def test_closed_passes_calls_through(self) -> None:
        """Successful calls keep the circuit closed and return results."""
        breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        assert breaker.execute(lambda: 42) == 42
        assert breaker.state == "closed"
        assert breaker.stats["total_failures"] == 0

    def test_opens_after_threshold_failures(self) -> None:
        """Consecutive failures open the circuit at the threshold."""
        breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)

        def fail() -> None:
            raise TimeoutError("unreachable")

        for _ in range(2):
            with pytest.raises(TimeoutError):
                breaker.execute(fail)
        assert breaker.state == "open"
        assert breaker.stats["consecutive_failures"] == 2

    def test_open_rejects_calls_fast(self) -> None:
        """An open circuit fails fast without invoking the remote call."""
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
        calls: list[int] = []

        def fail() -> None:
            calls.append(1)
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            breaker.execute(fail)
        with pytest.raises(CircuitOpenError):
            breaker.execute(fail)
        assert len(calls) == 1  # second call was rejected before execution
        assert breaker.stats["total_rejected"] == 1

    def test_half_open_recovers_after_cooldown(self) -> None:
        """After the cooldown a trial call re-closes the circuit on success."""
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)

        def fail() -> None:
            raise TimeoutError("down")

        with pytest.raises(TimeoutError):
            breaker.execute(fail)
        assert breaker.state == "open"

        import time

        time.sleep(0.08)
        assert breaker.execute(lambda: "recovered") == "recovered"
        assert breaker.state == "closed"

    def test_half_open_failure_reopens(self) -> None:
        """A failed trial re-opens the circuit."""
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)

        def fail() -> None:
            raise TimeoutError("down")

        with pytest.raises(TimeoutError):
            breaker.execute(fail)
        import time

        time.sleep(0.08)
        with pytest.raises(TimeoutError):
            breaker.execute(fail)
        assert breaker.state == "open"
        with pytest.raises(CircuitOpenError):
            breaker.execute(fail)

    def test_success_resets_failure_count(self) -> None:
        """A success before the threshold resets the failure streak."""
        breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)

        def flaky() -> str:
            if len(calls) == 1:
                raise TimeoutError("once")
            return "ok"

        calls: list[int] = []

        def call() -> str:
            calls.append(1)
            return flaky()

        with pytest.raises(TimeoutError):
            breaker.execute(call)
        assert breaker.execute(call) == "ok"
        assert breaker.stats["consecutive_failures"] == 0
        assert breaker.state == "closed"

    def test_invalid_parameters(self) -> None:
        """Threshold and cooldown bounds are validated."""
        with pytest.raises(ValueError):
            CircuitBreaker(failure_threshold=0)
        with pytest.raises(ValueError):
            CircuitBreaker(cooldown_seconds=-1.0)


# ---------------------------------------------------------------------------
# RemoteMLEngine (REST transport)
# ---------------------------------------------------------------------------


class TestRemoteMLEngine:
    """Remote engine over the REST JSON contract."""

    def test_search_roundtrip(self, stub_ml_server: str, sample_pair) -> None:
        """Search results deserialize into SearchResult-shaped dicts."""
        query, reference = sample_pair
        engine = RemoteMLEngine(
            "spec2vec", f"{stub_ml_server}/spec2vec", timeout_seconds=2.0
        )
        results = engine.search([query], [reference])
        assert len(results) == 1
        assert results[0]["query_id"] == "q1"
        assert results[0]["reference_id"] == "r1"
        assert results[0]["score"] == pytest.approx(0.9)
        assert results[0]["matched_peaks"] == 3

    def test_payload_fidelity(self, stub_ml_server: str, sample_pair) -> None:
        """The wire payload preserves float64 peak values and metadata."""
        query, reference = sample_pair
        engine = RemoteMLEngine(
            "ms2deepscore", f"{stub_ml_server}/ms2deepscore", timeout_seconds=2.0
        )
        engine.search([query], [reference], min_score=0.5, top_n=4)
        payload = _StubMLHandler.requests[-1]
        assert payload["algorithm"] == "ms2deepscore"
        assert payload["min_score"] == 0.5
        assert payload["top_n"] == 4
        assert payload["query_spectra"][0]["mz_array"] == [100.0, 200.0, 300.0]
        assert payload["query_spectra"][0]["intensity_array"] == [1.0, 1.0, 1.0]
        assert payload["query_spectra"][0]["metadata"]["precursor_mz"] == "400.0"

    def test_batch_score(self, stub_ml_server: str, sample_pair) -> None:
        """BatchScore returns aligned float64 scores."""
        query, reference = sample_pair
        engine = RemoteMLEngine(
            "spec2vec", f"{stub_ml_server}/spec2vec", timeout_seconds=2.0
        )
        scores = engine.batch_score([query, query], [reference, reference])
        assert scores.dtype == np.float64
        np.testing.assert_allclose(scores, [0.8, 0.6])

    def test_load_model_is_noop(self, stub_ml_server: str) -> None:
        """Model weights are owned by the remote service."""
        engine = RemoteMLEngine(
            "spec2vec", f"{stub_ml_server}/spec2vec", timeout_seconds=2.0
        )
        engine.load_model(Path("/nonexistent/model.bin"))  # must not raise

    def test_failures_open_circuit(self, sample_pair) -> None:
        """Unreachable endpoints trip the breaker and fail fast."""
        query, reference = sample_pair
        engine = RemoteMLEngine(
            "spec2vec",
            "http://127.0.0.1:1/spec2vec",
            timeout_seconds=0.5,
            circuit_failure_threshold=2,
            circuit_cooldown_seconds=30.0,
        )
        for _ in range(2):
            with pytest.raises(Exception):
                engine.search([query], [reference])
        assert engine.breaker.state == "open"
        with pytest.raises(CircuitOpenError):
            engine.search([query], [reference])

    def test_invalid_endpoint_scheme(self) -> None:
        """Only http(s):// and grpc:// schemes are accepted."""
        with pytest.raises(ValueError, match="scheme"):
            RemoteMLEngine("spec2vec", "ftp://host/spec2vec")


# ---------------------------------------------------------------------------
# Config & factory wiring
# ---------------------------------------------------------------------------


class TestMLEndpointConfig:
    """ml_endpoints configuration and factory routing."""

    def test_endpoints_validate(self, stub_ml_server: str) -> None:
        """Valid schemes pass; invalid schemes are rejected."""
        config = SimilarityConfig(
            algorithm="spec2vec",
            ml_endpoints={"spec2vec": f"{stub_ml_server}/spec2vec"},
        )
        assert config.ml_endpoints["spec2vec"].startswith("http")
        with pytest.raises(ValidationError, match="unsupported scheme"):
            SimilarityConfig(ml_endpoints={"spec2vec": "file:///tmp/spec2vec"})
        with pytest.raises(ValidationError, match="non-empty"):
            SimilarityConfig(ml_endpoints={"spec2vec": ""})

    def test_defaults(self) -> None:
        """Defaults: no endpoints, 10 s timeout, breaker at 3 failures."""
        config = SimilarityConfig()
        assert config.ml_endpoints == {}
        assert config.ml_request_timeout_seconds == 10.0
        assert config.ml_circuit_breaker_threshold == 3
        assert config.ml_circuit_breaker_cooldown_seconds == 60.0

    def test_factory_routes_to_remote_engine(self, stub_ml_server: str) -> None:
        """A configured endpoint takes priority over local engines."""
        config = SimilarityConfig(
            algorithm="spec2vec",
            ml_endpoints={"spec2vec": f"{stub_ml_server}/spec2vec"},
        )
        engine = get_similarity_engine(config)
        assert isinstance(engine, RemoteMLEngine)
        assert engine.algorithm == "spec2vec"

    def test_factory_local_registry_without_endpoint(self) -> None:
        """Without endpoints the local entry-point registry is used.

        In this environment the heavy dependencies are not installed, so
        the registry path raises the documented RuntimeError — proving the
        factory consulted the registry rather than building a remote
        engine (which never raises at construction).
        """
        config = SimilarityConfig(algorithm="spec2vec")
        with pytest.raises(RuntimeError, match="machine-learning extras"):
            get_similarity_engine(config)

    def test_classical_algorithms_ignore_endpoints(self) -> None:
        """Endpoints never affect classical engines."""
        config = SimilarityConfig(
            algorithm="modified_cosine",
            ml_endpoints={"modified_cosine": "http://127.0.0.1:1/x"},
        )
        assert isinstance(get_similarity_engine(config), SimilarityEngine)


# ---------------------------------------------------------------------------
# Graceful fallback in meta-engines
# ---------------------------------------------------------------------------


def _unreachable_ml_config(algorithm: str, **overrides) -> SimilarityConfig:
    """A config whose ML sub-engine points at an unreachable endpoint."""
    fields = dict(
        algorithm=algorithm,
        ms1_tolerance=0.02,
        ms2_tolerance=0.02,
        min_score=0.0,
        min_matched_peaks=1,
        ml_endpoints={"spec2vec": "http://127.0.0.1:1/spec2vec"},
        ml_request_timeout_seconds=0.5,
        ml_circuit_breaker_threshold=2,
        ml_circuit_breaker_cooldown_seconds=60.0,
    )
    fields.update(overrides)
    return SimilarityConfig(**fields)


class TestMetaEngineFallback:
    """Consensus/Cascade degrade to modified_cosine when ML is unavailable."""

    def test_consensus_falls_back_when_all_subengines_fail(self, sample_pair) -> None:
        """Unreachable remote sub-engine → modified_cosine results."""
        query, reference = sample_pair
        config = _unreachable_ml_config(
            "consensus",
            consensus_weights={"spec2vec": 1.0},
            consensus_min_engines=1,
        )
        engine = ConsensusEngine(config)
        results = engine.search([query], [reference], include_decoys=False)
        assert len(results) >= 1
        # The fallback engine scores the identical pair at 1.0.
        assert results[0]["score"] == pytest.approx(1.0)

    def test_consensus_falls_back_without_local_ml(self, sample_pair) -> None:
        """Missing local ML dependencies → modified_cosine results.

        The built-in Spec2Vec stub raises at search time when the [ml]
        extra is absent; the consensus engine must degrade instead of
        returning empty.
        """
        query, reference = sample_pair
        config = _unreachable_ml_config(
            "consensus",
            consensus_weights={"spec2vec": 1.0},
            consensus_min_engines=1,
        )
        # Drop the remote endpoint so the LOCAL stub engine is built.
        config.ml_endpoints = {}
        engine = ConsensusEngine(config)
        results = engine.search([query], [reference], include_decoys=False)
        assert len(results) >= 1
        assert results[0]["score"] == pytest.approx(1.0)

    def test_cascade_falls_back_when_first_stage_fails(self, sample_pair) -> None:
        """Unreachable first stage → classical fallback results."""
        query, reference = sample_pair
        config = _unreachable_ml_config(
            "cascade",
            cascade_stages=["spec2vec"],
            cascade_lower_bound=0.1,
            cascade_upper_bound=0.1,
        )
        engine = CascadeEngine(config)
        results = engine.search([query], [reference], include_decoys=False)
        assert len(results) >= 1
        assert results[0]["score"] == pytest.approx(1.0)

    def test_cascade_falls_back_with_no_stages(self, sample_pair) -> None:
        """A cascade with zero buildable stages uses the fallback."""
        query, reference = sample_pair
        config = _unreachable_ml_config(
            "cascade",
            cascade_stages=["spec2vec"],
            cascade_lower_bound=0.1,
            cascade_upper_bound=0.1,
        )
        engine = CascadeEngine(config)
        engine._stages = []  # simulate total build failure
        results = engine.search([query], [reference], include_decoys=False)
        assert len(results) >= 1
        assert results[0]["score"] == pytest.approx(1.0)

    def test_fallback_matches_direct_modified_cosine(self, sample_pair) -> None:
        """Fallback results equal a direct modified_cosine search."""
        query, reference = sample_pair
        config = _unreachable_ml_config(
            "cascade",
            cascade_stages=["spec2vec"],
            cascade_lower_bound=0.1,
            cascade_upper_bound=0.1,
        )
        engine = CascadeEngine(config)
        fallback_results = engine.search([query], [reference], include_decoys=False)

        direct = SimilarityEngine(
            SimilarityConfig(
                algorithm="modified_cosine",
                ms1_tolerance=0.02,
                ms2_tolerance=0.02,
                min_score=0.1,
                min_matched_peaks=1,
            )
        )
        direct_results = direct.search(
            [query], [reference], include_decoys=False, min_score=0.1
        )
        assert [(r["score"], r["reference_id"]) for r in fallback_results] == [
            (r["score"], r["reference_id"]) for r in direct_results
        ]


# ---------------------------------------------------------------------------
# Workflow-level fallback
# ---------------------------------------------------------------------------


class TestWorkflowFallback:
    """The orchestrator fails over to modified_cosine without crashing."""

    def test_process_single_file_falls_back(self, tmp_path: Path) -> None:
        """A failing configured engine degrades to classical scoring."""
        import MassFlow.workflow as workflow

        from MassFlow.streaming.queue import QueuedPacket  # noqa: F401

        # Build a query MGF and a reference pair.
        query_file = tmp_path / "query.mgf"
        query_file.write_text(
            """BEGIN IONS
PEPMASS=400.0
100.0 1.0
200.0 1.0
300.0 1.0
END IONS

"""
        )
        reference = make_spectrum("r1", [100.0, 200.0, 300.0], precursor_mz=400.0)

        config = MassFlowConfig(
            input={
                "input_path": str(query_file),
                "format": "mgf",
                "library_path": str(tmp_path / "lib.msp"),
            },
            processing={"min_peaks": 1, "filter_min_peaks": False},
            similarity={
                "algorithm": "cosine",
                "min_score": 0.1,
                "min_matched_peaks": 1,
                # A single-reference library produces no decoy hits, so the
                # conservative small-library FDR filter needs a permissive
                # threshold for the classical hit to survive.
                "fdr_threshold": 1.0,
            },
        )

        # Install an engine that always fails (simulates an unreachable
        # remote ML endpoint / missing heavy dependencies).
        class _FailingEngine:
            def search(self, *args, **kwargs):
                raise ConnectionError("ML service unreachable")

        # Preserve module state; this test mutates worker globals.
        saved_state = (
            workflow._worker_engine,
            workflow._worker_router,
            workflow._worker_references,
            workflow._worker_decoys,
            workflow._worker_ref_precursor_mzs,
            workflow._worker_ref_is_decoy,
            workflow._worker_fallback_engine,
        )
        try:
            # Minimal stand-in engine that always fails: exercises the
            # fallback path without implementing MLEngineProtocol.
            workflow._worker_engine = _FailingEngine()  # type: ignore[assignment]
            workflow._worker_router = None
            workflow._worker_references = [reference]
            workflow._worker_decoys = []
            workflow._worker_ref_precursor_mzs = np.array([400.0], dtype=np.float64)
            workflow._worker_ref_is_decoy = np.zeros(1, dtype=bool)
            workflow._worker_fallback_engine = None

            file_path, spectra, results = workflow._process_single_file(
                query_file, config
            )
        finally:
            (
                workflow._worker_engine,
                workflow._worker_router,
                workflow._worker_references,
                workflow._worker_decoys,
                workflow._worker_ref_precursor_mzs,
                workflow._worker_ref_is_decoy,
                workflow._worker_fallback_engine,
            ) = saved_state

        # The run completed and produced classical hits.
        assert file_path == query_file
        assert len(spectra) == 1
        assert any(result["reference_id"] == "r1" for result in results)
