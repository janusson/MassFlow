"""
Remote ML engine client — the massflow-ml satellite API boundary.

This module implements the client side of the decoupled machine-learning
architecture described in the post-v0.1 roadmap.  Heavy scoring engines
(Spec2Vec, MS2DeepScore) may live in a satellite repository or a remote
service; the MassFlow core talks to them through two transports:

* **REST** (``http://`` / ``https://`` endpoints) — a JSON contract mirroring
  the ``massflow.v1.ml`` protobuf messages, implemented with the standard
  library so the core gains no new dependencies.
* **gRPC** (``grpc://`` endpoints) — the ``MLEngineService`` defined in
  ``protos/massflow/v1/ml.proto`` (stubs under ``MassFlow.generated``).

Resilience
----------
Every remote call is wrapped in a :class:`CircuitBreaker`: after
``ml_circuit_breaker_threshold`` consecutive failures the breaker opens and
subsequent calls fail fast (without paying the network timeout), allowing
the orchestrator (``MLRouter``, ``ConsensusEngine``, ``CascadeEngine``) to
fall back to classical scoring immediately. After
``ml_circuit_breaker_cooldown_seconds`` the breaker allows one trial request
(half-open); success closes it again.

All scientific payloads use ``float64`` arrays — never ``float32``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List

import numpy as np
from matchms import Spectrum

from MassFlow.protocols import MLEngineProtocol

if TYPE_CHECKING:
    from MassFlow.similarity import SearchResult

logger = logging.getLogger(__name__)

# Import the generated gRPC stubs (created by scripts/protoc_gen.sh).
# ``ml_pb`` is referenced lazily inside the gRPC transport methods.
try:
    from MassFlow.generated.massflow.v1 import (  # type: ignore[import-untyped]  # noqa: F401
        ml_pb2 as ml_pb,
    )
    from MassFlow.generated.massflow.v1 import (  # type: ignore[import-untyped]
        ml_pb2_grpc as ml_pb_grpc,
    )

    _HAS_ML_GRPC_STUBS = True
except ImportError:  # pragma: no cover -- stubs are generated in-repo
    _HAS_ML_GRPC_STUBS = False
    logger.info(
        "ML gRPC stubs not found. Run 'scripts/protoc_gen.sh' to generate "
        "them; REST endpoints remain available without them."
    )


class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open and rejects a call outright."""


class CircuitBreaker:
    """Thread-safe circuit breaker for remote ML service calls.

    State machine:

    * ``closed`` — calls pass through; ``threshold`` consecutive failures
      open the circuit.
    * ``open`` — calls fail fast with :class:`CircuitOpenError` until the
      cooldown expires.
    * ``half-open`` — the first call after cooldown is allowed through as a
      trial; success closes the circuit, failure re-opens it.

    Parameters
    ----------
    failure_threshold : int
        Consecutive failures that trip the breaker.
    cooldown_seconds : float
        Seconds the circuit stays open before allowing a trial call.
    name : str, optional
        Human-readable name used in log messages.

    Examples
    --------
    >>> breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
    >>> breaker.execute(remote_call)
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        name: str = "remote-ml",
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1.")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0.")

        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._name = name
        self._lock = threading.Lock()
        self._state = "closed"  # "closed" | "open" | "half_open"
        self._consecutive_failures = 0
        self._opened_at_monotonic: float = 0.0
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejected = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current breaker state (``closed``, ``open``, or ``half_open``)."""
        with self._lock:
            return self._state

    @property
    def stats(self) -> dict[str, Any]:
        """Snapshot of call/failure/rejection counters."""
        with self._lock:
            return {
                "state": self._state,
                "consecutive_failures": self._consecutive_failures,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_rejected": self._total_rejected,
            }

    def execute(self, func: Callable[[], Any]) -> Any:
        """Run *func* through the breaker and return its result.

        Parameters
        ----------
        func : callable
            Zero-argument callable performing the remote operation.

        Returns
        -------
        Any
            The callable's return value.

        Raises
        ------
        CircuitOpenError
            If the circuit is open (fast fail).
        Exception
            Any exception raised by *func* (counted as a failure).
        """
        with self._lock:
            if self._state == "open":
                if time.monotonic() - self._opened_at_monotonic < (
                    self._cooldown_seconds
                ):
                    self._total_rejected += 1
                    logger.warning(
                        "Circuit breaker '%s' is open; rejecting call fast.",
                        self._name,
                    )
                    raise CircuitOpenError(
                        f"Circuit breaker '{self._name}' is open; call rejected."
                    )
                # Cooldown elapsed: allow one trial.
                self._state = "half_open"
                logger.info(
                    "Circuit breaker '%s' half-open; allowing trial call.",
                    self._name,
                )

            trial = self._state == "half_open"

        self._total_calls += 1
        try:
            result = func()
        except Exception:
            with self._lock:
                self._consecutive_failures += 1
                self._total_failures += 1
                if (
                    self._state != "open"
                    and self._consecutive_failures >= self._failure_threshold
                ):
                    self._state = "open"
                    self._opened_at_monotonic = time.monotonic()
                    logger.error(
                        "Circuit breaker '%s' OPENED after %d consecutive "
                        "failures; failing fast for %.1f s.",
                        self._name,
                        self._consecutive_failures,
                        self._cooldown_seconds,
                    )
                elif self._state == "open":
                    self._opened_at_monotonic = time.monotonic()
            raise

        with self._lock:
            if self._state != "closed" or self._consecutive_failures:
                logger.info(
                    "Circuit breaker '%s' closed after successful %s call.",
                    self._name,
                    "trial" if trial else "recovery",
                )
            self._state = "closed"
            self._consecutive_failures = 0
        return result

    def reset(self) -> None:
        """Force the breaker back to the closed state."""
        with self._lock:
            self._state = "closed"
            self._consecutive_failures = 0


# ---------------------------------------------------------------------------
# Payload (de)serialization
# ---------------------------------------------------------------------------


def _spectrum_to_payload(spectrum: Spectrum) -> dict[str, Any]:
    """Serialize a matchms.Spectrum into the wire-format dict."""
    metadata: dict[str, str] = {}
    for key, value in spectrum.metadata.items():
        metadata[str(key)] = str(value)
    return {
        "spectrum_id": str(spectrum.get("id", "")),
        "mz_array": [
            float(value) for value in np.asarray(spectrum.peaks.mz, dtype=np.float64)
        ],
        "intensity_array": [
            float(value)
            for value in np.asarray(spectrum.peaks.intensities, dtype=np.float64)
        ],
        "metadata": metadata,
    }


def _hits_to_search_results(
    query_id: str,
    payload: Any,
) -> List["SearchResult"]:
    """Convert a wire-format results payload into SearchResult dicts."""
    results: List["SearchResult"] = []
    for hit in payload:
        results.append(
            {
                "query_id": query_id,
                "query_precursor_mz": None,
                "reference_id": str(hit.get("reference_id", "")),
                "reference_name": str(hit.get("reference_name") or ""),
                "reference_precursor_mz": None,
                "score": float(hit.get("score", 0.0)),
                "matched_peaks": int(hit.get("matched_peaks", 0)),
                "smiles": str(hit.get("smiles") or "") or None,
                "inchikey": str(hit.get("inchikey") or "") or None,
                "is_decoy": False,
                "q_value": float(hit.get("q_value", 1.0)),
                "p_value": None,
                "annotation_tier": str(hit.get("annotation_tier") or "") or None,
                "structural_similarity": float(
                    hit.get("structural_similarity") or np.nan
                ),
                "mass_error_ppm": None,
                "score_breakdown": None,
            }
        )  # type: ignore[misc]
    return results


# ---------------------------------------------------------------------------
# Remote engine
# ---------------------------------------------------------------------------


class RemoteMLEngine(MLEngineProtocol):
    """Client-side implementation of ``MLEngineProtocol`` over REST or gRPC.

    Endpoints are configured per algorithm in ``SimilarityConfig.ml_endpoints``
    (e.g. ``{"spec2vec": "http://ml-host:8080/spec2vec"}`` or
    ``{"ms2deepscore": "grpc://ml-host:9090"}``).  Every call is guarded by a
    :class:`CircuitBreaker` so orchestrators can fail over to classical
    scoring the moment the service becomes unhealthy.

    Parameters
    ----------
    algorithm : str
        Algorithm name served by the remote engine (sent to the service).
    endpoint : str
        ``http://``, ``https://`` (REST/JSON) or ``grpc://`` (gRPC) URL.
    timeout_seconds : float, optional
        Per-request timeout for REST calls and gRPC RPCs.
    circuit_failure_threshold : int, optional
        Consecutive failures that open the circuit breaker.
    circuit_cooldown_seconds : float, optional
        Seconds the circuit stays open before a trial call.
    """

    def __init__(
        self,
        algorithm: str,
        endpoint: str,
        timeout_seconds: float = 10.0,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 60.0,
    ) -> None:
        self._algorithm = algorithm
        self._endpoint = endpoint.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._is_grpc = endpoint.startswith("grpc://")
        if not self._is_grpc and not (
            endpoint.startswith("http://") or endpoint.startswith("https://")
        ):
            raise ValueError(
                f"Unsupported ML endpoint scheme: '{endpoint}'. Use "
                f"http://, https://, or grpc://."
            )

        self._breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            cooldown_seconds=circuit_cooldown_seconds,
            name=f"remote-ml:{algorithm}",
        )
        self._grpc_channel: Any = None
        self._grpc_stub: Any = None

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        """The configured service endpoint."""
        return self._endpoint

    @property
    def algorithm(self) -> str:
        """The algorithm served by this remote engine."""
        return self._algorithm

    @property
    def breaker(self) -> CircuitBreaker:
        """The engine's circuit breaker (exposed for observability)."""
        return self._breaker

    # ------------------------------------------------------------------
    # gRPC client plumbing
    # ------------------------------------------------------------------

    def _get_grpc_stub(self) -> Any:
        """Return a (lazily created) gRPC stub for the endpoint."""
        if self._grpc_stub is not None:
            return self._grpc_stub
        if not _HAS_ML_GRPC_STUBS:
            raise RuntimeError(
                "ML gRPC stubs are not generated; run scripts/protoc_gen.sh "
                "or use an http(s) endpoint."
            )
        import grpc

        target = self._endpoint[len("grpc://") :]
        self._grpc_channel = grpc.insecure_channel(target)
        self._grpc_stub = ml_pb_grpc.MLEngineServiceStub(self._grpc_channel)
        return self._grpc_stub

    # ------------------------------------------------------------------
    # MLEngineProtocol
    # ------------------------------------------------------------------

    def search(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
    ) -> List["SearchResult"]:
        """Run a remote similarity search through the circuit breaker.

        Returns ``SearchResult``-shaped dicts.  ``ref_precursor_mzs`` and
        ``ref_is_decoy`` are accepted for interface compatibility but are
        derived remotely from ``reference_spectra``.
        """
        if not query_spectra or not list(reference_spectra):
            return []

        request_payload = {
            "algorithm": self._algorithm,
            "query_spectra": [_spectrum_to_payload(s) for s in query_spectra],
            "reference_spectra": [_spectrum_to_payload(s) for s in reference_spectra],
            "min_score": float(min_score if min_score is not None else 0.0),
            "top_n": int(top_n) if top_n is not None else 0,
            "include_decoys": bool(include_decoys),
        }

        if self._is_grpc:
            response = self._breaker.execute(lambda: self._grpc_search(request_payload))
            return self._grpc_response_to_results(response)

        response = self._breaker.execute(
            lambda: self._rest_post("/search", request_payload)
        )
        return self._rest_response_to_results(response)

    def batch_score(
        self,
        query_spectra: List[Spectrum],
        reference_spectra: List[Spectrum],
    ) -> np.ndarray:
        """Score pre-formed query-reference pairs remotely."""
        if len(query_spectra) != len(reference_spectra):
            raise ValueError(
                f"query_spectra and reference_spectra must have the same length; "
                f"got {len(query_spectra)} vs {len(reference_spectra)}"
            )
        if not query_spectra:
            return np.empty(0, dtype=np.float64)

        request_payload = {
            "algorithm": self._algorithm,
            "query_spectra": [_spectrum_to_payload(s) for s in query_spectra],
            "reference_spectra": [_spectrum_to_payload(s) for s in reference_spectra],
        }

        if self._is_grpc:
            response = self._breaker.execute(
                lambda: self._grpc_batch_score(request_payload)
            )
            return np.asarray(list(response.scores), dtype=np.float64)

        response = self._breaker.execute(
            lambda: self._rest_post("/batch_score", request_payload)
        )
        scores = response.get("scores", [])
        return np.asarray(scores, dtype=np.float64)

    def load_model(self, model_path: str | Path) -> None:
        """No-op: model weights are owned by the remote service."""
        logger.warning(
            "RemoteMLEngine '%s' does not load local models; ignoring "
            "load_model('%s').",
            self._algorithm,
            model_path,
        )

    # ------------------------------------------------------------------
    # REST transport
    # ------------------------------------------------------------------

    def _rest_post(self, path: str, payload: dict[str, Any]) -> Any:
        """POST *payload* as JSON to ``<endpoint><path>`` and parse JSON."""
        url = f"{self._endpoint}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _rest_response_to_results(self, response: Any) -> List["SearchResult"]:
        """Convert a REST search response into SearchResult dicts."""
        results: List["SearchResult"] = []
        for spectrum_result in response.get("results", []):
            query_id = str(spectrum_result.get("query_id", ""))
            results.extend(
                _hits_to_search_results(query_id, spectrum_result.get("hits", []))
            )
        return results

    # ------------------------------------------------------------------
    # gRPC transport
    # ------------------------------------------------------------------

    def _grpc_search(self, payload: dict[str, Any]) -> Any:
        """Call the gRPC Search RPC."""
        from MassFlow.generated.massflow.v1 import (  # type: ignore[import-untyped]
            ml_pb2 as pb,
        )

        request = pb.MLEngineRequest(  # type: ignore[attr-defined]
            algorithm=payload["algorithm"],
            query_spectra=[
                self._payload_to_pb_spectrum(s) for s in payload["query_spectra"]
            ],
            reference_spectra=[
                self._payload_to_pb_spectrum(s) for s in payload["reference_spectra"]
            ],
            min_score=payload["min_score"],
            top_n=payload["top_n"],
            include_decoys=payload["include_decoys"],
        )
        return self._get_grpc_stub().Search(request, timeout=self._timeout_seconds)

    def _grpc_batch_score(self, payload: dict[str, Any]) -> Any:
        """Call the gRPC BatchScore RPC."""
        from MassFlow.generated.massflow.v1 import (  # type: ignore[import-untyped]
            ml_pb2 as pb,
        )

        request = pb.BatchScoreRequest(  # type: ignore[attr-defined]
            algorithm=payload["algorithm"],
            query_spectra=[
                self._payload_to_pb_spectrum(s) for s in payload["query_spectra"]
            ],
            reference_spectra=[
                self._payload_to_pb_spectrum(s) for s in payload["reference_spectra"]
            ],
        )
        return self._get_grpc_stub().BatchScore(request, timeout=self._timeout_seconds)

    def _grpc_response_to_results(self, response: Any) -> List["SearchResult"]:
        """Convert a gRPC Search response into SearchResult dicts."""
        results: List["SearchResult"] = []
        for spectrum_result in response.results:
            hits_payload = [
                {
                    "reference_id": hit.reference_id,
                    "reference_name": hit.reference_name,
                    "score": hit.score,
                    "matched_peaks": hit.matched_peaks,
                    "smiles": hit.smiles,
                    "inchikey": hit.inchikey,
                    "q_value": hit.q_value,
                    "annotation_tier": hit.annotation_tier,
                    "structural_similarity": hit.structural_similarity,
                }
                for hit in spectrum_result.hits
            ]
            results.extend(
                _hits_to_search_results(spectrum_result.query_id, hits_payload)
            )
        return results

    @staticmethod
    def _payload_to_pb_spectrum(payload: dict[str, Any]) -> Any:
        """Build a protobuf SpectrumVector from a wire-format dict."""
        from MassFlow.generated.massflow.v1 import (  # type: ignore[import-untyped]
            ml_pb2 as pb,
        )

        return pb.SpectrumVector(  # type: ignore[attr-defined]
            spectrum_id=payload["spectrum_id"],
            mz_array=payload["mz_array"],
            intensity_array=payload["intensity_array"],
            metadata=payload["metadata"],
        )


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "RemoteMLEngine",
]
