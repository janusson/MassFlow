"""
gRPC reference server for the massflow-ml satellite (massflow.v1.ml).

Implements the ``MLEngineService`` contract defined in
``protos/massflow/v1/ml.proto`` on top of the in-repo generated stubs
(``MassFlow.generated.massflow.v1.ml_pb2_grpc``).  The servicer is
asynchronous (``grpc.aio``) and delegates scoring to the dummy binned-cosine
model in ``dummy_model.py``; a real satellite would swap in Spec2Vec or
MS2DeepScore here without changing the servicer.

Run from the repository root:

    uv run python examples/massflow-ml-satellite/grpc_server.py --port 9090
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import grpc
import numpy as np
from matchms import Spectrum

from MassFlow.generated.massflow.v1 import (  # type: ignore[import-untyped]
    ml_pb2 as _ml_pb2_module,
    ml_pb2_grpc,
)

from dummy_model import DummyBinnedCosineModel

# The protobuf runtime registers message classes dynamically at import
# time, so mypy cannot resolve them as attributes of the generated module.
# Route message construction through an Any-typed alias — the same
# convention used by ``MassFlow.ml_client`` for the generated stubs.
ml_pb2: Any = _ml_pb2_module

logger = logging.getLogger(__name__)

# Directory holding model.json (override with MASSFLOW_ML_MODEL_DIR).
MODEL_DIR = Path(
    os.environ.get("MASSFLOW_ML_MODEL_DIR", str(Path(__file__).resolve().parent))
)


# ---------------------------------------------------------------------------
# Protobuf <-> matchms translation
# ---------------------------------------------------------------------------


def _spectrum_from_pb(spectrum_vector: ml_pb2.SpectrumVector) -> Spectrum:
    """Translate a protobuf ``SpectrumVector`` into a matchms.Spectrum."""
    metadata: dict[str, object] = {
        str(key): str(value) for key, value in spectrum_vector.metadata.items()
    }
    spectrum = Spectrum(
        mz=np.asarray(spectrum_vector.mz_array, dtype=np.float64),
        intensities=np.asarray(spectrum_vector.intensity_array, dtype=np.float64),
        metadata=metadata,
    )
    if spectrum_vector.spectrum_id:
        spectrum.set("id", spectrum_vector.spectrum_id)
    return spectrum


def _results_to_pb(results: list[dict[str, Any]]) -> ml_pb2.MLEngineResponse:
    """Translate dummy-model search results into an ``MLEngineResponse``."""
    response = ml_pb2.MLEngineResponse()
    for result in results:
        spectrum_results = response.results.add()
        spectrum_results.query_id = str(result["query_id"])
        for hit in result["hits"]:
            pb_hit = spectrum_results.hits.add()
            pb_hit.reference_id = str(hit["reference_id"] or "")
            pb_hit.reference_name = str(hit["reference_name"] or "")
            pb_hit.score = float(hit["score"])
            pb_hit.matched_peaks = int(hit["matched_peaks"])
            pb_hit.smiles = str(hit["smiles"] or "")
            pb_hit.inchikey = str(hit["inchikey"] or "")
            pb_hit.q_value = float(hit["q_value"])
            pb_hit.annotation_tier = str(hit["annotation_tier"] or "")
            # Unset protobuf doubles default to 0.0; the core maps 0.0 to
            # NaN, so send NaN explicitly to preserve "not computed".
            pb_hit.structural_similarity = (
                float(hit["structural_similarity"])
                if hit["structural_similarity"] is not None
                else float("nan")
            )
    return response


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MLEngineService(ml_pb2_grpc.MLEngineServiceServicer):
    """Async gRPC servicer backed by the dummy binned-cosine model."""

    def __init__(self, model: DummyBinnedCosineModel) -> None:
        self._model = model

    async def Search(
        self,
        request: ml_pb2.MLEngineRequest,
        context: grpc.aio.ServicerContext,
    ) -> ml_pb2.MLEngineResponse:
        """Ranked similarity search over a reference library."""
        logger.info(
            "gRPC Search: algorithm=%s, %d queries x %d references (dummy model)",
            request.algorithm,
            len(request.query_spectra),
            len(request.reference_spectra),
        )
        query_spectra = [
            _spectrum_from_pb(spectrum_vector)
            for spectrum_vector in request.query_spectra
        ]
        reference_spectra = [
            _spectrum_from_pb(spectrum_vector)
            for spectrum_vector in request.reference_spectra
        ]
        results = self._model.search(
            query_spectra,
            reference_spectra,
            min_score=request.min_score,
            top_n=request.top_n,
            include_decoys=request.include_decoys,
        )
        return _results_to_pb(results)

    async def BatchScore(
        self,
        request: ml_pb2.BatchScoreRequest,
        context: grpc.aio.ServicerContext,
    ) -> ml_pb2.BatchScoreResponse:
        """Score pre-formed query-reference pairs in one batch call."""
        if len(request.query_spectra) != len(request.reference_spectra):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "query_spectra and reference_spectra must have the same "
                f"length; got {len(request.query_spectra)} vs "
                f"{len(request.reference_spectra)}",
            )
        logger.info(
            "gRPC BatchScore: algorithm=%s, %d pairs (dummy model)",
            request.algorithm,
            len(request.query_spectra),
        )
        query_spectra = [
            _spectrum_from_pb(spectrum_vector)
            for spectrum_vector in request.query_spectra
        ]
        reference_spectra = [
            _spectrum_from_pb(spectrum_vector)
            for spectrum_vector in request.reference_spectra
        ]
        scores = self._model.batch_score(query_spectra, reference_spectra)
        response = ml_pb2.BatchScoreResponse()
        response.scores.extend(float(score) for score in scores)
        return response


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


async def serve(model: DummyBinnedCosineModel, port: int, host: str = "[::]") -> None:
    """Run the gRPC server until termination (asyncio-managed lifecycle)."""
    server = grpc.aio.server()
    ml_pb2_grpc.add_MLEngineServiceServicer_to_server(MLEngineService(model), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info("massflow-ml satellite (gRPC) listening on %s:%d", host, port)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5.0)


def main() -> None:
    """CLI entry point: ``python grpc_server.py --port 9090``."""
    parser = argparse.ArgumentParser(
        description="massflow-ml satellite gRPC server (massflow.v1.ml)."
    )
    parser.add_argument(
        "--port", type=int, default=9090, help="TCP port (default: 9090)."
    )
    parser.add_argument("--host", default="[::]", help="Bind address (default: [::]).")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory holding model.json (default: this directory).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    model = DummyBinnedCosineModel.load(args.model_dir)
    asyncio.run(serve(model, port=args.port, host=args.host))


if __name__ == "__main__":
    main()
