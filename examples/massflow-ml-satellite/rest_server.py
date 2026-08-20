"""
FastAPI reference server for the massflow-ml satellite (REST transport).

Serves the JSON flavour of the ``massflow.v1.ml`` wire contract consumed by
``MassFlow.ml_client.RemoteMLEngine``:

* ``POST /{algorithm}/search``  — ranked similarity search
* ``POST /{algorithm}/batch_score`` — score pre-formed query-reference pairs
* ``GET  /health`` — liveness + model info

The ``{algorithm}`` path segment is accepted for any algorithm name; the
dummy model serves both ``spec2vec`` and ``ms2deepscore`` and logs which one
was requested, demonstrating how a real satellite would dispatch to
different engines.  All scientific payloads are float64.

Run from the repository root:

    uv run uvicorn rest_server:app --app-dir examples/massflow-ml-satellite \
        --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from matchms import Spectrum
from pydantic import BaseModel, Field

from dummy_model import DummyBinnedCosineModel

logger = logging.getLogger(__name__)

# Directory holding model.json (override with MASSFLOW_ML_MODEL_DIR).
MODEL_DIR = Path(
    os.environ.get("MASSFLOW_ML_MODEL_DIR", str(Path(__file__).resolve().parent))
)

app = FastAPI(
    title="massflow-ml satellite (reference implementation)",
    description=(
        "Reference server fulfilling the massflow.v1.ml contract for the "
        "decoupled MassFlow ML scoring boundary."
    ),
    version="0.1.0",
)

_model: DummyBinnedCosineModel | None = None


def get_model() -> DummyBinnedCosineModel:
    """Return the process-wide dummy model, loading it on first use."""
    global _model
    if _model is None:
        _model = DummyBinnedCosineModel.load(MODEL_DIR)
    return _model


# ---------------------------------------------------------------------------
# Wire-format Pydantic models (JSON mirror of massflow.v1.ml messages)
# ---------------------------------------------------------------------------


class SpectrumVectorPayload(BaseModel):
    """Serialized spectrum: float64 peak arrays + string metadata."""

    spectrum_id: str = ""
    mz_array: list[float]
    intensity_array: list[float]
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Mirror of ``MLEngineRequest`` (minus the service fields)."""

    algorithm: str = "spec2vec"
    query_spectra: list[SpectrumVectorPayload]
    reference_spectra: list[SpectrumVectorPayload]
    min_score: float = 0.0
    top_n: int = 0
    include_decoys: bool = False


class AnnotationHitResponse(BaseModel):
    """Mirror of ``AnnotationHit``."""

    reference_id: str
    reference_name: str | None = None
    score: float
    matched_peaks: int
    smiles: str | None = None
    inchikey: str | None = None
    q_value: float = 1.0
    annotation_tier: str | None = None
    structural_similarity: float | None = None


class SpectrumResultsResponse(BaseModel):
    """Mirror of ``SpectrumResults``."""

    query_id: str
    hits: list[AnnotationHitResponse]


class SearchResponse(BaseModel):
    """Mirror of ``MLEngineResponse``."""

    results: list[SpectrumResultsResponse]


class BatchScoreRequest(BaseModel):
    """Mirror of ``BatchScoreRequest``."""

    algorithm: str = "spec2vec"
    query_spectra: list[SpectrumVectorPayload]
    reference_spectra: list[SpectrumVectorPayload]


class BatchScoreResponse(BaseModel):
    """Mirror of ``BatchScoreResponse``."""

    scores: list[float]


class HealthResponse(BaseModel):
    """Liveness response with model provenance."""

    status: str
    model: str
    version: int
    bin_width: float
    mz_min: float
    mz_max: float


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _spectrum_from_payload(payload: SpectrumVectorPayload) -> Spectrum:
    """Translate a wire-format dict into a matchms.Spectrum (float64)."""
    spectrum = Spectrum(
        mz=np.asarray(payload.mz_array, dtype=np.float64),
        intensities=np.asarray(payload.intensity_array, dtype=np.float64),
        metadata=dict(payload.metadata),
    )
    if payload.spectrum_id:
        spectrum.set("id", payload.spectrum_id)
    return spectrum


def _hit_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one dummy-model hit dict into JSON-safe native types."""
    return {
        "reference_id": str(payload.get("reference_id") or ""),
        "reference_name": payload.get("reference_name"),
        "score": float(payload["score"]),
        "matched_peaks": int(payload["matched_peaks"]),
        "smiles": payload.get("smiles"),
        "inchikey": payload.get("inchikey"),
        "q_value": float(payload["q_value"]),
        "annotation_tier": payload.get("annotation_tier"),
        "structural_similarity": payload.get("structural_similarity"),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe with the loaded model's hyperparameters."""
    hyperparameters = get_model().hyperparameters
    return HealthResponse(
        status="ok",
        model=str(hyperparameters["model"]),
        version=int(hyperparameters["version"]),
        bin_width=float(hyperparameters["bin_width"]),
        mz_min=float(hyperparameters["mz_min"]),
        mz_max=float(hyperparameters["mz_max"]),
    )


@app.post("/{algorithm}/search", response_model=SearchResponse)
def search(algorithm: str, request: SearchRequest) -> SearchResponse:
    """Ranked similarity search served by the dummy model."""
    logger.info(
        "REST /%s/search: %d queries x %d references (dummy model)",
        algorithm,
        len(request.query_spectra),
        len(request.reference_spectra),
    )
    query_spectra = [
        _spectrum_from_payload(payload) for payload in request.query_spectra
    ]
    reference_spectra = [
        _spectrum_from_payload(payload) for payload in request.reference_spectra
    ]
    model_results = get_model().search(
        query_spectra,
        reference_spectra,
        min_score=request.min_score,
        top_n=request.top_n,
        include_decoys=request.include_decoys,
    )
    return SearchResponse(
        results=[
            SpectrumResultsResponse(
                query_id=str(result["query_id"]),
                hits=[
                    AnnotationHitResponse(**_hit_response(hit))
                    for hit in result["hits"]
                ],
            )
            for result in model_results
        ]
    )


@app.post("/{algorithm}/batch_score", response_model=BatchScoreResponse)
def batch_score(algorithm: str, request: BatchScoreRequest) -> BatchScoreResponse:
    """Score pre-formed query-reference pairs in one batch call."""
    logger.info(
        "REST /%s/batch_score: %d pairs (dummy model)",
        algorithm,
        len(request.query_spectra),
    )
    if len(request.query_spectra) != len(request.reference_spectra):
        raise HTTPException(
            status_code=422,
            detail=(
                "query_spectra and reference_spectra must have the same "
                "length; got "
                f"{len(request.query_spectra)} vs {len(request.reference_spectra)}"
            ),
        )
    query_spectra = [
        _spectrum_from_payload(payload) for payload in request.query_spectra
    ]
    reference_spectra = [
        _spectrum_from_payload(payload) for payload in request.reference_spectra
    ]
    scores = get_model().batch_score(query_spectra, reference_spectra)
    return BatchScoreResponse(scores=[float(score) for score in scores])
