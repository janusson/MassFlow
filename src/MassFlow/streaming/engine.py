"""
Streaming integration engine for real-time spectral annotation.

This module connects the gRPC ingestion pipeline to MassFlow's existing
``SimilarityEngine``.  It converts ``QueuedPacket`` instances into
``matchms.Spectrum`` objects, runs similarity search against a
pre-loaded reference library, and returns structured annotation
responses.

Design notes
------------
* The engine runs in a single async worker task to avoid GIL contention
  and keep latency predictable.  The similarity scoring itself is
  CPU-bound (NumPy) and releases the GIL where possible.
* Reference spectra are loaded once at server start and held in memory.
  A future optimisation could use a shared-memory ring-buffer for
  zero-copy handoff from the I/O thread.
"""

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np
from matchms import Spectrum

from MassFlow.config import MassFlowConfig
from MassFlow.similarity import SearchResult, get_similarity_engine

logger = logging.getLogger(__name__)


def _spectrum_from_packet(
    packet: "QueuedPacket",  # type: ignore[name-defined]  # noqa: F821
) -> Spectrum:
    """Convert a ``QueuedPacket`` into a ``matchms.Spectrum``.

    All numeric arrays preserve ``float64`` precision to comply with
    MassFlow's data-integrity conventions.  Missing values are
    represented as ``NaN`` (never coerced to zero).
    """
    mz = np.asarray(packet.mz_array, dtype=np.float64)
    intensities = np.asarray(packet.intensity_array, dtype=np.float64)

    # matchms Fragments requires m/z values in ascending order.
    if len(mz) > 0 and not np.all(mz[:-1] <= mz[1:]):
        sort_idx = np.argsort(mz)
        mz = mz[sort_idx]
        intensities = intensities[sort_idx]

    metadata = {
        "precursor_mz": float(packet.precursor_mz),
        "retention_time": float(packet.retention_time_seconds),
        "charge": int(packet.charge),
        "ionmode": packet.ion_mode if packet.ion_mode else "",
        "adduct": packet.adduct if packet.adduct else "",
        "collision_energy": float(packet.collision_energy),
    }

    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)


def _packet_to_search_result(
    packet: "QueuedPacket",  # type: ignore[name-defined]  # noqa: F821
    result: SearchResult,
) -> dict:
    """Convert a ``SearchResult`` and its originating packet into a dict
    suitable for building an ``AnnotationResponse`` protobuf message."""
    return {
        "spectrum_id": packet.spectrum_id,
        "reference_id": result["reference_id"],
        "reference_name": result.get("reference_name") or "",
        "smiles": result.get("smiles") or "",
        "inchikey": result.get("inchikey") or "",
        "score": float(result["score"]),
        "matched_peaks": int(result.get("matched_peaks", 0)),
        "precursor_mz_error_ppm": float(result.get("mass_error_ppm") or np.nan),
        "annotation_tier": result.get("annotation_tier") or "",
        "structural_similarity": float(result.get("structural_similarity") or np.nan),
        "fdr_q_value": float(result.get("q_value") or np.nan),
    }


class StreamingEngine:
    """Wraps ``SimilarityEngine`` for real-time, single-spectrum annotation.

    The engine holds a pre-loaded reference library and a configured
    ``SimilarityEngine``.  Each incoming spectrum is scored individually
    and the top-N hits are returned immediately.

    Parameters
    ----------
    config : MassFlowConfig
        Fully-parsed MassFlow configuration (used for similarity
        settings).
    reference_spectra : list of matchms.Spectrum
        Pre-processed reference library spectra.
    top_n : int
        Maximum number of annotation hits to return per spectrum.
    """

    def __init__(
        self,
        config: MassFlowConfig,
        reference_spectra: List[Spectrum],
        top_n: int = 5,
    ) -> None:
        self._config = config
        self._reference_spectra = reference_spectra
        self._top_n = top_n

        # Build the similarity engine and pre-compute hot-path arrays.
        self._engine = get_similarity_engine(config.similarity)

        # Pre-compute reference precursor m/z array (L2 cache pattern).
        self._ref_precursor_mzs = np.array(
            [s.get("precursor_mz", np.nan) for s in reference_spectra],
            dtype=np.float64,
        )
        self._ref_is_decoy = np.zeros(len(reference_spectra), dtype=bool)

        n_ref = len(reference_spectra)
        logger.info(
            "StreamingEngine initialised with %d reference spectra (top_n=%d).",
            n_ref,
            top_n,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        packet: "QueuedPacket",  # type: ignore[name-defined]  # noqa: F821
    ) -> dict:
        """Annotate a single spectrum and return a response dict.

        Parameters
        ----------
        packet : QueuedPacket
            The incoming spectral data.

        Returns
        -------
        dict
            A response payload with keys matching ``AnnotationResponse``
            fields: ``spectrum_id``, ``status``, ``error_message``,
            ``top_hits``, ``fdr_q_value``.
        """
        spectrum = _spectrum_from_packet(packet)
        t_start = time.perf_counter_ns()

        try:
            results: List[SearchResult] = self._engine.search(
                query_spectra=[spectrum],
                reference_spectra=self._reference_spectra,
                top_n=self._top_n,
                include_decoys=False,  # real-time path; decoys pre-computed if needed
                ref_precursor_mzs=self._ref_precursor_mzs,
                ref_is_decoy=self._ref_is_decoy,
            )
        except Exception:
            logger.exception(
                "Similarity search failed for spectrum %s.", packet.spectrum_id
            )
            return {
                "spectrum_id": packet.spectrum_id,
                "status": "error",
                "error_message": "Internal scoring engine failure.",
                "top_hits": [],
                "fdr_q_value": float("nan"),
            }

        latency_us = (time.perf_counter_ns() - t_start) / 1e3

        if not results:
            return {
                "spectrum_id": packet.spectrum_id,
                "status": "no_match",
                "error_message": "",
                "top_hits": [],
                "fdr_q_value": float("nan"),
            }

        hits = [_packet_to_search_result(packet, r) for r in results]
        best_q = float(results[0].get("q_value") or np.nan)

        logger.debug(
            "Annotated %s: %d hits in %.0f µs (best score=%.3f).",
            packet.spectrum_id,
            len(hits),
            latency_us,
            hits[0]["score"],
        )

        return {
            "spectrum_id": packet.spectrum_id,
            "status": "annotated",
            "error_message": "",
            "top_hits": hits,
            "fdr_q_value": best_q,
        }


def load_reference_library(
    config: MassFlowConfig,
) -> List[Spectrum]:
    """Load and pre-process the reference library from the config.

    This is a convenience wrapper that mirrors the first steps of the
    batch ``run_annotation_pipeline`` orchestrator.

    Parameters
    ----------
    config : MassFlowConfig
        Validated configuration containing ``library_path``.

    Returns
    -------
    list of matchms.Spectrum
        Processed reference spectra ready for similarity search.

    Raises
    ------
    RuntimeError
        If no spectra could be loaded from the configured library.
    """
    from MassFlow.io import load_spectra
    from MassFlow.processing import process_spectra

    library_path = config.input.library_path
    if library_path is None:
        raise RuntimeError("No library_path configured for streaming server.")

    logger.info("Loading reference library from %s", library_path)

    raw = load_spectra(library_path)
    raw_list = list(raw)
    if not raw_list:
        raise RuntimeError(f"No spectra loaded from library: {library_path}")

    processed = list(process_spectra(raw_list, config.processing))
    if not processed:
        raise RuntimeError("All spectra were discarded during library processing.")

    logger.info("Reference library ready: %d processed spectra.", len(processed))
    return processed
