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
* **Micro-batching** (``MicroBatcher``) accumulates individual spectra
  over a configurable time window or batch size before dispatching to
  ``SimilarityEngine.search()``, amortising overhead at high ingestion
  rates.
* **Streaming validation** (``validate_streaming_spectrum``) acts as a
  pre-scoring gate: it constructs a Pydantic ``SpectrumMetadata`` to
  enforce the 5-ppm precursor-m/z sanity check and applies matchms
  peak-level filtering so that malformed or empty MS2 scans are
  rejected before they reach the similarity engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from matchms import Spectrum

from MassFlow.config import MassFlowConfig, ProcessingConfig
from MassFlow.similarity import SearchResult, get_similarity_engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming validation gate
# ---------------------------------------------------------------------------


class StreamingValidationError(Exception):
    """Raised when a streaming spectrum fails the pre-scoring validation gate."""


def validate_streaming_spectrum(
    packet: "QueuedPacket",  # type: ignore[name-defined]  # noqa: F821
    processing_config: ProcessingConfig,
) -> Spectrum:
    """Validate and pre-process a streaming spectrum before scoring.

    This gate performs two checks before the spectrum enters the
    similarity queue:

    1. **Pydantic model validation** – a temporary ``SpectrumMetadata``
       is constructed to enforce ``precursor_mz > 0``, valid charge, and
       the 5-ppm precursor mass logic (gracefully skipped when molecular
       information is absent, as it will be for query spectra).

    2. **matchms peak filtering** – the spectrum is passed through
       ``require_minimum_number_of_peaks`` and ``select_by_intensity``
       so that un-centroided or empty MS2 arrays are rejected with a
       structured error rather than crashing the gRPC worker.

    Parameters
    ----------
    packet : QueuedPacket
        The incoming spectral data.
    processing_config : ProcessingConfig
        Peak-level filtering parameters (min_peaks, noise_threshold, …).

    Returns
    -------
    matchms.Spectrum
        A validated, sorted, centroid-like spectrum ready for scoring.

    Raises
    ------
    StreamingValidationError
        If the spectrum fails any preprocessing gate.
    """
    # --- Basic array sanity ---
    if not packet.mz_array or not packet.intensity_array:
        raise StreamingValidationError(
            f"Spectrum {packet.spectrum_id}: empty m/z or intensity array."
        )

    if len(packet.mz_array) != len(packet.intensity_array):
        raise StreamingValidationError(
            f"Spectrum {packet.spectrum_id}: m/z and intensity arrays "
            f"have mismatched lengths ({len(packet.mz_array)} vs "
            f"{len(packet.intensity_array)})."
        )

    mz = np.asarray(packet.mz_array, dtype=np.float64)
    intensities = np.asarray(packet.intensity_array, dtype=np.float64)

    # Reject arrays containing NaN or Inf.
    if np.any(np.isnan(mz)) or np.any(np.isnan(intensities)):
        raise StreamingValidationError(
            f"Spectrum {packet.spectrum_id}: NaN values in peak arrays."
        )
    if np.any(np.isinf(mz)) or np.any(np.isinf(intensities)):
        raise StreamingValidationError(
            f"Spectrum {packet.spectrum_id}: Inf values in peak arrays."
        )

    # Reject zero-length after NaN/Inf purge.
    if len(mz) == 0:
        raise StreamingValidationError(
            f"Spectrum {packet.spectrum_id}: no valid peaks after NaN/Inf removal."
        )

    # --- Pydantic model validation (5 ppm gate) ---
    _validate_precursor_ppm(packet)

    # --- matchms peak processing ---
    # Sort m/z ascending (required by matchms Fragments).
    if len(mz) > 0 and not np.all(mz[:-1] <= mz[1:]):
        sort_idx = np.argsort(mz)
        mz = mz[sort_idx]
        intensities = intensities[sort_idx]

    # Minimum peak count gate.
    # Always reject truly empty spectra (0 peaks) as a basic safety check.
    # The config-driven ``min_peaks`` threshold is only enforced when
    # ``filter_min_peaks`` is set, matching ``peak_processing`` behaviour.
    from matchms.filtering import require_minimum_number_of_peaks

    spec = Spectrum(mz=mz, intensities=intensities)
    # Basic safety: at least 1 peak.
    spec = require_minimum_number_of_peaks(spec, n_required=1)
    if spec is None:
        raise StreamingValidationError(
            f"Spectrum {packet.spectrum_id}: no peaks after basic filtering."
        )

    if getattr(processing_config, "filter_min_peaks", False):
        min_peaks = getattr(processing_config, "min_peaks", 1)
        if min_peaks > 1:
            spec = require_minimum_number_of_peaks(spec, n_required=min_peaks)
            if spec is None:
                raise StreamingValidationError(
                    f"Spectrum {packet.spectrum_id}: fewer than {min_peaks} "
                    f"peaks after filtering."
                )

    # Intensity filter (noise threshold).  Gate on ``filter_by_intensity``
    # to match the behaviour of ``peak_processing`` in processing.py.
    if getattr(processing_config, "filter_by_intensity", False):
        noise_threshold = getattr(processing_config, "noise_threshold", 0.0)
        if noise_threshold > 0:
            from matchms.filtering import select_by_intensity

            spec = select_by_intensity(
                spec, intensity_from=noise_threshold, intensity_to=float("inf")
            )
            if spec is None or len(spec.peaks) == 0:
                raise StreamingValidationError(
                    f"Spectrum {packet.spectrum_id}: all peaks below noise "
                    f"threshold ({noise_threshold})."
                )

            # Re-check minimum peaks post intensity filter.
            if getattr(processing_config, "filter_min_peaks", False):
                spec = require_minimum_number_of_peaks(spec, n_required=min_peaks)
                if spec is None:
                    raise StreamingValidationError(
                        f"Spectrum {packet.spectrum_id}: fewer than {min_peaks} "
                        f"peaks after intensity filter."
                    )

    # Attach metadata that the similarity engine expects.
    spec.set("precursor_mz", float(packet.precursor_mz))
    spec.set("retention_time", float(packet.retention_time_seconds))
    spec.set("charge", int(packet.charge))
    spec.set("ionmode", packet.ion_mode if packet.ion_mode else "")
    spec.set("adduct", packet.adduct if packet.adduct else "")
    spec.set("collision_energy", float(packet.collision_energy))
    spec.set("id", packet.spectrum_id)

    return spec


def _validate_precursor_ppm(
    packet: "QueuedPacket",  # type: ignore[name-defined]  # noqa: F821
) -> None:
    """Run the Pydantic ``SpectrumMetadata`` validator on the packet.

    For streaming query spectra the 5-ppm strict mass check is gracefully
    skipped because molecular information is absent, but basic field
    constraints (e.g. ``precursor_mz > 0``) are enforced by Pydantic.

    Raises
    ------
    StreamingValidationError
        If the spectrum fails Pydantic field validation.
    """
    try:
        from MassFlow.models import SpectrumMetadata

        SpectrumMetadata(
            spectrum_id=packet.spectrum_id,
            precursor_mz=packet.precursor_mz,
            retention_time=packet.retention_time_seconds,
            charge=packet.charge if packet.charge != 0 else None,
            ion_mode=packet.ion_mode if packet.ion_mode else None,
            adduct=packet.adduct if packet.adduct else None,
            collision_energy=packet.collision_energy,
        )
    except Exception as exc:
        raise StreamingValidationError(
            f"Spectrum {packet.spectrum_id}: precursor validation failed — {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Packet → Spectrum conversion
# ---------------------------------------------------------------------------


def _spectrum_from_packet(
    packet: "QueuedPacket",  # type: ignore[name-defined]  # noqa: F821
) -> Spectrum:
    """Convert a ``QueuedPacket`` into a ``matchms.Spectrum``.

    All numeric arrays preserve ``float64`` precision to comply with
    MassFlow's data-integrity conventions.  Missing values are
    represented as ``NaN`` (never coerced to zero).

    .. note::
       Prefer ``validate_streaming_spectrum()`` for the streaming path;
       this function is retained for backward compatibility.
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


# ---------------------------------------------------------------------------
# Micro-batcher
# ---------------------------------------------------------------------------


@dataclass
class MicroBatcher:
    """Accumulate spectra and dispatch batches to the similarity engine.

    Designed for the streaming ingestion path: individual spectra arrive
    at variable frequency (10–200 Hz).  The batcher groups them so that
    ``SimilarityEngine.search()`` sees a batch, amortising the per-call
    overhead of the matchms scoring machinery.

    Parameters
    ----------
    batch_max_size : int
        Maximum number of spectra per batch.  When this threshold is
        reached the batch is dispatched immediately.
    batch_timeout_seconds : float
        Maximum time (in seconds) to wait before dispatching a partial
        batch.  A smaller value reduces end-to-end latency at low
        ingestion rates; a larger value yields bigger batches.

    Attributes
    ----------
    pending_count : int
        Number of spectra currently accumulated in the internal buffer.
    """

    batch_max_size: int = 64
    batch_timeout_seconds: float = 0.050  # 50 ms
    _packets: list["QueuedPacket"] = field(default_factory=list)  # type: ignore[name-defined]  # noqa: F821
    _spectra: list[Spectrum] = field(default_factory=list)
    _deadline_ns: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def pending_count(self) -> int:
        return len(self._packets)

    async def add(
        self,
        packet: "QueuedPacket",  # type: ignore[name-defined]  # noqa: F821
        spectrum: Spectrum,
    ) -> tuple[list["QueuedPacket"], list[Spectrum]] | None:  # type: ignore[name-defined]  # noqa: F821
        """Add a validated packet+spectrum pair to the accumulator.

        Returns a ``(packets, spectra)`` tuple when the batch should be
        dispatched, or ``None`` if the item was buffered and the batch
        is not yet ready.

        The caller **must** hold the returned tuple and dispatch the
        batch to the engine before calling ``add()`` again (the lock
        ensures serial access).
        """
        async with self._lock:
            now_ns = time.time_ns()
            self._packets.append(packet)
            self._spectra.append(spectrum)

            # First item starts the deadline clock.
            if self._deadline_ns == 0:
                self._deadline_ns = now_ns + int(self.batch_timeout_seconds * 1e9)

            # Dispatch when batch is full or deadline has passed.
            if len(self._packets) >= self.batch_max_size or (
                now_ns >= self._deadline_ns
            ):
                return self._flush()

        return None

    async def flush(self) -> tuple[list["QueuedPacket"], list[Spectrum]] | None:  # type: ignore[name-defined]  # noqa: F821
        """Force-dispatch any accumulated items.

        Returns ``None`` if the accumulator is empty.
        """
        async with self._lock:
            if not self._packets:
                return None
            return self._flush()

    def _flush(
        self,
    ) -> tuple[list["QueuedPacket"], list[Spectrum]]:  # type: ignore[name-defined]  # noqa: F821
        """Release the current batch and reset state.  Caller must hold ``_lock``."""
        packets = self._packets
        spectra = self._spectra
        self._packets = []
        self._spectra = []
        self._deadline_ns = 0
        return packets, spectra


# ---------------------------------------------------------------------------
# Streaming Engine
# ---------------------------------------------------------------------------


class StreamingEngine:
    """Wraps ``SimilarityEngine`` for real-time spectral annotation.

    Supports both single-spectrum and micro-batched annotation paths.

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
                include_decoys=False,
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

        return _build_annotation_response(packet, results, latency_us)

    def annotate_batch(
        self,
        packets: List["QueuedPacket"],  # type: ignore[name-defined]  # noqa: F821
        spectra: List[Spectrum],
    ) -> List[dict]:
        """Annotate a micro-batch of spectra in a single engine call.

        The batch is dispatched to ``SimilarityEngine.search()`` as a
        single call, which computes the full N×M similarity matrix
        efficiently.  Results are then partitioned back to their
        originating packets.

        Parameters
        ----------
        packets : list of QueuedPacket
            Originating packets, in the same order as ``spectra``.
        spectra : list of matchms.Spectrum
            Pre-validated spectra ready for scoring.

        Returns
        -------
        list of dict
            One response dict per input packet (same order).
        """
        if not spectra:
            return []

        t_start = time.perf_counter_ns()
        batch_latency_us: float = 0.0

        try:
            all_results: List[SearchResult] = self._engine.search(
                query_spectra=spectra,
                reference_spectra=self._reference_spectra,
                top_n=self._top_n,
                include_decoys=False,
                ref_precursor_mzs=self._ref_precursor_mzs,
                ref_is_decoy=self._ref_is_decoy,
            )
        except Exception:
            logger.exception(
                "Batch similarity search failed (%d spectra).", len(spectra)
            )
            batch_latency_us = (time.perf_counter_ns() - t_start) / 1e3
            return [
                {
                    "spectrum_id": pkt.spectrum_id,
                    "status": "error",
                    "error_message": "Internal scoring engine failure.",
                    "top_hits": [],
                    "fdr_q_value": float("nan"),
                }
                for pkt in packets
            ]

        batch_latency_us = (time.perf_counter_ns() - t_start) / 1e3

        # Partition results by query spectrum using query_id indexing.
        # Build a mapping: query_id → packet
        pkt_by_id: dict[str, "QueuedPacket"] = {  # type: ignore[name-defined]  # noqa: F821
            p.spectrum_id: p for p in packets
        }

        # Group results by query_id.
        from collections import defaultdict

        hits_by_qid: dict[str, list[dict]] = defaultdict(list)
        for r in all_results:
            qid = r.get("query_id", "")
            if qid in pkt_by_id:
                hits_by_qid[qid].append(_packet_to_search_result(pkt_by_id[qid], r))

        # Build per-packet responses in input order.
        responses: list[dict] = []
        for pkt in packets:
            hits = hits_by_qid.get(pkt.spectrum_id, [])
            if not hits:
                responses.append(
                    {
                        "spectrum_id": pkt.spectrum_id,
                        "status": "no_match",
                        "error_message": "",
                        "top_hits": [],
                        "fdr_q_value": float("nan"),
                    }
                )
            else:
                best_q = float(hits[0].get("fdr_q_value") or np.nan)
                responses.append(
                    {
                        "spectrum_id": pkt.spectrum_id,
                        "status": "annotated",
                        "error_message": "",
                        "top_hits": hits,
                        "fdr_q_value": best_q,
                    }
                )

        per_spectrum_us = (
            batch_latency_us / len(spectra) if spectra else batch_latency_us
        )
        logger.debug(
            "Batch annotated %d spectra in %.0f µs (%.0f µs/spectrum).",
            len(spectra),
            batch_latency_us,
            per_spectrum_us,
        )

        return responses


def _build_annotation_response(
    packet: "QueuedPacket",  # type: ignore[name-defined]  # noqa: F821
    results: List[SearchResult],
    latency_us: float,
) -> dict:
    """Build a response dict from search results for a single packet."""
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


# ---------------------------------------------------------------------------
# Reference library loader
# ---------------------------------------------------------------------------


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
