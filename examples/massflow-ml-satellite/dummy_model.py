"""
Dummy binned-cosine scoring model for the massflow-ml satellite.

This module is the "model layer" of the reference satellite server.  It is
intentionally lightweight — a cosine scorer over fixed-width m/z bins whose
"weights" are nothing but binning hyperparameters read from ``model.json`` —
so the example stays runnable without PyTorch, Gensim, Spec2Vec, or
MS2DeepScore.  Swap the internals of this class for a real embedding model
without touching the wire contract.

Conventions shared with the MassFlow core:

* peak arrays are kept in ``float64`` precision (never ``float32``);
* search results use the exact wire-format key names consumed by
  ``MassFlow.ml_client.RemoteMLEngine._hits_to_search_results``
  (``reference_id``, ``reference_name``, ``score``, ``matched_peaks``,
  ``smiles``, ``inchikey``, ``q_value``, ``annotation_tier``,
  ``structural_similarity``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
from matchms import Spectrum

logger = logging.getLogger(__name__)

# Fallback hyperparameters used when no model.json is present.  Kept in sync
# with the checked-in model.json.
DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "model": "dummy-binned-cosine",
    "version": 1,
    "bin_width": 1.0,
    "mz_min": 0.0,
    "mz_max": 2000.0,
}


class DummyBinnedCosineModel:
    """Cosine similarity over fixed-width m/z bins.

    Each spectrum is collapsed into a float64 vector of length
    ``(mz_max - mz_min) / bin_width``: peaks falling into the same bin take
    the maximum intensity, intensities are sqrt-weighted (dampening the
    dominance of the base peak), and the vector is L2-normalized so the dot
    product of two vectors is their cosine similarity.

    Parameters
    ----------
    bin_width : float
        Width of each m/z bin in Da.
    mz_min : float
        Lower bound of the binned m/z range in Da.
    mz_max : float
        Upper bound of the binned m/z range in Da.
    """

    def __init__(self, *, bin_width: float, mz_min: float, mz_max: float) -> None:
        if bin_width <= 0.0:
            raise ValueError(f"bin_width must be positive; got {bin_width}.")
        if mz_max <= mz_min:
            raise ValueError(f"mz_max must exceed mz_min; got {mz_min}..{mz_max}.")
        self.bin_width = float(bin_width)
        self.mz_min = float(mz_min)
        self.mz_max = float(mz_max)
        self._bin_edges = np.arange(
            self.mz_min,
            self.mz_max + self.bin_width,
            self.bin_width,
            dtype=np.float64,
        )
        self._num_bins = int(len(self._bin_edges) - 1)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """The active binning hyperparameters (mirrors ``model.json``)."""
        return {
            "model": DEFAULT_HYPERPARAMETERS["model"],
            "version": DEFAULT_HYPERPARAMETERS["version"],
            "bin_width": self.bin_width,
            "mz_min": self.mz_min,
            "mz_max": self.mz_max,
        }

    @classmethod
    def load(cls, path: str | Path) -> "DummyBinnedCosineModel":
        """Load the model from a ``model.json`` file or its directory.

        Parameters
        ----------
        path : str or Path
            Path to ``model.json`` or to the directory containing it.  If
            the file is missing, the default hyperparameters are used.

        Returns
        -------
        DummyBinnedCosineModel
            A configured model instance.

        Raises
        ------
        ValueError
            If the model file is present but malformed.
        """
        model_path = Path(path)
        if model_path.is_dir():
            model_path = model_path / "model.json"
        if not model_path.exists():
            logger.warning(
                "Model file %s not found; using default hyperparameters %s.",
                model_path,
                DEFAULT_HYPERPARAMETERS,
            )
            hyperparameters = dict(DEFAULT_HYPERPARAMETERS)
        else:
            hyperparameters = json.loads(model_path.read_text(encoding="utf-8"))
            logger.info(
                "Loaded dummy model hyperparameters from %s: %s",
                model_path,
                hyperparameters,
            )
        try:
            return cls(
                bin_width=float(hyperparameters["bin_width"]),
                mz_min=float(hyperparameters["mz_min"]),
                mz_max=float(hyperparameters["mz_max"]),
            )
        except KeyError as error:
            raise ValueError(
                f"model file {model_path} is missing key {error}; expected "
                f"bin_width, mz_min, mz_max."
            ) from error

    # ------------------------------------------------------------------
    # Vectorization
    # ------------------------------------------------------------------

    def _bin_spectrum(self, spectrum: Spectrum) -> np.ndarray:
        """Return the sqrt-weighted, L2-normalized binned vector (float64)."""
        vector: np.ndarray = np.zeros(self._num_bins, dtype=np.float64)
        mz = np.asarray(spectrum.peaks.mz, dtype=np.float64)
        intensities = np.asarray(spectrum.peaks.intensities, dtype=np.float64)
        if mz.size == 0:
            return vector
        indices = np.floor((mz - self.mz_min) / self.bin_width).astype(np.int64)
        mask = (indices >= 0) & (indices < self._num_bins)
        # Collapse peaks sharing a bin with the maximum intensity.
        np.maximum.at(vector, indices[mask], intensities[mask])
        vector = np.sqrt(vector)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_pair(self, query: Spectrum, reference: Spectrum) -> tuple[float, int]:
        """Score one query-reference pair.

        Returns
        -------
        tuple[float, int]
            ``(cosine_score, matched_peaks)`` where *matched_peaks* is the
            number of bins occupied by both spectra.
        """
        query_vector = self._bin_spectrum(query)
        reference_vector = self._bin_spectrum(reference)
        if not np.any(query_vector) or not np.any(reference_vector):
            return 0.0, 0
        score = float(np.dot(query_vector, reference_vector))
        matched_peaks = int(
            np.count_nonzero((query_vector > 0.0) & (reference_vector > 0.0))
        )
        return score, matched_peaks

    def batch_score(
        self,
        query_spectra: Sequence[Spectrum],
        reference_spectra: Sequence[Spectrum],
    ) -> np.ndarray:
        """Score pre-formed query-reference pairs (elementwise alignment).

        Returns
        -------
        np.ndarray
            Float64 array of cosine scores, one per input pair.
        """
        scores = np.empty(len(query_spectra), dtype=np.float64)
        for index, (query, reference) in enumerate(
            zip(query_spectra, reference_spectra)
        ):
            scores[index], _ = self.score_pair(query, reference)
        return scores

    def search(
        self,
        query_spectra: Sequence[Spectrum],
        reference_spectra: Iterable[Spectrum],
        min_score: float = 0.0,
        top_n: int = 0,
        include_decoys: bool = False,
    ) -> List[dict[str, Any]]:
        """Rank references by cosine similarity for each query.

        Parameters
        ----------
        query_spectra : sequence of matchms.Spectrum
            Experimental spectra to annotate.
        reference_spectra : iterable of matchms.Spectrum
            Reference library spectra.
        min_score : float, optional
            Minimum cosine score for a hit to be retained (default 0.0).
        top_n : int, optional
            Maximum hits per query; 0 means unlimited (default 0).
        include_decoys : bool, optional
            Accepted for wire-contract compatibility.  The dummy model does
            not generate decoys; a real engine would append decoy hits here.

        Returns
        -------
        list of dict
            One entry per query: ``{"query_id": str, "hits": [ ... ]}``
            where each hit carries the ``massflow.v1.ml`` wire-format keys.
        """
        references = list(reference_spectra)
        results: List[dict[str, Any]] = []
        if not query_spectra or not references:
            return results
        if include_decoys:
            logger.info(
                "include_decoys=True: the dummy model does not generate "
                "decoys; real engines would append decoy hits here."
            )
        reference_vectors = np.stack(
            [self._bin_spectrum(reference) for reference in references]
        )
        for query in query_spectra:
            query_id = str(query.get("id") or "")
            query_vector = self._bin_spectrum(query)
            if not np.any(query_vector):
                results.append({"query_id": query_id, "hits": []})
                continue
            scores = np.einsum("ij,j->i", reference_vectors, query_vector)
            hits: List[dict[str, Any]] = []
            for reference, score in zip(references, scores):
                score = float(score)
                if score < float(min_score):
                    continue
                hits.append(self._hit_payload(query_vector, reference, score))
            hits.sort(key=lambda hit: hit["score"], reverse=True)
            if top_n > 0:
                hits = hits[:top_n]
            results.append({"query_id": query_id, "hits": hits})
        return results

    def _hit_payload(
        self, query_vector: np.ndarray, reference: Spectrum, score: float
    ) -> dict[str, Any]:
        """Build one wire-format hit dict for a scored reference."""
        reference_vector = self._bin_spectrum(reference)
        matched_peaks = int(
            np.count_nonzero((query_vector > 0.0) & (reference_vector > 0.0))
        )
        metadata = reference.metadata
        return {
            "reference_id": str(metadata.get("id") or ""),
            "reference_name": str(
                metadata.get("compound_name") or metadata.get("name") or ""
            )
            or None,
            "score": score,
            "matched_peaks": matched_peaks,
            "smiles": str(metadata.get("smiles") or "") or None,
            "inchikey": str(metadata.get("inchikey") or "") or None,
            # The dummy model applies no FDR; the core estimates q-values.
            "q_value": 1.0,
            "annotation_tier": None,
            # Not computed by this engine; the core maps null to NaN.
            "structural_similarity": None,
        }
