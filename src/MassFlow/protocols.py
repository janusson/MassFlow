"""
Protocol contracts for MassFlow plugin architecture.

This module defines the abstract base classes that external packages
(such as ``massflow-ml``) must implement to register ML-based similarity
scoring engines with the core MassFlow pipeline.

Core MassFlow has **zero** dependency on PyTorch, Gensim, spec2vec, or
ms2deepscore.  External plugins register their engine classes via the
``massflow.similarity_engines`` entry-point group, and the factory in
``similarity.py`` discovers them dynamically at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List

import numpy as np

if TYPE_CHECKING:
    from MassFlow.similarity import SearchResult
    from matchms import Spectrum


class MLEngineProtocol(ABC):
    """Abstract protocol that every external ML scoring engine must implement.

    Engines registered via the ``massflow.similarity_engines`` entry-point
    group are discovered at runtime by ``get_similarity_engine()``.  Any
    callable returned by an entry point MUST return an object conforming to
    this protocol.

    Subclasses must implement three methods:

    ``search()``
        Run similarity search of query spectra against a reference library.
    ``batch_score()``
        Score pre-formed query--reference pairs in a single batch call
        (useful for validation / numerical-parity testing).
    ``load_model()``
        Load pre-trained model weights or artefacts from disk.
    """

    @abstractmethod
    def search(
        self,
        query_spectra: List["Spectrum"],
        reference_spectra: Iterable["Spectrum"],
        min_score: float | None = None,
        top_n: int | None = None,
        include_decoys: bool = True,
        ref_precursor_mzs: np.ndarray | None = None,
        ref_is_decoy: np.ndarray | None = None,
        decoy_min_relative_intensity: float | None = None,
        decoy_mz_shift_da: float | None = None,
    ) -> List["SearchResult"]:
        """Run similarity search and return ranked results.

        Parameters
        ----------
        query_spectra : list of matchms.Spectrum
            Experimental spectra to annotate.
        reference_spectra : iterable of matchms.Spectrum
            Reference library spectra.
        min_score : float or None
            Minimum similarity score threshold (0.0–1.0).
        top_n : int or None
            Maximum hits to return per query spectrum.
        include_decoys : bool
            Whether to generate and score decoy spectra for FDR estimation.
        ref_precursor_mzs : np.ndarray or None
            Pre-computed reference precursor m/z array (float64).
        ref_is_decoy : np.ndarray or None
            Pre-computed boolean decoy flags for references.
        decoy_min_relative_intensity : float or None
            Decoy noise floor (fraction of the base peak); forwarded to
            engines that generate decoys locally. Ignored by engines that
            score decoys supplied in ``reference_spectra``.
        decoy_mz_shift_da : float or None
            Decoy m/z jitter (Da); forwarded to engines that generate
            decoys locally.

        Returns
        -------
        list of SearchResult
            Ranked list of ``SearchResult`` dicts.
        """
        ...

    @abstractmethod
    def batch_score(
        self,
        query_spectra: List["Spectrum"],
        reference_spectra: List["Spectrum"],
    ) -> np.ndarray:
        """Score pre-formed query--reference pairs in batch.

        Element *i* of ``query_spectra`` is paired with element *i* of
        ``reference_spectra``.  The returned array has shape ``(N,)`` where
        *N* is the number of input pairs.

        Parameters
        ----------
        query_spectra : list of matchms.Spectrum
            List of *N* query spectra.
        reference_spectra : list of matchms.Spectrum
            List of *N* reference spectra (same length as ``query_spectra``).

        Returns
        -------
        np.ndarray
            Array of *N* float64 similarity scores.
        """
        ...

    @abstractmethod
    def load_model(self, model_path: str | Path) -> None:
        """Load pre-trained model weights or artefacts.

        Parameters
        ----------
        model_path : str or Path
            Filesystem path to the model checkpoint directory or file.

        Raises
        ------
        FileNotFoundError
            If the model path does not exist or is inaccessible.
        RuntimeError
            If the model file is corrupted or incompatible.
        """
        ...
