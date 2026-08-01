"""
Abstract storage interface and factory for MassFlow spectral data persistence.

This module defines the formal :class:`SpectralStore` contract that all storage
backends must implement, alongside a factory function that resolves a backend
identifier (``"sqlite"`` or ``"zarr"``) to a concrete store instance.

Storage backends implement:
- Writing spectra (with optional batch insertion).
- Retrieving spectra as ``matchms.Spectrum`` iterators.
- Single-spectrum lookup by identifier.
- High-throughput batch array retrieval for similarity matrix construction.
- Aggregate metadata queries (total counts, category breakdowns, m/z ranges).

The default backend for v0.1 workflows is ``"sqlite"``, which uses the existing
``SpectralDatabase`` class. The ``"zarr"`` backend is opt-in via configuration
and is intended for horizontal-scaling and cloud-native distributed search
scenarios.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
from matchms import Spectrum


class SpectralStore(ABC):
    """
    Abstract interface for spectral data persistence.

    All storage backends (SQLite BLOB, Zarr, or future implementations) must
    implement this contract so downstream components (I/O layer, CLI, workflow)
    can operate against a uniform API.

    Parameters
    ----------
    store_path : Path
        File-system path to the underlying store (a ``.db`` file for SQLite, a
        directory for Zarr groups, etc.).

    Notes
    -----
    Implementations are responsible for their own resource lifecycle (open /
    close), schema migrations, and thread-safety guarantees. The contract
    does **not** prescribe transactional semantics—callers must not assume
    that writes within a single ``add_spectra`` call are atomic across all
    backends.
    """

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """
        Create or open the underlying storage structures.

        Called once during ``__init__``. Implementations should handle both
        first-time creation and re-opening of an existing store.
        """
        ...

    @abstractmethod
    def add_spectra(
        self,
        spectra: Iterator[Spectrum],
        category: str = "default",
        batch_size: int = 5000,
    ) -> int:
        """
        Persist a stream of spectra into the store.

        Parameters
        ----------
        spectra : Iterator[Spectrum]
            An iterator (potentially streaming/lazy) of ``matchms.Spectrum``
            objects to insert.
        category : str
            A category label to tag all inserted spectra (e.g. ``"library"``,
            ``"merged"``).
        batch_size : int
            Hint for internal batching / commit granularity. Implementations
            may use this to bound memory usage during large inserts.

        Returns
        -------
        int
            The number of spectra successfully persisted.
        """
        ...

    @abstractmethod
    def get_spectra(
        self,
        category: Optional[str] = None,
        name_pattern: Optional[str] = None,
    ) -> Iterator[Spectrum]:
        """
        Stream spectra from the store, optionally filtered.

        Parameters
        ----------
        category : str or None
            If provided, only return spectra whose ``category`` matches.
        name_pattern : str or None
            If provided, only return spectra whose compound name matches the
            SQL ``LIKE``-style pattern.

        Yields
        ------
        Spectrum
            Reconstructed ``matchms.Spectrum`` objects.
        """
        ...

    @abstractmethod
    def get_spectrum_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """
        Retrieve a single spectrum by its unique identifier.

        Parameters
        ----------
        spectrum_id : str
            The ``original_id`` or ``id`` of the desired spectrum.

        Returns
        -------
        Spectrum or None
            The spectrum if found, otherwise ``None``.
        """
        ...

    @abstractmethod
    def batch_get_arrays(
        self,
        spectrum_ids: Optional[list[str]] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Retrieve m/z and intensity arrays in batch for matrix operations.

        This method is designed for high-throughput similarity engines that
        need fast, parallel reads of spectral arrays without reconstructing
        full ``matchms.Spectrum`` objects.

        Parameters
        ----------
        spectrum_ids : list of str or None
            Specific spectrum IDs to retrieve. If ``None``, all spectra in the
            store are returned.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            A 2-tuple of ``(mz_arrays, intensity_arrays)``. Each list element
            is a ``float64`` numpy array for one spectrum. The two lists are
            aligned by index.
        """
        ...

    @abstractmethod
    def get_total_spectra_count(self) -> int:
        """
        Return the total number of spectra stored.

        Returns
        -------
        int
            Spectrum count.
        """
        ...

    @abstractmethod
    def get_category_counts(self) -> dict[str, int]:
        """
        Return the number of spectra per category label.

        Returns
        -------
        dict[str, int]
            Mapping of category name to count.
        """
        ...

    @abstractmethod
    def get_precursor_mz_range(self) -> tuple[float, float]:
        """
        Return the minimum and maximum precursor m/z values in the store.

        Returns
        -------
        tuple[float, float]
            ``(min_mz, max_mz)``. Returns ``(0.0, 0.0)`` when the store is
            empty.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Release any resources held by the store (file handles, connections).

        After calling ``close()`` the instance must not be used.
        """
        ...


def create_spectral_store(
    store_path: Path | str,
    backend: str = "sqlite",
    **kwargs: Any,
) -> SpectralStore:
    """
    Factory that instantiates a :class:`SpectralStore` for the given backend.

    Parameters
    ----------
    store_path : Path or str
        File-system path for the store.
    backend : str
        Backend identifier. Supported values:
        - ``"sqlite"`` — SQLite BLOB-backed store (default, v0.1 stable).
        - ``"zarr"`` — Zarr/Blosc-backed store (cloud-optimized).
    **kwargs
        Additional keyword arguments forwarded to the store constructor.

    Returns
    -------
    SpectralStore
        A concrete store instance.

    Raises
    ------
    ValueError
        If ``backend`` is not one of the supported identifiers.

    Examples
    --------
    >>> store = create_spectral_store("library.db", backend="sqlite")
    >>> store = create_spectral_store("library.zarr", backend="zarr")
    """
    backend = backend.lower()

    if backend == "sqlite":
        from MassFlow.database import SpectralDatabase

        allow_upgrade = kwargs.pop("allow_destructive_upgrade", False)
        return SpectralDatabase(
            Path(store_path), allow_destructive_upgrade=allow_upgrade
        )

    if backend == "zarr":
        from MassFlow.zarr_store import ZarrSpectralStore

        return ZarrSpectralStore(Path(store_path), **kwargs)

    raise ValueError(
        f"Unsupported storage backend: '{backend}'. "
        f"Supported backends: 'sqlite', 'zarr'."
    )


__all__ = [
    "SpectralStore",
    "create_spectral_store",
]
