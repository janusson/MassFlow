"""
HNSW (Hierarchical Navigable Small World) index wrapper for spectral data.

HNSW graphs provide sub-linear approximate nearest-neighbour search, which
bypasses the quadratic scaling bottleneck of all-vs-all molecular networking
on massive libraries. This module wraps ``hnswlib`` behind a MassFlow-native
API that operates on *binned spectral vectors*.

Two-channel vectorization
-------------------------
Each spectrum is encoded as a concatenated vector
``[binned exact m/z, binned neutral losses]`` (see
:func:`spectrum_to_binned_vector`). The neutral-loss channel is the domain
modified cosine matches on, so the index can retrieve shifted analogues
(identical neutral-loss profiles, disjoint exact m/z) that a raw-m/z-only
index would miss — without it, analogue searching against an HNSW index
suffers catastrophic recall failure.

Non-metric data caveat
----------------------
Spectral cosine similarity is not a metric, and modified cosine in particular
violates the triangle inequality (fragment matches depend on each pair's
precursor mass shift, so distances are not symmetric or transitive). HNSW
graph construction assumes a metric space, so this index is used **only for
candidate generation** — exact scoring must always follow (e.g. as the first
stage of :class:`MassFlow.similarity.CascadeEngine`).

To preserve recall on non-metric data the construction parameters are exposed
in ``SimilarityConfig`` and default to generous values:

- ``M`` — maximum connections per node per layer. Larger values densify the
  graph (better recall, more memory).
- ``ef_construction`` — size of the dynamic candidate list during graph
  build. Larger values make hnswlib's heuristic pruning gentler.
- ``ef_search`` — size of the dynamic candidate list at query time. Must be
  ``>= k``; larger values increase recall at query-time cost.

Users tuning these parameters should set ``ef_construction >= 2 * M`` and
``ef_search >= 4 * candidates_per_query`` for near-exhaustive recall on
non-metric spectral data, and validate recall against exact scoring on a
representative sample (see ``tests/test_acceleration.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from matchms import Spectrum

logger = logging.getLogger(__name__)

_HNSW_INSTALL_MSG = (
    "HNSW-accelerated candidate retrieval requires hnswlib. "
    "Install it with: pip install massflow[hnsw]"
)

try:
    import hnswlib  # type: ignore[import-not-found]

    _HAS_HNSWLIB = True
except ImportError:  # pragma: no cover -- exercised in no-hnsw environments
    _HAS_HNSWLIB = False
    logger.info(
        "hnswlib is not installed; HNSW-accelerated candidate retrieval is "
        "disabled and searches fall back to exact scoring. %s",
        _HNSW_INSTALL_MSG,
    )

# Construction/query defaults tuned for non-metric spectral data: a denser
# graph (M) and larger candidate lists (ef_*) trade memory/latency for recall.
DEFAULT_HNSW_M = 32
DEFAULT_HNSW_EF_CONSTRUCTION = 400
DEFAULT_HNSW_EF_SEARCH = 200

DEFAULT_HNSW_BIN_WIDTH = 1.0
DEFAULT_HNSW_MZ_MIN = 0.0
DEFAULT_HNSW_MZ_MAX = 2000.0

DEFAULT_HNSW_MAX_ELEMENTS = 1_000_000

# Sidecar filename for the label (id) mapping persisted next to the index.
_IDS_SIDECAR_SUFFIX = ".ids.json"


# ---------------------------------------------------------------------------
# Spectral vectorization (binning)
# ---------------------------------------------------------------------------


def spectrum_to_binned_vector(
    spectrum: Spectrum,
    bin_width: float = DEFAULT_HNSW_BIN_WIDTH,
    mz_min: float = DEFAULT_HNSW_MZ_MIN,
    mz_max: float = DEFAULT_HNSW_MZ_MAX,
) -> np.ndarray:
    """Vectorize a spectrum into a 2-channel concatenated binned vector.

    Channel layout (``2 * dim`` total, ``dim = ceil((mz_max - mz_min) /
    bin_width)``):

    * **Channel 0** — exact fragment m/z binned into ``[mz_min, mz_max)``;
      this is the domain classical cosine matches on.
    * **Channel 1** — neutral losses (``precursor_mz - fragment_mz``)
      binned into the same grid; this is the domain modified cosine
      matches on (its precursor-shifted alignment is exactly neutral-loss
      matching).

    Encoding both domains is required for analogue discovery: a shifted
    analogue shares its neutral-loss profile with the query but none of
    its exact m/z values, so an exact-m/z-only index would be blind to it.

    Each channel is independently L2-normalized (an empty channel stays
    zero-filled, e.g. when ``precursor_mz`` is missing or no neutral loss
    falls in range), and the concatenated vector is normalized to unit
    norm so both channels contribute equally to hnswlib's ``cosine`` space.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum to vectorize.
    bin_width : float, optional
        Width of each m/z bin in Da.
    mz_min : float, optional
        Lower bound of the binned range (inclusive).
    mz_max : float, optional
        Upper bound of the binned range (exclusive).

    Returns
    -------
    np.ndarray
        ``float32`` vector of shape ``(2 * dim,)`` — the first half is the
        exact-m/z channel, the second half the neutral-loss channel.
    """
    dimension = int(np.ceil((mz_max - mz_min) / bin_width))

    mz_array = np.asarray(spectrum.peaks.mz, dtype=np.float64)
    intensity_array = np.asarray(spectrum.peaks.intensities, dtype=np.float64)

    def _bin_channel(values: np.ndarray) -> np.ndarray:
        """Bin *values* (m/z or neutral losses) into a normalized channel."""
        channel = np.zeros(dimension, dtype=np.float32)
        in_range = (values >= mz_min) & (values < mz_max)
        if np.any(in_range):
            bin_indices = ((values[in_range] - mz_min) / bin_width).astype(np.int64)
            np.add.at(
                channel,
                bin_indices,
                intensity_array[in_range].astype(np.float32),
            )
            norm = float(np.linalg.norm(channel))
            if norm > 0.0:
                channel /= norm
        return channel

    exact_mz_channel = _bin_channel(mz_array)

    # Neutral-loss channel: precursor_mz - fragment_mz. Spectra without a
    # usable precursor get a zero channel (no neutral-loss information).
    precursor_mz = spectrum.get("precursor_mz")
    neutral_loss_channel = np.zeros(dimension, dtype=np.float32)
    if precursor_mz is not None:
        try:
            precursor_value = float(precursor_mz)
        except (ValueError, TypeError):
            precursor_value = np.nan
        if np.isfinite(precursor_value):
            neutral_loss_channel = _bin_channel(precursor_value - mz_array)

    vector = np.concatenate([exact_mz_channel, neutral_loss_channel])
    total_norm = float(np.linalg.norm(vector))
    if total_norm > 0.0:
        vector /= total_norm
    return vector


def bin_spectra(
    spectra: Sequence[Spectrum],
    bin_width: float = DEFAULT_HNSW_BIN_WIDTH,
    mz_min: float = DEFAULT_HNSW_MZ_MIN,
    mz_max: float = DEFAULT_HNSW_MZ_MAX,
) -> np.ndarray:
    """Vectorize a sequence of spectra into a 2-D ``float32`` matrix.

    Parameters
    ----------
    spectra : sequence of Spectrum
        Spectra to vectorize.
    bin_width : float, optional
        Width of each m/z bin in Da.
    mz_min : float, optional
        Lower bound of the binned range (inclusive).
    mz_max : float, optional
        Upper bound of the binned range (exclusive).

    Returns
    -------
    np.ndarray
        ``float32`` matrix of shape ``(n_spectra, 2 * dim)`` — per row the
        exact-m/z channel followed by the neutral-loss channel (see
        :func:`spectrum_to_binned_vector`). Rows are unit-normalized;
        empty spectra map to zero rows.
    """
    if not spectra:
        dimension = int(np.ceil((mz_max - mz_min) / bin_width))
        return np.empty((0, dimension), dtype=np.float32)
    return np.stack(
        [
            spectrum_to_binned_vector(
                spectrum,
                bin_width=bin_width,
                mz_min=mz_min,
                mz_max=mz_max,
            )
            for spectrum in spectra
        ]
    ).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# HNSW index wrapper
# ---------------------------------------------------------------------------


class HNSWSpectralIndex:
    """hnswlib-backed approximate nearest-neighbour index over binned spectra.

    Spectra are vectorized with :func:`spectrum_to_binned_vector` into
    **two-channel** vectors — ``[binned exact m/z, binned neutral losses]`` —
    so the index can retrieve both classical matches and shifted analogues
    whose neutral-loss profiles match but whose exact m/z values do not
    overlap (modified cosine territory).

    The index maps each spectrum to an integer label internally; the public
    API uses the spectra's ``id`` strings (or positional indices when ids are
    missing).  Build-time parameters ``M`` and ``ef_construction`` control
    graph density and pruning aggressiveness; query-time ``ef_search`` trades
    recall against latency.

    Parameters
    ----------
    dim : int
        Vector dimensionality (number of m/z bins).
    m : int, optional
        Maximum connections per node per layer (hnswlib ``M``).
    ef_construction : int, optional
        Dynamic candidate list size during index construction.
    space : str, optional
        hnswlib metric space; ``"cosine"`` for binned spectral vectors.
    random_seed : int, optional
        Seed for the graph construction RNG.
    max_elements : int, optional
        Upper bound on the number of items added (hnswlib requires this
        up-front).

    Raises
    ------
    RuntimeError
        If ``hnswlib`` is not installed.
    ValueError
        If construction parameters are invalid (``m < 1`` or
        ``ef_construction < m``).

    Examples
    --------
    >>> index = HNSWSpectralIndex.from_spectra(references)
    >>> candidate_ids, distances = index.query(query_vectors, k=50)
    """

    def __init__(
        self,
        dim: int,
        m: int = DEFAULT_HNSW_M,
        ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
        space: str = "cosine",
        random_seed: int = 42,
        max_elements: int = DEFAULT_HNSW_MAX_ELEMENTS,
    ) -> None:
        if not _HAS_HNSWLIB:
            raise RuntimeError(_HNSW_INSTALL_MSG)
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}.")
        if dim % 2 != 0:
            raise ValueError(
                f"dim must be even (two-channel [m/z, neutral-loss] layout); got {dim}."
            )
        if m < 1:
            raise ValueError(f"m must be >= 1; got {m}.")
        if ef_construction < m:
            raise ValueError(
                f"ef_construction must be >= m for stable graph construction; "
                f"got ef_construction={ef_construction} < m={m}. Non-metric "
                f"spectral data benefits from generous ef_construction values."
            )

        self._dim = dim
        self._m = m
        self._ef_construction = ef_construction
        self._space = space
        self._random_seed = random_seed
        self._max_elements = max_elements

        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=m,
            random_seed=random_seed,
        )
        # Default query-time ef; per-query overrides supported in query().
        self._index.set_ef(ef_construction)

        self._ids: list[str] = []
        self._id_to_label: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_spectra(
        cls,
        spectra: Sequence[Spectrum],
        bin_width: float = DEFAULT_HNSW_BIN_WIDTH,
        mz_min: float = DEFAULT_HNSW_MZ_MIN,
        mz_max: float = DEFAULT_HNSW_MZ_MAX,
        m: int = DEFAULT_HNSW_M,
        ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
        random_seed: int = 42,
        max_elements: Optional[int] = None,
    ) -> "HNSWSpectralIndex":
        """Build an index over *spectra* using binned spectral vectors.

        Parameters
        ----------
        spectra : sequence of Spectrum
            Spectra to index.
        bin_width, mz_min, mz_max : float, optional
            Binning parameters forwarded to :func:`bin_spectra`.
        m, ef_construction, random_seed : optional
            Graph construction parameters (see class docstring).
        max_elements : int or None, optional
            Upper bound on indexed items; defaults to ``max(len(spectra), 1)``.

        Returns
        -------
        HNSWSpectralIndex
            Populated index over ``2 * dim``-dimensional two-channel
            vectors (exact m/z + neutral loss).
        """
        dimension = 2 * int(np.ceil((mz_max - mz_min) / bin_width))
        index = cls(
            dim=dimension,
            m=m,
            ef_construction=ef_construction,
            random_seed=random_seed,
            max_elements=max(1, int(max_elements or len(spectra))),
        )
        vectors = bin_spectra(
            spectra, bin_width=bin_width, mz_min=mz_min, mz_max=mz_max
        )
        ids = [
            str(spectrum.get("id", position))
            for position, spectrum in enumerate(spectra)
        ]
        index.add_items(vectors, ids)
        return index

    def add_items(
        self,
        vectors: np.ndarray,
        ids: Sequence[str],
    ) -> None:
        """Add *vectors* under the given string *ids*.

        Parameters
        ----------
        vectors : np.ndarray
            ``float32`` matrix of shape ``(n, dim)``.
        ids : sequence of str
            One identifier per row. Identifiers should be unique within the
            index; later duplicates overwrite the earlier label mapping.

        Returns
        -------
        None
        """
        vectors_float32 = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors_float32.ndim != 2 or vectors_float32.shape[1] != self._dim:
            raise ValueError(
                f"vectors must have shape (n, {self._dim}); "
                f"got {vectors_float32.shape}."
            )
        if vectors_float32.shape[0] != len(ids):
            raise ValueError(
                "vectors and ids must have the same length; "
                f"got {vectors_float32.shape[0]} vs {len(ids)}."
            )

        for vector, spectrum_id in zip(vectors_float32, ids):
            if spectrum_id in self._id_to_label:
                label = self._id_to_label[spectrum_id]
            else:
                label = len(self._ids)
                self._ids.append(str(spectrum_id))
                self._id_to_label[str(spectrum_id)] = label
            self._index.add_items(
                vector.reshape(1, self._dim), np.array([label], dtype=np.uint64)
            )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        vectors: np.ndarray,
        k: int,
        ef_search: Optional[int] = None,
    ) -> tuple[list[list[str]], np.ndarray]:
        """Retrieve the ``k`` nearest neighbours for each query vector.

        Parameters
        ----------
        vectors : np.ndarray
            ``float32`` matrix of query vectors of shape ``(n, dim)``.
        k : int
            Number of candidates to retrieve per query. Must not exceed the
            number of indexed items.
        ef_search : int or None, optional
            Query-time candidate list size. When ``None`` the construction
            ``ef`` is used. Must be ``>= k``.

        Returns
        -------
        tuple[list[list[str]], np.ndarray]
            ``(candidate_ids, distances)``: per-query lists of neighbour id
            strings, and the ``float32`` cosine distances
            (``distance = 1 - cosine``).

        Raises
        ------
        ValueError
            If ``k`` exceeds the indexed item count or ``ef_search < k``.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}.")
        if k > len(self._ids):
            raise ValueError(
                f"k={k} exceeds the number of indexed items ({len(self._ids)})."
            )
        if ef_search is not None and ef_search < k:
            raise ValueError(
                f"ef_search must be >= k for meaningful recall; "
                f"got ef_search={ef_search} < k={k}."
            )

        vectors_float32 = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors_float32.ndim != 2 or vectors_float32.shape[1] != self._dim:
            raise ValueError(
                f"vectors must have shape (n, {self._dim}); "
                f"got {vectors_float32.shape}."
            )

        if ef_search is not None:
            self._index.set_ef(int(ef_search))
        elif len(self._ids) > self._ef_construction:
            # Ensure the default ef can serve k even when the index is small.
            self._index.set_ef(max(self._ef_construction, k))

        labels, distances = self._index.knn_query(vectors_float32, k=int(k))
        candidate_ids = [[self._ids[int(label)] for label in row] for row in labels]
        return candidate_ids, distances

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist the index graph and its id mapping to disk.

        Writes the hnswlib index to *path* and the id/label mapping to a JSON
        sidecar (``<path>.ids.json``). hnswlib itself only persists integer
        labels, so the sidecar is required to restore string ids.

        Parameters
        ----------
        path : Path
            Destination file for the index graph.

        Returns
        -------
        None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(path))
        sidecar = {
            "ids": self._ids,
            "dim": self._dim,
            "space": self._space,
            "m": self._m,
            "ef_construction": self._ef_construction,
            "random_seed": self._random_seed,
        }
        sidecar_path = Path(str(path) + _IDS_SIDECAR_SUFFIX)
        with open(sidecar_path, "w") as fh:
            json.dump(sidecar, fh)

    @classmethod
    def load(cls, path: Path) -> "HNSWSpectralIndex":
        """Restore an index previously written by :meth:`save`.

        Parameters
        ----------
        path : Path
            Index file written by :meth:`save`.

        Returns
        -------
        HNSWSpectralIndex
            Restored index.

        Raises
        ------
        FileNotFoundError
            If the index file or its id sidecar is missing.
        """
        if not _HAS_HNSWLIB:
            raise RuntimeError(_HNSW_INSTALL_MSG)

        sidecar_path = Path(str(path) + _IDS_SIDECAR_SUFFIX)
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Index id sidecar not found: {sidecar_path}")
        with open(sidecar_path, "r") as fh:
            sidecar = json.load(fh)

        index = cls.__new__(cls)
        index._dim = int(sidecar["dim"])
        index._space = str(sidecar["space"])
        index._m = int(sidecar["m"])
        index._ef_construction = int(sidecar["ef_construction"])
        index._random_seed = int(sidecar["random_seed"])
        index._ids = [str(spectrum_id) for spectrum_id in sidecar["ids"]]
        index._id_to_label = {
            spectrum_id: label for label, spectrum_id in enumerate(index._ids)
        }
        index._max_elements = max(1, len(index._ids))
        index._index = hnswlib.Index(space=index._space, dim=index._dim)
        index._index.load_index(str(path), max_elements=index._max_elements)
        index._index.set_ef(index._ef_construction)
        return index

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of indexed items."""
        return len(self._ids)

    @property
    def dim(self) -> int:
        """Vector dimensionality of the index."""
        return self._dim

    @property
    def m(self) -> int:
        """Graph construction parameter ``M``."""
        return self._m

    @property
    def ef_construction(self) -> int:
        """Graph construction parameter ``ef_construction``."""
        return self._ef_construction


__all__ = [
    "_HAS_HNSWLIB",
    "HNSWSpectralIndex",
    "bin_spectra",
    "spectrum_to_binned_vector",
]
