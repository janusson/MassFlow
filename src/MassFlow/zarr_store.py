"""
Zarr-backed spectral storage for MassFlow with cloud-native support.

This module implements :class:`ZarrSpectralStore`, a cloud-optimized storage
backend that persists fragment m/z and intensity vectors as compressed,
chunked Zarr arrays alongside native Zarr metadata arrays --- eliminating the
``metadata_index.json`` sidecar that existed in earlier versions.

Two complementary storage layouts are supported:

``"flat"`` (default, backward-compatible v0.1)
    ``peaks/mz_flat``, ``peaks/intensity_flat``, ``peaks/boundaries`` ---
    concatenated 1-D arrays indexed by a cumulative boundary array.  Chunking
    is along the single flat dimension so contiguous spectrum ranges map to
    a contiguous slice of the underlying chunks.

``"tensor"`` (cloud-optimized, experimental)
    ``peaks/tensor`` --- shape ``(2, B, M)`` where channel 0 = m/z, channel 1
    = intensity, *B* is the number of batch-chunks, and *M* is the configured
    peak capacity per spectrum (``max_peaks_per_spectrum``).  Each batch slice
    ``tensor[:, b, :]`` contains all peak data for one batch of spectra,
    zero-padded to *M*.  ``peaks/peak_counts`` records the true number of
    peaks per spectrum so callers can strip padding.  The 2-D chunking maps
    directly to pairwise similarity batch-processing: reading one reference
    batch reads exactly the tensor slices needed with no amplification.

Peak capacity and scientific fidelity
-------------------------------------
Storing a spectrum must never silently change its scientific data.  The
flat layout stores every peak exactly (arbitrary peak counts).  The tensor
layout has a fixed per-spectrum capacity *M*; what happens to an
over-capacity spectrum is governed by the explicit ``peak_reduction``
policy:

``"none"`` (default)
    ``add_spectra`` raises :class:`PeakCapacityError` naming the spectrum
    and its peak count; the batch is rejected atomically.  No data is
    written, no data is altered.

``"topN"``
    The spectrum is explicitly reduced to its *M* most intense peaks
    (deterministic, stable tie-breaking) and the reduction is recorded in
    the spectrum's stored provenance metadata (``peak_reduction`` inside
    ``extra_metadata``) so every consumer of the store can see that a
    transformation was applied.

The active layout, capacity, and reduction policy are persisted in a store
manifest (``.zattrs``) and validated on every open, so a store cannot be
re-opened with parameters that would silently reinterpret its data.

Cloud storage is enabled by passing an ``fsspec``-compatible URL as
``store_path`` (e.g. ``s3://my-bucket/library.zarr`` or
``https://data.example.org/library.zarr``).  When a remote URL is detected
the store wraps the connection with :class:`zarr.storage.FSStore` and applies
exponential-backoff retries to every remote read.

Metadata caching
----------------
A read-through, thread-safe cache is layered over all metadata array reads.
The cache is invalidated on write and has a configurable TTL.  This prevents
redundant network I/O during precursor m/z window pre-filtering (100 Da /
0.25 Da gates) where the same ``precursor_mz``, ``category``, and ``id``
arrays are consulted repeatedly across worker batches.

.. note::
    This backend is **experimental** for v0.1 and must be explicitly enabled
    via ``storage_backend: "zarr"`` in the MassFlow YAML configuration.  The
    default backend remains the SQLite BLOB implementation in
    :mod:`MassFlow.database`.

Thread-safety
-------------
- **Reads** are lock-free and safe across threads and processes.  Zarr v3
  guarantees that array reads are immutable snapshots.  For cloud stores each
  thread/worker should obtain its own store wrapper (handled internally via
  ``_get_store`` which re-opens an fsspec filesystem per call when needed).
- **Writes** continue to use the dual-lock strategy (``threading.Lock`` +
  ``fcntl.flock``) and are mutually exclusive with other writes.
- Cloud locking for writes is **not** implemented --- Zarr stores accessed
  over HTTP/S3 are assumed to be read-only or single-writer.  Multi-writer
  cloud scenarios should use a coordination layer outside this module.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, TYPE_CHECKING

import numpy as np
from matchms import Spectrum

from MassFlow.storage import SpectralStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Float64 elements per chunk in the flat peak arrays.
# ~1M elements ≈ 8 MB per chunk — balances compression ratio against read
# amplification for batch similarity workloads.
_DEFAULT_PEAK_CHUNK_SIZE: int = 1_048_576

# Spectra per chunk for 1-D metadata arrays (flat layout).
_DEFAULT_METADATA_CHUNK_SIZE: int = 4096

# Spectra per chunk for the boundaries array in the hybrid SQLite+Zarr backend.
_DEFAULT_BOUNDARY_CHUNK_SIZE: int = 4096

# Spectra per batch for the tensor layout.
_DEFAULT_TENSOR_BATCH_SIZE: int = 1024

# Default peak capacity per spectrum in tensor mode.  Spectra exceeding
# this capacity are NEVER silently altered: the active ``peak_reduction``
# policy decides between an explicit error and an explicit, provenance-
# recorded Top-N reduction.
_DEFAULT_MAX_PEAKS: int = 512

# Root-group attribute holding the persisted store manifest (layout,
# capacity, reduction policy).  Validated on every open so a store cannot
# be re-opened with parameters that silently reinterpret its data.
_MANIFEST_ATTR: str = "massflow_store_manifest"

# Legacy sidecar filename — checked during initialisation for automatic
# migration to the native-metadata format.
_LEGACY_METADATA_INDEX = "metadata_index.json"

# Lock filename for cross-process append coordination.
_LOCK_FILENAME = ".lock"

# Remote fetch defaults.
_DEFAULT_REMOTE_TIMEOUT: float = 30.0  # seconds
_DEFAULT_REMOTE_RETRIES: int = 3
_DEFAULT_CACHE_TTL: float = 300.0  # seconds

# Backoff factor for exponential retry (seconds).
_BACKOFF_BASE: float = 1.0
_BACKOFF_MULTIPLIER: float = 2.0
_BACKOFF_MAX: float = 60.0


class PeakCapacityError(ValueError):
    """A spectrum exceeds the tensor-layout peak capacity.

    Raised by :meth:`ZarrSpectralStore.add_spectra` when a spectrum has more
    peaks than ``max_peaks_per_spectrum`` and no explicit peak-reduction
    policy (``peak_reduction="topN"``) is active.  The batch is rejected
    atomically — nothing is written — so a storage limit can never silently
    alter scientific data.
    """


def _top_n_peaks(
    mz: np.ndarray,
    intensities: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the ``n`` most intense peaks, deterministically.

    This is an explicit scientific transformation: unlike the legacy
    behaviour (silently keeping the first ``n`` peaks in m/z order), it keeps
    the peaks that dominate cosine scoring.  Ties in intensity are broken by
    original order (stable sort), and the retained peaks are returned in
    ascending m/z order, matching the input convention.  When ``n`` >= the
    number of peaks the arrays are returned unchanged (copies).

    Parameters
    ----------
    mz : np.ndarray
        float64 m/z values, expected m/z-ascending.
    intensities : np.ndarray
        float64 intensities aligned with ``mz``.
    n : int
        Maximum number of peaks to retain (must be >= 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The retained (m/z, intensity) arrays, m/z-ascending.
    """
    mz = np.asarray(mz, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if n >= mz.size:
        return mz.copy(), intensities.copy()
    order = np.argsort(-intensities, kind="stable")
    keep = np.sort(order[:n])
    return mz[keep].copy(), intensities[keep].copy()


def _default_compressor() -> Optional[Any]:
    """Return the default Blosc+zstd compressor, or ``None`` if unavailable.

    zarr v3 exposes codecs via :mod:`zarr.codecs`.  Blosc with zstd and
    byte-shuffling at clevel 3 gives a good compression/speed trade-off for
    float64 spectral arrays (~8 MB chunks).
    """
    try:
        import zarr.codecs

        return zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=3,
            shuffle="shuffle",
        )
    except (ImportError, AttributeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Retry configuration & decorator
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """Configuration for exponential-backoff retries on remote reads."""

    max_retries: int = _DEFAULT_REMOTE_RETRIES
    base_delay: float = _BACKOFF_BASE
    multiplier: float = _BACKOFF_MULTIPLIER
    max_delay: float = _BACKOFF_MAX
    retryable_exceptions: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        OSError,
    )


def _retry_with_backoff(config: RetryConfig) -> Callable:
    """Decorator factory: retry a function with exponential backoff.

    Only *config.retryable_exceptions* trigger a retry; all other exceptions
    propagate immediately.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == config.max_retries:
                        break
                    delay = min(
                        config.base_delay * (config.multiplier**attempt),
                        config.max_delay,
                    )
                    logger.warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1,
                        config.max_retries,
                        func.__name__,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Metadata read-through cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """A single cache entry with expiry."""

    data: Any
    expires_at: float


class MetadataQueryCache:
    """Thread-safe read-through cache for metadata array slices.

    The cache is keyed by ``(field_name, start_idx, end_idx)`` and stores
    the fetched numpy array.  Entries expire after *ttl_seconds* to prevent
    stale data when the store is updated by another process.

    Parameters
    ----------
    ttl_seconds : float
        Time-to-live for cache entries in seconds.
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_CACHE_TTL) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[tuple[str, int, int], _CacheEntry] = {}
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    def get(
        self,
        key: tuple[str, int, int],
        loader: Callable[[], np.ndarray],
    ) -> Any:
        """Return cached data or load via *loader* and cache it.

        Parameters
        ----------
        key : tuple[str, int, int]
            ``(field_name, start, end)`` identifying the slice.
        loader : callable
            Zero-argument callable that produces the data on a cache miss.

        Returns
        -------
        numpy.ndarray
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and entry.expires_at > now:
                self._hits += 1
                return entry.data

        # Cache miss — load outside the lock to avoid blocking other threads.
        self._misses += 1
        data = loader()
        with self._lock:
            self._cache[key] = _CacheEntry(data=data, expires_at=now + self._ttl)
        return data

    def invalidate(self) -> None:
        """Clear all cached entries (called after writes)."""
        with self._lock:
            self._cache.clear()

    def invalidate_field(self, field_name: str) -> None:
        """Clear cached entries for a specific metadata field."""
        with self._lock:
            keys_to_delete = [k for k in self._cache if k[0] == field_name]
            for k in keys_to_delete:
                del self._cache[k]

    @property
    def stats(self) -> dict[str, int]:
        """Cache hit/miss statistics (informational)."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
            }

    def clear(self) -> None:
        """Alias for ``invalidate``."""
        self.invalidate()


# ---------------------------------------------------------------------------
# URL / store helpers
# ---------------------------------------------------------------------------


def _is_remote_url(path: str) -> bool:
    """Return True if *path* looks like a remote fsspec-compatible URL."""
    return bool(
        path.startswith("s3://")
        or path.startswith("gs://")
        or path.startswith("abfs://")
        or path.startswith("az://")
        or path.startswith("http://")
        or path.startswith("https://")
    )


def _make_fsspec_store(
    store_path: str,
    storage_options: Optional[dict[str, Any]] = None,
    read_only: bool = True,
) -> Any:  # zarr.storage.FsspecStore
    """Create a :class:`zarr.storage.FsspecStore` for a remote URL.

    Uses ``fsspec`` to construct an async filesystem, then wraps it in
    zarr v3's :class:`FsspecStore`.  Each call creates a fresh filesystem
    instance so threads get independent connection pools.
    """
    import zarr.storage

    try:
        import fsspec  # noqa: F401
    except ImportError:
        raise ImportError(
            "Remote Zarr stores require fsspec.  Install it with: "
            "pip install fsspec s3fs requests aiohttp"
        )

    opts = dict(storage_options) if storage_options else {}

    # Build a sync (thread-safe) fsspec filesystem.  FsspecStore
    # handles the async bridge internally.
    protocol = _fsspec_protocol(store_path)
    # For HTTP/HTTPS we need to tell fsspec to use the http filesystem.
    if protocol in ("http", "https"):
        fs = fsspec.filesystem("http", **opts)
        path_clean = store_path
    else:
        fs = fsspec.filesystem(protocol, **opts)
        path_clean = _fsspec_strip_protocol(store_path)

    return zarr.storage.FsspecStore(
        fs=fs,
        read_only=read_only,
        path=path_clean,
    )


def _fsspec_protocol(url: str) -> str:
    """Extract the fsspec protocol from a URL."""
    for proto in ("s3", "gs", "abfs", "az", "http", "https"):
        if url.startswith(f"{proto}://"):
            return proto
    return "file"


def _fsspec_strip_protocol(url: str) -> str:
    """Strip the protocol prefix for FSStore (fsspec handles it separately)."""
    for proto in ("s3://", "gs://", "abfs://", "az://"):
        if url.startswith(proto):
            return url[len(proto) :]
    return url


# ---------------------------------------------------------------------------
# String-array construction helpers
# ---------------------------------------------------------------------------


def _detect_string_backend() -> str:
    """Return the best available string-storage strategy.

    Returns one of ``"stringdtype"``, ``"vlenutf8"``, or ``"fixed"``.
    """
    # Strategy 1: NumPy StringDType (≥ 2.0) — zero extra dependencies.
    try:
        np.dtypes.StringDType()
        return "stringdtype"
    except (AttributeError, TypeError):
        pass

    # Strategy 2: numcodecs VLenUTF8.
    try:
        import numcodecs  # noqa: F401

        return "vlenutf8"
    except ImportError:
        pass

    # Strategy 3: fall back to fixed-length Unicode (wastes space but works
    # everywhere).
    return "fixed"


def _create_string_array(
    group: Any,  # zarr.Group
    name: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    backend: str,
    compressor: Any = None,
    fill_value: str = "",
) -> Any:  # zarr.Array
    """Create a chunked Zarr array suitable for variable-length strings.

    Parameters
    ----------
    group : zarr.Group
        Parent group for the new array.
    name : str
        Array name within *group*.
    shape : tuple of int
        Initial shape (typically ``(0,)`` for empty, later resized).
    chunks : tuple of int
        Chunk shape.
    backend : str
        One of ``"stringdtype"``, ``"vlenutf8"``, ``"fixed"``.
    compressor : optional
        Zarr compressor (applied when the backend supports it).
    fill_value : str
        Default value for uninitialised elements.

    Returns
    -------
    zarr.Array
    """
    kwargs: dict[str, Any] = {}
    if compressor is not None:
        kwargs["compressors"] = [compressor]

    if backend == "stringdtype":
        return group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=np.dtypes.StringDType(),
            fill_value=fill_value,
            **kwargs,
        )

    if backend == "vlenutf8":
        from numcodecs import VLenUTF8

        return group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=object,
            filters=[VLenUTF8()],
            object_codec=VLenUTF8(),
            fill_value=fill_value,
            **kwargs,
        )

    # Fixed-length fallback: use a generous max field width.
    return group.create_array(
        name,
        shape=shape,
        chunks=chunks,
        dtype="U512",
        fill_value=fill_value,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Helper for acquiring a cross-process file lock
# ---------------------------------------------------------------------------


def _acquire_file_lock(lock_path: Path) -> int | None:
    """Acquire an exclusive advisory lock on *lock_path*.

    Returns an open file descriptor (which must be closed to release the
    lock), or ``None`` on platforms where ``fcntl`` is unavailable.
    """
    try:
        import fcntl

        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except (ImportError, OSError):
        return None


def _release_file_lock(fd: int | None) -> None:
    """Release a file lock acquired by :func:`_acquire_file_lock`."""
    if fd is None:
        return
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


# ===================================================================
# ZarrSpectralStore
# ===================================================================


class ZarrSpectralStore(SpectralStore):
    """Persist mass spectra using compressed, chunked Zarr arrays.

    The on-disk layout is a Zarr group (a directory with ``.zarray`` /
    ``.zgroup`` metadata files for local stores, or a key-prefixed object
    path for cloud stores) containing:

    **Flat layout** (``layout="flat"``, default):

    ``peaks/``
        ``mz_flat`` (float64), ``intensity_flat`` (float64), and
        ``boundaries`` (int64) — the flat peak storage arrays.

    **Tensor layout** (``layout="tensor"``, experimental):

    ``peaks/``
        ``tensor`` (float64, shape ``(2, B, M)``) — 3-D spectral tensor.
        ``peak_counts`` (int32, shape ``(N,)``) — actual peak count per
        spectrum (N = total spectra).
        ``batch_index``, ``batch_boundaries`` — optional batch indexing.

    ``metadata/``
        One chunked 1-D Zarr array per metadata field:

        * ``id``, ``name``, ``ionmode``, ``adduct``, ``category`` —
          variable-length string arrays.
        * ``precursor_mz`` (float64), ``charge`` (int32) — scalar arrays.
        * ``extra_metadata`` — JSON-serialised free-form metadata blob
          (variable-length string).

    Thread-safe parallel reads are lock-free.  Writes use the existing
    dual-lock strategy (``threading.Lock`` + ``fcntl.flock``).

    Parameters
    ----------
    store_path : Path
        Directory path for local stores, or an fsspec-compatible URL
        (``s3://``, ``gs://``, ``https://``) for cloud stores.
    peak_chunk_size : int, optional
        Float64 elements per chunk in flat peak arrays (ignored in
        tensor mode).
    metadata_chunk_size : int, optional
        Spectra per chunk in the metadata arrays.
    compressor : optional
        Zarr compressor (defaults to Blosc+zstd, clevel=3).
    overwrite : bool, optional
        If ``True``, delete any existing store before creating a fresh one.
    storage_options : dict, optional
        Keyword arguments forwarded to the ``fsspec`` filesystem constructor
        (e.g. ``{"key": "...", "secret": "...", "endpoint_url": "..."}``).
    layout : str, optional
        Storage layout: ``"flat"`` (default) or ``"tensor"``.
    tensor_batch_size : int, optional
        Spectra per batch in tensor layout.
    max_peaks_per_spectrum : int, optional
        Peak capacity per spectrum in tensor layout (shorter spectra are
        zero-padded to this size).  Over-capacity spectra are never silently
        altered: see ``peak_reduction``.
    peak_reduction : str, optional
        Explicit policy for spectra exceeding ``max_peaks_per_spectrum`` in
        tensor layout.  ``"none"`` (default) raises :class:`PeakCapacityError`
        and rejects the batch atomically.  ``"topN"`` retains the N most
        intense peaks and records the reduction in the spectrum's stored
        provenance metadata (``extra_metadata.peak_reduction``).  Requires
        ``layout="tensor"``.
    remote_timeout : float, optional
        Timeout in seconds for individual remote read operations.
    remote_retries : int, optional
        Maximum retry attempts for remote reads (exponential backoff).
    cache_ttl : float, optional
        Cache time-to-live in seconds for metadata queries.
    """

    # ------------------------------------------------------------------
    # Metadata field definitions
    # ------------------------------------------------------------------
    _SCALAR_FIELDS: dict[str, np.dtype] = {
        "precursor_mz": np.dtype(np.float64),
        "charge": np.dtype(np.int32),
    }

    _STRING_FIELDS: list[str] = [
        "id",
        "name",
        "ionmode",
        "adduct",
        "category",
        "extra_metadata",
    ]

    _ALL_METADATA_FIELDS: list[str] = [
        *_SCALAR_FIELDS.keys(),
        *_STRING_FIELDS,
    ]

    # Sub-group names within the root Zarr group.
    _PEAKS_GROUP = "peaks"
    _METADATA_GROUP = "metadata"

    def __init__(
        self,
        store_path: Path,
        peak_chunk_size: int = _DEFAULT_PEAK_CHUNK_SIZE,
        metadata_chunk_size: int = _DEFAULT_METADATA_CHUNK_SIZE,
        compressor: Optional[Any] = None,
        overwrite: bool = False,
        storage_options: Optional[dict[str, Any]] = None,
        layout: str = "flat",
        tensor_batch_size: int = _DEFAULT_TENSOR_BATCH_SIZE,
        max_peaks_per_spectrum: int = _DEFAULT_MAX_PEAKS,
        peak_reduction: str = "none",
        remote_timeout: float = _DEFAULT_REMOTE_TIMEOUT,
        remote_retries: int = _DEFAULT_REMOTE_RETRIES,
        cache_ttl: float = _DEFAULT_CACHE_TTL,
    ) -> None:
        # --- Store / I/O config ---
        self._peak_chunk_size = peak_chunk_size
        self._metadata_chunk_size = metadata_chunk_size
        self._compressor = compressor
        self._overwrite = overwrite
        self._storage_options = storage_options
        self._layout = layout

        # --- Tensor layout config ---
        self._tensor_batch_size = tensor_batch_size
        self._max_peaks = max_peaks_per_spectrum
        self._peak_reduction = peak_reduction

        # --- Remote / retry config ---
        self._remote_timeout = remote_timeout
        self._retry_config = RetryConfig(
            max_retries=remote_retries,
        )
        self._is_remote = _is_remote_url(str(store_path))

        # --- Cache ---
        self._metadata_cache = MetadataQueryCache(ttl_seconds=cache_ttl)

        # --- Internal state ---
        self._root: Optional[Any] = None  # zarr.Group
        self._n_spectra: int = 0
        self._string_backend: str = "fixed"
        self._store: Optional[Any] = None  # underlying zarr.Store
        # In-process lock for coordinating appends within the same process.
        self._lock = threading.Lock()

        # Validate layout early.
        if self._layout not in ("flat", "tensor"):
            raise ValueError(
                f"Unsupported layout: '{self._layout}'. Must be 'flat' or 'tensor'."
            )
        if peak_reduction not in ("none", "topN"):
            raise ValueError(
                f"Unsupported peak_reduction: '{peak_reduction}'. "
                "Must be 'none' or 'topN'."
            )
        if peak_reduction == "topN" and self._layout != "tensor":
            raise ValueError(
                "peak_reduction='topN' requires layout='tensor': the flat "
                "layout stores every peak exactly and has no peak capacity."
            )
        if self._layout == "tensor" and max_peaks_per_spectrum < 1:
            raise ValueError(
                "max_peaks_per_spectrum must be >= 1 for the tensor layout."
            )

        super().__init__(store_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Open or create the Zarr group and all sub-groups/arrays."""
        import zarr

        store_path_str = str(self.store_path)

        if self._overwrite and not self._is_remote and Path(store_path_str).exists():
            import shutil

            shutil.rmtree(store_path_str)

        if not self._is_remote:
            self.store_path.mkdir(parents=True, exist_ok=True)

        # Open the store: local paths use zarr.open_group directly;
        # remote URLs use FsspecStore.
        if self._is_remote:
            self._store = _make_fsspec_store(
                store_path_str,
                storage_options=self._storage_options,
                read_only=True,
            )
            self._root = zarr.open_group(store=self._store, mode="r")
        else:
            self._root = zarr.open_group(store_path_str, mode="a")
            self._store = None  # Local stores managed by zarr internally.

        # Detect the best available string backend.
        self._string_backend = _detect_string_backend()
        logger.debug("String storage backend: %s", self._string_backend)

        # Set up compressor default using zarr's built-in Blosc codec.
        if self._compressor is None:
            try:
                import zarr.codecs

                self._compressor = zarr.codecs.BloscCodec(
                    cname="zstd",
                    clevel=3,
                    shuffle="shuffle",
                )
            except (ImportError, AttributeError, TypeError):
                self._compressor = None

        # Ensure sub-groups exist.
        for grp_name in (self._PEAKS_GROUP, self._METADATA_GROUP):
            if grp_name not in self._root:
                try:
                    self._root.create_group(grp_name)
                except Exception:
                    # Group may already exist in a concurrent session.
                    pass

        # Validate (or adopt) the persisted store manifest.  The layout,
        # peak capacity, and reduction policy must match the constructor
        # arguments, or a later read would silently reinterpret the data.
        self._validate_or_write_manifest()

        # Attempt migration from legacy metadata_index.json if present.
        self._maybe_migrate_legacy_metadata()

        # Open or create metadata arrays, inferring n_spectra from shape.
        self._open_or_create_metadata_arrays()

        logger.info(
            "ZarrSpectralStore opened: path=%s layout=%s remote=%s n_spectra=%d",
            str(self.store_path),
            self._layout,
            self._is_remote,
            self._n_spectra,
        )

    def close(self) -> None:
        """Release resources and clear caches.

        Zarr data is already durable on disk/object store.  This method
        drops the in-memory group reference and clears the metadata cache.
        """
        self._metadata_cache.clear()
        self._root = None
        self._store = None

    # ------------------------------------------------------------------
    # Store manifest (layout / capacity / reduction policy persistence)
    # ------------------------------------------------------------------

    def _validate_or_write_manifest(self) -> None:
        """Validate the persisted store manifest, or adopt one on first open.

        Stores created before the manifest existed (legacy) are inferred from
        the on-disk arrays: a store with a ``peaks/tensor`` array is a tensor
        store whose capacity is the tensor's third dimension.  A mismatch
        between the constructor arguments and the persisted manifest raises
        ``ValueError`` instead of silently misreading the data.
        """
        manifest = self._read_manifest()

        if manifest is None:
            inferred_layout = self._infer_layout_from_arrays()
            if inferred_layout != self._layout:
                raise ValueError(
                    f"Store at {self.store_path} was created with layout="
                    f"{inferred_layout!r} but is being opened with layout="
                    f"{self._layout!r}. Reopen with the original layout or "
                    "rebuild the store."
                )
            if self._layout == "tensor":
                pg = self._peaks_group()
                # A fresh store has no tensor array yet; only existing
                # (legacy) stores carry the capacity in their shape.
                if "tensor" in pg:
                    inferred_m = int(pg["tensor"].shape[2])
                    if inferred_m != self._max_peaks:
                        raise ValueError(
                            f"Store at {self.store_path} was created with "
                            f"max_peaks_per_spectrum={inferred_m} but is being "
                            f"opened with max_peaks_per_spectrum={self._max_peaks}. "
                            "Reopen with the original capacity or rebuild the store."
                        )
                    # Legacy tensor stores were built before the reduction policy
                    # existed; some may contain silently truncated spectra.  The
                    # only safe policy for such stores is the strict one.
                    if self._peak_reduction != "none":
                        raise ValueError(
                            f"Legacy tensor store at {self.store_path} has no "
                            "persisted peak-reduction policy; only "
                            "peak_reduction='none' may be used with it."
                        )
                    logger.warning(
                        "Legacy tensor store %s adopted with capacity %d. "
                        "Note: pre-manifest stores may contain spectra that were "
                        "silently truncated at write time; rebuild the store to "
                        "guarantee peak fidelity.",
                        self.store_path,
                        inferred_m,
                    )
            if not self._is_remote:
                self._write_manifest()
            return

        mismatches: list[str] = []
        if manifest.get("layout") != self._layout:
            mismatches.append(f"layout {manifest.get('layout')!r} != {self._layout!r}")
        if (
            self._layout == "tensor"
            and manifest.get("max_peaks_per_spectrum") != self._max_peaks
        ):
            mismatches.append(
                f"max_peaks_per_spectrum {manifest.get('max_peaks_per_spectrum')!r}"
                f" != {self._max_peaks!r}"
            )
        if manifest.get("peak_reduction") != self._peak_reduction:
            mismatches.append(
                f"peak_reduction {manifest.get('peak_reduction')!r}"
                f" != {self._peak_reduction!r}"
            )
        if mismatches:
            raise ValueError(
                f"Store at {self.store_path} was created with a different "
                f"configuration ({'; '.join(mismatches)}). Reopen with the "
                "original parameters or rebuild the store."
            )

    def _read_manifest(self) -> Optional[dict[str, Any]]:
        """Return the persisted store manifest, or ``None`` for legacy stores."""
        if self._root is None:
            raise RuntimeError("ZarrSpectralStore is not open.")
        try:
            attrs = dict(self._root.attrs)
        except Exception:
            return None
        manifest = attrs.get(_MANIFEST_ATTR)
        return manifest if isinstance(manifest, dict) else None

    def _write_manifest(self) -> None:
        """Persist the active layout / capacity / policy as the store manifest."""
        if self._root is None:
            raise RuntimeError("ZarrSpectralStore is not open.")
        self._root.attrs[_MANIFEST_ATTR] = {
            "layout": self._layout,
            "max_peaks_per_spectrum": self._max_peaks,
            "peak_reduction": self._peak_reduction,
        }

    def _infer_layout_from_arrays(self) -> str:
        """Infer the layout of a legacy (pre-manifest) store from its arrays."""
        pg = self._peaks_group()
        if "tensor" in pg:
            return "tensor"
        if "mz_flat" in pg:
            return "flat"
        return self._layout  # Empty store: nothing to infer.

    # ------------------------------------------------------------------
    # Store re-open helper (for read-only concurrent access)
    # ------------------------------------------------------------------

    def _open_read_store(self) -> Any:  # zarr.Group
        """Return a read-only Zarr group handle.

        For cloud stores this creates a fresh fsspec filesystem +
        FsspecStore per call, ensuring each thread gets its own
        connection pool.  For local stores it reuses the existing group
        (file handles are shared safely via the OS).
        """
        if self._root is None:
            raise RuntimeError("ZarrSpectralStore is not open.")

        if self._is_remote:
            import zarr

            fsspec_store = _make_fsspec_store(
                str(self.store_path),
                storage_options=self._storage_options,
                read_only=True,
            )
            return zarr.open_group(store=fsspec_store, mode="r")

        return self._root

    # ------------------------------------------------------------------
    # Sub-group accessors
    # ------------------------------------------------------------------

    def _peaks_group(self) -> Any:  # zarr.Group
        if self._root is None:
            raise RuntimeError("ZarrSpectralStore is not open.")
        return self._root[self._PEAKS_GROUP]  # type: ignore[return-value]

    def _metadata_group(self) -> Any:  # zarr.Group
        if self._root is None:
            raise RuntimeError("ZarrSpectralStore is not open.")
        return self._root[self._METADATA_GROUP]  # type: ignore[return-value]

    def _metadata_group_ro(self) -> Any:  # zarr.Group
        """Return the metadata group from a read-only store handle."""
        root = self._open_read_store()
        return root[self._METADATA_GROUP]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Retry-wrapped remote read
    # ------------------------------------------------------------------

    def _retry_read(self, func: Callable[[], Any]) -> Any:
        """Execute *func* with exponential-backoff retry for remote stores.

        For local stores this is a transparent pass-through.
        """
        if not self._is_remote:
            return func()
        return _retry_with_backoff(self._retry_config)(func)()

    # ------------------------------------------------------------------
    # Metadata arrays — creation / open / migration
    # ------------------------------------------------------------------

    def _open_or_create_metadata_arrays(self) -> None:
        """Ensure all metadata Zarr arrays exist and determine ``_n_spectra``."""
        mg = self._metadata_group()

        existing_len: int | None = None
        for field in self._ALL_METADATA_FIELDS:
            if field in mg:
                existing_len = int(mg[field].shape[0])
                break

        if existing_len is not None:
            self._n_spectra = existing_len
            self._ensure_metadata_arrays(existing_len)
            return

        # Fresh store: create empty metadata arrays.
        self._ensure_metadata_arrays(0)
        self._n_spectra = 0

    def _ensure_metadata_arrays(self, size: int) -> None:
        """Create any missing metadata arrays with the given *size*."""
        mg = self._metadata_group()
        chunk = (min(max(size, 1), self._metadata_chunk_size),)
        compressor_kwargs: dict[str, Any] = {}
        if self._compressor is not None:
            compressor_kwargs["compressors"] = [self._compressor]

        for field, dtype in self._SCALAR_FIELDS.items():
            if field not in mg:
                mg.create_array(
                    field,
                    shape=(size,),
                    chunks=chunk,
                    dtype=dtype,
                    fill_value=0 if dtype.kind in ("i", "u") else np.nan,
                    **compressor_kwargs,
                )

        for field in self._STRING_FIELDS:
            if field not in mg:
                _create_string_array(
                    mg,
                    field,
                    shape=(size,),
                    chunks=chunk,
                    backend=self._string_backend,
                    compressor=self._compressor
                    if self._string_backend == "stringdtype"
                    else None,
                )

    def _maybe_migrate_legacy_metadata(self) -> None:
        """Migrate from the old ``metadata_index.json`` sidecar if present."""
        if self._is_remote:
            return  # Cannot migrate remote stores.

        legacy_path = self.store_path / _LEGACY_METADATA_INDEX
        if not legacy_path.exists():
            return

        mg = self._metadata_group()
        if "id" in mg:
            logger.info(
                "Legacy %s found but native metadata arrays already "
                "exist — skipping migration.",
                _LEGACY_METADATA_INDEX,
            )
            return

        logger.info(
            "Migrating legacy %s to native Zarr metadata arrays...",
            _LEGACY_METADATA_INDEX,
        )

        with open(legacy_path, "r") as fh:
            legacy_data: list[dict[str, Any]] = json.load(fh)

        n = len(legacy_data)
        if n == 0:
            return

        self._ensure_metadata_arrays(n)
        self._n_spectra = n

        self._write_metadata_rows(mg, 0, legacy_data)

        backup = legacy_path.with_suffix(".json.bak")
        legacy_path.rename(backup)
        logger.info(
            "Migration complete (%d spectra). Legacy index backed up to %s.",
            n,
            backup,
        )

    # ------------------------------------------------------------------
    # Low-level metadata read / write
    # ------------------------------------------------------------------

    def _write_metadata_rows(
        self,
        mg: Any,  # zarr.Group
        start_idx: int,
        entries: list[dict[str, Any]],
    ) -> None:
        """Write *entries* into the metadata arrays starting at *start_idx*."""
        n = len(entries)
        if n == 0:
            return
        end_idx = start_idx + n

        for field, dtype in self._SCALAR_FIELDS.items():
            arr = mg[field]
            values = np.empty(n, dtype=dtype)
            for i, entry in enumerate(entries):
                raw = entry.get(field)
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    values[i] = 0 if dtype.kind in ("i", "u") else np.nan
                else:
                    values[i] = dtype.type(raw)
            arr[start_idx:end_idx] = values

        for field in self._STRING_FIELDS:
            arr = mg[field]
            for i, entry in enumerate(entries):
                arr[start_idx + i] = str(entry.get(field, ""))

    def _read_metadata_entry(self, mg: Any, index: int) -> dict[str, Any]:
        """Read a single metadata row at *index* into a dict."""
        entry: dict[str, Any] = {}

        for field in self._SCALAR_FIELDS:
            val = mg[field][index]
            if hasattr(val, "item"):
                val = val.item()
            entry[field] = val

        for field in self._STRING_FIELDS:
            val = mg[field][index]
            if hasattr(val, "item"):
                val = val.item()
            entry[field] = str(val) if val is not None else ""

        return entry

    # ------------------------------------------------------------------
    # Metadata query (batch, cached) — cloud-optimized
    # ------------------------------------------------------------------

    def metadata_query(
        self,
        fields: list[str],
        indices: Optional[np.ndarray] = None,
        category: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """Batch-read metadata fields with read-through caching.

        This is the primary interface for cloud-optimized pre-filtering.
        Instead of extracting ``precursor_mz`` from Spectrum objects one
        at a time, the similarity engine can call this method once per
        batch to obtain flat numpy arrays of all required metadata.

        Parameters
        ----------
        fields : list of str
            Metadata field names to retrieve.  Must be a subset of
            ``_ALL_METADATA_FIELDS``.
        indices : np.ndarray or None
            Spectrum indices to query (``int64``).  If ``None``, all
            spectra are returned.
        category : str or None
            If provided, only return rows where the ``category`` field
            matches this value.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of field name to a ``float64`` or ``str`` numpy array.

        Raises
        ------
        ValueError
            If any requested field is not present in the store.
        """
        mg = self._metadata_group_ro()
        n = self._n_spectra
        if n == 0:
            return {f: np.array([], dtype=np.float64) for f in fields}

        # Validate fields.
        for field in fields:
            if field not in mg:
                raise ValueError(
                    f"Unknown metadata field: '{field}'. "
                    f"Available: {list(mg.array_keys())}"
                )

        # Determine the index range to fetch.
        if indices is None:
            start, end = 0, n
        else:
            idx_arr = np.asarray(indices, dtype=np.int64)
            if idx_arr.size == 0:
                return {f: np.array([], dtype=np.float64) for f in fields}
            start = int(np.min(idx_arr))
            end = int(np.max(idx_arr)) + 1

        # If a category filter is requested, pre-fetch the category column
        # and build a boolean mask.
        cat_mask: Optional[np.ndarray] = None
        if category is not None:
            cat_data = self._cached_slice(mg, "category", 0, n)
            cat_mask = np.array([str(cat_data[i]) == category for i in range(n)])
            if indices is not None:
                # Combine the index-based filter with the category filter.
                idx_set = set(int(i) for i in indices)
                cat_mask = np.array([cat_mask[i] and (i in idx_set) for i in range(n)])

        # Fetch each requested field via the cache.
        result: dict[str, np.ndarray] = {}
        for field in fields:
            data = self._cached_slice(mg, field, start, end)
            if cat_mask is not None:
                data = data[cat_mask[start:end]]
            elif indices is not None:
                # Map global indices to the [start:end) window.
                idx_arr = np.asarray(indices, dtype=np.int64)
                rel_indices = idx_arr - start
                data = data[rel_indices]
            result[field] = np.asarray(data)

        return result

    def _cached_slice(
        self,
        mg: Any,  # zarr.Group
        field: str,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Read a metadata slice, consulting the cache first."""
        key = (field, start, end)

        def loader() -> np.ndarray:
            arr = mg[field]
            return self._retry_read(lambda: np.asarray(arr[start:end]))

        return self._metadata_cache.get(key, loader)

    # ------------------------------------------------------------------
    # Peak group helpers
    # ------------------------------------------------------------------

    def _ensure_peak_arrays_flat(self, total_peaks: int) -> None:
        """Create the flat peak arrays and boundaries in ``peaks/`` if absent."""
        pg = self._peaks_group()
        kwargs: dict[str, Any] = {}
        if self._compressor is not None:
            kwargs["compressors"] = [self._compressor]
        if "mz_flat" not in pg:
            pg.create_array(
                "mz_flat",
                shape=(total_peaks,),
                chunks=(min(total_peaks, self._peak_chunk_size),),
                dtype=np.float64,
                **kwargs,
            )
        if "intensity_flat" not in pg:
            pg.create_array(
                "intensity_flat",
                shape=(total_peaks,),
                chunks=(min(total_peaks, self._peak_chunk_size),),
                dtype=np.float64,
                **kwargs,
            )

    def _ensure_peak_arrays_tensor(
        self,
        n_spectra: int,
        n_batches: int,
    ) -> None:
        """Create the 3-D tensor and peak_counts arrays in ``peaks/``."""
        pg = self._peaks_group()
        kwargs: dict[str, Any] = {}
        if self._compressor is not None:
            kwargs["compressors"] = [self._compressor]

        M = self._max_peaks

        if "tensor" not in pg:
            pg.create_array(
                "tensor",
                shape=(2, n_batches, M),
                chunks=(2, 1, min(M, 256)),
                dtype=np.float64,
                fill_value=0.0,
                **kwargs,
            )

        if "peak_counts" not in pg:
            pg.create_array(
                "peak_counts",
                shape=(n_spectra,),
                chunks=(min(max(n_spectra, 1), self._tensor_batch_size),),
                dtype=np.int32,
                fill_value=0,
            )

    # ------------------------------------------------------------------
    # Thread-safe append coordination
    # ------------------------------------------------------------------

    def _acquire_write_lock(self) -> tuple[int | None, bool]:
        """Acquire both in-process and cross-process write locks.

        Returns
        -------
        tuple[int | None, bool]
            ``(file_descriptor, in_process_lock_acquired)``.
            Pass both to :meth:`_release_write_lock`.
        """
        if self._is_remote:
            # Cloud stores: only use the in-process lock; there is no
            # cross-process fcntl on remote filesystems.
            acquired = self._lock.acquire(blocking=True)
            return None, acquired

        lock_path = self.store_path / _LOCK_FILENAME
        fd = _acquire_file_lock(lock_path)
        acquired = self._lock.acquire(blocking=True)
        return fd, acquired

    def _release_write_lock(self, fd: int | None, acquired: bool) -> None:
        """Release locks acquired by :meth:`_acquire_write_lock`."""
        if acquired:
            self._lock.release()
        if not self._is_remote:
            _release_file_lock(fd)

    # ------------------------------------------------------------------
    # Spectra insertion
    # ------------------------------------------------------------------

    def add_spectra(
        self,
        spectra: Iterator[Spectrum],
        category: str = "default",
        batch_size: int = 5000,
    ) -> int:
        """Append spectra to the Zarr store (thread-safe).

        Spectra are stored exactly as given: no peak is ever added, dropped,
        or reordered implicitly.  The tensor layout stores spectra up to
        ``max_peaks_per_spectrum`` peaks and zero-pads shorter ones; an
        over-capacity spectrum is either rejected with
        :class:`PeakCapacityError` (default) or explicitly reduced to its
        most intense peaks with the reduction recorded in provenance
        (``peak_reduction="topN"``).  The flat layout has no peak limit.
        """
        if self._is_remote:
            raise RuntimeError(
                "Writing to remote Zarr stores is not supported. "
                "Build the store locally and upload the resulting "
                "directory/prefix to your cloud storage."
            )

        spectra_list: list[Spectrum] = []
        for spec in spectra:
            if spec is None:
                continue
            spectra_list.append(spec)

        if not spectra_list:
            return 0

        if self._layout == "tensor":
            return self._add_spectra_tensor(spectra_list, category)
        else:
            return self._add_spectra_flat(spectra_list, category)

    def _add_spectra_flat(
        self,
        spectra_list: list[Spectrum],
        category: str,
    ) -> int:
        """Append spectra using the flat 1-D layout."""
        n_new = len(spectra_list)

        # Phase 1: prepare data (lock-free).
        total_new_peaks = sum(int(spec.peaks.mz.size) for spec in spectra_list)

        new_boundaries = np.zeros(n_new + 1, dtype=np.int64)
        new_mz_flat = np.empty(total_new_peaks, dtype=np.float64)
        new_intensity_flat = np.empty(total_new_peaks, dtype=np.float64)
        metadata_entries: list[dict[str, Any]] = []

        peak_offset = 0
        for i, spec in enumerate(spectra_list):
            mz_arr = np.asarray(spec.peaks.mz, dtype=np.float64)
            intensity_arr = np.asarray(spec.peaks.intensities, dtype=np.float64)
            n_peaks = mz_arr.size

            new_boundaries[i + 1] = new_boundaries[i] + n_peaks

            end = peak_offset + n_peaks
            new_mz_flat[peak_offset:end] = mz_arr
            new_intensity_flat[peak_offset:end] = intensity_arr
            peak_offset = end

            entry = self._make_metadata_entry(spec, category)
            metadata_entries.append(entry)

        # Phase 2: append under write lock.
        fd, lock_acquired = self._acquire_write_lock()
        try:
            pg = self._peaks_group()
            mg = self._metadata_group()

            existing_peaks: int = 0
            if "mz_flat" in pg:
                existing_peaks = int(pg["mz_flat"].shape[0])
            total_peaks = existing_peaks + total_new_peaks

            if existing_peaks == 0 and "mz_flat" not in pg:
                self._ensure_peak_arrays_flat(total_peaks)
                pg["mz_flat"][:] = new_mz_flat
                pg["intensity_flat"][:] = new_intensity_flat
            else:
                pg["mz_flat"].resize(total_peaks)
                pg["intensity_flat"].resize(total_peaks)
                pg["mz_flat"][existing_peaks:total_peaks] = new_mz_flat
                pg["intensity_flat"][existing_peaks:total_peaks] = new_intensity_flat

            # Boundaries array.
            old_n_spectra = self._n_spectra
            new_boundaries_global = new_boundaries[1:] + existing_peaks

            if "boundaries" not in pg:
                full_bounds = np.zeros(old_n_spectra + n_new + 1, dtype=np.int64)
                full_bounds[1:] = new_boundaries_global
                pg.create_array(
                    "boundaries",
                    shape=(old_n_spectra + n_new + 1,),
                    chunks=(min(old_n_spectra + n_new + 1, 4096),),
                    dtype=np.int64,
                )
                pg["boundaries"][:] = full_bounds
            else:
                old_bounds = pg["boundaries"][:]
                full_bounds = np.concatenate([old_bounds, new_boundaries_global])
                pg["boundaries"].resize(old_n_spectra + n_new + 1)
                pg["boundaries"][:] = full_bounds

            # Metadata arrays.
            new_total = old_n_spectra + n_new
            self._ensure_metadata_arrays(new_total)
            for field in self._ALL_METADATA_FIELDS:
                arr = mg[field]
                if arr.shape[0] < new_total:
                    arr.resize(new_total)

            self._write_metadata_rows(mg, old_n_spectra, metadata_entries)
            self._n_spectra = new_total

        finally:
            self._release_write_lock(fd, lock_acquired)

        # Invalidate metadata cache after writes.
        self._metadata_cache.invalidate()

        return n_new

    def _add_spectra_tensor(
        self,
        spectra_list: list[Spectrum],
        category: str,
    ) -> int:
        """Append spectra using the 3-D tensor layout.

        Over-capacity spectra are handled explicitly before any write (the
        whole batch fails atomically): ``peak_reduction="none"`` raises
        :class:`PeakCapacityError`; ``peak_reduction="topN"`` retains the
        most intense peaks and records the reduction in provenance metadata.
        """
        n_new = len(spectra_list)
        M = self._max_peaks

        metadata_entries: list[dict[str, Any]] = []
        peak_counts = np.zeros(n_new, dtype=np.int32)
        tensor_data = np.zeros((2, n_new, M), dtype=np.float64)

        for i, spec in enumerate(spectra_list):
            mz_arr = np.asarray(spec.peaks.mz, dtype=np.float64)
            intensity_arr = np.asarray(spec.peaks.intensities, dtype=np.float64)
            original_count = mz_arr.size

            reduction_record: Optional[dict[str, Any]] = None
            if original_count > M:
                if self._peak_reduction == "none":
                    raise PeakCapacityError(
                        f"Spectrum {spec.get('id', '<no id>')!r} has "
                        f"{original_count} peaks, exceeding the tensor-layout "
                        f"capacity of {M} (max_peaks_per_spectrum). No peak-"
                        "reduction policy is active, so the batch was rejected "
                        "unchanged: nothing was written. Set "
                        "peak_reduction='topN' to explicitly reduce "
                        "over-capacity spectra (recorded in provenance)."
                    )
                mz_arr, intensity_arr = _top_n_peaks(mz_arr, intensity_arr, M)
                reduction_record = {
                    "mode": "topN",
                    "max_peaks": M,
                    "original_peak_count": int(original_count),
                }

            n_peaks = mz_arr.size
            peak_counts[i] = n_peaks

            tensor_data[0, i, :n_peaks] = mz_arr
            tensor_data[1, i, :n_peaks] = intensity_arr

            entry = self._make_metadata_entry(
                spec, category, peak_reduction=reduction_record
            )
            metadata_entries.append(entry)

        fd, lock_acquired = self._acquire_write_lock()
        try:
            pg = self._peaks_group()
            mg = self._metadata_group()

            old_n_spectra = self._n_spectra
            new_total = old_n_spectra + n_new

            if "tensor" not in pg:
                self._ensure_peak_arrays_tensor(new_total, new_total)
            else:
                # Resize the tensor along the batch dimension.
                pg["tensor"].resize(2, new_total, M)
                if "peak_counts" in pg:
                    pg["peak_counts"].resize(new_total)

            # Ensure peak_counts exists.
            if "peak_counts" not in pg:
                pg.create_array(
                    "peak_counts",
                    shape=(new_total,),
                    chunks=(min(max(new_total, 1), self._tensor_batch_size),),
                    dtype=np.int32,
                    fill_value=0,
                )

            pg["tensor"][:, old_n_spectra:new_total, :] = tensor_data
            pg["peak_counts"][old_n_spectra:new_total] = peak_counts

            # Metadata.
            self._ensure_metadata_arrays(new_total)
            for field in self._ALL_METADATA_FIELDS:
                arr = mg[field]
                if arr.shape[0] < new_total:
                    arr.resize(new_total)

            self._write_metadata_rows(mg, old_n_spectra, metadata_entries)
            self._n_spectra = new_total

        finally:
            self._release_write_lock(fd, lock_acquired)

        self._metadata_cache.invalidate()
        return n_new

    # ------------------------------------------------------------------
    # Metadata entry builder
    # ------------------------------------------------------------------

    @staticmethod
    def _make_metadata_entry(
        spec: Spectrum,
        category: str,
        peak_reduction: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build a metadata dict from a Spectrum for storage.

        ``peak_reduction`` (when given) is recorded inside the spectrum's
        ``extra_metadata`` under the ``peak_reduction`` key: any intentional
        storage-stage transformation is part of the stored provenance and
        round-trips into reconstructed spectra.
        """
        pmz = spec.get("precursor_mz")
        extra_meta = spec.metadata.copy()
        for reserved in (
            "id",
            "compound_name",
            "name",
            "precursor_mz",
            "charge",
            "ionmode",
            "adduct",
        ):
            extra_meta.pop(reserved, None)
        if peak_reduction is not None:
            extra_meta["peak_reduction"] = peak_reduction
        extra_json = json.dumps(extra_meta, default=str)

        return {
            "id": str(spec.get("id", "")),
            "name": str(spec.get("compound_name") or spec.get("name") or ""),
            "precursor_mz": float(pmz) if pmz is not None else np.nan,
            "charge": int(spec.get("charge", 0) or 0),
            "ionmode": str(spec.get("ionmode") or ""),
            "adduct": str(spec.get("adduct") or ""),
            "category": str(category),
            "extra_metadata": extra_json,
        }

    # ------------------------------------------------------------------
    # Spectrum retrieval
    # ------------------------------------------------------------------

    def _peak_slice(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mz, intensity) arrays for the spectrum at *index*.

        Uses the layout-appropriate access pattern.
        """
        if self._layout == "tensor":
            return self._peak_slice_tensor(index)

        pg = self._peaks_group()
        boundaries = self._retry_read(lambda: pg["boundaries"][:])
        start = int(boundaries[index])
        end = int(boundaries[index + 1])

        def read_mz() -> np.ndarray:
            return np.asarray(pg["mz_flat"][start:end], dtype=np.float64)

        def read_int() -> np.ndarray:
            return np.asarray(pg["intensity_flat"][start:end], dtype=np.float64)

        mz = self._retry_read(read_mz)
        intensity = self._retry_read(read_int)
        return mz, intensity

    def _peak_slice_tensor(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mz, intensity) for tensor-layout spectrum at *index*."""
        pg = self._peaks_group()
        n_peaks = int(pg["peak_counts"][index])

        def read_tensor() -> np.ndarray:
            return np.asarray(pg["tensor"][:, index, :n_peaks], dtype=np.float64)

        data = self._retry_read(read_tensor)
        return data[0, :], data[1, :]

    def _row_to_spectrum(self, index: int) -> Spectrum:
        """Reconstruct a ``matchms.Spectrum`` from the store at *index*."""
        mg = self._metadata_group_ro()
        entry = self._read_metadata_entry(mg, index)

        extra_json = entry.get("extra_metadata", "{}")
        try:
            extra_meta: dict[str, Any] = json.loads(extra_json)
        except (json.JSONDecodeError, TypeError):
            extra_meta = {}

        metadata: dict[str, Any] = dict(extra_meta)
        metadata["id"] = entry.get("id", "")
        if "compound_name" not in metadata and entry.get("name"):
            metadata["compound_name"] = entry["name"]
        if entry.get("precursor_mz") is not None and not (
            isinstance(entry["precursor_mz"], float) and np.isnan(entry["precursor_mz"])
        ):
            metadata["precursor_mz"] = entry["precursor_mz"]
        if entry.get("charge"):
            metadata["charge"] = entry["charge"]
        if entry.get("ionmode"):
            metadata["ionmode"] = entry["ionmode"]
        if entry.get("adduct"):
            metadata["adduct"] = entry["adduct"]

        mz_arr, intensity_arr = self._peak_slice(index)
        return Spectrum(mz=mz_arr, intensities=intensity_arr, metadata=metadata)

    def get_spectra(
        self,
        category: Optional[str] = None,
        name_pattern: Optional[str] = None,
    ) -> Iterator[Spectrum]:
        """Stream spectra, optionally filtered by category and/or name pattern."""
        mg = self._metadata_group_ro()
        n = self._n_spectra
        if n == 0:
            return

        if category is not None or name_pattern is not None:
            cat_arr = mg["category"] if category is not None else None
            name_arr = mg["name"] if name_pattern is not None else None

            import re

            regex = None
            if name_pattern is not None:
                pattern = (
                    re.escape(name_pattern).replace(r"\%", ".*").replace(r"\_", ".")
                )
                regex = re.compile(pattern)

            for i in range(n):
                if cat_arr is not None:
                    if str(cat_arr[i]) != category:
                        continue
                if regex is not None:
                    assert name_arr is not None
                    if not regex.match(str(name_arr[i])):
                        continue
                yield self._row_to_spectrum(i)
        else:
            for i in range(n):
                yield self._row_to_spectrum(i)

    def get_spectrum_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """Look up a spectrum by its unique identifier."""
        mg = self._metadata_group_ro()
        id_arr = mg["id"]
        n = self._n_spectra
        for i in range(n):
            if str(id_arr[i]) == spectrum_id:
                return self._row_to_spectrum(i)
        return None

    def batch_get_arrays(
        self,
        spectrum_ids: Optional[list[str]] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Batch-read m/z and intensity arrays for similarity matrix construction.

        When *spectrum_ids* is ``None``, all spectra are returned in store
        order.  For the tensor layout, slices are read directly from the
        3-D tensor with zero-copy views of the uncompressed chunks.
        """
        n = self._n_spectra
        if n == 0:
            return [], []

        if self._layout == "tensor":
            return self._batch_get_arrays_tensor(spectrum_ids)

        pg = self._peaks_group()
        if "boundaries" not in pg:
            return [], []

        if spectrum_ids is not None:
            mg = self._metadata_group_ro()
            id_arr = mg["id"]
            id_to_idx: dict[str, int] = {}
            for i in range(n):
                sid = str(id_arr[i])
                if sid:
                    id_to_idx[sid] = i
            indices = [id_to_idx[sid] for sid in spectrum_ids if sid in id_to_idx]
        else:
            indices = list(range(n))

        boundaries = self._retry_read(lambda: pg["boundaries"][:])
        mz_flat = pg["mz_flat"]
        int_flat = pg["intensity_flat"]

        mz_arrays: list[np.ndarray] = []
        intensity_arrays: list[np.ndarray] = []
        for idx in indices:
            start = int(boundaries[idx])
            end = int(boundaries[idx + 1])
            mz_arrays.append(
                self._retry_read(
                    lambda s=start, e=end: np.asarray(mz_flat[s:e], dtype=np.float64)  # type: ignore[misc]
                )
            )
            intensity_arrays.append(
                self._retry_read(
                    lambda s=start, e=end: np.asarray(int_flat[s:e], dtype=np.float64)  # type: ignore[misc]
                )
            )

        return mz_arrays, intensity_arrays

    def _batch_get_arrays_tensor(
        self,
        spectrum_ids: Optional[list[str]] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Batch-read from the 3-D tensor layout."""
        pg = self._peaks_group()
        n = self._n_spectra

        if spectrum_ids is not None:
            mg = self._metadata_group_ro()
            id_arr = mg["id"]
            id_to_idx: dict[str, int] = {}
            for i in range(n):
                sid = str(id_arr[i])
                if sid:
                    id_to_idx[sid] = i
            indices = [id_to_idx[sid] for sid in spectrum_ids if sid in id_to_idx]
        else:
            indices = list(range(n))

        peak_counts = self._retry_read(lambda: np.asarray(pg["peak_counts"][:]))

        mz_arrays: list[np.ndarray] = []
        intensity_arrays: list[np.ndarray] = []
        for idx in indices:
            n_peaks = int(peak_counts[idx])
            data = self._retry_read(
                lambda i=idx, p=n_peaks: np.asarray(  # type: ignore[misc]
                    pg["tensor"][:, i, :p], dtype=np.float64
                )
            )
            mz_arrays.append(data[0, :].copy())
            intensity_arrays.append(data[1, :].copy())

        return mz_arrays, intensity_arrays

    # ------------------------------------------------------------------
    # Aggregate queries (cached where beneficial)
    # ------------------------------------------------------------------

    def get_total_spectra_count(self) -> int:
        """Total number of spectra."""
        return self._n_spectra

    def get_category_counts(self) -> dict[str, int]:
        """Spectra count per category (cached)."""
        cache_key = ("__category_counts__", 0, 0)

        def loader() -> np.ndarray:
            mg = self._metadata_group_ro()
            cat_arr = mg["category"]
            counts: dict[str, int] = {}
            n = self._n_spectra
            for i in range(n):
                cat = str(cat_arr[i])
                counts[cat] = counts.get(cat, 0) + 1
            # Return as object array for cache storage.
            return np.array([counts], dtype=object)

        result = self._metadata_cache.get(cache_key, loader)
        # The loader stores a single-element object array.
        if isinstance(result, np.ndarray) and result.dtype == object:
            return result.item()
        return result

    def get_precursor_mz_range(self) -> tuple[float, float]:
        """Min and max precursor m/z (cached)."""
        if self._n_spectra == 0:
            return (0.0, 0.0)

        cache_key = ("__precursor_mz_range__", 0, 0)

        def loader() -> np.ndarray:
            mg = self._metadata_group_ro()
            pmz_arr = mg["precursor_mz"]
            valid = pmz_arr[~np.isnan(pmz_arr[:])]
            if valid.size == 0:
                return np.array([0.0, 0.0], dtype=np.float64)
            return np.array(
                [float(np.min(valid)), float(np.max(valid))], dtype=np.float64
            )

        result = self._metadata_cache.get(cache_key, loader)
        if isinstance(result, np.ndarray) and result.size == 2:
            return (float(result[0]), float(result[1]))
        return (0.0, 0.0)

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return metadata cache hit/miss statistics."""
        return self._metadata_cache.stats

    @property
    def is_remote(self) -> bool:
        """Return ``True`` if the store is backed by a remote URL."""
        return self._is_remote

    @property
    def layout(self) -> str:
        """Return the active storage layout (``"flat"`` or ``"tensor"``)."""
        return self._layout

    def backend_provenance(self) -> dict[str, Any]:
        """Describe this backend's identity for run provenance.

        Returns
        -------
        dict
            ``{"backend": "zarr", "path": str, "spectrum_count": int}``.
        """
        return {
            "backend": "zarr",
            "path": str(self.store_path),
            "spectrum_count": self.get_total_spectra_count(),
        }


# ===================================================================
# ZarrPeakArrayStore
# ===================================================================


class ZarrPeakArrayStore:
    """Chunked Zarr storage for fragment peak arrays (m/z + intensity).

    This class is the array-storage half of the hybrid SQLite + Zarr backend.
    SQLite retains spectrum metadata rows plus a ``zarr_ref`` / ``zarr_index``
    reference pair; this class persists the ``float64`` fragment vectors in a
    flat, chunked layout:

    ``peaks/``
        ``mz_flat`` (float64) and ``intensity_flat`` (float64) — concatenated
        per-spectrum vectors indexed by ``peaks/boundaries`` (int64), where
        ``boundaries[i]`` is the flat offset of spectrum *i*.  ``boundaries``
        holds ``n_spectra + 1`` entries and ``boundaries[0] == 0``.

    The root group attributes record ``store_uuid`` — a unique identifier that
    the SQLite ``zarr_ref`` column references so database rows can be matched
    to the array store that owns their peaks.

    Chunk sizing is configurable per array dimension:

    - ``peak_chunk_size`` — float64 elements per chunk in ``mz_flat`` and
      ``intensity_flat`` (default ~8 MB per chunk).  Chunks that are too small
      make per-chunk metadata overhead dominate; chunks that are too large
      cause decompression memory spikes during parallel reads.
    - ``boundary_chunk_size`` — spectra per chunk in ``boundaries``.

    Reads are lock-free and safe for multiprocessing: every read opens a fresh
    read-only Zarr handle, and appends only write *new* chunks (the
    ``boundaries`` tail is appended without rewriting existing chunks), so
    concurrent readers always observe a consistent snapshot of previously
    appended spectra.  Writes serialize through the same dual-lock strategy
    used by :class:`ZarrSpectralStore` (``threading.Lock`` +
    ``fcntl.flock``).

    Parameters
    ----------
    store_path : Path
        Directory path for local stores, or an fsspec-compatible URL
        (``s3://``, ``gs://``, ``https://``) for cloud stores.  Remote stores
        are read-only.
    peak_chunk_size : int, optional
        Float64 elements per chunk in ``mz_flat`` / ``intensity_flat``.
    boundary_chunk_size : int, optional
        Spectra per chunk in ``boundaries``.
    compressor : optional
        Zarr compressor (defaults to Blosc+zstd, clevel=3).
    overwrite : bool, optional
        If ``True``, delete any existing local store before creating a
        fresh one.
    storage_options : dict, optional
        Keyword arguments forwarded to the ``fsspec`` filesystem constructor.
    remote_timeout : float, optional
        Timeout in seconds for individual remote read operations.
    remote_retries : int, optional
        Maximum retry attempts for remote reads (exponential backoff).

    Examples
    --------
    >>> store = ZarrPeakArrayStore("library.zarr", peak_chunk_size=262144)
    >>> start = store.append_spectra([mz_array], [intensity_array])
    >>> mz_read, intensity_read = store.read_spectrum(start)
    >>> store.close()
    """

    _PEAKS_GROUP = "peaks"
    _MZ_ARRAY = "mz_flat"
    _INTENSITY_ARRAY = "intensity_flat"
    _BOUNDARIES_ARRAY = "boundaries"
    _UUID_ATTR = "store_uuid"

    def __init__(
        self,
        store_path: Path,
        peak_chunk_size: int = _DEFAULT_PEAK_CHUNK_SIZE,
        boundary_chunk_size: int = _DEFAULT_BOUNDARY_CHUNK_SIZE,
        compressor: Optional[Any] = None,
        overwrite: bool = False,
        storage_options: Optional[dict[str, Any]] = None,
        remote_timeout: float = _DEFAULT_REMOTE_TIMEOUT,
        remote_retries: int = _DEFAULT_REMOTE_RETRIES,
    ) -> None:
        if peak_chunk_size < 1:
            raise ValueError("peak_chunk_size must be >= 1.")
        if boundary_chunk_size < 1:
            raise ValueError("boundary_chunk_size must be >= 1.")

        self.store_path = Path(store_path)
        self._peak_chunk_size = peak_chunk_size
        self._boundary_chunk_size = boundary_chunk_size
        self._compressor = compressor
        self._storage_options = storage_options
        self._is_remote = _is_remote_url(str(store_path))
        self._retry_config = RetryConfig(max_retries=remote_retries)
        self._remote_timeout = remote_timeout

        self._root: Optional[Any] = None  # zarr.Group
        self._store: Optional[Any] = None  # underlying zarr.Store
        self._n_spectra: int = 0
        self._store_uuid: str = ""
        # In-process lock for coordinating appends within the same process.
        self._lock = threading.Lock()

        self._initialize(overwrite)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _initialize(self, overwrite: bool) -> None:
        """Open or create the Zarr group and peak arrays."""
        import zarr

        store_path_str = str(self.store_path)

        if overwrite and not self._is_remote and self.store_path.exists():
            import shutil

            shutil.rmtree(store_path_str)

        if not self._is_remote:
            self.store_path.mkdir(parents=True, exist_ok=True)

        if self._is_remote:
            self._store = _make_fsspec_store(
                store_path_str,
                storage_options=self._storage_options,
                read_only=True,
            )
            self._root = zarr.open_group(store=self._store, mode="r")
        else:
            self._root = zarr.open_group(store_path_str, mode="a")
            self._store = None

        if self._compressor is None:
            self._compressor = _default_compressor()

        # Ensure the peaks sub-group and flat arrays exist.
        if self._PEAKS_GROUP not in self._root:
            try:
                self._root.create_group(self._PEAKS_GROUP)
            except Exception:
                # Group may already exist in a concurrent session.
                pass

        pg = self._root[self._PEAKS_GROUP]  # type: ignore[return-value]
        compressor_kwargs: dict[str, Any] = {}
        if self._compressor is not None:
            compressor_kwargs["compressors"] = [self._compressor]

        for name in (self._MZ_ARRAY, self._INTENSITY_ARRAY):
            if name not in pg:  # type: ignore[operator]
                pg.create_array(  # type: ignore[union-attr]
                    name,
                    shape=(0,),
                    chunks=(self._peak_chunk_size,),
                    dtype=np.float64,
                    fill_value=0.0,
                    **compressor_kwargs,
                )

        if self._BOUNDARIES_ARRAY not in pg:  # type: ignore[operator]
            pg.create_array(  # type: ignore[union-attr]
                self._BOUNDARIES_ARRAY,
                shape=(0,),
                chunks=(self._boundary_chunk_size,),
                dtype=np.int64,
                fill_value=0,
            )

        # Store UUID — generated on first creation, stable afterwards.
        if self._UUID_ATTR in self._root.attrs:
            self._store_uuid = str(self._root.attrs[self._UUID_ATTR])
        else:
            if self._is_remote:
                raise RuntimeError(
                    "Remote Zarr store is missing the 'store_uuid' attribute; "
                    "cannot validate the database reference."
                )
            import uuid

            self._store_uuid = uuid.uuid4().hex
            self._root.attrs[self._UUID_ATTR] = self._store_uuid

        boundaries = pg[self._BOUNDARIES_ARRAY]  # type: ignore[index]
        self._n_spectra = max(0, int(boundaries.shape[0]) - 1)  # type: ignore[union-attr]

        logger.info(
            "ZarrPeakArrayStore opened: path=%s remote=%s n_spectra=%d "
            "peak_chunk_size=%d boundary_chunk_size=%d",
            store_path_str,
            self._is_remote,
            self._n_spectra,
            self._peak_chunk_size,
            self._boundary_chunk_size,
        )

    def close(self) -> None:
        """Release Zarr handles.  Data is already durable on disk/object store."""
        self._root = None
        self._store = None

    @property
    def is_open(self) -> bool:
        """Return ``True`` while the store has an open Zarr handle."""
        return self._root is not None

    # ------------------------------------------------------------------
    # Read helpers — lock-free, fresh read-only handle per call
    # ------------------------------------------------------------------

    def _open_read_group(self) -> Any:  # zarr.Group
        """Open a fresh read-only Zarr group for a single read operation.

        Opening a new handle per call keeps reads lock-free and safe across
        processes: no file handle, cache, or chunk state is shared between
        threads, worker processes, or the writer.
        """
        import zarr

        if self._is_remote:
            fsspec_store = _make_fsspec_store(
                str(self.store_path),
                storage_options=self._storage_options,
                read_only=True,
            )
            return zarr.open_group(store=fsspec_store, mode="r")
        return zarr.open_group(str(self.store_path), mode="r")

    def _retry_read(self, func: Callable[[], Any]) -> Any:
        """Execute *func* with exponential-backoff retry for remote stores."""
        if not self._is_remote:
            return func()
        return _retry_with_backoff(self._retry_config)(func)()

    # ------------------------------------------------------------------
    # Writes — append-only under dual lock
    # ------------------------------------------------------------------

    def _acquire_write_lock(self) -> tuple[int | None, bool]:
        """Acquire in-process and cross-process write locks."""
        if self._is_remote:
            acquired = self._lock.acquire(blocking=True)
            return None, acquired
        lock_path = self.store_path / _LOCK_FILENAME
        fd = _acquire_file_lock(lock_path)
        acquired = self._lock.acquire(blocking=True)
        return fd, acquired

    def _release_write_lock(self, fd: int | None, acquired: bool) -> None:
        """Release locks acquired by :meth:`_acquire_write_lock`."""
        if acquired:
            self._lock.release()
        if not self._is_remote:
            _release_file_lock(fd)

    def append_spectra(
        self,
        mz_arrays: Sequence[np.ndarray],
        intensity_arrays: Sequence[np.ndarray],
    ) -> int:
        """Append spectrum peak vectors and return the first new index.

        The returned index is the spectrum position that callers must persist
        alongside their metadata rows (``zarr_index``).

        Parameters
        ----------
        mz_arrays : sequence of np.ndarray
            Fragment m/z vectors, one per spectrum (``float64``).
        intensity_arrays : sequence of np.ndarray
            Fragment intensity vectors, one per spectrum (``float64``).

        Returns
        -------
        int
            The start index of the first appended spectrum.

        Raises
        ------
        ValueError
            If the two sequences have different lengths or an array is not
            one-dimensional.
        RuntimeError
            If the store is remote (remote stores are read-only).
        """
        if self._is_remote:
            raise RuntimeError(
                "Writing to remote Zarr stores is not supported. "
                "Build the store locally and upload the resulting "
                "directory/prefix to your cloud storage."
            )

        mz_list = [np.asarray(a, dtype=np.float64) for a in mz_arrays]
        intensity_list = [np.asarray(a, dtype=np.float64) for a in intensity_arrays]

        if len(mz_list) != len(intensity_list):
            raise ValueError(
                "mz_arrays and intensity_arrays must have the same length."
            )
        for mz_arr, intensity_arr in zip(mz_list, intensity_list):
            if mz_arr.ndim != 1 or intensity_arr.ndim != 1:
                raise ValueError("Peak arrays must be one-dimensional.")
            if mz_arr.size != intensity_arr.size:
                raise ValueError("Each m/z and intensity pair must match in size.")

        if not mz_list:
            return self._n_spectra

        peak_counts = np.fromiter(
            (arr.size for arr in mz_list), dtype=np.int64, count=len(mz_list)
        )
        total_new_peaks = int(peak_counts.sum())
        if total_new_peaks > 0:
            new_mz_flat = np.concatenate(mz_list)
            new_intensity_flat = np.concatenate(intensity_list)
        else:
            new_mz_flat = np.empty(0, dtype=np.float64)
            new_intensity_flat = np.empty(0, dtype=np.float64)
        new_tail = np.cumsum(peak_counts, dtype=np.int64)

        fd, lock_acquired = self._acquire_write_lock()
        try:
            if self._root is None:
                raise RuntimeError("ZarrPeakArrayStore is not open.")
            pg = self._root[self._PEAKS_GROUP]
            mz_flat = pg[self._MZ_ARRAY]
            intensity_flat = pg[self._INTENSITY_ARRAY]
            boundaries = pg[self._BOUNDARIES_ARRAY]

            existing_spectra = max(0, int(boundaries.shape[0]) - 1)
            existing_peaks = (
                int(boundaries[existing_spectra]) if existing_spectra > 0 else 0
            )
            total_peaks = existing_peaks + total_new_peaks

            mz_flat.resize(total_peaks)
            intensity_flat.resize(total_peaks)
            if total_new_peaks > 0:
                mz_flat[existing_peaks:total_peaks] = new_mz_flat
                intensity_flat[existing_peaks:total_peaks] = new_intensity_flat

            # Append only the new boundary tail — never rewrite existing
            # chunks so concurrent lock-free readers stay consistent.
            new_n_spectra = existing_spectra + len(mz_list)
            boundaries.resize(new_n_spectra + 1)
            boundaries[existing_spectra + 1 : new_n_spectra + 1] = (
                existing_peaks + new_tail
            )

            start_index = existing_spectra
            self._n_spectra = new_n_spectra
        finally:
            self._release_write_lock(fd, lock_acquired)

        return start_index

    def truncate(self, n_spectra: int) -> None:
        """
        Remove trailing spectra so the store holds exactly *n_spectra*.

        Used by the BLOB→Zarr migration to discard orphaned arrays left by
        an interrupted run (arrays appended but never referenced by SQLite
        rows). Zarr v3 ``resize`` drops the trailing chunks entirely.

        Parameters
        ----------
        n_spectra : int
            Target number of spectra. Must be between ``0`` and the current
            count.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If *n_spectra* is negative or larger than the current count.
        RuntimeError
            If the store is remote (remote stores are read-only).

        Examples
        --------
        >>> store.truncate(10)
        """
        if self._is_remote:
            raise RuntimeError(
                "Writing to remote Zarr stores is not supported. "
                "Build the store locally and upload the resulting "
                "directory/prefix to your cloud storage."
            )
        if n_spectra < 0 or n_spectra > self._n_spectra:
            raise ValueError(
                f"n_spectra must be between 0 and {self._n_spectra}; got {n_spectra}."
            )

        fd, lock_acquired = self._acquire_write_lock()
        try:
            if self._root is None:
                raise RuntimeError("ZarrPeakArrayStore is not open.")
            pg = self._root[self._PEAKS_GROUP]
            mz_flat = pg[self._MZ_ARRAY]
            intensity_flat = pg[self._INTENSITY_ARRAY]
            boundaries = pg[self._BOUNDARIES_ARRAY]

            new_total_peaks = int(boundaries[n_spectra]) if n_spectra > 0 else 0
            mz_flat.resize(new_total_peaks)
            intensity_flat.resize(new_total_peaks)
            if n_spectra == 0:
                # Keep a single zero sentinel so future appends have
                # boundaries[0] == 0 to anchor their cumulative offsets.
                boundaries.resize(1)
                boundaries[0] = 0
            else:
                boundaries.resize(n_spectra + 1)
            self._n_spectra = n_spectra
        finally:
            self._release_write_lock(fd, lock_acquired)

    # ------------------------------------------------------------------
    # Reads — lock-free
    # ------------------------------------------------------------------

    def read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read the ``(mz, intensity)`` pair for the spectrum at *index*.

        Both returned arrays are ``float64`` and owned by the caller.
        """
        mz_arrays, intensity_arrays = self.read_spectra([index])
        return mz_arrays[0], intensity_arrays[0]

    def read_spectra(
        self,
        indices: Optional[Sequence[int]] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Batch-read peak vectors for the requested spectrum indices.

        Parameters
        ----------
        indices : sequence of int or None, optional
            Spectrum indices to read.  If ``None``, all spectra are returned
            in store order.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            Aligned lists of ``float64`` m/z and intensity arrays.
        """
        group = self._open_read_group()
        pg = group[self._PEAKS_GROUP]
        boundaries = self._retry_read(
            lambda: np.asarray(pg[self._BOUNDARIES_ARRAY][:], dtype=np.int64)
        )
        n_spectra = max(0, int(boundaries.shape[0]) - 1)

        if indices is None:
            index_list = list(range(n_spectra))
        else:
            index_list = [int(i) for i in indices]
            for idx in index_list:
                if idx < 0 or idx >= n_spectra:
                    raise IndexError(
                        f"Spectrum index {idx} out of range for store with "
                        f"{n_spectra} spectra."
                    )

        if not index_list:
            return [], []

        start_min = int(np.min([boundaries[i] for i in index_list]))
        end_max = int(np.max([boundaries[i + 1] for i in index_list]))

        def read_mz_span() -> np.ndarray:
            return np.asarray(pg[self._MZ_ARRAY][start_min:end_max], dtype=np.float64)

        def read_intensity_span() -> np.ndarray:
            return np.asarray(
                pg[self._INTENSITY_ARRAY][start_min:end_max], dtype=np.float64
            )

        mz_span = self._retry_read(read_mz_span)
        intensity_span = self._retry_read(read_intensity_span)

        mz_arrays: list[np.ndarray] = []
        intensity_arrays: list[np.ndarray] = []
        for idx in index_list:
            start = int(boundaries[idx]) - start_min
            end = int(boundaries[idx + 1]) - start_min
            mz_arrays.append(np.asarray(mz_span[start:end], dtype=np.float64).copy())
            intensity_arrays.append(
                np.asarray(intensity_span[start:end], dtype=np.float64).copy()
            )

        return mz_arrays, intensity_arrays

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    @property
    def n_spectra(self) -> int:
        """Number of spectra currently stored."""
        return self._n_spectra

    @property
    def store_uuid(self) -> str:
        """Unique identifier of this store, referenced by ``zarr_ref``."""
        return self._store_uuid

    @property
    def peak_chunk_size(self) -> int:
        """Configured float64-elements-per-chunk for the peak arrays."""
        return self._peak_chunk_size

    @property
    def boundary_chunk_size(self) -> int:
        """Configured spectra-per-chunk for the boundaries array."""
        return self._boundary_chunk_size

    @property
    def chunk_sizes(self) -> dict[str, int]:
        """Actual chunk sizes of the underlying arrays.

        Useful for verifying chunk-size configurability against an open
        store (arrays created with a given chunk shape keep it forever).
        """
        if self._root is None:
            return {}
        pg = self._root[self._PEAKS_GROUP]
        return {
            self._MZ_ARRAY: int(pg[self._MZ_ARRAY].chunks[0]),
            self._INTENSITY_ARRAY: int(pg[self._INTENSITY_ARRAY].chunks[0]),
            self._BOUNDARIES_ARRAY: int(pg[self._BOUNDARIES_ARRAY].chunks[0]),
        }

    @property
    def is_remote(self) -> bool:
        """Return ``True`` if the store is backed by a remote URL."""
        return self._is_remote
