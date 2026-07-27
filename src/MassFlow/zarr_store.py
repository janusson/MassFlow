"""
Zarr-backed spectral storage for MassFlow.

This module implements :class:`ZarrSpectralStore`, a cloud-optimized storage
backend that persists fragment m/z and intensity vectors as compressed,
chunked Zarr arrays alongside native Zarr metadata arrays — eliminating the
``metadata_index.json`` sidecar that existed in earlier versions.

The store is designed for high-throughput 1-D slice reads: the concatenated
(flat) peak arrays are chunked along the single dimension so that reading the
peaks for a contiguous range of spectra requires touching only a few chunks,
each decompressed independently in parallel.

Metadata is stored as separate chunked Zarr arrays (one per field) within a
``metadata/`` sub-group.  Variable-length string fields use NumPy
``StringDType`` when available (NumPy ≥ 2.0), falling back to
``numcodecs.VLenUTF8``.  The free-form ``metadata`` dict attached to each
``matchms.Spectrum`` is stored as a JSON-serialized blob in a companion
string array.

Thread-safe parallel appends are coordinated through a dual-lock strategy:
an in-process ``threading.Lock`` for same-process workers and an advisory
``fcntl.flock`` on a ``.lock`` file for cross-process coordination.  Read
operations require no locking.

.. note::
    This backend is **experimental** for v1.0 and must be explicitly enabled
    via ``storage_backend: "zarr"`` in the MassFlow YAML configuration. The
    default backend remains the SQLite BLOB implementation in
    :mod:`MassFlow.database`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Iterator, Optional, TYPE_CHECKING

import numpy as np
from matchms import Spectrum

from MassFlow.storage import SpectralStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Number of float64 elements per chunk in the flat peak arrays.
# At 8 bytes per element, ~1M elements ≈ 8 MB per chunk—a sweet spot
# balancing compression ratio against read amplification for batch
# similarity workloads.
_DEFAULT_PEAK_CHUNK_SIZE: int = 1_048_576

# Spectra per chunk for the 1-D metadata arrays.
_DEFAULT_METADATA_CHUNK_SIZE: int = 4096

# Legacy sidecar filename — checked during initialisation for automatic
# migration to the native-metadata format.
_LEGACY_METADATA_INDEX = "metadata_index.json"

# Lock filename for cross-process append coordination.
_LOCK_FILENAME = ".lock"

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
    ``.zgroup`` metadata files) containing:

    ``peaks/``
        ``mz_flat`` (float64), ``intensity_flat`` (float64), and
        ``boundaries`` (int64) — the flat peak storage arrays.

    ``metadata/``
        One chunked 1-D Zarr array per metadata field:

        * ``id``, ``name``, ``ionmode``, ``adduct``, ``category`` —
          variable-length string arrays.
        * ``precursor_mz`` (float64), ``charge`` (int32) — scalar arrays.
        * ``extra_metadata`` — JSON-serialised free-form metadata blob
          (variable-length string).

    Thread-safe parallel appends use a dual-lock strategy
    (``threading.Lock`` + ``fcntl.flock``).  Reads are lock-free.

    Parameters
    ----------
    store_path : Path
        Directory path for the Zarr group.
    peak_chunk_size : int, optional
        Float64 elements per chunk in the flat peak arrays.
    metadata_chunk_size : int, optional
        Spectra per chunk in the metadata arrays.
    compressor : optional
        Zarr compressor (defaults to Blosc+zstd, clevel=3).
    overwrite : bool, optional
        If ``True``, delete any existing store before creating a fresh one.
    """

    # ------------------------------------------------------------------
    # Metadata field definitions
    # ------------------------------------------------------------------
    # Scalar (numeric) fields.
    _SCALAR_FIELDS: dict[str, np.dtype] = {
        "precursor_mz": np.dtype(np.float64),
        "charge": np.dtype(np.int32),
    }

    # String fields stored as variable-length arrays.
    _STRING_FIELDS: list[str] = [
        "id",
        "name",
        "ionmode",
        "adduct",
        "category",
        "extra_metadata",
    ]

    # Ordered list of all metadata array names (used for resize/write loops).
    _ALL_METADATA_FIELDS: list[str] = [
        *list(_SCALAR_FIELDS.keys()),
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
    ) -> None:
        self._peak_chunk_size = peak_chunk_size
        self._metadata_chunk_size = metadata_chunk_size
        self._compressor = compressor
        self._overwrite = overwrite
        self._root: Optional[Any] = None  # zarr.Group
        self._n_spectra: int = 0
        self._string_backend: str = "fixed"
        # In-process lock for coordinating appends within the same process.
        self._lock = threading.Lock()
        super().__init__(store_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Open or create the Zarr group and all sub-groups/arrays."""
        import zarr

        store_path_str = str(self.store_path)

        if self._overwrite and Path(store_path_str).exists():
            import shutil

            shutil.rmtree(store_path_str)

        self.store_path.mkdir(parents=True, exist_ok=True)
        self._root = zarr.open_group(store_path_str, mode="a")

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
                self._root.create_group(grp_name)

        # Attempt migration from legacy metadata_index.json if present.
        self._maybe_migrate_legacy_metadata()

        # Open or create metadata arrays, inferring n_spectra from shape.
        self._open_or_create_metadata_arrays()

    def close(self) -> None:
        """Release resources.  (Zarr data is already durable on disk.)"""
        self._root = None

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

    # ------------------------------------------------------------------
    # Metadata arrays — creation / open / migration
    # ------------------------------------------------------------------

    def _open_or_create_metadata_arrays(self) -> None:
        """Ensure all metadata Zarr arrays exist and determine ``_n_spectra``."""
        mg = self._metadata_group()

        # Determine current spectrum count from an existing array (all arrays
        # are always kept at the same length).  Prefer a scalar field for a
        # fast O(1) shape check.
        existing_len: int | None = None
        for field in self._ALL_METADATA_FIELDS:
            if field in mg:
                existing_len = int(mg[field].shape[0])
                break

        if existing_len is not None:
            self._n_spectra = existing_len
            # Ensure all other arrays also exist (idempotent repair).
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
        # Compression is only applied to numeric arrays; string codecs
        # handle their own compression internally (or Blosc is not
        # applicable to object/VLen arrays).
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
        legacy_path = self.store_path / _LEGACY_METADATA_INDEX
        if not legacy_path.exists():
            return

        mg = self._metadata_group()
        # Only migrate if the metadata arrays don't already exist.
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

        # Batch-write all metadata fields.
        self._write_metadata_rows(mg, 0, legacy_data)

        # Rename the legacy file so we never re-migrate.
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
            # Zarr may return a 0-d array; convert to a Python scalar.
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
    # Peak group helpers
    # ------------------------------------------------------------------

    def _ensure_peak_arrays(self, total_peaks: int) -> None:
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
        lock_path = self.store_path / _LOCK_FILENAME
        fd = _acquire_file_lock(lock_path)
        acquired = self._lock.acquire(blocking=True)
        return fd, acquired

    def _release_write_lock(self, fd: int | None, acquired: bool) -> None:
        """Release locks acquired by :meth:`_acquire_write_lock`."""
        if acquired:
            self._lock.release()
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

        Builds the new peak data and metadata entries under a dual lock,
        then atomically appends to all Zarr arrays.  Multiple workers
        (threads or processes) can safely call this concurrently.
        """
        # Materialise the iterator so we know sizes up-front.
        spectra_list: list[Spectrum] = []
        for spec in spectra:
            if spec is None:
                continue
            spectra_list.append(spec)

        if not spectra_list:
            return 0

        n_new = len(spectra_list)

        # ------------------------------------------------------------------
        # Phase 1: prepare new peak & metadata data (lock-free).
        # ------------------------------------------------------------------
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

            pmz = spec.get("precursor_mz")
            # Serialise the free-form metadata dict to JSON for storage.
            extra_meta = spec.metadata.copy()
            # Remove keys we store in dedicated fields to avoid duplication.
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
            extra_json = json.dumps(extra_meta, default=str)

            entry: dict[str, Any] = {
                "id": str(spec.get("id", "")),
                "name": str(spec.get("compound_name") or spec.get("name") or ""),
                "precursor_mz": float(pmz) if pmz is not None else np.nan,
                "charge": int(spec.get("charge", 0) or 0),
                "ionmode": str(spec.get("ionmode") or ""),
                "adduct": str(spec.get("adduct") or ""),
                "category": str(category),
                "extra_metadata": extra_json,
            }
            metadata_entries.append(entry)

        # ------------------------------------------------------------------
        # Phase 2: append under write lock (thread-/process-safe).
        # ------------------------------------------------------------------
        fd, lock_acquired = self._acquire_write_lock()
        try:
            pg = self._peaks_group()
            mg = self._metadata_group()

            # --- Peak arrays ---
            existing_peaks: int = 0
            if "mz_flat" in pg:
                existing_peaks = int(pg["mz_flat"].shape[0])
            total_peaks = existing_peaks + total_new_peaks

            if existing_peaks == 0 and "mz_flat" not in pg:
                self._ensure_peak_arrays(total_peaks)
                pg["mz_flat"][:] = new_mz_flat
                pg["intensity_flat"][:] = new_intensity_flat
            else:
                pg["mz_flat"].resize(total_peaks)
                pg["intensity_flat"].resize(total_peaks)
                pg["mz_flat"][existing_peaks:total_peaks] = new_mz_flat
                pg["intensity_flat"][existing_peaks:total_peaks] = new_intensity_flat

            # --- Boundaries array ---
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

            # --- Metadata arrays ---
            new_total = old_n_spectra + n_new
            self._ensure_metadata_arrays(new_total)
            # Resize any arrays that were created at a smaller size.
            for field in self._ALL_METADATA_FIELDS:
                arr = mg[field]
                if arr.shape[0] < new_total:
                    arr.resize(new_total)

            self._write_metadata_rows(mg, old_n_spectra, metadata_entries)
            self._n_spectra = new_total

        finally:
            self._release_write_lock(fd, lock_acquired)

        return n_new

    # ------------------------------------------------------------------
    # Spectrum retrieval
    # ------------------------------------------------------------------

    def _peak_slice(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mz, intensity) arrays for the spectrum at *index*."""
        pg = self._peaks_group()
        boundaries = pg["boundaries"][:]
        start = int(boundaries[index])
        end = int(boundaries[index + 1])
        mz = pg["mz_flat"][start:end]
        intensity = pg["intensity_flat"][start:end]
        return (
            np.asarray(mz, dtype=np.float64),
            np.asarray(intensity, dtype=np.float64),
        )

    def _row_to_spectrum(self, index: int) -> Spectrum:
        """Reconstruct a ``matchms.Spectrum`` from the store at *index*."""
        mg = self._metadata_group()
        entry = self._read_metadata_entry(mg, index)

        # Reconstruct the free-form metadata dict.
        extra_json = entry.get("extra_metadata", "{}")
        try:
            extra_meta: dict[str, Any] = json.loads(extra_json)
        except (json.JSONDecodeError, TypeError):
            extra_meta = {}

        # Merge fixed fields into the metadata dict for compatibility.
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
        mg = self._metadata_group()
        n = self._n_spectra
        if n == 0:
            return

        # Pre-fetch category and name arrays for vectorised filtering when
        # a filter is requested.
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
                    assert name_arr is not None  # implied by name_pattern check
                    if not regex.match(str(name_arr[i])):
                        continue
                yield self._row_to_spectrum(i)
        else:
            for i in range(n):
                yield self._row_to_spectrum(i)

    def get_spectrum_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """Look up a spectrum by its unique identifier."""
        mg = self._metadata_group()
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
        order.
        """
        pg = self._peaks_group()
        if "boundaries" not in pg:
            return [], []

        n = self._n_spectra
        if n == 0:
            return [], []

        if spectrum_ids is not None:
            mg = self._metadata_group()
            id_arr = mg["id"]
            # Scan IDs once.
            id_to_idx: dict[str, int] = {}
            for i in range(n):
                sid = str(id_arr[i])
                if sid:
                    id_to_idx[sid] = i
            indices = [id_to_idx[sid] for sid in spectrum_ids if sid in id_to_idx]
        else:
            indices = list(range(n))

        boundaries = pg["boundaries"][:]
        mz_flat = pg["mz_flat"]
        int_flat = pg["intensity_flat"]

        mz_arrays: list[np.ndarray] = []
        intensity_arrays: list[np.ndarray] = []
        for idx in indices:
            start = int(boundaries[idx])
            end = int(boundaries[idx + 1])
            mz_arrays.append(np.asarray(mz_flat[start:end], dtype=np.float64))
            intensity_arrays.append(np.asarray(int_flat[start:end], dtype=np.float64))

        return mz_arrays, intensity_arrays

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    def get_total_spectra_count(self) -> int:
        """Total number of spectra."""
        return self._n_spectra

    def get_category_counts(self) -> dict[str, int]:
        """Spectra count per category."""
        mg = self._metadata_group()
        cat_arr = mg["category"]
        counts: dict[str, int] = {}
        n = self._n_spectra
        # Use a simple loop — for very large stores this could be optimised
        # with numpy.unique, but the string dtype makes that tricky.
        for i in range(n):
            cat = str(cat_arr[i])
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def get_precursor_mz_range(self) -> tuple[float, float]:
        """Min and max precursor m/z."""
        if self._n_spectra == 0:
            return (0.0, 0.0)
        mg = self._metadata_group()
        pmz_arr = mg["precursor_mz"]
        # Filter out NaN values.
        valid = pmz_arr[~np.isnan(pmz_arr[:])]
        if valid.size == 0:
            return (0.0, 0.0)
        return (float(np.min(valid)), float(np.max(valid)))


__all__ = [
    "ZarrSpectralStore",
]
