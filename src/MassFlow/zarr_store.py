"""
Zarr-backed spectral storage for MassFlow.

This module implements :class:`ZarrSpectralStore`, a cloud-optimized storage
backend that persists fragment m/z and intensity vectors as compressed,
chunked Zarr arrays while retaining relational metadata in a lightweight
JSON index file co-located within the same store directory.

The store is designed for high-throughput 1-D slice reads: the concatenated
(flat) peak arrays are chunked along the single dimension so that reading the
peaks for a contiguous range of spectra requires touching only a few chunks,
each decompressed independently in parallel.

.. note::
    This backend is **experimental** for v1.0 and must be explicitly enabled
    via ``storage_backend: "zarr"`` in the MassFlow YAML configuration. The
    default backend remains the SQLite BLOB implementation in
    :mod:`MassFlow.database`.
"""

from __future__ import annotations

import json
import logging
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

# Filename for the JSON metadata index within the Zarr store directory.
_METADATA_INDEX_FILENAME = "metadata_index.json"


class ZarrSpectralStore(SpectralStore):
    """
    Persist mass spectra using compressed, chunked Zarr arrays.

    The on-disk layout is a Zarr group (a directory with ``.zarray`` /
    ``.zgroup`` metadata files) containing:

    - ``peaks/`` — a pair of flat float64 arrays (``mz_flat``,
      ``intensity_flat``) and an integer ``boundaries`` array that maps each
      spectrum to its slice within the flat arrays.
    - ``metadata_index.json`` — a JSON sidecar file with per-spectrum
      metadata (id, name, precursor_mz, charge, ionmode, adduct, category,
      plus the original metadata dict).

    Metadata is kept in a separate JSON file to avoid the ``numcodecs``
    dependency that would otherwise be required for zarr 3.x string/object
    arrays.

    Chunk shape is controlled via ``peak_chunk_size`` and applies to the
    single dimension of the flat peak arrays. Reads of contiguous spectra
    translate into contiguous reads of the flat arrays, which touch far fewer
    chunks than a per-spectrum layout would.

    Compression uses zarr's built-in Blosc+zstd defaults (clevel=3,
    shuffle enabled), which are applied automatically to float64 arrays.

    Parameters
    ----------
    store_path : Path
        Directory path for the Zarr group. Created if it does not exist.
    peak_chunk_size : int, optional
        Number of float64 elements per chunk in the flat peak arrays.
        Defaults to 1,048,576 (≈ 8 MB per chunk).
    overwrite : bool, optional
        If ``True``, delete any existing store at ``store_path`` before
        creating a fresh one.

    Notes
    -----
    ``ZarrSpectralStore`` is **append-only** in the current implementation.
    Spectra cannot be removed individually and spectrum IDs must be unique.
    """

    def __init__(
        self,
        store_path: Path,
        peak_chunk_size: int = _DEFAULT_PEAK_CHUNK_SIZE,
        compressor: Optional[Any] = None,
        overwrite: bool = False,
    ) -> None:
        self._peak_chunk_size = peak_chunk_size
        self._compressor = compressor
        self._overwrite = overwrite
        self._root: Optional[Any] = None  # zarr.Group
        self._metadata: list[dict[str, Any]] = []
        super().__init__(store_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Open or create the Zarr group and sub-groups."""
        import zarr

        store_path_str = str(self.store_path)

        if self._overwrite and Path(store_path_str).exists():
            import shutil

            shutil.rmtree(store_path_str)

        self.store_path.mkdir(parents=True, exist_ok=True)
        self._root = zarr.open_group(store_path_str, mode="a")

        # Set up compressor default using zarr's built-in Blosc codec
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

        # Ensure subgroups exist
        if "peaks" not in self._root:
            self._root.create_group("peaks")

        # Load existing metadata index
        self._load_metadata_index()

    def close(self) -> None:
        """Flush metadata and close the Zarr store."""
        self._save_metadata_index()
        self._root = None

    # ------------------------------------------------------------------
    # Metadata index (JSON sidecar)
    # ------------------------------------------------------------------

    def _metadata_path(self) -> Path:
        """Return the path to the JSON metadata index file."""
        return self.store_path / _METADATA_INDEX_FILENAME

    def _load_metadata_index(self) -> None:
        """Load the metadata index from disk, or start empty."""
        path = self._metadata_path()
        if path.exists():
            with open(path, "r") as fh:
                self._metadata = json.load(fh)
        else:
            self._metadata = []

    def _save_metadata_index(self) -> None:
        """Persist the metadata index to disk."""
        path = self._metadata_path()
        with open(path, "w") as fh:
            json.dump(self._metadata, fh, default=str, indent=None)

    # ------------------------------------------------------------------
    # Peak group helpers
    # ------------------------------------------------------------------

    def _peaks_group(self) -> Any:  # returns zarr.Group
        """Return the ``peaks/`` sub-group."""
        if self._root is None:
            raise RuntimeError("ZarrSpectralStore is not open.")
        return self._root["peaks"]  # type: ignore[return-value]

    def _ensure_peak_arrays(self, total_peaks: int) -> None:
        """Create the flat peak arrays and boundaries in ``peaks/``."""
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
    # Spectra insertion
    # ------------------------------------------------------------------

    def add_spectra(
        self,
        spectra: Iterator[Spectrum],
        category: str = "default",
        batch_size: int = 5000,
    ) -> int:
        """Append spectra to the Zarr store."""
        spectra_list: list[Spectrum] = []
        for spec in spectra:
            if spec is None:
                continue
            spectra_list.append(spec)

        if not spectra_list:
            return 0

        pg = self._peaks_group()

        existing_peaks: int = 0
        if "mz_flat" in pg:
            existing_peaks = int(pg["mz_flat"].shape[0])

        n_new = len(spectra_list)
        total_new_peaks = sum(int(spec.peaks.mz.size) for spec in spectra_list)
        total_peaks = existing_peaks + total_new_peaks

        # Build flat arrays and metadata
        new_boundaries = np.zeros(n_new + 1, dtype=np.int64)
        new_mz_flat = np.empty(total_new_peaks, dtype=np.float64)
        new_intensity_flat = np.empty(total_new_peaks, dtype=np.float64)

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

            # Build metadata entry
            pmz = spec.get("precursor_mz")
            entry: dict[str, Any] = {
                "id": str(spec.get("id", "")),
                "name": str(spec.get("compound_name") or spec.get("name") or ""),
                "precursor_mz": float(pmz) if pmz is not None else None,
                "charge": int(spec.get("charge", 0) or 0),
                "ionmode": str(spec.get("ionmode") or ""),
                "adduct": str(spec.get("adduct") or ""),
                "category": str(category),
                "metadata": spec.metadata.copy(),
            }
            self._metadata.append(entry)

        # Write peak arrays
        if existing_peaks == 0 and "mz_flat" not in pg:
            self._ensure_peak_arrays(total_peaks)
            pg["mz_flat"][existing_peaks:total_peaks] = new_mz_flat
            pg["intensity_flat"][existing_peaks:total_peaks] = new_intensity_flat
        else:
            pg["mz_flat"].resize(total_peaks)
            pg["intensity_flat"].resize(total_peaks)
            pg["mz_flat"][existing_peaks:total_peaks] = new_mz_flat
            pg["intensity_flat"][existing_peaks:total_peaks] = new_intensity_flat

        # Write boundaries
        if "boundaries" not in pg:
            full_boundaries = np.zeros(len(self._metadata) + 1, dtype=np.int64)
            full_boundaries[1:] = new_boundaries[1:] + existing_peaks
            pg.create_array(
                "boundaries",
                shape=(len(self._metadata) + 1,),
                chunks=(min(len(self._metadata) + 1, 4096),),
                dtype=np.int64,
            )
            pg["boundaries"][:] = full_boundaries
        else:
            old_boundaries = pg["boundaries"][:]
            last_boundary = int(old_boundaries[-1])
            extended = new_boundaries[1:] + last_boundary
            full_boundaries = np.concatenate([old_boundaries, extended])
            pg["boundaries"].resize(len(self._metadata) + 1)
            pg["boundaries"][:] = full_boundaries

        self._save_metadata_index()
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
        return np.asarray(mz, dtype=np.float64), np.asarray(intensity, dtype=np.float64)

    def _row_to_spectrum(self, index: int) -> Spectrum:
        """Reconstruct a ``matchms.Spectrum`` from the store at *index*."""
        entry = self._metadata[index]
        metadata = entry.get("metadata", {}).copy()
        if "id" not in metadata:
            metadata["id"] = entry.get("id", "")
        if "precursor_mz" not in metadata:
            metadata["precursor_mz"] = entry.get("precursor_mz")

        mz_arr, intensity_arr = self._peak_slice(index)
        return Spectrum(mz=mz_arr, intensities=intensity_arr, metadata=metadata)

    def get_spectra(
        self,
        category: Optional[str] = None,
        name_pattern: Optional[str] = None,
    ) -> Iterator[Spectrum]:
        """Stream spectra, optionally filtered by category and/or name pattern."""
        for i, entry in enumerate(self._metadata):
            if category is not None:
                if entry.get("category") != category:
                    continue
            if name_pattern is not None:
                name = str(entry.get("name", ""))
                import re

                pattern = (
                    re.escape(name_pattern).replace(r"\%", ".*").replace(r"\_", ".")
                )
                if not re.match(pattern, name):
                    continue
            yield self._row_to_spectrum(i)

    def get_spectrum_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """Look up a spectrum by its unique identifier."""
        for i, entry in enumerate(self._metadata):
            if entry.get("id") == spectrum_id:
                return self._row_to_spectrum(i)
        return None

    def batch_get_arrays(
        self,
        spectrum_ids: Optional[list[str]] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Batch-read m/z and intensity arrays for similarity matrix construction.

        When *spectrum_ids* is ``None``, all spectra are returned in store
        order. The flat-array layout with chunked reads makes this operation
        particularly efficient compared to per-row BLOB extraction.
        """
        pg = self._peaks_group()
        if "boundaries" not in pg:
            return [], []

        n = len(self._metadata)
        if n == 0:
            return [], []

        if spectrum_ids is not None:
            id_to_idx: dict[str, int] = {}
            for i, entry in enumerate(self._metadata):
                id_to_idx[str(entry.get("id", ""))] = i
            indices = [id_to_idx[sid] for sid in spectrum_ids if sid in id_to_idx]
        else:
            indices = list(range(n))

        # Pre-read all boundaries once
        boundaries = pg["boundaries"][:]
        mz_flat = pg["mz_flat"]
        int_flat = pg["intensity_flat"]

        mz_arrays = []
        intensity_arrays = []
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
        return len(self._metadata)

    def get_category_counts(self) -> dict[str, int]:
        """Spectra count per category."""
        counts: dict[str, int] = {}
        for entry in self._metadata:
            cat = str(entry.get("category", ""))
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def get_precursor_mz_range(self) -> tuple[float, float]:
        """Min and max precursor m/z."""
        if not self._metadata:
            return (0.0, 0.0)
        pmzs = [
            e["precursor_mz"]
            for e in self._metadata
            if e.get("precursor_mz") is not None
        ]
        if not pmzs:
            return (0.0, 0.0)
        return (float(min(pmzs)), float(max(pmzs)))


__all__ = [
    "ZarrSpectralStore",
]
