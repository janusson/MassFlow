"""
Library preparation and the worker-owned backend model.

The multiprocessing failure of the previous design was serializing the FULL
``list[Spectrum]`` library into every ``ProcessPoolExecutor`` worker via
``initializer`` initargs: on ``spawn`` (macOS/Windows) each worker received a
private pickled copy (measured: ~116 MiB pickled + ~1.0 GiB RSS per worker at
100k spectra), and even on ``fork`` the copy-on-write pages were touched during
scoring, so RAM scaled linearly with worker count.

This module replaces that with a **worker-owned backend** model:

* ``prepare_library()`` runs once in the parent and normalizes any raw
  spectral library (mzML/mzXML/MGF/MSP) into a MassFlow store whose backend
  is selected by ``config.input.storage_backend`` (``sqlite`` default,
  ``zarr``, or ``hybrid``) — the setting has ONE meaning everywhere.  Store
  inputs (``.db``/``.sqlite``/``.zarr``) are used directly in their own
  backend.
* Only a compact, pickle-safe :class:`LibrarySpec` (a path plus two strings)
  crosses the process boundary.
* Every worker opens the store itself via :func:`open_library` and streams
  processed spectra in bounded chunks (10k spectra), so per-worker RAM is
  bounded by the chunk size, not the library size.
* The backend interface is the single :class:`MassFlow.storage.SpectralStore`
  contract: the annotation layer, CLI database commands, and streaming server
  all consume the same interface and never care whether the underlying
  library is SQLite or Zarr.
* Results are identical across backends because the store round-trips
  spectra byte-for-byte (float64 peak arrays + full metadata JSON; verified
  by ``tests/test_library.py`` and ``tests/test_storage_contract.py``) and
  decoy generation is chunk-invariant.

No distributed computing is introduced: everything stays on one machine, and
the store is a local file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import numpy as np
from matchms import Spectrum

from MassFlow import io, processing
from MassFlow.config import MassFlowConfig, ProcessingConfig
from MassFlow.storage import SpectralStore, create_spectral_store

logger = logging.getLogger(__name__)

# Extension of MassFlow-native stores (opened directly by workers).
_STORE_EXTENSIONS = {".db", ".sqlite", ".zarr"}


@dataclass(frozen=True)
class LibrarySpec:
    """Compact, pickle-safe description of a reference library.

    This is the ONLY library-related object that crosses the process
    boundary. The full spectral payload never does.
    """

    path: Path
    kind: Literal["store", "file"]
    storage_backend: Optional[str] = None
    """For ``kind == "store"``: ``sqlite``, ``zarr``, or ``hybrid``."""


class RawFileLibraryStore(SpectralStore):
    """Read-only :class:`SpectralStore` adapter over a raw open-format file.

    Implements the unified backend interface by parsing and processing the
    file on demand (mzML/mzXML/MGF/MSP).  Used only for direct API calls that
    pass a ``LibrarySpec(kind="file")``; the pipeline itself normalizes raw
    files into MassFlow stores via :func:`prepare_library`.  Writes are
    rejected explicitly — a raw file is never a write target.
    """

    def __init__(self, path: Path, processing_config: ProcessingConfig) -> None:
        self._processing_config = processing_config
        super().__init__(path)

    def _initialize(self) -> None:
        """Nothing to open: the file is parsed lazily on each read."""

    def _iter_processed(self) -> Iterator[Spectrum]:
        raw = io.load_spectra(self.store_path)
        for spectrum in processing.process_spectra(raw, self._processing_config):
            if spectrum is None:
                continue
            yield spectrum

    def add_spectra(
        self,
        spectra: Iterator[Spectrum],
        category: str = "default",
        batch_size: int = 5000,
    ) -> int:
        raise NotImplementedError(
            "Raw spectral files are read-only; build a MassFlow store first "
            "(e.g. via MassFlow.library.prepare_library or `massflow db build`)."
        )

    def get_spectra(
        self,
        category: Optional[str] = None,
        name_pattern: Optional[str] = None,
    ) -> Iterator[Spectrum]:
        regex = None
        if name_pattern is not None:
            pattern = re.escape(name_pattern).replace(r"\%", ".*").replace(r"\_", ".")
            regex = re.compile(pattern)
        for spectrum in self._iter_processed():
            if (
                category is not None
                and str(spectrum.get("category", "raw")) != category
            ):
                continue
            if regex is not None:
                name = str(spectrum.get("compound_name") or spectrum.get("name") or "")
                if not regex.match(name):
                    continue
            yield spectrum

    def get_spectrum_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        for spectrum in self._iter_processed():
            if str(spectrum.get("id", "")) == spectrum_id:
                return spectrum
        return None

    def batch_get_arrays(
        self,
        spectrum_ids: Optional[list[str]] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        spectra = list(self._iter_processed())
        if spectrum_ids is not None:
            wanted = set(spectrum_ids)
            spectra = [s for s in spectra if str(s.get("id", "")) in wanted]
        mz_arrays = [np.asarray(s.peaks.mz, dtype=np.float64) for s in spectra]
        intensity_arrays = [
            np.asarray(s.peaks.intensities, dtype=np.float64) for s in spectra
        ]
        return mz_arrays, intensity_arrays

    def get_total_spectra_count(self) -> int:
        return sum(1 for _ in self._iter_processed())

    def get_category_counts(self) -> dict[str, int]:
        return {"raw": self.get_total_spectra_count()}

    def get_precursor_mz_range(self) -> tuple[float, float]:
        values = [
            float(s.get("precursor_mz"))
            for s in self._iter_processed()
            if s.get("precursor_mz") is not None
        ]
        if not values:
            return (0.0, 0.0)
        return (min(values), max(values))

    _RAW_METADATA_FIELDS = [
        "id",
        "name",
        "precursor_mz",
        "charge",
        "ionmode",
        "adduct",
        "category",
    ]

    def metadata_query(
        self,
        fields: list[str],
        indices: Optional[np.ndarray] = None,
        category: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        for field in fields:
            if field not in self._RAW_METADATA_FIELDS:
                raise ValueError(
                    f"Unknown metadata field: '{field}'. "
                    f"Available: {self._RAW_METADATA_FIELDS}"
                )
        spectra = list(self._iter_processed())
        if category is not None:
            spectra = [s for s in spectra if str(s.get("category", "raw")) == category]
        result: dict[str, np.ndarray] = {}
        for field in fields:
            if field == "precursor_mz":
                result[field] = np.asarray(
                    [
                        float(s.get("precursor_mz"))
                        if s.get("precursor_mz") is not None
                        else np.nan
                        for s in spectra
                    ],
                    dtype=np.float64,
                )
            elif field == "charge":
                result[field] = np.asarray(
                    [int(s.get("charge") or 0) for s in spectra], dtype=np.int64
                )
            elif field == "name":
                result[field] = np.asarray(
                    [
                        str(s.get("compound_name") or s.get("name") or "")
                        for s in spectra
                    ],
                    dtype=object,
                )
            else:
                result[field] = np.asarray(
                    [str(s.get(field) or "") for s in spectra], dtype=object
                )
        if indices is not None:
            idx = np.asarray(indices, dtype=np.int64)
            result = {f: arr[idx] for f, arr in result.items()}
        return result

    def close(self) -> None:
        """Nothing to release (parses are ephemeral)."""

    def backend_provenance(self) -> dict[str, Any]:
        return {
            "backend": "raw-file",
            "path": str(self.store_path),
            "spectrum_count": self.get_total_spectra_count(),
        }


def _detect_store_backend(path: Path) -> str:
    """Detect the storage backend of a MassFlow store path."""
    if path.suffix.lower() == ".zarr" or path.is_dir():
        return "zarr"
    if path.with_suffix(".zarr").is_dir():
        return "hybrid"
    return "sqlite"


def library_spec_for_config(config: MassFlowConfig) -> LibrarySpec:
    """Build the :class:`LibrarySpec` for the configured library path.

    Store inputs (``.db``/``.sqlite``/``.zarr``) become ``kind="store"``
    specs opened in their own backend; raw open-format files become
    ``kind="file"`` specs streamed through :class:`RawFileLibraryStore`.
    """
    library_path = config.input.library_path
    if library_path is None:
        raise ValueError("Library path is not configured.")
    library_path = Path(library_path)
    if not library_path.exists():
        raise ValueError(f"Library path does not exist: {library_path}")
    if library_path.suffix.lower() in _STORE_EXTENSIONS:
        return LibrarySpec(
            path=library_path,
            kind="store",
            storage_backend=_detect_store_backend(library_path),
        )
    return LibrarySpec(path=library_path, kind="file")


def open_library(
    spec: LibrarySpec, processing_config: ProcessingConfig
) -> SpectralStore:
    """Open a :class:`MassFlow.storage.SpectralStore` for a :class:`LibrarySpec`.

    ``kind == "store"`` opens the SQLite/Zarr/hybrid store via the unified
    backend factory (workers open the library themselves); ``kind == "file"``
    returns the read-only :class:`RawFileLibraryStore` adapter.

    Returns
    -------
    SpectralStore
        The unified backend interface.  The annotation layer never needs to
        know which concrete backend it is.
    """
    if spec.kind == "store":
        backend = spec.storage_backend or _detect_store_backend(spec.path)
        return create_spectral_store(spec.path, backend=backend)
    return RawFileLibraryStore(spec.path, processing_config)


def _processing_fingerprint(processing_config: ProcessingConfig) -> str:
    """Stable fingerprint of the processing pipeline (cache invalidation)."""
    return hashlib.sha256(
        processing_config.model_dump_json().encode("utf-8")
    ).hexdigest()


def _store_is_fresh(
    store_path: Path,
    source_path: Path,
    fingerprint: str,
    backend: str,
) -> bool:
    """A cached temp store is reusable only if the source, the processing
    pipeline, AND the backend that built it are unchanged."""
    meta_path = store_path.with_suffix(store_path.suffix + ".meta.json")
    if not store_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    source_stat = source_path.stat()
    return (
        meta.get("source") == str(source_path)
        and meta.get("source_mtime_ns") == source_stat.st_mtime_ns
        and meta.get("source_size") == source_stat.st_size
        and meta.get("fingerprint") == fingerprint
        and meta.get("backend") == backend
    )


def prepare_library(
    config: MassFlowConfig, output_directory: Path
) -> tuple[LibrarySpec, int]:
    """Normalize the configured reference library into a worker-openable store.

    Runs once in the parent process:

    * Store inputs (``.db``/``.sqlite``/``.zarr``) are used directly in
      their own backend.
    * Raw open-format files are processed and written to a temporary store
      in ``output_directory`` whose backend is ``config.input.storage_backend``
      (``<stem>_library.db`` for ``sqlite``/``hybrid``, ``<stem>_library.zarr``
      for ``zarr``).  The store is reused on later runs when the source file,
      the processing pipeline, AND the backend are unchanged.

    Returns the compact :class:`LibrarySpec` handed to workers (pickle-safe,
    constant-size) and the exact processed spectrum count. Memory stays
    bounded: the conversion streams, it never materializes the library.

    Parameters
    ----------
    config : MassFlowConfig
        Full pipeline configuration (``input.library_path``,
        ``input.storage_backend``, ``processing``).
    output_directory : Path
        Directory for the temporary store (created if missing).

    Returns
    -------
    tuple[LibrarySpec, int]
        The worker-openable library spec and its processed spectrum count.

    Raises
    ------
    ValueError
        If the library path is missing or yields no valid spectra.
    """
    if not config.input.library_path:
        raise ValueError("Library path not specified in configuration.")
    library_path = Path(config.input.library_path)
    if not library_path.exists():
        raise ValueError(f"Library path does not exist: {library_path}")

    extension = library_path.suffix.lower()
    if extension in _STORE_EXTENSIONS:
        spec = LibrarySpec(
            path=library_path,
            kind="store",
            storage_backend=_detect_store_backend(library_path),
        )
        backend = open_library(spec, config.processing)
        try:
            count = backend.spectrum_count()
        finally:
            backend.close()
        return spec, count

    # Raw open-format file: build (or reuse) a temporary store in the
    # configured backend.  `storage_backend` has exactly one meaning here:
    # it selects the backend of the library store this run builds.
    backend_name = config.input.storage_backend
    if backend_name == "zarr":
        store_path = output_directory / f"{library_path.stem}_library.zarr"
    else:
        store_path = output_directory / f"{library_path.stem}_library.db"
    output_directory.mkdir(parents=True, exist_ok=True)
    fingerprint = _processing_fingerprint(config.processing)

    if not _store_is_fresh(store_path, library_path, fingerprint, backend_name):
        logger.info(
            "Building worker library store (%s): %s (source %s, %d MiB)",
            backend_name,
            store_path,
            library_path,
            library_path.stat().st_size // (1024 * 1024),
        )
        store = create_spectral_store(store_path, backend=backend_name)
        try:
            raw = io.load_spectra(library_path)
            processed = processing.process_spectra(raw, config.processing)
            count = store.add_spectra(processed, category="library")
        finally:
            store.close()
        if count == 0:
            raise ValueError("No valid spectra found in library.")
        source_stat = library_path.stat()
        meta_path = store_path.with_suffix(store_path.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "source": str(library_path),
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "source_size": source_stat.st_size,
                    "fingerprint": fingerprint,
                    "backend": backend_name,
                }
            )
        )
        logger.info("Library store built with %d spectra.", count)
    else:
        backend = open_library(
            LibrarySpec(path=store_path, kind="store", storage_backend=backend_name),
            config.processing,
        )
        try:
            count = backend.spectrum_count()
        finally:
            backend.close()
        logger.info("Reusing cached library store %s (%d spectra).", store_path, count)

    return LibrarySpec(
        path=store_path, kind="store", storage_backend=backend_name
    ), count
