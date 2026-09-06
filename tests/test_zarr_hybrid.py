"""
Tests for the Phase 1 hybrid SQLite + Zarr storage backend.

Covers:
- :class:`ZarrPeakArrayStore`: chunk-size configurability, bitwise data
  integrity, truncation, validation, and concurrent reads across threads and
  processes (lock-free multiprocessing reads).
- :class:`SpectralDatabase` hybrid mode: metadata-only SQLite schema,
  ``zarr_ref``/``zarr_index`` references, BLOB fallback for pre-migration
  rows, and error handling for mismatched/missing Zarr stores.
- :func:`migrate_blobs_to_zarr`: verified BLOB → Zarr migration,
  idempotency, orphan recovery, and the 0002 migration script wrapper.

These tests cover the experimental hybrid backend and run in the default
release suite (no special marker).
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import sqlite3
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.database import (
    SpectralDatabase,
    _ensure_zarr_columns,
    create_current_spectra_table,
    get_spectra_table_columns,
    is_current_spectra_schema,
    migrate_blobs_to_zarr,
)
from MassFlow.storage import create_spectral_store
from MassFlow.zarr_store import ZarrPeakArrayStore


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def make_sample_spectra(count: int, seed: int = 42) -> list[Spectrum]:
    """
    Build a deterministic list of spectra for round-trip tests.

    Parameters
    ----------
    count : int
        Number of spectra to generate.
    seed : int, optional
        RNG seed for reproducible arrays.

    Returns
    -------
    list[Spectrum]
        Generated spectra with explicit ``id`` and ``precursor_mz`` metadata.
    """
    rng = np.random.default_rng(seed)
    spectra: list[Spectrum] = []
    for i in range(count):
        n_peaks = max(1, int(rng.poisson(25)))
        mz = np.sort(rng.uniform(50.0, 1500.0, size=n_peaks)).astype(np.float64)
        intensity = rng.uniform(0.001, 1.0, size=n_peaks).astype(np.float64)
        spectra.append(
            Spectrum(
                mz=mz,
                intensities=intensity,
                metadata={
                    "id": f"spec_{i:04d}",
                    "compound_name": f"Compound_{i}",
                    "precursor_mz": float(rng.uniform(100.0, 1000.0)),
                    "charge": int(rng.choice([1, 2])),
                    "ionmode": str(rng.choice(["positive", "negative"])),
                    "adduct": str(rng.choice(["[M+H]+", "[M-H]-", "[M+Na]+"])),
                },
            )
        )
    return spectra


def expected_peaks(
    spectra: list[Spectrum],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract ``(mz, intensity)`` pairs from spectra as ``float64`` arrays.

    Parameters
    ----------
    spectra : list[Spectrum]
        Spectra to extract peaks from.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        ``(mz, intensity)`` pairs.
    """
    return [
        (
            np.asarray(spec.peaks.mz, dtype=np.float64),
            np.asarray(spec.peaks.intensities, dtype=np.float64),
        )
        for spec in spectra
    ]


@pytest.fixture
def sample_spectra() -> list[Spectrum]:
    """A deterministic batch of 40 spectra."""
    return make_sample_spectra(40)


@pytest.fixture
def blob_db_path(tmp_path: Path, sample_spectra: list[Spectrum]) -> Path:
    """A BLOB-mode SQLite database pre-populated with 40 spectra."""
    db_path = tmp_path / "blob.db"
    db = SpectralDatabase(db_path)
    db.add_spectra(iter(sample_spectra), category="library")
    db.close()
    return db_path


def _repo_root() -> Path:
    """Return the repository root (parent of ``tests/``)."""
    return Path(__file__).resolve().parents[1]


def _process_pool(**kwargs):
    """Build a spawn-based ProcessPoolExecutor with an importable test module."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_repo_root()) + os.pathsep + env.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]
    context = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(mp_context=context, **kwargs)


# ---------------------------------------------------------------------------
# Module-level worker functions for spawn-based multiprocessing tests.
# (Spawn re-imports this module in each worker; the functions must be
# importable by qualified name.)
# ---------------------------------------------------------------------------


def _worker_read_store(store_path: str, indices: list[int]) -> list[list[list[float]]]:
    """Read spectra from a ZarrPeakArrayStore inside a worker process."""
    store = ZarrPeakArrayStore(Path(store_path))
    try:
        mz_arrays, intensity_arrays = store.read_spectra(indices)
    finally:
        store.close()
    return [
        [mz_arr.tolist(), intensity_arr.tolist()]
        for mz_arr, intensity_arr in zip(mz_arrays, intensity_arrays)
    ]


def _worker_read_hybrid_batch(
    db_path: str,
    zarr_path: str,
    indices: list[int],
) -> list[list[list[float]]]:
    """Read spectra from a hybrid SpectralDatabase inside a worker process."""
    db = SpectralDatabase(Path(db_path), zarr_path=Path(zarr_path))
    try:
        mz_arrays, intensity_arrays = db.batch_get_arrays()
    finally:
        db.close()
    return [
        [mz_arrays[idx].tolist(), intensity_arrays[idx].tolist()] for idx in indices
    ]


def _worker_read_hybrid_by_id(
    db_path: str,
    zarr_path: str,
    spectrum_id: str,
) -> list[list[float]]:
    """Read one spectrum by id from a hybrid database inside a worker."""
    db = SpectralDatabase(Path(db_path), zarr_path=Path(zarr_path))
    try:
        spectrum = db.get_spectrum_by_id(spectrum_id)
    finally:
        db.close()
    if spectrum is None:
        return [[], []]
    return [
        np.asarray(spectrum.peaks.mz, dtype=np.float64).tolist(),
        np.asarray(spectrum.peaks.intensities, dtype=np.float64).tolist(),
    ]


# ---------------------------------------------------------------------------
# ZarrPeakArrayStore — chunk-size configurability
# ---------------------------------------------------------------------------


class TestPeakStoreChunkSizes:
    """Chunk-size configurability of the Zarr peak array store."""

    def test_small_chunk_sizes_are_applied(self, tmp_path: Path) -> None:
        """
        Configured chunk sizes are used for the underlying Zarr arrays.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = ZarrPeakArrayStore(
            tmp_path / "small_chunks.zarr",
            peak_chunk_size=7,
            boundary_chunk_size=3,
        )
        assert store.chunk_sizes == {
            "mz_flat": 7,
            "intensity_flat": 7,
            "boundaries": 3,
        }
        assert store.peak_chunk_size == 7
        assert store.boundary_chunk_size == 3
        store.close()

    def test_default_chunk_sizes(self, tmp_path: Path) -> None:
        """
        Defaults produce ~8 MB peak chunks and 4096-spectrum boundary chunks.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = ZarrPeakArrayStore(tmp_path / "default_chunks.zarr")
        assert store.peak_chunk_size == 1_048_576
        assert store.boundary_chunk_size == 4096
        store.close()

    def test_invalid_chunk_sizes_raise(self, tmp_path: Path) -> None:
        """
        Non-positive chunk sizes are rejected at construction time.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        with pytest.raises(ValueError):
            ZarrPeakArrayStore(tmp_path / "bad_peak.zarr", peak_chunk_size=0)
        with pytest.raises(ValueError):
            ZarrPeakArrayStore(tmp_path / "bad_bound.zarr", boundary_chunk_size=-1)

    def test_chunk_sizes_persist_across_reopen(self, tmp_path: Path) -> None:
        """
        Chunk shapes are stored in the array metadata and survive a reopen.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        path = tmp_path / "persist_chunks.zarr"
        store = ZarrPeakArrayStore(path, peak_chunk_size=11, boundary_chunk_size=5)
        store.close()

        reopened = ZarrPeakArrayStore(
            path, peak_chunk_size=999, boundary_chunk_size=999
        )
        assert reopened.chunk_sizes == {
            "mz_flat": 11,
            "intensity_flat": 11,
            "boundaries": 5,
        }
        reopened.close()


# ---------------------------------------------------------------------------
# ZarrPeakArrayStore — data integrity
# ---------------------------------------------------------------------------


class TestPeakStoreIntegrity:
    """Bitwise round-trip integrity of the Zarr peak array store."""

    def test_bitwise_roundtrip_spans_chunks(self, tmp_path: Path) -> None:
        """
        Random float64 vectors survive the round-trip exactly.

        The test uses small chunk sizes so the data spans many chunks and
        exercises chunk-boundary slicing.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(7)
        mz_inputs = [
            rng.uniform(-1000.0, 5000.0, size=40).astype(np.float64) for _ in range(50)
        ]
        intensity_inputs = [
            rng.uniform(0.0, 1e6, size=40).astype(np.float64) for _ in range(50)
        ]

        store = ZarrPeakArrayStore(
            tmp_path / "integrity.zarr",
            peak_chunk_size=64,
            boundary_chunk_size=7,
        )
        store.append_spectra(mz_inputs, intensity_inputs)

        mz_read, intensity_read = store.read_spectra()
        assert len(mz_read) == 50
        for index, (mz_arr, intensity_arr) in enumerate(zip(mz_read, intensity_read)):
            assert mz_arr.dtype == np.float64
            assert intensity_arr.dtype == np.float64
            assert np.array_equal(mz_arr, mz_inputs[index])
            assert np.array_equal(intensity_arr, intensity_inputs[index])
        store.close()

    def test_empty_spectra_roundtrip(self, tmp_path: Path) -> None:
        """
        Zero-peak spectra round-trip as empty float64 arrays.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = ZarrPeakArrayStore(tmp_path / "empty_spec.zarr")
        start = store.append_spectra(
            [np.array([], dtype=np.float64)],
            [np.array([], dtype=np.float64)],
        )
        assert start == 0
        mz_arr, intensity_arr = store.read_spectrum(0)
        assert mz_arr.shape == (0,)
        assert intensity_arr.shape == (0,)
        assert mz_arr.dtype == np.float64
        store.close()

    def test_append_returns_monotonic_indices(self, tmp_path: Path) -> None:
        """
        Successive appends return the starting index of each new block.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = ZarrPeakArrayStore(tmp_path / "indices.zarr")
        mz = np.array([1.0, 2.0], dtype=np.float64)
        intensity = np.array([0.5, 0.5], dtype=np.float64)
        first = store.append_spectra([mz, mz], [intensity, intensity])
        second = store.append_spectra([mz], [intensity])
        third = store.append_spectra([mz, mz, mz], [intensity, intensity, intensity])
        assert first == 0
        assert second == 2
        assert third == 3
        assert store.n_spectra == 6
        store.close()

    def test_validation_errors(self, tmp_path: Path) -> None:
        """
        Mismatched inputs and out-of-range reads raise ValueError/IndexError.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = ZarrPeakArrayStore(tmp_path / "validation.zarr")
        mz = np.array([1.0, 2.0], dtype=np.float64)
        intensity = np.array([0.5, 0.5], dtype=np.float64)

        with pytest.raises(ValueError):
            store.append_spectra([mz, mz], [intensity])
        with pytest.raises(ValueError):
            store.append_spectra([mz.reshape(1, 2)], [intensity.reshape(1, 2)])
        with pytest.raises(ValueError):
            store.append_spectra([mz], [intensity[:1]])

        store.append_spectra([mz], [intensity])
        with pytest.raises(IndexError):
            store.read_spectrum(1)
        with pytest.raises(IndexError):
            store.read_spectra([0, -1])
        store.close()

    def test_truncate_drops_trailing_spectra(self, tmp_path: Path) -> None:
        """
        Truncation shrinks the store and keeps earlier spectra intact.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = ZarrPeakArrayStore(tmp_path / "truncate.zarr")
        mz = [np.array([float(i), float(i) + 0.5]) for i in range(5)]
        intensity = [np.full(2, 0.25 + 0.1 * i) for i in range(5)]
        store.append_spectra(mz, intensity)
        assert store.n_spectra == 5

        store.truncate(2)
        assert store.n_spectra == 2
        read_mz, read_intensity = store.read_spectrum(1)
        assert np.array_equal(read_mz, mz[1])
        assert np.array_equal(read_intensity, intensity[1])
        with pytest.raises(IndexError):
            store.read_spectrum(2)

        # The store remains appendable after truncation.
        start = store.append_spectra([np.array([9.0])], [np.array([1.0])])
        assert start == 2
        assert store.n_spectra == 3
        assert np.array_equal(store.read_spectrum(2)[0], np.array([9.0]))
        store.close()

    def test_truncate_validation(self, tmp_path: Path) -> None:
        """
        Truncation to invalid sizes raises ValueError.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = ZarrPeakArrayStore(tmp_path / "truncate_val.zarr")
        store.append_spectra([np.array([1.0])], [np.array([1.0])])
        with pytest.raises(ValueError):
            store.truncate(-1)
        with pytest.raises(ValueError):
            store.truncate(2)
        store.truncate(0)
        assert store.n_spectra == 0
        store.close()


# ---------------------------------------------------------------------------
# ZarrPeakArrayStore — concurrent, lock-free reads
# ---------------------------------------------------------------------------


class TestPeakStoreConcurrency:
    """Concurrent read/write behavior of the Zarr peak array store."""

    @pytest.fixture
    def populated_store_path(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> Path:
        """A Zarr store pre-populated with the sample spectra."""
        path = tmp_path / "concurrent.zarr"
        store = ZarrPeakArrayStore(path, peak_chunk_size=256, boundary_chunk_size=16)
        peaks = expected_peaks(sample_spectra)
        store.append_spectra(
            [mz_arr for mz_arr, _ in peaks],
            [intensity_arr for _, intensity_arr in peaks],
        )
        store.close()
        return path

    def test_concurrent_thread_reads(
        self,
        populated_store_path: Path,
        sample_spectra: list[Spectrum],
    ) -> None:
        """
        Many threads reading overlapping index ranges see identical data.

        Parameters
        ----------
        populated_store_path : Path
            Path of the pre-populated Zarr store.
        sample_spectra : list[Spectrum]
            The spectra used to populate the store.

        Returns
        -------
        None
        """
        expected = expected_peaks(sample_spectra)
        store = ZarrPeakArrayStore(populated_store_path)
        errors: list[Exception] = []

        def reader(worker_id: int) -> None:
            try:
                rng = np.random.default_rng(worker_id)
                for _ in range(25):
                    indices = rng.choice(store.n_spectra, size=8, replace=False)
                    mz_arrays, intensity_arrays = store.read_spectra(list(indices))
                    for idx, mz_arr, intensity_arr in zip(
                        indices, mz_arrays, intensity_arrays
                    ):
                        exp_mz, exp_intensity = expected[int(idx)]
                        if not np.array_equal(mz_arr, exp_mz):
                            raise AssertionError(f"m/z mismatch at index {idx}")
                        if not np.array_equal(intensity_arr, exp_intensity):
                            raise AssertionError(f"intensity mismatch at index {idx}")
            except Exception as error:  # pragma: no cover - failure path
                errors.append(error)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(reader, w) for w in range(8)]
            for future in futures:
                future.result()

        store.close()
        assert not errors

    def test_concurrent_process_reads(
        self,
        populated_store_path: Path,
        sample_spectra: list[Spectrum],
    ) -> None:
        """
        Spawned worker processes read the store concurrently and correctly.

        Each worker opens its own :class:`ZarrPeakArrayStore` handle, which
        exercises the fresh-handle-per-read multiprocessing path.

        Parameters
        ----------
        populated_store_path : Path
            Path of the pre-populated Zarr store.
        sample_spectra : list[Spectrum]
            The spectra used to populate the store.

        Returns
        -------
        None
        """
        expected = expected_peaks(sample_spectra)
        n_spectra = len(sample_spectra)
        store_path_str = str(populated_store_path)

        # Four workers read overlapping, interleaved index ranges.
        slices = [
            list(range(0, n_spectra, 3)),
            list(range(1, n_spectra, 3)),
            list(range(2, n_spectra, 3)),
            list(range(0, n_spectra, 5)),
        ]
        with _process_pool(max_workers=4) as executor:
            results = list(
                executor.map(_worker_read_store, [store_path_str] * 4, slices)
            )

        for indices, worker_results in zip(slices, results):
            assert len(worker_results) == len(indices)
            for idx, worker_result in zip(indices, worker_results):
                exp_mz, exp_intensity = expected[idx]
                assert np.array_equal(
                    np.asarray(worker_result[0], dtype=np.float64), exp_mz
                )
                assert np.array_equal(
                    np.asarray(worker_result[1], dtype=np.float64), exp_intensity
                )

    def test_concurrent_thread_appends_lose_no_data(self, tmp_path: Path) -> None:
        """
        Concurrent appends from multiple threads preserve all data.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = ZarrPeakArrayStore(tmp_path / "append_threads.zarr")
        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        lock = threading.Lock()

        def appender(worker_id: int) -> None:
            block_mz = [
                np.array([float(worker_id), float(worker_id) + 1.0, 100.0 + worker_id])
            ] * 25
            block_intensity = [np.full(3, 0.5)] * 25
            start = store.append_spectra(block_mz, block_intensity)
            with lock:
                results[start] = (block_mz[0], block_intensity[0])

        threads = [
            threading.Thread(target=appender, args=(worker_id,))
            for worker_id in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert store.n_spectra == 100
        # Every appended block is contiguous and intact at its start index.
        for start, (expected_mz, expected_intensity) in results.items():
            read_mz, read_intensity = store.read_spectrum(start)
            assert np.array_equal(read_mz, expected_mz)
            assert np.array_equal(read_intensity, expected_intensity)
        store.close()

    def test_reads_during_appends_are_consistent(self, tmp_path: Path) -> None:
        """
        Readers observing committed appends always see consistent data.

        The writer publishes a watermark after each :meth:`append_spectra`
        call; readers only read below the watermark. Because appends only
        write new chunks, previously committed spectra are immutable.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = ZarrPeakArrayStore(
            tmp_path / "read_while_append.zarr", peak_chunk_size=64
        )
        watermark = 0
        watermark_lock = threading.Lock()
        stop_event = threading.Event()
        errors: list[Exception] = []

        def writer() -> None:
            nonlocal watermark
            for batch_index in range(10):
                block_mz = [
                    np.linspace(batch_index * 100.0, batch_index * 100.0 + 9.0, 10)
                    for _ in range(10)
                ]
                block_intensity = [np.full(10, 0.1 * batch_index)] * 10
                start = store.append_spectra(block_mz, block_intensity)
                with watermark_lock:
                    watermark = start + 10

        def reader(worker_id: int) -> None:
            try:
                rng = np.random.default_rng(worker_id + 1000)
                while not stop_event.is_set():
                    with watermark_lock:
                        current = watermark
                    if current == 0:
                        time.sleep(0.001)
                        continue
                    indices = list(
                        rng.choice(current, size=min(4, current), replace=False)
                    )
                    mz_arrays, intensity_arrays = store.read_spectra(indices)
                    for idx, mz_arr, intensity_arr in zip(
                        indices, mz_arrays, intensity_arrays
                    ):
                        batch_index = idx // 10
                        offset = idx % 10
                        expected_mz = batch_index * 100.0 + offset
                        if mz_arr.size != 10 or mz_arr[offset] != expected_mz:
                            raise AssertionError(
                                f"inconsistent read at index {idx}: {mz_arr}"
                            )
                        if (
                            intensity_arr.size != 10
                            or intensity_arr[0] != 0.1 * batch_index
                        ):
                            raise AssertionError(
                                f"inconsistent intensity at index {idx}"
                            )
            except Exception as error:  # pragma: no cover - failure path
                errors.append(error)
            finally:
                stop_event.set()

        writer_thread = threading.Thread(target=writer)
        reader_threads = [
            threading.Thread(target=reader, args=(worker_id,)) for worker_id in range(4)
        ]
        writer_thread.start()
        for thread in reader_threads:
            thread.start()
        writer_thread.join()
        stop_event.set()
        for thread in reader_threads:
            thread.join()

        assert not errors
        assert store.n_spectra == 100
        full_mz, full_intensity = store.read_spectra()
        assert len(full_mz) == 100
        for idx, mz_arr in enumerate(full_mz):
            batch_index = idx // 10
            offset = idx % 10
            assert mz_arr[offset] == batch_index * 100.0 + offset
            assert full_intensity[idx][0] == 0.1 * batch_index
        store.close()


# ---------------------------------------------------------------------------
# Hybrid SpectralDatabase — schema
# ---------------------------------------------------------------------------


class TestHybridSchema:
    """Schema layout of the hybrid SQLite + Zarr backend."""

    def test_fresh_hybrid_table_is_metadata_only(self, tmp_path: Path) -> None:
        """
        A fresh hybrid database has no BLOB columns, only references.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        db = SpectralDatabase(
            tmp_path / "hybrid.db", zarr_path=tmp_path / "hybrid.zarr"
        )
        assert db.conn is not None
        columns = get_spectra_table_columns(db.conn)
        assert "zarr_ref" in columns
        assert "zarr_index" in columns
        assert "mz_array" not in columns
        assert "intensity_array" not in columns
        assert is_current_spectra_schema(db.conn)
        db.close()

    def test_blob_table_is_still_current_schema(self, tmp_path: Path) -> None:
        """
        The BLOB-mode schema remains valid for the current-schema check.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        connection = sqlite3.connect(tmp_path / "blob.db")
        create_current_spectra_table(connection, include_peak_blobs=True)
        assert is_current_spectra_schema(connection)
        columns = set(get_spectra_table_columns(connection))
        assert {"mz_array", "intensity_array"}.issubset(columns)
        connection.close()

    def test_hybrid_table_variant_is_current_schema(self, tmp_path: Path) -> None:
        """
        The hybrid table variant passes the current-schema check.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        connection = sqlite3.connect(tmp_path / "hybrid.db")
        create_current_spectra_table(connection, include_peak_blobs=False)
        assert is_current_spectra_schema(connection)
        columns = set(get_spectra_table_columns(connection))
        assert {"zarr_ref", "zarr_index"}.issubset(columns)
        assert "mz_array" not in columns
        connection.close()

    def test_ensure_zarr_columns_adds_to_blob_table(self, tmp_path: Path) -> None:
        """
        Opening a BLOB table in hybrid mode adds the reference columns.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        db_path = tmp_path / "blob.db"
        db = SpectralDatabase(db_path)
        db.close()

        connection = sqlite3.connect(db_path)
        _ensure_zarr_columns(connection)
        columns = set(get_spectra_table_columns(connection))
        assert {"zarr_ref", "zarr_index"}.issubset(columns)
        assert {"mz_array", "intensity_array"}.issubset(columns)
        connection.close()

    def test_hybrid_chunk_sizes_flow_through(self, tmp_path: Path) -> None:
        """
        Chunk-size arguments reach the attached Zarr store.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        db = SpectralDatabase(
            tmp_path / "chunked.db",
            zarr_path=tmp_path / "chunked.zarr",
            peak_chunk_size=17,
            boundary_chunk_size=5,
        )
        assert db._zarr_arrays is not None
        assert db._zarr_arrays.chunk_sizes == {
            "mz_flat": 17,
            "intensity_flat": 17,
            "boundaries": 5,
        }
        db.close()


# ---------------------------------------------------------------------------
# Hybrid SpectralDatabase — data integrity & queries
# ---------------------------------------------------------------------------


class TestHybridRoundtrip:
    """Round-trip fidelity of the hybrid backend."""

    @pytest.fixture
    def hybrid_db(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> Generator[SpectralDatabase, None, None]:
        """A hybrid database pre-populated with the sample spectra."""
        db = SpectralDatabase(
            tmp_path / "roundtrip.db",
            zarr_path=tmp_path / "roundtrip.zarr",
            peak_chunk_size=256,
            boundary_chunk_size=16,
        )
        db.add_spectra(iter(sample_spectra), category="library")
        yield db
        db.close()

    def test_spectra_roundtrip_bitwise(
        self, hybrid_db: SpectralDatabase, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Streamed spectra reconstruct the exact float64 peaks and metadata.

        Parameters
        ----------
        hybrid_db : SpectralDatabase
            Pre-populated hybrid database fixture.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        expected = expected_peaks(sample_spectra)
        read_back = list(hybrid_db.get_spectra())
        assert len(read_back) == len(sample_spectra)
        for index, spectrum in enumerate(read_back):
            exp_mz, exp_intensity = expected[index]
            read_mz = np.asarray(spectrum.peaks.mz, dtype=np.float64)
            read_intensity = np.asarray(spectrum.peaks.intensities, dtype=np.float64)
            assert np.array_equal(read_mz, exp_mz)
            assert np.array_equal(read_intensity, exp_intensity)
            assert spectrum.get("id") == sample_spectra[index].get("id")
            assert spectrum.get("compound_name") == sample_spectra[index].get(
                "compound_name"
            )
            assert np.isclose(
                float(spectrum.get("precursor_mz")),
                float(sample_spectra[index].get("precursor_mz")),
            )

    def test_batch_get_arrays(
        self, hybrid_db: SpectralDatabase, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Batch array reads match the original data bitwise.

        Parameters
        ----------
        hybrid_db : SpectralDatabase
            Pre-populated hybrid database fixture.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        expected = expected_peaks(sample_spectra)
        mz_arrays, intensity_arrays = hybrid_db.batch_get_arrays()
        assert len(mz_arrays) == len(sample_spectra)
        for index, (mz_arr, intensity_arr) in enumerate(
            zip(mz_arrays, intensity_arrays)
        ):
            exp_mz, exp_intensity = expected[index]
            assert mz_arr.dtype == np.float64
            assert np.array_equal(mz_arr, exp_mz)
            assert np.array_equal(intensity_arr, exp_intensity)

        # Subset query by id.
        ids = ["spec_0005", "spec_0001", "spec_0017"]
        subset_mz, subset_intensity = hybrid_db.batch_get_arrays(spectrum_ids=ids)
        assert len(subset_mz) == 3
        for requested_id, mz_arr in zip(ids, subset_mz):
            idx = int(requested_id.split("_")[1])
            exp_mz, _ = expected[idx]
            assert np.array_equal(mz_arr, exp_mz)

    def test_metadata_queries(
        self, hybrid_db: SpectralDatabase, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Aggregate queries operate on SQLite metadata alone.

        Parameters
        ----------
        hybrid_db : SpectralDatabase
            Pre-populated hybrid database fixture.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        assert hybrid_db.get_total_spectra_count() == len(sample_spectra)
        assert hybrid_db.get_category_counts() == {"library": len(sample_spectra)}
        precursor_values = [float(spec.get("precursor_mz")) for spec in sample_spectra]
        mz_min, mz_max = hybrid_db.get_precursor_mz_range()
        assert np.isclose(mz_min, min(precursor_values))
        assert np.isclose(mz_max, max(precursor_values))

    def test_get_spectrum_by_id(
        self, hybrid_db: SpectralDatabase, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Single-spectrum lookup by id returns the correct spectrum.

        Parameters
        ----------
        hybrid_db : SpectralDatabase
            Pre-populated hybrid database fixture.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        found = hybrid_db.get_spectrum_by_id("spec_0003")
        assert found is not None
        exp_mz, exp_intensity = expected_peaks(sample_spectra)[3]
        assert np.array_equal(np.asarray(found.peaks.mz, dtype=np.float64), exp_mz)
        assert np.array_equal(
            np.asarray(found.peaks.intensities, dtype=np.float64), exp_intensity
        )
        assert hybrid_db.get_spectrum_by_id("does_not_exist") is None

    def test_reopen_persistence(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Data persists across close/reopen of a hybrid database.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        db_path = tmp_path / "persist.db"
        zarr_path = tmp_path / "persist.zarr"
        db = SpectralDatabase(db_path, zarr_path=zarr_path)
        db.add_spectra(iter(sample_spectra), category="library")
        expected = expected_peaks(sample_spectra)
        ref = db.zarr_ref
        db.close()

        reopened = SpectralDatabase(db_path, zarr_path=zarr_path)
        assert reopened.zarr_ref == ref
        mz_arrays, intensity_arrays = reopened.batch_get_arrays()
        assert len(mz_arrays) == len(sample_spectra)
        for index, (mz_arr, intensity_arr) in enumerate(
            zip(mz_arrays, intensity_arrays)
        ):
            exp_mz, exp_intensity = expected[index]
            assert np.array_equal(mz_arr, exp_mz)
            assert np.array_equal(intensity_arr, exp_intensity)
        reopened.close()


# ---------------------------------------------------------------------------
# Hybrid SpectralDatabase — BLOB fallback & error handling
# ---------------------------------------------------------------------------


class TestHybridFallbacks:
    """Interoperation between BLOB rows and Zarr-referenced rows."""

    def test_blob_rows_readable_after_hybrid_reopen(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Pre-migration BLOB rows fall back to the BLOB read path.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        db_path = tmp_path / "mixed.db"
        zarr_path = tmp_path / "mixed.zarr"

        # Phase 1: BLOB-mode rows.
        db = SpectralDatabase(db_path)
        db.add_spectra(iter(sample_spectra[:10]), category="blob")
        db.close()

        # Phase 2: reopen in hybrid mode and add Zarr-backed rows.
        hybrid = SpectralDatabase(db_path, zarr_path=zarr_path)
        hybrid.add_spectra(iter(sample_spectra[10:20]), category="zarr")
        expected = expected_peaks(sample_spectra[:20])

        mz_arrays, intensity_arrays = hybrid.batch_get_arrays()
        assert len(mz_arrays) == 20
        for index, (mz_arr, intensity_arr) in enumerate(
            zip(mz_arrays, intensity_arrays)
        ):
            exp_mz, exp_intensity = expected[index]
            assert np.array_equal(mz_arr, exp_mz)
            assert np.array_equal(intensity_arr, exp_intensity)
        hybrid.close()

    def test_blob_mode_reading_zarr_row_raises(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        A BLOB-mode instance cannot resolve Zarr-referenced rows.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        db_path = tmp_path / "zarr_only.db"
        zarr_path = tmp_path / "zarr_only.zarr"
        db = SpectralDatabase(db_path, zarr_path=zarr_path)
        db.add_spectra(iter(sample_spectra[:5]), category="library")
        db.close()

        blob_mode = SpectralDatabase(db_path)
        with pytest.raises(RuntimeError, match="zarr_path"):
            list(blob_mode.get_spectra())
        blob_mode.close()

    def test_zarr_store_uuid_mismatch_raises(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Attaching a different Zarr store than the referenced one raises.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        db_path = tmp_path / "mismatch.db"
        zarr_path_a = tmp_path / "store_a.zarr"
        zarr_path_b = tmp_path / "store_b.zarr"

        db = SpectralDatabase(db_path, zarr_path=zarr_path_a)
        db.add_spectra(iter(sample_spectra[:5]), category="library")
        db.close()

        # Create an unrelated store with its own UUID.
        other = ZarrPeakArrayStore(zarr_path_b)
        other.close()

        with pytest.raises(RuntimeError, match="identity mismatch"):
            SpectralDatabase(db_path, zarr_path=zarr_path_b)

    def test_missing_arrays_raise(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Referenced indices beyond the array store's coverage raise.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        db_path = tmp_path / "missing.db"
        zarr_path = tmp_path / "missing.zarr"
        db = SpectralDatabase(db_path, zarr_path=zarr_path)
        db.add_spectra(iter(sample_spectra[:5]), category="library")
        ref = db.zarr_ref
        db.close()

        # Manually insert a row that references an out-of-range index.
        connection = sqlite3.connect(db_path)
        connection.execute(
            "INSERT INTO spectra (original_id, name, category, metadata, "
            "zarr_ref, zarr_index, triage_flags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ghost", "Ghost", "library", "{}", ref, 999, "{}"),
        )
        connection.commit()
        connection.close()

        with pytest.raises(RuntimeError, match="missing data"):
            SpectralDatabase(db_path, zarr_path=zarr_path)


# ---------------------------------------------------------------------------
# Hybrid SpectralDatabase — factory & merge guards
# ---------------------------------------------------------------------------


class TestHybridIntegration:
    """Factory wiring and merge safety for the hybrid backend."""

    def test_factory_creates_hybrid(self, tmp_path: Path) -> None:
        """
        The factory resolves the 'hybrid' backend to a hybrid database.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = create_spectral_store(tmp_path / "factory.db", backend="hybrid")
        assert isinstance(store, SpectralDatabase)
        assert store.zarr_path == tmp_path / "factory.zarr"
        store.close()

    def test_factory_aliases_sqlite_zarr(self, tmp_path: Path) -> None:
        """
        The 'sqlite-zarr' alias resolves to the same hybrid backend.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        store = create_spectral_store(tmp_path / "alias.db", backend="sqlite-zarr")
        assert isinstance(store, SpectralDatabase)
        assert store.zarr_path is not None
        store.close()

    def test_merge_from_sqlite_hybrid_target_raises(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        The bulk SQL merge is refused when the target is a hybrid database.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the source database.

        Returns
        -------
        None
        """
        source_path = tmp_path / "source.db"
        source = SpectralDatabase(source_path)
        source.add_spectra(iter(sample_spectra[:5]), category="source")
        source.close()

        target = SpectralDatabase(
            tmp_path / "target.db", zarr_path=tmp_path / "target.zarr"
        )
        with pytest.raises(RuntimeError, match="iterator-based"):
            target.merge_from_sqlite(source_path)
        target.close()

    def test_merge_from_sqlite_hybrid_source_raises(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        A hybrid source cannot be bulk-copied into a BLOB target.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the source database.

        Returns
        -------
        None
        """
        source_path = tmp_path / "source_hybrid.db"
        source = SpectralDatabase(
            source_path, zarr_path=tmp_path / "source_hybrid.zarr"
        )
        source.add_spectra(iter(sample_spectra[:5]), category="source")
        source.close()

        target = SpectralDatabase(tmp_path / "target_blob.db")
        with pytest.raises(RuntimeError, match="iterator-based"):
            target.merge_from_sqlite(source_path)
        target.close()

    def test_iterator_merge_into_hybrid_works(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        The iterator-based cross-backend merge works for hybrid targets.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the source database.

        Returns
        -------
        None
        """
        source_path = tmp_path / "iter_source.db"
        source = SpectralDatabase(source_path)
        source.add_spectra(iter(sample_spectra[:5]), category="source")
        source.close()

        target = SpectralDatabase(
            tmp_path / "iter_target.db", zarr_path=tmp_path / "iter_target.zarr"
        )
        try:
            in_store = SpectralDatabase(source_path)
            try:
                added = target.add_spectra(in_store.get_spectra(), category="merged")
            finally:
                in_store.close()
        finally:
            target.close()
        assert added == 5


# ---------------------------------------------------------------------------
# Hybrid SpectralDatabase — multiprocessing concurrent reads
# ---------------------------------------------------------------------------


class TestHybridMultiprocessing:
    """Lock-free concurrent reads from spawned worker processes."""

    def test_concurrent_process_batch_reads(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Worker processes read the hybrid database concurrently and correctly.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        db_path = tmp_path / "mp.db"
        zarr_path = tmp_path / "mp.zarr"
        db = SpectralDatabase(db_path, zarr_path=zarr_path, peak_chunk_size=256)
        db.add_spectra(iter(sample_spectra), category="library")
        expected = expected_peaks(sample_spectra)
        db.close()

        n_spectra = len(sample_spectra)
        slices = [
            list(range(0, n_spectra, 2)),
            list(range(1, n_spectra, 2)),
            list(range(0, n_spectra, 4)),
        ]
        with _process_pool(max_workers=3) as executor:
            results = list(
                executor.map(
                    _worker_read_hybrid_batch,
                    [str(db_path)] * 3,
                    [str(zarr_path)] * 3,
                    slices,
                )
            )

        for indices, worker_results in zip(slices, results):
            assert len(worker_results) == len(indices)
            for idx, worker_result in zip(indices, worker_results):
                exp_mz, exp_intensity = expected[idx]
                assert np.array_equal(
                    np.asarray(worker_result[0], dtype=np.float64), exp_mz
                )
                assert np.array_equal(
                    np.asarray(worker_result[1], dtype=np.float64), exp_intensity
                )

    def test_concurrent_process_by_id_reads(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Worker processes resolving spectra by id see identical data.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        db_path = tmp_path / "mp_id.db"
        zarr_path = tmp_path / "mp_id.zarr"
        db = SpectralDatabase(db_path, zarr_path=zarr_path)
        db.add_spectra(iter(sample_spectra), category="library")
        expected = expected_peaks(sample_spectra)
        db.close()

        spectrum_ids = [f"spec_{i:04d}" for i in range(0, len(sample_spectra), 5)]
        with _process_pool(max_workers=4) as executor:
            results = list(
                executor.map(
                    _worker_read_hybrid_by_id,
                    [str(db_path)] * len(spectrum_ids),
                    [str(zarr_path)] * len(spectrum_ids),
                    spectrum_ids,
                )
            )

        for spectrum_id, worker_result in zip(spectrum_ids, results):
            idx = int(spectrum_id.split("_")[1])
            exp_mz, exp_intensity = expected[idx]
            assert np.array_equal(
                np.asarray(worker_result[0], dtype=np.float64), exp_mz
            )
            assert np.array_equal(
                np.asarray(worker_result[1], dtype=np.float64), exp_intensity
            )


# ---------------------------------------------------------------------------
# migrate_blobs_to_zarr
# ---------------------------------------------------------------------------


class TestMigrateBlobsToZarr:
    """Verified BLOB → Zarr migration."""

    def test_migration_preserves_data_bitwise(
        self, blob_db_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Migrated rows read back bitwise-identical through the hybrid path.

        Parameters
        ----------
        blob_db_path : Path
            Pre-populated BLOB-mode database fixture.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        expected = expected_peaks(sample_spectra)
        summary = migrate_blobs_to_zarr(
            blob_db_path,
            peak_chunk_size=128,
            boundary_chunk_size=16,
        )
        assert summary["status"] == "migrated"
        assert summary["migrated_row_count"] == len(sample_spectra)
        assert summary["blobs_nulled"] is True
        assert isinstance(summary["zarr_ref"], str) and summary["zarr_ref"]

        hybrid = SpectralDatabase(
            blob_db_path, zarr_path=blob_db_path.with_suffix(".zarr")
        )
        assert hybrid.zarr_ref == summary["zarr_ref"]
        mz_arrays, intensity_arrays = hybrid.batch_get_arrays()
        assert len(mz_arrays) == len(expected)
        for index, (mz_arr, intensity_arr) in enumerate(
            zip(mz_arrays, intensity_arrays)
        ):
            exp_mz, exp_intensity = expected[index]
            assert np.array_equal(mz_arr, exp_mz)
            assert np.array_equal(intensity_arr, exp_intensity)
        hybrid.close()

        # BLOB columns were NULLed: SQLite retains metadata + reference only.
        connection = sqlite3.connect(blob_db_path)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT mz_array, intensity_array, zarr_ref, zarr_index FROM spectra"
        ).fetchall()
        assert len(rows) == len(sample_spectra)
        for row in rows:
            assert row["mz_array"] is None
            assert row["intensity_array"] is None
            assert row["zarr_ref"] == summary["zarr_ref"]
            assert row["zarr_index"] is not None
        connection.close()

    def test_migration_is_idempotent(self, blob_db_path: Path) -> None:
        """
        Re-running the migration on a migrated database is a no-op.

        Parameters
        ----------
        blob_db_path : Path
            Pre-populated BLOB-mode database fixture.

        Returns
        -------
        None
        """
        first = migrate_blobs_to_zarr(blob_db_path)
        assert first["status"] == "migrated"

        second = migrate_blobs_to_zarr(blob_db_path)
        assert second["status"] == "already_migrated"
        assert second["migrated_row_count"] == 0

    def test_migration_keep_blobs(
        self, blob_db_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        ``null_blobs=False`` keeps BLOB copies alongside the references.

        Parameters
        ----------
        blob_db_path : Path
            Pre-populated BLOB-mode database fixture.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        summary = migrate_blobs_to_zarr(blob_db_path, null_blobs=False)
        assert summary["status"] == "migrated"
        assert summary["blobs_nulled"] is False

        connection = sqlite3.connect(blob_db_path)
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT mz_array, zarr_index FROM spectra ORDER BY id LIMIT 1"
        ).fetchone()
        assert row["mz_array"] is not None
        assert row["zarr_index"] is not None
        connection.close()

    def test_migration_custom_chunk_sizes(
        self, blob_db_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Migration honors custom chunk sizes for the Zarr arrays.

        Parameters
        ----------
        blob_db_path : Path
            Pre-populated BLOB-mode database fixture.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        migrate_blobs_to_zarr(
            blob_db_path,
            peak_chunk_size=31,
            boundary_chunk_size=7,
        )
        store = ZarrPeakArrayStore(blob_db_path.with_suffix(".zarr"))
        assert store.chunk_sizes == {
            "mz_flat": 31,
            "intensity_flat": 31,
            "boundaries": 7,
        }
        assert store.n_spectra == len(sample_spectra)
        store.close()

    def test_migration_overwrite_rebuilds_store(
        self, blob_db_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        ``overwrite=True`` rebuilds the array store from intact BLOBs.

        Parameters
        ----------
        blob_db_path : Path
            Pre-populated BLOB-mode database fixture.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        first = migrate_blobs_to_zarr(blob_db_path, null_blobs=False)
        rebuilt = migrate_blobs_to_zarr(
            blob_db_path,
            null_blobs=False,
            overwrite=True,
        )
        assert rebuilt["status"] == "migrated"
        assert rebuilt["migrated_row_count"] == len(sample_spectra)
        assert rebuilt["zarr_ref"] != first["zarr_ref"]

        hybrid = SpectralDatabase(
            blob_db_path, zarr_path=blob_db_path.with_suffix(".zarr")
        )
        assert hybrid.zarr_ref == rebuilt["zarr_ref"]
        assert hybrid.get_total_spectra_count() == len(sample_spectra)
        hybrid.close()

    def test_migration_orphan_recovery(
        self, blob_db_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        Orphaned arrays from an interrupted run are truncated on re-run.

        Parameters
        ----------
        blob_db_path : Path
            Pre-populated BLOB-mode database fixture.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        migrate_blobs_to_zarr(blob_db_path)

        # Simulate an interrupted run: extra arrays appended to the store
        # that no SQLite row references.
        zarr_path = blob_db_path.with_suffix(".zarr")
        store = ZarrPeakArrayStore(zarr_path)
        store.append_spectra(
            [np.array([1.0, 2.0, 3.0])] * 3,
            [np.array([0.5, 0.5, 0.5])] * 3,
        )
        store.close()

        rerun = migrate_blobs_to_zarr(blob_db_path)
        assert rerun["status"] == "already_migrated"

        store_after = ZarrPeakArrayStore(zarr_path)
        assert store_after.n_spectra == len(sample_spectra)
        store_after.close()

    def test_migration_legacy_database_raises(self, tmp_path: Path) -> None:
        """
        Legacy ``peaks``-schema databases must run migration 0001 first.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.

        Returns
        -------
        None
        """
        legacy_path = tmp_path / "legacy.db"
        connection = sqlite3.connect(legacy_path)
        connection.execute(
            "CREATE TABLE spectra (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT, peaks TEXT)"
        )
        connection.execute(
            "INSERT INTO spectra (name, peaks) VALUES (?, ?)",
            ("legacy", "[[100.0, 1.0]]"),
        )
        connection.commit()
        connection.close()

        with pytest.raises(RuntimeError, match="0001"):
            migrate_blobs_to_zarr(legacy_path)

    def test_migration_script_main(
        self, tmp_path: Path, sample_spectra: list[Spectrum]
    ) -> None:
        """
        The 0002 migration script migrates a database and exits cleanly.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory provided by pytest.
        sample_spectra : list[Spectrum]
            The spectra used to populate the database.

        Returns
        -------
        None
        """
        db_path = tmp_path / "script.db"
        db = SpectralDatabase(db_path)
        db.add_spectra(iter(sample_spectra), category="library")
        db.close()

        script_path = _repo_root() / "scripts" / "migrations" / "0002_blobs_to_zarr.py"
        spec = importlib.util.spec_from_file_location(
            "massflow_migration_0002", script_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        zarr_output = tmp_path / "script.zarr"
        exit_code = module.main(
            ["--input", str(db_path), "--zarr-output", str(zarr_output)]
        )
        assert exit_code == 0

        hybrid = SpectralDatabase(db_path, zarr_path=zarr_output)
        assert hybrid.get_total_spectra_count() == len(sample_spectra)
        hybrid.close()

        # Missing input yields exit code 2.
        exit_code_missing = module.main(
            ["--input", str(tmp_path / "does_not_exist.db")]
        )
        assert exit_code_missing == 2
