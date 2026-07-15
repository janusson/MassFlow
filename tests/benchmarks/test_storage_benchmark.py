"""
Benchmark tests comparing SQLite BLOB vs Zarr storage backends.

These benchmarks measure read/write throughput and matrix construction
latency across a simulated dataset of 10,000 spectra. They are not run
as part of the standard test suite; use ``pytest --benchmark-enable``
or invoke individually with ``pytest tests/benchmarks/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.storage import SpectralStore, create_spectral_store

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

N_SPECTRA = 10_000
MEAN_PEAKS = 50
RANDOM_SEED = 42


def _generate_synthetic_spectra(
    n: int = N_SPECTRA,
    mean_peaks: int = MEAN_PEAKS,
    seed: int = RANDOM_SEED,
) -> list[Spectrum]:
    """Generate *n* synthetic spectra with random peak data."""
    rng = np.random.default_rng(seed)
    spectra: list[Spectrum] = []
    for i in range(n):
        n_peaks = max(1, int(rng.poisson(mean_peaks)))
        mz = np.sort(rng.uniform(50.0, 1500.0, size=n_peaks).astype(np.float64))
        intensity = rng.uniform(0.01, 1.0, size=n_peaks).astype(np.float64)
        metadata = {
            "id": f"spec_{i:06d}",
            "compound_name": f"Compound_{i}",
            "precursor_mz": float(rng.uniform(100.0, 1000.0)),
            "charge": int(rng.choice([1, 2])),
            "ionmode": rng.choice(["positive", "negative"]),
            "adduct": rng.choice(["[M+H]+", "[M-H]-", "[M+Na]+"]),
        }
        spectra.append(Spectrum(mz=mz, intensities=intensity, metadata=metadata))
    return spectra


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_spectra() -> list[Spectrum]:
    """Module-scoped fixture: generate spectra once per benchmark session."""
    return _generate_synthetic_spectra()


def _time_write(spectra: list[Spectrum], store_path: Path, backend: str) -> None:
    """Write helper: creates a fresh store each call so the benchmark is re-entrant."""
    kwargs: dict[str, Any] = {}
    if backend == "zarr":
        kwargs["overwrite"] = True
    store = create_spectral_store(store_path, backend=backend, **kwargs)
    store.add_spectra(iter(spectra), category="benchmark", batch_size=1000)
    store.close()


def _time_read_all(store: SpectralStore) -> None:
    """Read-all helper: materialises every spectrum from the store.

    Does **not** close the store — lifecycle is managed by the caller so the
    benchmark fixture can invoke this function repeatedly.
    """
    count = 0
    for _ in store.get_spectra():
        count += 1
    assert count > 0, "No spectra read"


def _time_batch_arrays(store: SpectralStore, ids: list[str]) -> None:
    """Batch array retrieval helper.

    Does **not** close the store — lifecycle is managed by the caller.
    """
    mz_list, int_list = store.batch_get_arrays(ids)
    assert len(mz_list) == len(ids)


# ---------------------------------------------------------------------------
# SQLite benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_sqlite_write_throughput(
    synthetic_spectra: list[Spectrum], tmp_path: Path, benchmark
) -> None:
    """Benchmark SQLite BLOB write throughput for 10,000 spectra."""
    db_path = tmp_path / "bench_sqlite.db"
    benchmark(_time_write, synthetic_spectra, db_path, "sqlite")


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_sqlite_read_throughput(
    synthetic_spectra: list[Spectrum], tmp_path: Path, benchmark
) -> None:
    """Benchmark SQLite BLOB read throughput (iterate all spectra)."""
    db_path = tmp_path / "bench_sqlite_read.db"
    store = create_spectral_store(db_path, backend="sqlite")
    store.add_spectra(iter(synthetic_spectra), category="benchmark")
    store.close()

    store2 = create_spectral_store(db_path, backend="sqlite")
    benchmark(_time_read_all, store2)
    store2.close()


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_sqlite_batch_array_latency(
    synthetic_spectra: list[Spectrum], tmp_path: Path, benchmark
) -> None:
    """Benchmark SQLite batch array retrieval for matrix construction."""
    db_path = tmp_path / "bench_sqlite_batch.db"
    store = create_spectral_store(db_path, backend="sqlite")
    store.add_spectra(iter(synthetic_spectra), category="benchmark")
    store.close()

    ids = [f"spec_{i:06d}" for i in range(min(1000, N_SPECTRA))]
    store2 = create_spectral_store(db_path, backend="sqlite")
    benchmark(_time_batch_arrays, store2, ids)
    store2.close()


# ---------------------------------------------------------------------------
# Zarr benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_zarr_write_throughput(
    synthetic_spectra: list[Spectrum], tmp_path: Path, benchmark
) -> None:
    """Benchmark Zarr write throughput for 10,000 spectra."""
    zarr_path = tmp_path / "bench_zarr.zarr"
    benchmark(_time_write, synthetic_spectra, zarr_path, "zarr")


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_zarr_read_throughput(
    synthetic_spectra: list[Spectrum], tmp_path: Path, benchmark
) -> None:
    """Benchmark Zarr read throughput (iterate all spectra)."""
    zarr_path = tmp_path / "bench_zarr_read.zarr"
    store = create_spectral_store(zarr_path, backend="zarr", overwrite=True)
    store.add_spectra(iter(synthetic_spectra), category="benchmark")
    store.close()

    store2 = create_spectral_store(zarr_path, backend="zarr")
    benchmark(_time_read_all, store2)
    store2.close()


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_zarr_batch_array_latency(
    synthetic_spectra: list[Spectrum], tmp_path: Path, benchmark
) -> None:
    """Benchmark Zarr batch array retrieval for matrix construction."""
    zarr_path = tmp_path / "bench_zarr_batch.zarr"
    store = create_spectral_store(zarr_path, backend="zarr", overwrite=True)
    store.add_spectra(iter(synthetic_spectra), category="benchmark")
    store.close()

    ids = [f"spec_{i:06d}" for i in range(min(1000, N_SPECTRA))]
    store2 = create_spectral_store(zarr_path, backend="zarr")
    benchmark(_time_batch_arrays, store2, ids)
    store2.close()


# ---------------------------------------------------------------------------
# Cross-backend correctness tests (not benchmarks)
# ---------------------------------------------------------------------------


def test_cross_backend_roundtrip(
    synthetic_spectra: list[Spectrum], tmp_path: Path
) -> None:
    """Verify both backends produce identical spectra after round-trip."""
    sqlite_path = tmp_path / "roundtrip_sqlite.db"
    zarr_path = tmp_path / "roundtrip_zarr.zarr"

    for backend, path in [("sqlite", sqlite_path), ("zarr", zarr_path)]:
        store = create_spectral_store(
            path, backend=backend, overwrite=(backend == "zarr")
        )
        store.add_spectra(iter(synthetic_spectra[:100]), category="test")
        store.close()

        store2 = create_spectral_store(path, backend=backend)
        retrieved = list(store2.get_spectra())
        store2.close()

        assert len(retrieved) == 100
        for orig, ret in zip(synthetic_spectra[:100], retrieved):
            assert np.allclose(orig.peaks.mz, ret.peaks.mz)
            assert np.allclose(orig.peaks.intensities, ret.peaks.intensities)
            assert ret.get("precursor_mz") is not None


def test_store_factory_unknown_backend(tmp_path: Path) -> None:
    """The factory raises ValueError for unknown backends."""
    with pytest.raises(ValueError, match="Unsupported storage backend"):
        create_spectral_store(tmp_path / "test.xyz", backend="hdf5")


def test_store_factory_sqlite_default(tmp_path: Path) -> None:
    """Default factory creates an SQLite store."""
    store = create_spectral_store(tmp_path / "default.db")
    assert "SpectralDatabase" in type(store).__name__
    store.close()


def test_store_factory_zarr(tmp_path: Path) -> None:
    """Factory creates a Zarr store when requested."""
    store = create_spectral_store(
        tmp_path / "test.zarr", backend="zarr", overwrite=True
    )
    assert "ZarrSpectralStore" in type(store).__name__
    store.close()


def test_zarr_empty_store(tmp_path: Path) -> None:
    """Zarr store reports zero counts when empty."""
    store = create_spectral_store(
        tmp_path / "empty.zarr", backend="zarr", overwrite=True
    )
    assert store.get_total_spectra_count() == 0
    assert store.get_category_counts() == {}
    assert store.get_precursor_mz_range() == (0.0, 0.0)
    assert store.get_spectrum_by_id("nonexistent") is None
    mz, inten = store.batch_get_arrays()
    assert mz == [] and inten == []
    store.close()
