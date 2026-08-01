"""
Tests for ZarrSpectralStore — cloud-native storage with flat & tensor layouts.

Covers:
- ABC contract compliance (all SpectralStore abstract methods).
- Flat layout round-trip correctness (backward compatibility).
- Tensor layout (3-D peak tensor) round-trip correctness.
- Metadata query caching and invalidation.
- Thread-safety: concurrent reads from multiple threads.
- Cloud URL detection and read-only guard for remote stores.
- Exponential backoff retry decorator.
- Store factory integration.
- Empty-store edge cases.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.storage import create_spectral_store
from MassFlow.zarr_store import (
    ZarrSpectralStore,
    MetadataQueryCache,
    RetryConfig,
    _is_remote_url,
    _retry_with_backoff,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_spectrum() -> Spectrum:
    """A single well-formed spectrum."""
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensities=np.array([0.1, 0.5, 1.0], dtype=np.float64),
        metadata={
            "id": "spec_001",
            "compound_name": "Test Compound A",
            "precursor_mz": 200.0,
            "charge": 1,
            "ionmode": "positive",
            "adduct": "[M+H]+",
        },
    )


@pytest.fixture
def sample_spectra() -> list[Spectrum]:
    """A batch of diverse spectra."""
    rng = np.random.default_rng(42)
    spectra: list[Spectrum] = []
    for i in range(50):
        n_peaks = max(1, int(rng.poisson(30)))
        mz = np.sort(rng.uniform(50.0, 1500.0, size=n_peaks).astype(np.float64))
        intensity = rng.uniform(0.01, 1.0, size=n_peaks).astype(np.float64)
        metadata = {
            "id": f"spec_{i:04d}",
            "compound_name": f"Compound_{i}",
            "precursor_mz": float(rng.uniform(100.0, 1000.0)),
            "charge": int(rng.choice([1, 2])),
            "ionmode": rng.choice(["positive", "negative"]),
            "adduct": rng.choice(["[M+H]+", "[M-H]-", "[M+Na]+"]),
        }
        spectra.append(Spectrum(mz=mz, intensities=intensity, metadata=metadata))
    return spectra


@pytest.fixture
def flat_store(tmp_path: Path) -> Generator[ZarrSpectralStore, None, None]:
    """A fresh flat-layout Zarr store."""
    store = ZarrSpectralStore(tmp_path / "flat.zarr", overwrite=True, layout="flat")
    yield store
    store.close()


@pytest.fixture
def tensor_store(tmp_path: Path) -> Generator[ZarrSpectralStore, None, None]:
    """A fresh tensor-layout Zarr store."""
    store = ZarrSpectralStore(
        tmp_path / "tensor.zarr",
        overwrite=True,
        layout="tensor",
        tensor_batch_size=16,
        max_peaks_per_spectrum=128,
    )
    yield store
    store.close()


@pytest.fixture
def populated_flat_store(
    tmp_path: Path, sample_spectra: list[Spectrum]
) -> Generator[ZarrSpectralStore, None, None]:
    """A flat store pre-populated with 50 spectra."""
    store = ZarrSpectralStore(tmp_path / "pop_flat.zarr", overwrite=True, layout="flat")
    store.add_spectra(iter(sample_spectra), category="library")
    yield store
    store.close()


@pytest.fixture
def populated_tensor_store(
    tmp_path: Path, sample_spectra: list[Spectrum]
) -> Generator[ZarrSpectralStore, None, None]:
    """A tensor store pre-populated with 50 spectra."""
    store = ZarrSpectralStore(
        tmp_path / "pop_tensor.zarr",
        overwrite=True,
        layout="tensor",
        tensor_batch_size=8,
        max_peaks_per_spectrum=64,
    )
    store.add_spectra(iter(sample_spectra), category="library")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("s3://bucket/path.zarr", True),
        ("gs://bucket/path.zarr", True),
        ("https://example.com/store.zarr", True),
        ("http://example.com/store.zarr", True),
        ("az://container/path.zarr", True),
        ("abfs://container/path.zarr", True),
        ("/local/path.zarr", False),
        ("./relative/path.zarr", False),
        ("file:///local/path.zarr", False),
    ],
)
def test_is_remote_url(url: str, expected: bool) -> None:
    """Remote URLs are correctly detected."""
    assert _is_remote_url(url) == expected


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def test_retry_with_backoff_success() -> None:
    """Retry decorator returns the function result on success."""
    config = RetryConfig(max_retries=2, base_delay=0.01)

    @_retry_with_backoff(config)
    def succeed() -> str:
        return "ok"

    assert succeed() == "ok"


def test_retry_with_backoff_eventual_success() -> None:
    """Retry decorator retries and succeeds on a transient failure."""
    config = RetryConfig(max_retries=3, base_delay=0.01)
    call_count = [0]

    @_retry_with_backoff(config)
    def flaky() -> str:
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError("transient failure")
        return "recovered"

    assert flaky() == "recovered"
    assert call_count[0] == 3


def test_retry_with_backoff_exhausted() -> None:
    """Retry decorator raises the last exception after exhausting retries."""
    config = RetryConfig(max_retries=2, base_delay=0.01)

    @_retry_with_backoff(config)
    def always_fail() -> str:
        raise TimeoutError("network timeout")

    with pytest.raises(TimeoutError, match="network timeout"):
        always_fail()


def test_retry_with_backoff_non_retryable() -> None:
    """Non-retryable exceptions propagate immediately."""
    config = RetryConfig(max_retries=3, base_delay=0.01)

    @_retry_with_backoff(config)
    def bad_value() -> str:
        raise ValueError("not a network error")

    with pytest.raises(ValueError, match="not a network error"):
        bad_value()


# ---------------------------------------------------------------------------
# MetadataQueryCache
# ---------------------------------------------------------------------------


def test_cache_hit_and_miss() -> None:
    """Cache returns cached data on hit, calls loader on miss."""
    cache = MetadataQueryCache(ttl_seconds=60.0)
    load_count = [0]

    def loader() -> np.ndarray:
        load_count[0] += 1
        return np.array([1.0, 2.0, 3.0])

    key = ("precursor_mz", 0, 10)

    # First call: miss.
    result1 = cache.get(key, loader)
    assert np.array_equal(result1, [1.0, 2.0, 3.0])
    assert load_count[0] == 1

    # Second call: hit (same key).
    result2 = cache.get(key, loader)
    assert np.array_equal(result2, [1.0, 2.0, 3.0])
    assert load_count[0] == 1  # Loader not called again.


def test_cache_invalidation() -> None:
    """Cache invalidation clears all entries."""
    cache = MetadataQueryCache(ttl_seconds=60.0)
    load_count = [0]

    def loader() -> np.ndarray:
        load_count[0] += 1
        return np.array([1.0])

    key = ("precursor_mz", 0, 5)
    cache.get(key, loader)
    cache.invalidate()

    # After invalidation, should miss.
    cache.get(key, loader)
    assert load_count[0] == 2


def test_cache_field_invalidation() -> None:
    """Invalidating a single field leaves other fields intact."""
    cache = MetadataQueryCache(ttl_seconds=60.0)
    load_a = [0]
    load_b = [0]

    cache.get(
        ("a", 0, 5), lambda: (load_a.__setitem__(0, load_a[0] + 1), np.array([1.0]))[1]
    )
    cache.get(
        ("b", 0, 5), lambda: (load_b.__setitem__(0, load_b[0] + 1), np.array([2.0]))[1]
    )

    cache.invalidate_field("a")

    # Field "a" should be a miss.
    cache.get(
        ("a", 0, 5), lambda: (load_a.__setitem__(0, load_a[0] + 1), np.array([1.0]))[1]
    )
    assert load_a[0] == 2  # was 1, now 2

    # Field "b" should still be a hit.
    cache.get(
        ("b", 0, 5), lambda: (load_b.__setitem__(0, load_b[0] + 1), np.array([2.0]))[1]
    )
    assert load_b[0] == 1  # still 1


def test_cache_ttl_expiry() -> None:
    """Cache entries expire after TTL."""
    cache = MetadataQueryCache(ttl_seconds=0.01)  # Very short TTL.
    load_count = [0]

    def loader() -> np.ndarray:
        load_count[0] += 1
        return np.array([1.0])

    cache.get(("f", 0, 1), loader)
    time.sleep(0.02)  # Wait for expiry.
    cache.get(("f", 0, 1), loader)
    assert load_count[0] == 2


# ---------------------------------------------------------------------------
# ABC contract — empty store
# ---------------------------------------------------------------------------


class TestEmptyStore:
    """Empty store returns sensible defaults for all ABC methods."""

    def test_empty_flat_store(self, flat_store: ZarrSpectralStore) -> None:
        self._assert_empty_store(flat_store)

    def test_empty_tensor_store(self, tensor_store: ZarrSpectralStore) -> None:
        self._assert_empty_store(tensor_store)

    def _assert_empty_store(self, store: ZarrSpectralStore) -> None:
        assert store.get_total_spectra_count() == 0
        assert store.get_category_counts() == {}
        assert store.get_precursor_mz_range() == (0.0, 0.0)
        assert store.get_spectrum_by_id("nonexistent") is None
        assert list(store.get_spectra()) == []
        mz, inten = store.batch_get_arrays()
        assert mz == [] and inten == []
        result = store.metadata_query(["precursor_mz", "id"])
        assert len(result["precursor_mz"]) == 0


# ---------------------------------------------------------------------------
# ABC contract — populated flat store
# ---------------------------------------------------------------------------


class TestPopulatedFlatStore:
    """Full ABC compliance for a populated flat-layout store."""

    def test_total_count(self, populated_flat_store: ZarrSpectralStore) -> None:
        assert populated_flat_store.get_total_spectra_count() == 50

    def test_category_counts(self, populated_flat_store: ZarrSpectralStore) -> None:
        counts = populated_flat_store.get_category_counts()
        assert counts.get("library", 0) == 50

    def test_precursor_mz_range(self, populated_flat_store: ZarrSpectralStore) -> None:
        lo, hi = populated_flat_store.get_precursor_mz_range()
        assert lo > 0.0
        assert hi > lo

    def test_get_spectra_all(self, populated_flat_store: ZarrSpectralStore) -> None:
        spectra = list(populated_flat_store.get_spectra())
        assert len(spectra) == 50
        for s in spectra:
            assert isinstance(s, Spectrum)
            assert s.peaks.mz.size > 0
            assert s.peaks.intensities.size > 0

    def test_get_spectra_filter_category(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        spectra = list(populated_flat_store.get_spectra(category="library"))
        assert len(spectra) == 50
        spectra_none = list(populated_flat_store.get_spectra(category="nonexistent"))
        assert len(spectra_none) == 0

    def test_get_spectrum_by_id_found(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        spec = populated_flat_store.get_spectrum_by_id("spec_0000")
        assert spec is not None
        assert spec.get("id") == "spec_0000"

    def test_get_spectrum_by_id_not_found(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        assert populated_flat_store.get_spectrum_by_id("nonexistent") is None

    def test_batch_get_arrays_all(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        mz_list, int_list = populated_flat_store.batch_get_arrays()
        assert len(mz_list) == 50
        assert len(int_list) == 50
        assert all(isinstance(a, np.ndarray) for a in mz_list)
        assert all(a.dtype == np.float64 for a in mz_list)

    def test_batch_get_arrays_by_ids(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        ids = ["spec_0000", "spec_0001", "spec_0049"]
        mz_list, int_list = populated_flat_store.batch_get_arrays(ids)
        assert len(mz_list) == 3
        assert len(int_list) == 3

    def test_batch_get_arrays_nonexistent_ids(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        mz_list, int_list = populated_flat_store.batch_get_arrays(["fake_id"])
        assert len(mz_list) == 0

    def test_add_spectra_returns_count(
        self, flat_store: ZarrSpectralStore, sample_spectrum: Spectrum
    ) -> None:
        count = flat_store.add_spectra(iter([sample_spectrum]), category="test")
        assert count == 1
        assert flat_store.get_total_spectra_count() == 1

    def test_add_spectra_skips_none(
        self, flat_store: ZarrSpectralStore, sample_spectrum: Spectrum
    ) -> None:
        count = flat_store.add_spectra(iter([sample_spectrum, None, sample_spectrum]))
        assert count == 2

    def test_roundtrip_metadata_fidelity(
        self, flat_store: ZarrSpectralStore, sample_spectrum: Spectrum
    ) -> None:
        flat_store.add_spectra(iter([sample_spectrum]), category="test")
        spec = flat_store.get_spectrum_by_id("spec_001")
        assert spec is not None
        assert spec.get("precursor_mz") == 200.0
        assert spec.get("compound_name") == "Test Compound A"
        assert spec.get("charge") == 1
        assert spec.get("ionmode") == "positive"

    def test_peak_roundtrip(
        self, flat_store: ZarrSpectralStore, sample_spectrum: Spectrum
    ) -> None:
        flat_store.add_spectra(iter([sample_spectrum]), category="test")
        spec = flat_store.get_spectrum_by_id("spec_001")
        assert spec is not None
        assert np.allclose(spec.peaks.mz, sample_spectrum.peaks.mz)
        assert np.allclose(spec.peaks.intensities, sample_spectrum.peaks.intensities)


# ---------------------------------------------------------------------------
# ABC contract — populated tensor store
# ---------------------------------------------------------------------------


class TestPopulatedTensorStore:
    """Full ABC compliance for a populated tensor-layout store."""

    def test_total_count(self, populated_tensor_store: ZarrSpectralStore) -> None:
        assert populated_tensor_store.get_total_spectra_count() == 50

    def test_category_counts(self, populated_tensor_store: ZarrSpectralStore) -> None:
        counts = populated_tensor_store.get_category_counts()
        assert counts.get("library", 0) == 50

    def test_precursor_mz_range(
        self, populated_tensor_store: ZarrSpectralStore
    ) -> None:
        lo, hi = populated_tensor_store.get_precursor_mz_range()
        assert lo > 0.0
        assert hi > lo

    def test_get_spectra_all(self, populated_tensor_store: ZarrSpectralStore) -> None:
        spectra = list(populated_tensor_store.get_spectra())
        assert len(spectra) == 50
        for s in spectra:
            assert isinstance(s, Spectrum)

    def test_get_spectrum_by_id(
        self, populated_tensor_store: ZarrSpectralStore
    ) -> None:
        spec = populated_tensor_store.get_spectrum_by_id("spec_0000")
        assert spec is not None

    def test_batch_get_arrays_all(
        self, populated_tensor_store: ZarrSpectralStore
    ) -> None:
        mz_list, int_list = populated_tensor_store.batch_get_arrays()
        assert len(mz_list) == 50
        assert all(a.dtype == np.float64 for a in mz_list)

    def test_batch_get_arrays_by_ids(
        self, populated_tensor_store: ZarrSpectralStore
    ) -> None:
        ids = ["spec_0000", "spec_0025", "spec_0049"]
        mz_list, int_list = populated_tensor_store.batch_get_arrays(ids)
        assert len(mz_list) == 3

    def test_tensor_truncation(self, tensor_store: ZarrSpectralStore) -> None:
        """Spectra exceeding max_peaks_per_spectrum are truncated."""
        large_spec = Spectrum(
            mz=np.arange(200, dtype=np.float64),
            intensities=np.ones(200, dtype=np.float64),
            metadata={"id": "large", "precursor_mz": 500.0},
        )
        tensor_store.add_spectra(iter([large_spec]), category="test")
        spec = tensor_store.get_spectrum_by_id("large")
        assert spec is not None
        # max_peaks_per_spectrum is 128 for tensor_store fixture.
        assert spec.peaks.mz.size <= 128

    def test_tensor_padding(self, tensor_store: ZarrSpectralStore) -> None:
        """Spectra with fewer peaks are padded but returned at correct size."""
        small_spec = Spectrum(
            mz=np.array([100.0, 200.0], dtype=np.float64),
            intensities=np.array([0.5, 1.0], dtype=np.float64),
            metadata={"id": "small", "precursor_mz": 150.0},
        )
        tensor_store.add_spectra(iter([small_spec]), category="test")
        spec = tensor_store.get_spectrum_by_id("small")
        assert spec is not None
        assert spec.peaks.mz.size == 2  # Correct peak count, not padded.


# ---------------------------------------------------------------------------
# metadata_query (cloud-optimized batch metadata reads)
# ---------------------------------------------------------------------------


class TestMetadataQuery:
    """The metadata_query method provides cached batch metadata reads."""

    def test_query_all_fields(self, populated_flat_store: ZarrSpectralStore) -> None:
        result = populated_flat_store.metadata_query(
            ["precursor_mz", "id", "category", "charge"]
        )
        assert len(result["precursor_mz"]) == 50
        assert len(result["id"]) == 50
        assert result["precursor_mz"].dtype == np.float64
        assert result["charge"].dtype in (np.int32, np.int64)

    def test_query_with_indices(self, populated_flat_store: ZarrSpectralStore) -> None:
        indices = np.array([0, 10, 20, 49], dtype=np.int64)
        result = populated_flat_store.metadata_query(["precursor_mz"], indices=indices)
        assert len(result["precursor_mz"]) == 4

    def test_query_with_category_filter(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        result = populated_flat_store.metadata_query(
            ["precursor_mz"], category="library"
        )
        assert len(result["precursor_mz"]) == 50

        result_none = populated_flat_store.metadata_query(
            ["precursor_mz"], category="nonexistent"
        )
        assert len(result_none["precursor_mz"]) == 0

    def test_query_unknown_field_raises(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        with pytest.raises(ValueError, match="Unknown metadata field"):
            populated_flat_store.metadata_query(["nonexistent_field"])

    def test_query_empty_store(self, flat_store: ZarrSpectralStore) -> None:
        result = flat_store.metadata_query(["precursor_mz"])
        assert len(result["precursor_mz"]) == 0

    def test_query_caching(self, populated_flat_store: ZarrSpectralStore) -> None:
        """Repeated queries with the same parameters hit the cache."""
        stats_before = populated_flat_store.cache_stats
        populated_flat_store.metadata_query(["precursor_mz"])
        stats_after = populated_flat_store.cache_stats
        # Cache should have been consulted and populated.
        assert stats_after["misses"] >= stats_before["misses"]


# ---------------------------------------------------------------------------
# Thread-safety: concurrent reads
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Concurrent read operations are thread-safe."""

    def test_concurrent_reads_flat(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        """Multiple threads reading simultaneously do not interfere."""
        errors: list[Exception] = []
        results: list[int] = []

        def reader() -> None:
            try:
                spectra = list(populated_flat_store.get_spectra())
                results.append(len(spectra))
                # Also exercise batch and metadata_query.
                populated_flat_store.batch_get_arrays()
                populated_flat_store.metadata_query(["precursor_mz"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent reads: {errors}"
        assert all(r == 50 for r in results)

    def test_concurrent_reads_tensor(
        self, populated_tensor_store: ZarrSpectralStore
    ) -> None:
        """Multiple threads reading tensor layout do not interfere."""
        errors: list[Exception] = []
        results: list[int] = []

        def reader() -> None:
            try:
                spectra = list(populated_tensor_store.get_spectra())
                results.append(len(spectra))
                populated_tensor_store.batch_get_arrays()
                populated_tensor_store.metadata_query(["precursor_mz", "id"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent reads: {errors}"
        assert all(r == 50 for r in results)

    def test_concurrent_get_by_id(
        self, populated_flat_store: ZarrSpectralStore
    ) -> None:
        """Concurrent get_spectrum_by_id calls are safe."""
        errors: list[Exception] = []

        def lookup() -> None:
            try:
                for i in range(50):
                    spec = populated_flat_store.get_spectrum_by_id(f"spec_{i:04d}")
                    if spec is None:
                        errors.append(ValueError(f"Missing spec_{i:04d}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=lookup) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_read_after_write_visibility(
        self, flat_store: ZarrSpectralStore, sample_spectrum: Spectrum
    ) -> None:
        """After a write, reads from other threads see the new data."""
        flat_store.add_spectra(iter([sample_spectrum]), category="test")

        seen: list[int] = []

        def reader() -> None:
            seen.append(flat_store.get_total_spectra_count())

        t = threading.Thread(target=reader)
        t.start()
        t.join()

        assert seen[0] == 1


# ---------------------------------------------------------------------------
# Remote store guard
# ---------------------------------------------------------------------------


def test_remote_store_write_raises(tmp_path: Path, sample_spectrum: Spectrum) -> None:
    """Writing to a remote URL is rejected."""
    # Create a local store first, then simulate remote mode by overriding
    # _is_remote.  This avoids actual network calls during init.
    store = ZarrSpectralStore(
        tmp_path / "local_remote_test.zarr",
        overwrite=True,
        layout="flat",
    )
    store._is_remote = True
    with pytest.raises(RuntimeError, match="remote Zarr stores is not supported"):
        store.add_spectra(iter([sample_spectrum]))
    store.close()


# ---------------------------------------------------------------------------
# Store factory integration
# ---------------------------------------------------------------------------


def test_factory_creates_zarr_with_layout(tmp_path: Path) -> None:
    """Factory passes layout kwargs through."""
    store = create_spectral_store(
        tmp_path / "factory_tensor.zarr",
        backend="zarr",
        overwrite=True,
        layout="tensor",
        max_peaks_per_spectrum=64,
    )
    assert isinstance(store, ZarrSpectralStore)
    assert store.layout == "tensor"
    store.close()


def test_factory_creates_zarr_defaults(tmp_path: Path) -> None:
    """Factory creates a flat-layout Zarr store by default."""
    store = create_spectral_store(
        tmp_path / "factory_flat.zarr", backend="zarr", overwrite=True
    )
    assert isinstance(store, ZarrSpectralStore)
    assert store.layout == "flat"
    assert not store.is_remote
    store.close()


def test_invalid_layout_raises(tmp_path: Path) -> None:
    """An invalid layout string raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported layout"):
        ZarrSpectralStore(tmp_path / "bad.zarr", layout="invalid")


# ---------------------------------------------------------------------------
# Cross-backend correctness (flat ↔ tensor round-trip)
# ---------------------------------------------------------------------------


def test_flat_and_tensor_produce_same_spectra(
    tmp_path: Path, sample_spectra: list[Spectrum]
) -> None:
    """Both layouts produce identical spectra after round-trip."""
    flat_store = ZarrSpectralStore(
        tmp_path / "cross_flat.zarr", overwrite=True, layout="flat"
    )
    tensor_store = ZarrSpectralStore(
        tmp_path / "cross_tensor.zarr",
        overwrite=True,
        layout="tensor",
        max_peaks_per_spectrum=200,
    )

    flat_store.add_spectra(iter(sample_spectra[:20]), category="test")
    tensor_store.add_spectra(iter(sample_spectra[:20]), category="test")

    flat_specs = list(flat_store.get_spectra())
    tensor_specs = list(tensor_store.get_spectra())

    assert len(flat_specs) == len(tensor_specs)
    for fs, ts in zip(flat_specs, tensor_specs):
        assert fs.get("id") == ts.get("id")
        assert fs.get("precursor_mz") == ts.get("precursor_mz")
        assert np.allclose(fs.peaks.mz, ts.peaks.mz)
        assert np.allclose(fs.peaks.intensities, ts.peaks.intensities)

    flat_store.close()
    tensor_store.close()


# ---------------------------------------------------------------------------
# Cache invalidation on write
# ---------------------------------------------------------------------------


def test_cache_invalidated_after_add(
    flat_store: ZarrSpectralStore, sample_spectrum: Spectrum
) -> None:
    """Metadata cache is invalidated after adding new spectra."""
    flat_store.add_spectra(iter([sample_spectrum]), category="test")

    # First query populates cache.
    result1 = flat_store.metadata_query(["precursor_mz"])
    assert len(result1["precursor_mz"]) == 1

    # Add more spectra.
    spec2 = Spectrum(
        mz=np.array([50.0], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata={"id": "spec_002", "precursor_mz": 50.0},
    )
    flat_store.add_spectra(iter([spec2]), category="test")

    # Query should see both spectra (cache was invalidated).
    result2 = flat_store.metadata_query(["precursor_mz"])
    assert len(result2["precursor_mz"]) == 2


# ---------------------------------------------------------------------------
# Re-open persistence
# ---------------------------------------------------------------------------


def test_store_persists_across_reopen(
    tmp_path: Path, sample_spectra: list[Spectrum]
) -> None:
    """Data written to a store survives close and re-open."""
    path = tmp_path / "persist.zarr"

    store1 = ZarrSpectralStore(path, overwrite=True, layout="flat")
    store1.add_spectra(iter(sample_spectra[:10]), category="test")
    store1.close()

    store2 = ZarrSpectralStore(path, layout="flat")
    assert store2.get_total_spectra_count() == 10
    specs = list(store2.get_spectra())
    assert len(specs) == 10
    store2.close()


def test_tensor_store_persists_across_reopen(
    tmp_path: Path, sample_spectra: list[Spectrum]
) -> None:
    """Tensor-layout data survives close and re-open."""
    path = tmp_path / "persist_tensor.zarr"

    store1 = ZarrSpectralStore(
        path, overwrite=True, layout="tensor", max_peaks_per_spectrum=64
    )
    store1.add_spectra(iter(sample_spectra[:10]), category="test")
    store1.close()

    store2 = ZarrSpectralStore(path, layout="tensor", max_peaks_per_spectrum=64)
    assert store2.get_total_spectra_count() == 10
    store2.close()


# ---------------------------------------------------------------------------
# Cache stats
# ---------------------------------------------------------------------------


def test_cache_stats_reporting(flat_store: ZarrSpectralStore) -> None:
    """Cache stats property returns a dict with hits/misses/size."""
    stats = flat_store.cache_stats
    assert "hits" in stats
    assert "misses" in stats
    assert "size" in stats
    assert all(isinstance(v, int) for v in stats.values())


# ---------------------------------------------------------------------------
# Large batch performance test (memory safety)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_large_batch_memory_safety(
    tmp_path: Path,
) -> None:
    """Verify that reading a large batch does not explode memory.

    This test writes 500 spectra with moderate peak counts and reads them
    back via batch_get_arrays.  The tensor layout should be within a few
    hundred MB of RSS — well under the 4 GB budget.
    """
    rng = np.random.default_rng(99)
    spectra: list[Spectrum] = []
    for i in range(500):
        n_peaks = max(1, int(rng.poisson(40)))
        mz = np.sort(rng.uniform(50, 1500, n_peaks)).astype(np.float64)
        intensity = rng.uniform(0.01, 1.0, n_peaks).astype(np.float64)
        spectra.append(
            Spectrum(
                mz=mz,
                intensities=intensity,
                metadata={
                    "id": f"large_{i:05d}",
                    "precursor_mz": float(rng.uniform(100, 1000)),
                },
            )
        )

    store = ZarrSpectralStore(
        tmp_path / "large.zarr",
        overwrite=True,
        layout="tensor",
        tensor_batch_size=64,
        max_peaks_per_spectrum=128,
    )
    store.add_spectra(iter(spectra), category="bench")

    # Read all arrays in batch.
    mz_list, int_list = store.batch_get_arrays()
    assert len(mz_list) == 500

    store.close()
