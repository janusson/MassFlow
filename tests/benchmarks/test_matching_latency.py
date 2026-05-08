import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import get_similarity_engine


def generate_mock_spectra(num, mz_len=100):
    spectra = []
    for i in range(num):
        mz = np.linspace(100.0, 1000.0, mz_len) + np.random.uniform(-0.1, 0.1, mz_len)
        intensities = np.random.uniform(0.1, 1.0, mz_len)
        spec = Spectrum(
            mz=np.sort(mz),
            intensities=intensities,
            metadata={"id": f"spec_{i}", "precursor_mz": np.random.uniform(200, 800)},
        )
        spectra.append(spec)
    return spectra


@pytest.fixture(scope="session")
def large_library():
    # 5,000 spectra library
    return generate_mock_spectra(5000, mz_len=50)


@pytest.fixture(scope="session")
def queries(large_library):
    # Create 100 queries based on the first 100 library spectra, with slight noise
    query_spectra = []
    for i in range(100):
        base_spec = large_library[i]
        new_meta = base_spec.metadata.copy()
        new_meta["id"] = f"query_{i}"

        # Add tiny shift to precursor to simulate real data
        new_meta["precursor_mz"] = float(
            base_spec.get("precursor_mz")
        ) + np.random.uniform(-0.01, 0.01)

        new_spec = Spectrum(
            mz=base_spec.peaks.mz.copy(),
            intensities=base_spec.peaks.intensities.copy(),
            metadata=new_meta,
        )
        query_spectra.append(new_spec)
    return query_spectra


@pytest.mark.benchmark(group="similarity_engine")
def test_matching_latency_cosine(benchmark, large_library, queries):
    """
    Benchmark the latency of the Cosine similarity engine
    against a large library.
    """
    config = SimilarityConfig(
        algorithm="cosine",
        ms1_tolerance=0.05,
        ms2_tolerance=0.05,
        min_score=0.5,
        min_matched_peaks=3,
    )
    engine = get_similarity_engine(config)

    # We do not include decoys in the raw benchmark to purely measure matrix comp time
    result = benchmark.pedantic(
        engine.search,
        kwargs={
            "query_spectra": queries,
            "reference_spectra": large_library,
            "include_decoys": False,
        },
        rounds=3,
        iterations=1,
    )
    assert len(result) > 0


@pytest.mark.benchmark(group="similarity_engine")
def test_matching_latency_modified_cosine(benchmark, large_library, queries):
    """
    Benchmark the latency of the Modified Cosine similarity engine.
    """
    config = SimilarityConfig(
        algorithm="modified_cosine",
        ms2_tolerance=0.05,
        min_score=0.5,
        min_matched_peaks=3,
    )
    engine = get_similarity_engine(config)

    result = benchmark.pedantic(
        engine.search,
        kwargs={
            "query_spectra": queries,
            "reference_spectra": large_library,
            "include_decoys": False,
        },
        rounds=3,
        iterations=1,
    )
    assert len(result) > 0
