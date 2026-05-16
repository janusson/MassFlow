import pytest

from MassFlow.io import _build_results_dataframe

from .mock_data import generate_mock_spectra_and_results


@pytest.mark.benchmark(group="dataframe_construction")
def test_build_results_dataframe_performance(benchmark):
    """
    Benchmark the construction and labeling of the results DataFrame.
    Simulates a high-resolution dataset with 50,000 queries.
    """
    # Setup: Generate 50,000 query spectra and matches
    # This happens outside the timed loop by default in pytest-benchmark
    queries, results = generate_mock_spectra_and_results(
        num_queries=50000, match_rate=0.4
    )

    # Execute benchmark
    result_df = benchmark(
        _build_results_dataframe, results=results, query_spectra=queries
    )

    # Assertions to ensure the logic remains correct
    assert result_df is not None
    assert len(result_df) == 50000
    assert "Annotation_Status" in result_df.columns


@pytest.mark.benchmark(group="dataframe_construction")
def test_build_results_dataframe_minimal_results(benchmark):
    """
    Benchmark the construction with very few matches (sparse results).
    """
    queries, results = generate_mock_spectra_and_results(
        num_queries=50000, match_rate=0.01
    )

    result_df = benchmark(
        _build_results_dataframe, results=results, query_spectra=queries
    )

    assert result_df is not None
    assert len(result_df) == 50000
