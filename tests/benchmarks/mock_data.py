import random

import numpy as np
from matchms import Spectrum


def generate_mock_spectra_and_results(num_queries=1000, match_rate=0.1):
    """
    Generate synthetic spectra and a list of match result dictionaries for benchmarking.
    """
    queries = []
    results = []

    for i in range(num_queries):
        query_id = f"query_{i}"
        mz = random.uniform(100, 1000)
        rt = random.uniform(0, 1800)

        spec = Spectrum(
            mz=np.array([100.0, 200.0]),
            intensities=np.array([0.5, 1.0]),
            metadata={"id": query_id, "precursor_mz": mz, "retention_time": rt},
        )
        queries.append(spec)

        if random.random() < match_rate:
            results.append(
                {
                    "query_id": query_id,
                    "reference_id": f"ref_{random.randint(0, 5000)}",
                    "reference_name": "Mock Compound",
                    "score": random.uniform(0.6, 1.0),
                    "matched_peaks": random.randint(1, 10),
                }
            )

    return queries, results
