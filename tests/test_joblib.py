import time
import numpy as np
from matchms import Spectrum, calculate_scores
from matchms.similarity import ModifiedCosineGreedy
from joblib import Parallel, delayed


def generate_mock_spectra(num, mz_len=50):
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


def calculate_chunk(refs, queries_chunk):
    sim = ModifiedCosineGreedy(tolerance=0.05)
    scores = calculate_scores(refs, queries_chunk, sim, is_symmetric=False)
    scores_data = scores.scores
    if hasattr(scores_data, "to_array"):
        return scores_data.to_array()
    return np.asarray(scores_data)


if __name__ == "__main__":
    refs = generate_mock_spectra(5000)
    queries = generate_mock_spectra(100)

    t0 = time.time()
    n_jobs = 8
    chunk_size = max(1, len(queries) // n_jobs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_chunk)(refs, queries[i : i + chunk_size])
        for i in range(0, len(queries), chunk_size)
    )
    res = np.hstack(results)
    t1 = time.time()
    print(f"joblib Parallel took: {t1-t0:.2f}s")
