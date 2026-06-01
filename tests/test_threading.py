import time
import numpy as np
from matchms import Spectrum, calculate_scores
from matchms.similarity import ModifiedCosineGreedy
from concurrent.futures import ThreadPoolExecutor


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


def calculate_chunk(args):
    refs, queries = args
    scores = calculate_scores(
        refs, queries, ModifiedCosineGreedy(tolerance=0.05), is_symmetric=False
    )
    scores_data = scores.scores
    if hasattr(scores_data, "to_array"):
        return scores_data.to_array()
    return np.asarray(scores_data)


if __name__ == "__main__":
    refs = generate_mock_spectra(1000)
    queries = generate_mock_spectra(50)

    t0 = time.time()
    calculate_chunk((refs, queries))
    t1 = time.time()
    print(f"Sequential took: {t1 - t0:.2f}s")

    t0 = time.time()
    chunks = []
    for i in range(0, len(queries), 10):
        chunks.append((refs, queries[i : i + 10]))
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(calculate_chunk, chunks))
    t1 = time.time()
    print(f"Parallel Threading took: {t1 - t0:.2f}s")
