import time
import numpy as np
from matchms import Spectrum, calculate_scores
from matchms.similarity import ModifiedCosineGreedy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


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


global_refs = None


def init_worker(refs):
    global global_refs
    global_refs = refs


def calculate_chunk(queries_chunk):
    sim = ModifiedCosineGreedy(tolerance=0.05)
    scores = calculate_scores(
        global_refs, queries_chunk, sim, is_symmetric=False, array_type="sparse"
    )
    scores_data = scores.scores
    if hasattr(scores_data, "to_array"):
        return scores_data.to_array()
    return np.asarray(scores_data)


if __name__ == "__main__":
    refs = generate_mock_spectra(5000)
    queries = generate_mock_spectra(100)

    t0 = time.time()
    cores = multiprocessing.cpu_count()
    chunk_size = max(1, len(queries) // cores)
    chunks = []
    for i in range(0, len(queries), chunk_size):
        chunks.append(queries[i : i + chunk_size])

    with ProcessPoolExecutor(
        max_workers=cores, initializer=init_worker, initargs=(refs,)
    ) as executor:
        results = list(executor.map(calculate_chunk, chunks))
    full_array = np.hstack(results)
    t1 = time.time()
    print(f"Parallel Process with init took: {t1-t0:.2f}s")
