import time
import numpy as np
from matchms import Spectrum
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


def compute_chunk(args):
    sim, refs, queries, idx_r, idx_c = args
    return sim.sparse_array(refs, queries, idx_r, idx_c, is_symmetric=False)


if __name__ == "__main__":
    refs = generate_mock_spectra(5000)
    queries = generate_mock_spectra(100)

    sim = ModifiedCosineGreedy(tolerance=0.05)

    t0 = time.time()
    idx_row, idx_col = np.where(np.ones((len(refs), len(queries)), dtype=bool))
    chunk_size = len(idx_row) // 8
    chunks = []
    for i in range(0, len(idx_row), chunk_size):
        chunks.append(
            (
                sim,
                refs,
                queries,
                idx_row[i : i + chunk_size],
                idx_col[i : i + chunk_size],
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(compute_chunk, chunks))

    res = np.vstack(results)
    t1 = time.time()
    print(f"ThreadPool sparse_array took: {t1-t0:.2f}s")
