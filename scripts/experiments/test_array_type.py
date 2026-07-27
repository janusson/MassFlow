import time
from matchms import Spectrum, calculate_scores
from matchms.similarity import ModifiedCosineGreedy
import numpy as np


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


if __name__ == "__main__":
    refs = generate_mock_spectra(1000)
    queries = generate_mock_spectra(50)
    sim = ModifiedCosineGreedy(tolerance=0.05)

    t0 = time.time()
    calculate_scores(refs, queries, sim, is_symmetric=False)
    t1 = time.time()
    print(f"numpy array took: {t1 - t0:.2f}s")

    t0 = time.time()
    calculate_scores(refs, queries, sim, is_symmetric=False, array_type="sparse")
    t1 = time.time()
    print(f"sparse array took: {t1 - t0:.2f}s")
