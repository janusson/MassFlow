import time
import numpy as np
from matchms import Spectrum
from numba import njit


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


@njit
def prefilter_pairs(ref_mzs, ref_nls, query_mzs, query_nls, tol, min_matches):
    n_refs = len(ref_mzs)
    n_queries = len(query_mzs)

    row_idx = []
    col_idx = []

    for r in range(n_refs):
        rmz = ref_mzs[r]
        rnl = ref_nls[r]
        for q in range(n_queries):
            qmz = query_mzs[q]
            qnl = query_nls[q]

            matches = 0

            # Check mz matches
            i, j = 0, 0
            while i < len(rmz) and j < len(qmz):
                diff = rmz[i] - qmz[j]
                if abs(diff) <= tol:
                    matches += 1
                    i += 1
                    j += 1
                elif diff < 0:
                    i += 1
                else:
                    j += 1

            if matches >= min_matches:
                row_idx.append(r)
                col_idx.append(q)
                continue

            # Check neutral loss matches
            i, j = 0, 0
            while i < len(rnl) and j < len(qnl):
                diff = rnl[i] - qnl[j]
                if abs(diff) <= tol:
                    matches += 1
                    i += 1
                    j += 1
                elif diff < 0:
                    i += 1
                else:
                    j += 1

            if matches >= min_matches:
                row_idx.append(r)
                col_idx.append(q)

    return row_idx, col_idx


if __name__ == "__main__":
    refs = generate_mock_spectra(5000)
    queries = generate_mock_spectra(100)

    t0 = time.time()
    tol = 0.05
    min_matches = 3

    ref_mzs = tuple(s.peaks.mz for s in refs)
    ref_nls = tuple(s.peaks.mz - s.get("precursor_mz") for s in refs)

    query_mzs = tuple(s.peaks.mz for s in queries)
    query_nls = tuple(s.peaks.mz - s.get("precursor_mz") for s in queries)

    # Numba needs typed lists or just a loop in python if it's fast enough.
    # Numba doesn't like tuples of arrays easily if they vary in length unless using typed lists
    t1 = time.time()
    print("Setup took", t1 - t0)
