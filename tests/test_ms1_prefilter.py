import time
import numpy as np


def ms1_prefilter_old(ref_mzs, query_mzs, ms1_tolerance):
    diff = np.abs(ref_mzs[:, None] - query_mzs[None, :])
    mask = diff <= ms1_tolerance
    return np.where(mask)


def ms1_prefilter_new(ref_mzs, query_mzs, ms1_tolerance):
    query_mzs_indexed = list(enumerate(query_mzs))
    ref_mzs_sorted_indices = np.argsort(ref_mzs)
    ref_mzs_sorted = ref_mzs[ref_mzs_sorted_indices]

    rows, cols = [], []
    for query_idx, query_mz in query_mzs_indexed:
        min_mz, max_mz = query_mz - ms1_tolerance, query_mz + ms1_tolerance

        start_idx = np.searchsorted(ref_mzs_sorted, min_mz, side="left")
        end_idx = np.searchsorted(ref_mzs_sorted, max_mz, side="right")

        original_indices = ref_mzs_sorted_indices[start_idx:end_idx]
        for ref_idx in original_indices:
            rows.append(ref_idx)
            cols.append(query_idx)

    return np.array(rows), np.array(cols)


ref_mzs = np.random.uniform(100, 1000, 10000)
query_mzs = np.random.uniform(100, 1000, 50000)

t0 = time.time()
ms1_prefilter_old(ref_mzs, query_mzs, 0.05)
t1 = time.time()
print(f"Old took: {t1 - t0:.4f}s")

t0 = time.time()
ms1_prefilter_new(ref_mzs, query_mzs, 0.05)
t1 = time.time()
print(f"New took: {t1 - t0:.4f}s")
