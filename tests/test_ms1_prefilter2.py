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
        if query_mz > 0:
            min_mz, max_mz = query_mz - ms1_tolerance, query_mz + ms1_tolerance
            start_idx = np.searchsorted(ref_mzs_sorted, min_mz, side="left")
            end_idx = np.searchsorted(ref_mzs_sorted, max_mz, side="right")

            # Using extend is faster than append in a loop
            original_indices = ref_mzs_sorted_indices[start_idx:end_idx]
            rows.extend(original_indices)
            cols.extend([query_idx] * len(original_indices))

    return np.array(rows), np.array(cols)


ref_mzs = np.random.uniform(100, 1000, 50000)
query_mzs = np.random.uniform(100, 1000, 10000)

t0 = time.time()
r_old, c_old = ms1_prefilter_old(ref_mzs, query_mzs, 0.05)
t1 = time.time()
print(f"Old took: {t1-t0:.4f}s")

t0 = time.time()
r_new, c_new = ms1_prefilter_new(ref_mzs, query_mzs, 0.05)
t1 = time.time()
print(f"New took: {t1-t0:.4f}s")

assert len(r_old) == len(r_new)
