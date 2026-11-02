# Mixed-frame modified-cosine recall defect

## Finding

On main `ffef50bff913fcf400cb3de72595dab50e4f3fd8`, the default Numba
prefilter silently removes some pairs that pass full modified-cosine scoring.
With the default `min_matched_peaks=3`, two direct-fragment matches plus one
precursor-shifted match yield score **1.0, three matched peaks**, but no search
result with acceleration enabled. The same input returns a hit with
`enable_numba_prefilter=False`. Both the JIT kernel and its Python mirror have
the incorrect gate; the engine activates the gate when Numba is available.

This is a demonstrated false negative and a reproducibility defect, since
enabling the prefilter changes the accepted search results. Its frequency in measured
data is unknown.

## Discriminating experiment

All spectra below are **explicitly synthetic**. They do not represent measured
caffeine or theobromine fragmentation, authentic standards, or confirmed
identifications. The precursor formulas and illustrative fragment formulas
provide plausible small-molecule mass ranges while isolating the algorithm.
Masses use real `pyteomics.mass.calculate_mass(..., charge=1)` calculations;
the fixtures exercise the real formula/adduct/5-ppm integrity gate.

- Reference precursor: `C7H8N4O2`, `[M+H]+`, charge +1,
  m/z **181.07200198165003**.
- Query precursor: `C8H10N4O2`, `[M+H]+`, charge +1,
  m/z **195.08765204578998**.
- Precursor difference: **14.015650064139948 Da**.
- Reference peaks: **67.02907454051, 97.03963922421,
  109.03963922420998**, derived from `C3H2N2`, `C4H4N2O`, `C5H4N2O`
  with charge +1.
- Query peaks: add the precursor difference to the final one or two peaks.
  All intensities are 1.0; arrays are `float64`.
- Scoring: `modified_cosine`, MS2 tolerance **0.02 Da**, minimum score **0.7**,
  minimum matched peaks **3**. No decoys are used: this experiment isolates
  candidate recall, not confidence estimation.

For three disjoint matched pairs with unit intensities, the expected cosine is
`3 / (sqrt(3) * sqrt(3)) = 1`. The installed matchms 0.33.1 scorer independently
returns that value and three matched peaks.

| Synthetic query | Direct matches | Shifted matches | Full score / peaks | Main: prefilter off | Main: prefilter on | Corrected: prefilter on |
| --- | ---: | ---: | --- | --- | --- | --- |
| All direct | 3 | 0 | 1.0 / 3 | Hit | Hit | Hit |
| All shifted | 0 | 3 | 1.0 / 3 | Hit | Hit | Hit |
| Two direct, one shifted | 2 | 1 | 1.0 / 3 | Hit | **No hit** | Hit |
| One direct, two shifted | 1 | 2 | 1.0 / 3 | Hit | **No hit** | Hit |
| One direct, one shifted, one unmatched | 1 | 1 | 2/3 / 2 | No hit | No hit | No hit |

The last control displaces the second query peak by 3 Da rather than preserving
it. Regression tests also reverse query/reference roles, exercise the inclusive
tolerance boundary with exactly representable masses, and verify that overlapping
frames cannot inflate the final one-to-one match count.

## Root cause and correction

`_prefilter_pairs_numba` and `_prefilter_pairs_python` previously required
`max(direct_count, loss_count) >= min_matches`, in both their allocation and
emission passes. Modified cosine assigns peaks over the **union** of direct and
shifted candidate edges, as documented in the
[matchms scorer](https://matchms.readthedocs.io/en/stable/_modules/matchms/similarity/ModifiedCosineGreedy.html).
MassFlow's independent golden-fixture reference scorer also uses this union.

For each sorted frame, the two-pointer count is a maximum-cardinality matching.
Partition the scorer's selected edges by frame; each subset is bounded by its
frame's maximum count. Thus their **sum** is an upper bound on the combined
selected count, while their maximum is not. The correction uses the sum in all
four gate sites, keeping the existing early exit when direct matches suffice.
Cross-frame double counting can admit extra candidates, but full scoring still
resolves peak reuse and applies the final matched-peak and score thresholds.
This is a conservative pruning bound, not a replacement score or a scientific
heuristic. The cosine-only gate is unchanged.

Here, full scoring means the existing matchms `ModifiedCosineGreedy` scorer,
which uses greedy peak assignment, rather than an optimal Hungarian assignment.
The count bound assumes that counters and scorer recognize the same
tolerance-compatible peak pairs. Floating-point differences in how they compare
masses at a tolerance boundary remain a separate, pre-existing limitation;
this correction does not establish recall equivalence for every float64 input.
The regression boundary fixtures use exactly representable masses to isolate
the mixed-frame counting defect.

## Reproduction and scope

Run the committed experiment/regression tests:

```bash
uv sync --locked --all-extras
uv run pytest tests/test_mixed_frame_prefilter.py -v
```

Copy that test file onto the stated main revision to reproduce the before state:
**18 failures, 8 passes**. On the correction all **26 pass**. The baseline full
suite passed **1356 tests**, with **7 skipped**, **35 deselected**, and
**82.33% coverage**, demonstrating the existing coverage gap. The corrected full
suite passed **1382 tests**, with **7 skipped**, **35 deselected**, and
**82.46% coverage** using the mandatory command:

```bash
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v
```

These results were freshly reproduced on **2026-10-21 UTC**, on macOS arm64,
Python 3.13.12, matchms 0.33.1, NumPy 2.2.6, Numba 0.61.2. Lockfile validation,
Ruff lint/format, mypy, all pre-commit hooks, and strict MkDocs build passed.
An independent read-only review checked all four gate sites and the matching
bound; its documentation findings are reflected in the limitations above.
Hosted Linux/macOS CI remains a separate check on the resulting PR head.

This correction restores the demonstrated candidate recall. Additional candidates
can affect downstream rankings and target-decoy estimates; the experiment does
not establish identification accuracy, empirical FDR calibration, or the
scientific reliability of analogue annotations. No golden expected values were
regenerated, and no confidence estimator was changed.
