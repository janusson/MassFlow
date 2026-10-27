# Duplicate fragment rows can manufacture matched-peak support

## Finding

On committed main `ffef50bff913fcf400cb3de72595dab50e4f3fd8`, raw MGF and MSP
spectra containing exact duplicate fragment m/z rows pass ingestion. A synthetic
pair sharing only one distinct ion then passes the default three-matched-peak
gate when that ion is represented by three rows in each spectrum. The cosine
score is **100/101 = 0.9900990099009901** both before and after row splitting.

This is an ingestion defect relative to the documented physical-evidence purpose
of `min_matched_peaks`. The upstream scoring algorithm correctly counts assigned
array rows; those rows do not necessarily represent distinct ions. It is not an
experimentally demonstrated false compound identification or an FDR result.

## Experiment and independent expectation

The input spectra are explicitly synthetic caffeine-formula `[M+H]+` records,
not measured caffeine spectra. Real Pyteomics calculations give precursor m/z
approximately 195.087652. Both spectra contain a hypothetical protonated `C6H7N3O`
fragment at m/z 138.066188 with intensity 1. The reference also contains
protonated `C5H7N3` at m/z 110.071274 with intensity 0.1; the query instead
contains protonated `C4H6N2` at m/z 83.060375 with intensity 0.1. These
formulas are subcompositions of the precursor; no fragmentation yield or pathway
is asserted. The real formula/adduct/5-ppm validators accept the precursors.

Replace every peak by `k` identical m/z rows, each with intensity `I/k`. This
conserves integrated signal for each ion and adds no distinct fragment masses.
For equal splitting on both sides, the normalized dot product remains
`1 / (1 + 0.1**2) = 100/101`, while the number of assigned rows becomes `k`.

| Rows per ion, both spectra | Distinct shared ions | Score | Reported matches | Candidate passes default gates |
| --- | --- | --- | --- | --- |
| 1 | 1 | 100/101 | 1 | No |
| 2 | 1 | 100/101 | 2 | No |
| 3 | 1 | 100/101 | 3 | Yes |
| 5 | 1 | 100/101 | 5 | Yes |

A 96-condition baseline matrix tested cosine/modified cosine, prefilter on/off,
both directions, 0.001/0.02 Da fragment tolerances, the four equal-splitting
conditions above and one-sided `(1,3)`/`(3,1)` controls. All observations agreed
with the row-space calculation. The 32 equally split three-/five-row conditions
passed the gate. One-sided splitting cannot create three assigned matches.
Actual MGF/MSP loading followed by default processing retained duplicate rows.
The three-distinct-fragment positive controls remained eligible.

The deliberately challenged interpretation, that one shared ion cannot satisfy
three distinct fragments, failed in all 32 primary conditions. A separate
127-test characterization, including signal conservation, real imports,
three-distinct-ion controls and 4/6-ppm physics controls, passed. These are
constructed software experiments; they establish no empirical frequency.

## Root cause and correction

`io._validate_spectra_iterator` accepted nondecreasing arrays through
`np.diff(mz) >= 0`. Processing retained equal masses. `SimilarityEngine.search`
then compared the upstream assigned-row count with `min_matched_peaks`.
[matchms 0.33.1 also accepts equal adjacent masses](https://matchms.readthedocs.io/en/latest/_modules/matchms/Fragments.html),
and [its cosine API returns a matched-peak count](https://matchms.readthedocs.io/en/latest/api/matchms.similarity.CosineGreedy.html).

The correction quarantines the entire spectrum when adjacent fragment masses
are exactly equal, using `Duplicate fragment m/z values` through the existing
rejection callback and quarantine log. It does not silently sum intensities or
choose an empirical peak-coalescing tolerance. Distinct masses, including a
one-ULP separation, remain accepted. Mixed query files record the rejection
count in their sidecar; an all-rejected query file fails explicitly, including
the reason in its failure sidecar, and writes no result CSV.

The managed raw-library cache fingerprint now includes
`io.SPECTRUM_VALIDATION_REVISION`. Tests seed legacy SQLite, Zarr and hybrid
caches with duplicate rows and verify that rebuilding removes the invalid
record while preserving the valid one. This policy revision is independent of
the processing-semantics revisions in the existing stabilization PRs; integration
must retain both inputs to cache invalidation.

## Reproduction and qualification

The regression tests build the spectra, perform real chemistry validation,
write/read actual MGF/MSP files and exercise actual scoring, workflow reports
and legacy cache rebuilding:

```sh
uv run pytest tests/test_duplicate_fragment_rows.py -v
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v
```

The final 21-test regression file, run against a clean pinned baseline with
verified imports, gives **15 failed / 6 passed**. The six passing cases protect
distinct-mass boundaries. The corrected file gives **21 passed**. The wider
ingestion/library selection gives **52 passed**. No dependency, score formula,
confidence estimator or golden expectation is changed.

The mandatory full coverage command passes on the corrected tree with
**1,377 passed, 7 skipped, 35 deselected, 82 warnings and 82.41% coverage**.
The unchanged baseline passed with 1,356 passed, 7 skipped, 35 deselected,
81 warnings and 82.39% coverage. Ruff, mypy, lockfile checking, all pre-commit
hooks and the strict documentation build pass. Independent read-only review
found no remaining blocker and reran all 21 regressions, including with the
quarantine logger configured not to propagate to the root logger.

## Limits

Exact duplicates are a data-representation problem; their prevalence in real
libraries is unknown. Distinct nearby peaks, unresolved profile data, chemical
independence of fragments and isotopic information require separate acquisition
and processing decisions. Unique m/z values alone do not prove independent
chemical evidence, correct identity or calibrated confidence.

Explicit native stores used directly by `prepare_library` and in-memory spectra
passed directly to the scoring API remain caller-managed. The ingestion check
applies when `load_spectra` is used, including native stores read through that
function; it does not retroactively scan or modify every existing native store.
Rebuild or inspect such inputs before relying on distinct matched-ion support.
The PR is based on main, not a qualified combined stabilization candidate.
