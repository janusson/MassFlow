# Frozen MassBank retrieval-funnel diagnostic rerun

**Candidate SHA:** `d66719f7b28350369afbe9da0f99e0020c9bb85c` (clean run)
**Frozen source:** MassBank-data commit `705afb7bccc3b2c42410a744eef73674716a60ef`, archive SHA-256 `cb30a4aab9d06dc5978c08e73397106a104955f1c31a57671ee5c15025985529`
**Manifest SHA-256:** `66e27364488feb3e4dc74a452d964d0240d2763e1d1dd40c94ff33c400410b49`
**Parameters SHA-256:** `096a6ea137ec010d28c5b657a7225bf57c274cc15fa0e3125166b0b5dd446738`

This is a distinct experiment from the MoNA/Vaniya-Fiehn LTQ/QTOF cohort in
[`../retrieval-diagnostics-v1/INVESTIGATION.md`](../retrieval-diagnostics-v1/INVESTIGATION.md).
The frozen source, manifest, selection, eligibility, excluded potential-leakage query,
processing settings, scoring thresholds, and denominators were unchanged. The runner
only enables `collect_retrieval_diagnostics`, which is non-scoring. Machine-readable
per-query attribution, score histograms, provenance hashes, and comparisons are in
[`retrieval-diagnostic-comparison.json`](retrieval-diagnostic-comparison.json).

## Retrieval results

| Arm | Exact top-1 | Exact top-5 | Candidate calls | Incorrect top-1 for absent truths | Confidence state |
|---|---:|---:|---:|---:|---|
| Present cosine | 52/99 | 53/99 | 55 | — | `unavailable` (0 surviving decoys) |
| Present modified cosine | 27/99 | 46/99 | 92 | — | `estimated_uncalibrated` (1 surviving decoy) |
| Mixed cosine | 26/49 | 27/49 | 29 | 2/50 | `unavailable` (0 surviving decoys) |
| Mixed modified cosine | 16/49 | 21/49 | 91 | 46/50 | `estimated_uncalibrated` (1 surviving decoy) |

Each arm attempted 100 queries, processed 92, and rejected the same eight queries for
insufficient peaks. The primary present denominator is 99 (one prespecified potential
leakage exclusion); mixed arms contain 49 eligible library-present and 50
library-absent truths. The absent-truth call rate is cohort-specific, not an unknown-
sample false-positive rate.

## Observable pair funnel

| Arm | Target query-reference pairs | Pairs scored | Score gate passed | Matched-fragment gate passed / candidates returned |
|---|---:|---:|---:|---:|
| Present cosine | 636,364 | 1,390 | 403 | 247 |
| Present modified cosine | 636,364 | 636,364 | 62,072 | 3,490 |
| Mixed cosine | 583,740 | 821 | 194 | 108 |
| Mixed modified cosine | 583,740 | 583,740 | 53,824 | 2,961 |

The cosine MS1 prefilter reduces the pair domain; modified cosine retains its distinct
full-library pair domain in these frozen settings. The reported histograms summarize
scores submitted to each scorer before score and matched-peak gates. The per-query
funnel distinguishes no precursor candidates, score rejection, insufficient fragments,
confidence state, and returned candidates. It does not infer a cause when a stage was
not observed.

Truth-only replay uses the exact full InChIKey, separately from production annotations.
Among primary-eligible queries, the main truth outcomes were:

| Arm | Correct top-1 | Correct top-5, not top-1 | Score below threshold | Insufficient fragments | Truth ranked below top-5 | Absent from mixed library |
|---|---:|---:|---:|---:|---:|---:|
| Present cosine | 52 | 1 | 6 | 33 | 0 | — |
| Present modified cosine | 27 | 19 | 6 | 33 | 7 | — |
| Mixed cosine | 26 | 1 | 3 | 15 | 0 | 50 |
| Mixed modified cosine | 16 | 5 | 3 | 15 | 6 | 50 |

Truth-reference pairs are replayed through the same effective processing and scoring
gates, and compared to the full-library candidate ranking. The report also records when
truth records are absent from the prepared store, per-query rank, and the reason when
all truth candidates fail a measurable gate. Benchmark truth never enters normal
MassFlow output.

Reference record accession metadata is present on every captured candidate row in the
four runs. The library/reference provenance and per-query lineage remain in the
external run files and are indexed by SHA-256 in the comparison JSON.

## Frozen-baseline comparison and confidence

The comparison uses the previous clean integrated candidate at
`dd418131c87d26eb84c69d1b3733c764d8ee14ff`. For all four arms, the full metric
dictionaries, query accounting, result rows, candidate sets, rankings, scores,
matched-peak counts, q-values, and confidence states are identical. Every arm reports
zero query-ledger changes and zero changed result rankings. The only effective
configuration addition is the diagnostics flag; the frozen scoring fields compare
equal.

The target/decoy state remains uncalibrated: cosine has zero surviving decoy
competitions, while modified cosine has one. This does not establish calibrated
q-values or FDR control. The frozen Numba prefilter is disabled, so this MassBank
experiment does not exercise its approximate-pruning behavior.

## Reproduction

```sh
uv run python -m scripts.validation_benchmark.runner run \
  --cache /Users/ericjanusson/.cache/massflow-validation \
  --output /tmp/massflow-massbank-retrieval-diagnostics-d66719f \
  --manifest benchmarks/massbank-v1/manifest.json \
  --manifest-sha256 66e27364488feb3e4dc74a452d964d0240d2763e1d1dd40c94ff33c400410b49 \
  --parameters benchmarks/massbank-v1/parameters.json \
  --parameters-sha256 096a6ea137ec010d28c5b657a7225bf57c274cc15fa0e3125166b0b5dd446738
```

The full source archive and native run stores remain outside Git. The source archive is
pinned by the SHA above; generated artifacts are reproducible with the frozen command.
These results characterize this deposited cohort only and do not certify unknown-sample
identification or FDR.
