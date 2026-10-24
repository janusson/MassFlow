# Retrieval-funnel investigation (candidate retrieval, not identification)

**Candidate code:** `2f72911d297a186b94791c9f97ceee64f3ccace6`
**Environment:** macOS arm64, Python 3.13.12, `uv 0.10.2`
**Machine-readable evidence:** [`cross-instrument-report.json`](cross-instrument-report.json)

The report contains the exact run configuration, normalized-data fingerprint,
source-file SHA-256 hashes, per-source processing/search funnels, bounded per-query
stages and score histograms, truth-only pair replays, baseline row comparisons, and
the joint narrow/wide comparison. Truth labels are used only in the benchmark audit;
normal annotation diagnostics do not require truth.

## Reproduction and identity

The October 18 `/tmp/massflow-benchmark-20261018/` run products are absent. The raw
source MSP files remain available at
`~/Projects/MassFlow/assets/spectral-libraries/msp`. Rebuilding the selector inputs
with seed `20260921`, 300 selected QTOF and 300 selected LTQ queries, and the
QExactive library reproduces the handoff's normalized **dataset fingerprint** exactly:

```text
ac4428a0c5bb54eb9ce3ec7c81deefcf88290305f6e699c6bc4095f78a61331e
```

The source hashes and selected-data hashes are in the JSON report. This verifies the
cohort definition, not the missing historical run outputs/config provenance. The
older 70.3% LTQ-coverage observation therefore remains unreconciled.

Both arms retained the handoff's settings: cosine, `min_score=0.4`,
`min_matched_peaks=3`, `min_query_peaks=3`, `fdr_threshold=1.0`; MS1/MS2 were
`0.01/0.01 Da` (narrow) and `0.02/0.02 Da` (wide). No thresholds, input selection,
or denominators were changed.

## Results

| Arm | Selected / rejected / searched | Calls | Exact-connectivity top-1 | True connectivity among calls | Candidate rows |
|---|---:|---:|---:|---:|---:|
| Narrow | 600 / 100 / 500 | 148 | 116/600 (end-to-end) | 134 | 472 |
| Wide | 600 / 100 / 500 | 155 | 116/600 (end-to-end) | 136 | 508 |

The 100 preprocessing rejections are 83 LTQ and 17 QTOF spectra with fewer than
three input peaks. The 116/600 top-1 numerator is unchanged in both arms. The paired
narrow/wide comparison is descriptive only: **both** MS1 and MS2 tolerances changed.
The saved per-query table reports all comparison changes; the separate comparison to
the pre-diagnostics CSV reports zero candidate-set, ranking, score, matched-peak,
q-value, or confidence-state changes for each arm.

## LTQ zero-call attribution

All 217 LTQ queries that passed the peak-count gate still received zero calls in
both arms. The production funnel gives the following mutually exclusive query
outcomes:

| Arm | No library precursor inside cosine MS1 window | Scored candidates, all below score gate | Score-qualified but insufficient fragment matches | Calls |
|---|---:|---:|---:|---:|
| Narrow | 85 | 132 | 0 | 0 |
| Wide | 54 | 161 | 2 | 0 |

The truth-only replay narrows the explanation. All 300 selected LTQ queries have a
QExactive library connectivity label by benchmark construction. Across their 3,918
truth query-reference pairs, 2,877 pairs belong to the 217 searched queries. At the
narrow `0.01 Da` MS1 window, only 96/2,877 searched truth pairs pass; just 29/217
searched queries have any truth record inside the window. The other 188 searched
queries have no eligible truth precursor candidate. All 29 truth-eligible queries
then fail the configured score/match gates. At `0.02 Da`, 189/2,877 pairs and
59/217 queries pass the precursor window; 58 truth-eligible queries fail the score
gate and one fails the minimum matched-fragment gate. No threshold relaxation is
proposed.

The serialized LTQ query-to-truth-reference absolute precursor differences have
median `17.047695 Da`, p90 `39.943246 Da`, p95 `40.005463 Da`, and maximum
`394.016710 Da`. By comparison, QTOF has 831/3,516 searched truth pairs inside both
windows, with at least one in-window truth reference for 255/283 searched queries.
These measurements support an MS1 measurement-domain incompatibility between the
low-resolution LTQ precursor values and the QExactive reference masses as the
largest observed cause. They do **not** prove that this is the only cause of every
missing LTQ match.

For the 1,551 narrow LTQ target pairs that actually reach scoring, the score histogram
contains 1,537 scores in `[0.0, 0.1)`, 7 in `[0.1, 0.2)`, 6 in `[0.2, 0.3)`, and 1
in `[0.3, 0.4)`; none reaches `0.4`. In the wide arm, 3/3,036 scored LTQ target pairs
reach `0.4`, but none passes the three-fragment gate. This is an observed score
shortfall after the configured candidate domain, not evidence of reference absence.

### Metadata and alternative explanations

The selected LTQ records contain 185 `P` / 115 `N` ion-mode labels and precursor
types `M-H` (234), `M+H` (30), `M+Na` (20), `M+NH4` (5), `M+K` (3), plus six other or
unknown values. QTOF contains 197 `P` / 103 `N`; its most common precursor types are
`[M+H]+` (103), `[M-H]-` (97), `[M+Na]+` (67), and `[M+NH4]+` (27). The raw QExactive
library reports 21,591 positive / 6,669 negative source-mode records and uses several
precursor types (`[M+H]+`, `[M+K]+`, `[M+Na]+`, `[M+NH4]+`, `[M-H]-`). The raw query
files have no separate charge or explicit `adduct` field in the parsed metadata.

The benchmark formatters write `IONMODE=Positive` and `CHARGE=1` for every
normalized query/library record. They copy adducts only from `adduct` or
`adductionname`; they do not map `Precursor_type`/`precursor_type` into the search
adduct field. The query MGF writer does not emit an adduct. Thus source polarity,
charge, and precursor-type evidence is not carried into this experiment's search
metadata, and these results cannot test source polarity/adduct filtering. RT is also
not serialized and `rt_tolerance` is unset. These omissions are limitations of the
benchmark normalization, not evidence that opposite source modes are compatible.

Retention-time tolerance is unset, so RT filtering did not cause these zero calls.
All searched LTQ queries were observed by the pipeline; there are no unknown pipeline
failures in the selected cohort. Confidence is `unavailable` in both arms because no
decoy competition survives the configured gates; this does not establish FDR control.

## Frozen MassBank experiment

The frozen MassBank experiment is separate and preserves its own source archive,
manifest, configuration, selection, and denominators. Its diagnostic rerun, truth
attribution, pre-diagnostics comparison, and confidence-state counts are recorded in
[`../massbank-v1/retrieval-diagnostic-comparison.json`](../massbank-v1/retrieval-diagnostic-comparison.json).
It must not be pooled with the Vaniya-Fiehn cross-instrument experiment.

## Interpretation boundary

The reproduced LTQ failure is principally a **configured precursor-eligibility
limit** followed by low observed scores—not evidence that the selected compounds are
absent from the reference library. The full-library search also records 85/54 LTQ
queries with no precursor-eligible reference in narrow/wide and 132/161 with scored
candidates below threshold. No instrument-specific preprocessing, candidate-search
algorithm, threshold, or denominator was changed. Similarity scores are not identity
probabilities; the decoy evidence is insufficient or unavailable for calibrated FDR.
