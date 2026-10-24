# Benchmarking reference-library retrieval

`scripts/benchmark_real_corpus.py` evaluates cross-instrument retrieval: spectra
from one acquisition are searched against a known spectral library, and a
retrieved reference is called correct when its InChIKey connectivity block
matches the known query label. This is a retrieval benchmark. It does not
measure identification of compounds in an unknown sample.

## Metric contract

The report uses query-level units. “Selected” means every labeled query chosen
for the benchmark, including queries that the preprocessing peak-count gate
rejects before scoring. “Searched” means selected queries that pass that gate.
An accepted call is a query with at least one candidate at the stated score,
matched-peak, and (when enabled) q-value operating point.

| Report field | Numerator | Denominator | Meaning |
|---|---|---|---|
| `end_to_end_top1_accuracy` | Correct top-ranked compound calls | All selected queries | Headline retrieval accuracy, including preprocessing rejections and no-candidate queries as failures |
| `conditional_top1_accuracy` | Correct top-ranked compound calls | Searched queries | Ranking/retrieval performance conditional on passing preprocessing |
| `call_coverage_selected` | Queries with an accepted candidate | All selected queries | Fraction of the selected set receiving an accepted call |
| `call_coverage_searched` | Queries with an accepted candidate | Searched queries | Fraction of searched queries receiving an accepted call |
| `accepted_call_accuracy` | Correct top-ranked compound calls | Queries with an accepted candidate | Accuracy among accepted calls; undefined (`null`) when there are no calls |
| `true_compound_retrieval_rate_selected` | Queries with the true compound anywhere among retained candidates | All selected queries | Candidate-generation reachability across the selected set |
| `rank1_of_reachable` | Correct top-ranked compound calls | Queries with the true compound among retained candidates | Ranking success conditional on candidate reachability |
| `candidate_rows_exported` | Candidate rows in the saved result table | Not a rate; this is a count | Export volume, not a query metric |

For a headline, use `end_to_end_top1_accuracy` and print its correct-call and
selected-query counts beside it, along with preprocessing-rejected and
searched counts. Label it **cross-instrument reference-library retrieval** and
state that it is not unknown-sample identification. Report instrument-specific
values as well as the pooled result when the query set spans instruments. A
pooled metric can conceal a source-specific failure; investigate and disclose
large coverage differences or a source with zero calls before presenting the
pooled result as representative.

The candidate-row ratio for a paired report is
`current candidate_rows_exported / baseline candidate_rows_exported`. It is a
row-volume ratio, not a multiplicative change in candidates per query or an
accuracy change. The report also gives the percent wall-time change. A zero
baseline row count makes the ratio undefined (`null`). The paired comparison
also flags operating-system, CPU architecture, and Python-version differences;
interpret wall-time changes cautiously when any of these differ.

## Reproducing and comparing runs

Run the benchmark from the repository root with a fresh work directory. For
300 selected queries per input instrument, pass `--max-queries 300`; the
selection can be smaller if the source corpus has fewer eligible records.
The default preprocessing gate is `--min-query-peaks 3`. Its rejected queries
remain in the end-to-end denominator. The October 18 cross-instrument cohort is
recreated with seed `20260921`, 300 queries per QTOF/LTQ file, QExactive as the
library, cosine, `min_score=0.4`, `min_matched_peaks=3`, `fdr_threshold=1.0`,
and matched MS1/MS2 tolerances of `0.01/0.01 Da` (narrow) or `0.02/0.02 Da`
(wide). These settings are not defaults; keep them explicit when reproducing
that experiment. Example:

```bash
uv run python scripts/benchmark_real_corpus.py \
  --corpus /path/to/msp \
  --workdir /tmp/massflow-benchmark-baseline \
  --max-queries 300 \
  --run-tag _baseline
```

Keep the resulting `report.json`, `stats.json`, normalized `library.msp`,
`queries.mgf`, `labels.json`, and run CSVs together. Schema-v2 report metadata
records SHA-256 hashes for normalized inputs and selected query IDs, git
revision and dirty state, Python and platform, run configuration, exported
candidate-row counts, and wall time. `--keep-data` verifies schema, labels, and
the input fingerprint; older or changed prepared data must be rebuilt.

To compare a new run, use a separate work directory and point `--compare-to`
at the baseline report:

```bash
uv run python scripts/benchmark_real_corpus.py \
  --corpus /path/to/msp \
  --workdir /tmp/massflow-benchmark-current \
  --max-queries 300 \
  --run-tag _current \
  --compare-to /tmp/massflow-benchmark-baseline/report.json
```

The selection seed and exact normalized library/query inputs must match; the
report comparison refuses different dataset fingerprints. If both MS1 and MS2
tolerances changed, describe the comparison as a joint configuration change.
It cannot establish the causal effect of either tolerance separately. To
attribute effects, hold other settings fixed and vary one tolerance at a time,
or run a 2×2 MS1/MS2 tolerance design. The score and q-value sweeps evaluate
only candidates retained and exported by the actual run gates; they cannot
recover rows filtered out during annotation.

## FDR wording for benchmark reports

When the decoy null is empty at a gate, state:

> The decoy null was empty at this gate; the reported q/FDR column is an
> uncalibrated rank bound and does not establish FDR control.

Do not describe the rank-bound column as a measured or calibrated FDR, claim
that the run controls FDR at the selected threshold, or say that accepted
candidates have a verified false-match rate at or below that threshold.

At the permissive fixture operating point with 400 target and 375 decoy
competitions and 3,780 rows at `q <= 0.05`, state:

> At the permissive gate, 400 target and 375 decoy competitions produced
> 3,780 rows with q <= 0.05 under the implemented ranking procedure; this is a
> fixture-specific rank-based result and is not evidence of calibrated FDR
> control.

Do not state that 3,780 compounds were identified, that 5% of those rows are
known false, or that the observed counts prove empirical control of the true
false-discovery rate. The q-value is query-level under the target-decoy
contract; repeated candidate rows from one query are not independent
discoveries. See [Scoring Logic](scoring_logic.md) for assumptions and
degraded-mode behavior.

## Saved-output accounting check

Before quoting a metric, reconcile every selected query into exactly one of
three states: preprocessing-rejected, searched with no accepted candidate, or
searched with an accepted candidate. Then check that correct top-1 calls do
not exceed calls, calls do not exceed searched queries, and searched queries
plus preprocessing rejections equal selected queries. Recompute the metric
numerators and denominators from the query partition and saved candidate rows;
never derive accepted-call accuracy from classification metrics over invented
labels for wrong predictions.

## Retrieval-funnel diagnostics

Set `similarity.collect_retrieval_diagnostics: true` to write a versioned
`<results-stem>.diagnostics.json` beside the normal YAML report. It contains
input/library fingerprints when available, the effective configuration and
run SHA/dirty state, bounded typed preprocessing-rejection samples, per-query
terminal outcomes, and cumulative target/decoy pair counts at candidate
prefilter, score, matched-fragment, polarity, RT, top-N, and confidence stages.
The JSON sidecar is referenced from the YAML provenance report. Collection is
opt-in and does not alter scores or result ordering.

The `search.pair_stages` counts are **query-reference pairs**, not query
counts. Each stage is a cumulative survivor count; `rejection_counts` records
disjoint differences between adjacent stages. Query terminal outcomes are
mutually exclusive. `unknown_or_unattributable` is reported separately and
must not be added as a second terminal state. `no_precursor_candidates` means
no reference passed the configured cosine precursor window; it does not mean
the query compound is absent from the library. An empty candidate list alone
never establishes missing chemical coverage.

Benchmark-only `truth_attribution` is kept separate from production
annotations. It replays each labelled query against only its declared truth
records using the same effective processing and search gates, then compares
that reachability with the full-library ranking. It can distinguish records
not retained in the prepared store, precursor/prefilter loss, score or fragment
threshold loss, polarity/RT loss, confidence filtering, and ranking among
returned candidates. Any reason not directly observed is recorded as unknown;
truth labels are not passed into the annotation pipeline.

The real-corpus harness currently normalizes every query with `IONMODE=Positive`
and `CHARGE=1` and does not emit source precursor-type metadata as an adduct.
Its source metadata audit records the original polarity/adduct values separately
and warns that these normalized cross-instrument runs do not test the original
source polarity/adduct gates. Do not interpret them as evidence that LTQ/QTOF
polarity metadata were compatible.

When a search configuration allows zero-score/zero-match rows (`min_score <= 0`
and `min_matched_peaks == 0`), the legacy sparse-cosine output can contain a
zero-valued row for a pair that was not submitted to scoring. The diagnostic
marks such stage arithmetic with `stage_accounting_consistent: false` and
`unattributable_pairs`; it does not invent a score- or prefilter-rejection
cause. This edge behavior is preserved for compatibility, and is not exercised
by the cross-instrument benchmark's `0.4` score / 3 matched-peak gates.

Each pair-stage class also includes `score_distribution`, a fixed 0.1-wide
histogram of finite scores submitted to scoring and before the configured score
cutoff. The final bin includes score 1.0; below-range, above-range, and
non-finite values are counted separately. Histogram counts plus those three
exception counts equal `scored_pairs`; prefilter-excluded pairs are not included
in the distribution. This bounded summary is sufficient to distinguish “scored
but below threshold” from “never reached scoring” without retaining a full
query-reference score matrix.

The benchmark report adds `retrieval_diagnostics_by_source`: per-instrument
selected/preprocessing-rejected/searched/no-call counts, target/decoy pair
funnel totals, score histograms, confidence states, and terminal outcomes. Its
preprocessing-rejected denominator is derived from benchmark peak-count labels
and checked against the production query-ID funnel; unobserved non-preprocessing
records are kept as unknown rather than silently classified.
