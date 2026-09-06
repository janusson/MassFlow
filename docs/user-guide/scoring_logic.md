# MassFlow Scoring & FDR Statistical Contract

> **STATUS: authoritative.** This document defines the statistical model behind
> every `q_value`, `p_value`, and FDR-based filter decision in MassFlow. The
> executable form of this contract is `tests/test_fdr_statistics.py`,
> `tests/test_fdr.py`, and the golden known-answer suite
> `tests/test_scientific_validation.py` (fixtures in
> `tests/scientific_validation/`); the implementation lives in
> `MassFlow.similarity.calculate_fdr`, `MassFlow.similarity.calibrate_query_level_fdr`,
> and the per-file FDR block of `MassFlow.workflow._process_single_file`.

## 1. The score being calibrated

The calibrated quantity is the **similarity score** produced by the configured
engine: `cosine` / `modified_cosine` (stable core), or `consensus` / `cascade` /
routed ML scores (experimental). Every engine returns scores on the [0, 1]
interval, but the calibration does **not** assume any particular score
distribution — it only assumes **exchangeability of null target hits and decoy
hits scored by the same engine** (see §6).

Scores are pre-filtered by `min_score` and `min_matched_peaks` **before**
calibration. All statements below are therefore conditional on the configured
thresholds: q-values control error among *retained* candidates.

## 2. The competition unit: the query spectrum

**One query spectrum = one competition.** For each query `q`:

* `T_q` = best (maximum) score over all of `q`'s **target** hits that passed the
  thresholds.
* `D_q` = best score over all of `q`'s **decoy** hits that passed the thresholds.
* A query with no target hit is not a potential discovery and contributes
  nothing to the target side; a query with no decoy hit has `D_q = -∞` and
  contributes nothing to the decoy side.

Multiple hits of the same query are **never** counted more than once. This is
what makes the estimate robust to correlated hits: a query that produces 50
threshold-passing hits competes exactly once, identically to a query that
produces one hit.

## 3. The FDR estimate and the q-value

For a score threshold `t`:

```
FDR(t) = (1 + #{queries : D_q >= t}) / #{queries : T_q >= t}
```

clipped to [0, 1]. The `+1` pseudo-count in the numerator is the standard
conservative target-decoy correction (Elias & Gygi, 2007): it prevents
optimistic `0.0` estimates and keeps small-library estimates conservative.

The **q-value** of a query whose best target score is `s` is the monotone
closure

```
q(s) = min over t <= s of FDR(t)
```

i.e. the smallest FDR level at which the query's top annotation is accepted.

**What a reported q-value means:** if all queries with `q <= α` are accepted,
the expected fraction of accepted queries whose top annotation is a false match
is at most `α`, *under* the exchangeability assumption of §6. It is a property
of the **query**, not of an individual hit. Every exported row of the same
query carries the same q-value; the q-value calibrates the query's *best*
annotation, and lower-ranked hits of the same query are part of the same
discovery unit (they are not independently calibrated).

**Ties are handled conservatively:** on equal scores, decoys rank before
targets, so a tied decoy is always counted against the target. A tie can never
lower a target's q-value.

**Duplicate scores:** the q-value is a function of the score; queries with
identical best-target scores receive identical q-values. The per-score lookup
uses the last (lowest-ranked) occurrence of the score, i.e. the largest
q-value within the tie block.

## 4. Empirical p-values: diagnostic only

The exported `p_value` column is

```
p(s) = (1 + #{queries : D_q >= s}) / (1 + #{queries with a decoy hit})
```

i.e. the fraction of decoy competitions that matched or beat score `s`, with a
`+1` pseudo-count. Ties count against the target.

**The p-value is never used as a filter.** The only acceptance filter is
`q_value <= fdr_threshold`. The p-value is exported as a per-query diagnostic
so users can see how the query's score ranks against the empirical decoy null.

## 5. When is FDR used; what happens for small libraries

**FDR (TDC q-values) is always the calibration and always the filter** — for
libraries of every size. There is no automatic switch to another statistical
concept.

For very small libraries (`< 2000` target spectra) the q-value resolution is
coarse (the pseudo-count dominates, and q-values often collapse to 1.0). The
workflow emits a **CRITICAL SCIENTIFIC WARNING** and recommends relaxing
`fdr_threshold` or using a larger library; it does *not* silently replace the
q-value with a Bonferroni-corrected p-value or any other error measure. A
Bonferroni-style bound appears only in one degenerate case: when **no decoy
hits exist at all**, no calibration is possible and every query receives
`q = 1/N` (the rank-based bound shared by all N competing queries), with
`p = 1.0`.

If there are **no target hits**, nothing is exported and no q-values are
produced.

## 6. Heterogeneous engines (consensus, cascade, routing)

A query is scored by exactly **one** engine (the router assigns easy queries to
a classical engine and hard queries to an ML/consensus engine; consensus and
cascade internally use the same engine for all queries they score). The query's
target hit and decoy hit therefore always come from **the same engine** and the
same score scale.

The exchangeability requirement of §3 is: for a query with no true match, the
distribution of its best target score equals the distribution of its best decoy
score. This holds per engine — decoys are scored through the identical
pipeline (for cascade: the same stages, including candidate winnowing; for
consensus: the same weighted aggregation). Pooling per-query counts across
engines is then valid because every query contributes its target count and its
decoy count on the same internal scale; a decoy that beats its own query's
target enters the null exactly once.

Consensus and cascade scores therefore have an **interpretable null**: the
null distribution is the empirical distribution of per-query best *decoy*
scores under the same engine. No assumption about the shape of the null (e.g.
normality) is made.

## 7. Execution-mode equivalence

Single-file runs, multi-file (worker) runs, and streaming-library runs must
produce **identical statistical behavior**:

* Decoys are identical across modes: each decoy is derived from a stable hash
  of `(random_seed, m/z array, intensity array)`, so chunked streaming
  generates exactly the same decoys as one-pass generation.
* The engine-generated decoys (single-file / streaming path) use the
  configured `processing.decoy_min_relative_intensity` and
  `processing.decoy_mz_shift_da` — the same parameters the parent process uses
  in the multi-file path.
* The true reference-library size (targets only) is known in every mode:
  passed explicitly by the orchestrator, or derived from worker state, or
  counted while the streamed library is consumed. The small-library decision
  is therefore identical across modes.

## 8. What changed (audit of the previous behavior)

The previous implementation pooled **every threshold-passing hit of every
query** into one ranking and:

* let queries with many hits inflate the target count (over-optimistic
  q-values);
* broke ties in favor of targets (anti-conservative);
* silently switched small libraries to Bonferroni-corrected hit-level
  p-values compared against an FDR threshold (concept mismatch);
* derived the library size from worker-local state, which was empty in the
  single-file and streaming paths (every such run took the small-library
  branch);
* never scored decoys in the cascade engine's `include_decoys=True` path;
* generated different decoys under chunked streaming than in one pass;
* reported a placeholder `fdr_q_value = 1.0` from the streaming gRPC path
  (now `NaN` = uncalibrated, since per-packet FDR is undefined).

## 9. Mathematical reference

| Symbol | Meaning |
| --- | --- |
| `T_q` | best target score of query `q` (after thresholds) |
| `D_q` | best decoy score of query `q` (after thresholds); `-∞` if none |
| `FDR(t)` | `(1 + #{q: D_q ≥ t}) / #{q: T_q ≥ t}`, clipped to [0, 1] |
| `q(s)` | `min_{t ≤ s} FDR(t)` — monotone closure |
| `p(s)` | `(1 + #{q: D_q ≥ s}) / (1 + #{q: D_q finite})` — diagnostic only |
| Filter | keep hit iff its query's `q(T_q) <= fdr_threshold` |
