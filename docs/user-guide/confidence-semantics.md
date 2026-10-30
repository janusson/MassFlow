# Confidence estimates and evidence states

MassFlow's similarity score and its statistical estimate are different quantities.
A cosine or modified-cosine score measures spectral agreement under a particular
preprocessing, peak-assignment and tolerance policy. It is not an identification
probability. A numerical q estimate does not establish experimentally demonstrated
FDR control or confirmed chemical identity.

## Machine-readable contract

| Field/state | Meaning |
| --- | --- |
| `score`, `matched_peaks` | Numerical spectral matching outputs, unchanged by this confidence reporting layer |
| `confidence_state=estimated_uncalibrated` | At least one qualifying target and decoy query maximum permits the implemented calculation; empirical calibration is pending |
| `confidence_state=unavailable` | No target observations or no surviving decoy observations; no statistical estimate is available |
| `confidence_state=invalid` | An unattributable candidate, nonfinite/nonnumeric score or malformed numerical input invalidates the family; the workflow fails explicitly |
| `confidence_q_value` | Nullable numerical estimate for a query with a target when the family is estimated_uncalibrated; null otherwise |
| `confidence_method` | `pooled_query_maxima_plus_one_v1` |
| `confidence_scope` | `query_best_target`; every candidate of that query inherits the query assessment, not candidate-specific confidence |
| `confidence_limitations` | JSON array in tabular exports; machine-readable reasons and unvalidated assumptions |
| Sidecar `confidence` | Versioned method, formula, discovery unit, family, counts, tie policy, limitations and `empirical_calibration: pending` |

There is no `calibrated` state produced by this implementation. A nonempty null is
necessary for this numerical estimate and is insufficient to establish calibration.
One observed decoy is explicitly reported as `single_decoy_observation`; no magic
sample size or library size converts an estimate into a validated one.
Missing estimates serialize as null/empty cells, never as zero. A supplied or
computed zero must never be interpreted as certainty.

## Compatibility and filtering

The historical `q_value`/`p_value` columns and existing `fdr_threshold` filtering
retain their API names. A proven partial-tie undercount is corrected, so values
and threshold selection can change for tied target/decoy scores.
On an empty decoy null, historical q is `1/N_target` and p is 1: these are arithmetic
diagnostics, **not an estimated false discovery rate or a conservative error bound**.
Use `confidence_state` and `confidence_q_value` to determine estimate availability.
Changing query-file partitioning changes N and can change the historical filter.
Do not use the legacy column without checking the explicit state.

Newly assessed candidate rows are labelled `Uncalibrated` even when a decoy survived.
No-hit queries remain `Unknown` with unavailable confidence. Existing direct export
callers without assessments retain legacy score-status behavior for compatibility;
those labels cannot be interpreted as a validated confidence level.

## Exact method

For each query, retain the maximum target and maximum decoy score after configured
candidate eligibility, score and matched-peak gates. Pool these two sets; this is
not one target-or-decoy winner per query. At score threshold t:

```text
T(t) = number of finite best target scores >= t
D(t) = number of finite best decoy scores >= t
FDR_hat(t) = min(1, (1 + D(t))/T(t))      [1 when T(t)=0]
q(s) = min_{t<=s} FDR_hat(t)
p(s) = (1 + number of best decoy scores >= s)/(1 + N_decoy)
```

The arrays are ranked by descending score, decoys first on ties, followed by a
reverse cumulative minimum over complete equal-score blocks. Partial tie ranks
are never treated as attainable thresholds. Equal best target scores share q. The p column is a
diagnostic and is not used for FDR thresholding. The calculation family is one
input file. Arrays must be one-dimensional, real and finite; invalid values cannot
silently disappear. Candidate scoring remains independent of this validation.

## Assumptions and pending calibration

Incorrect target matches and generated decoys must be comparable after every gate.
Precursor windows, adduct/polarity restrictions, shifted-fragment candidates,
library composition, repeated spectra, query dependence and instrument/CE changes
can alter those populations. Preserving entropy or precursor m/z does not prove
exchangeability; uniform fragment jitter can also leave some peaks within scoring
tolerance. Scores from routed engines are not automatically on comparable scales.

Synthetic arithmetic tests verify formulas, ties, monotonicity and invalid-input
behavior. They do not verify chemical error rates. Existing saved cross-instrument
MoNA results include empty-decoy regimes; they cannot establish FDR calibration.
Independent known-truth present/absent experiments, group-level uncertainty and
frozen query/file/library definitions are required. Calibration remains pending.
See Scheubert et al., [10.1038/s41467-017-01318-5](https://doi.org/10.1038/s41467-017-01318-5),
for a distinct published decoy-method evaluation; it does not validate MassFlow's
generator. Li et al., [10.1038/s41592-021-01331-z](https://doi.org/10.1038/s41592-021-01331-z),
addresses entropy similarity and discusses limitations of computational decoys.
