# Scoring and numerical FDR contract

Classical cosine and modified cosine use matchms greedy one-to-one peak assignment
with configured fragment tolerance. Modified cosine also considers precursor-shifted
matches and can retrieve different structures. Spectral agreement is not structural
identity, and a score is not a probability.

The exact query-maxima formula, tie policy, empty-null behavior, explicit confidence
states, machine-readable limitations and compatibility rules are documented in
[Confidence estimates and evidence states](confidence-semantics.md).

## Statistical scope

MassFlow retains one best qualifying target score and one best qualifying decoy
score per query, pools both sets, computes `(1+D)/T`, and applies a reverse minimum
to obtain q-like numerical estimates. This is not a paired winner-take-all contest.
Every exported candidate inherits the statistic determined by its query's best
target, so additional rows are not independently calibrated discoveries.

Both candidate generation and configured gates condition the target/decoy samples.
A surviving decoy permits a numerical estimate, not a claim that the null models
incorrect chemical matches. Empty-null legacy values are rank diagnostics, not
statistical bounds. Increasing a library or relaxing a threshold does not repair
missing calibration evidence. Empirical p-values remain diagnostics only.

## Tests and interpretation

`test_fdr.py`, `test_fdr_statistics.py`, `test_fdr_invariants.py` and
`test_confidence_semantics.py` verify arithmetic and reporting contracts, including
independent complete-threshold enumeration, ties, sparse/empty/null cases and
nonfinite input rejection. Golden integration fixtures verify repeatable numerical
outputs. None is a chemical-identification calibration experiment.

The current procedure has no registered empirically calibrated domain. Confidence
reporting therefore uses estimated_uncalibrated, unavailable or invalid. No code path
promotes confidence to calibrated based on a score threshold, library size,
nonempty null or passing software tests.
