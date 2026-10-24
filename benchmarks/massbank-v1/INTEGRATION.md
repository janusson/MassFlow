# Integrated candidate MassBank rerun — 2026-10-23

**Application candidate:** `dd418131c87d26eb84c69d1b3733c764d8ee14ff` (clean worktree)

This reruns PR #111's frozen diagnostic harness after integrating PRs #101, #103–#107,
#110 and the restricted-scope contract in #112. It uses the original MassBank-data
2026.03 source commit `705afb7bccc3b2c42410a744eef73674716a60ef`, archive SHA-256
`cb30a4aab9d06dc5978c08e73397106a104955f1c31a57671ee5c15025985529`, the existing
manifest digest `66e27364488feb3e4dc74a452d964d0240d2763e1d1dd40c94ff33c400410b49`, and
parameters digest `096a6ea137ec010d28c5b657a7225bf57c274cc15fa0e3125166b0b5dd446738`.
No selection, corpus, threshold, scoring, or denominator changes were made.

The run was executed with the frozen command in [`README.md`](README.md), using a new
output path under
`/Users/ericjanusson/.cache/massflow-validation/runs/ae-v1-final-dd41813`. All four
run provenance records identify the same clean candidate SHA. The baseline output
hashes were checked against the committed `measured-2026-10-22/artifacts.json` index.

## Before / after

Exact-structure endpoints use the existing eligible denominators (99 present truths;
49 present and 50 absent in mixed arms). Top five counts distinct full InChIKeys.
Candidate calls are qualified target calls. “Absent calls” is incorrect top-one
assignment among the 50 eligible truths removed from the mixed library.

| Arm | Exact top-1 before → after | Exact top-5 before → after | Candidate calls before → after | Absent calls before → after |
| --- | --- | --- | --- | --- |
| Present cosine | 52/99 → 52/99 | 53/99 → 53/99 | 55 → 55 | — |
| Present modified cosine | 27/99 → 27/99 | 46/99 → 46/99 | 92 → 92 | — |
| Mixed cosine | 26/49 → 26/49 | 27/49 → 27/49 | 29 → 29 | 2/50 → 2/50 |
| Mixed modified cosine | 16/49 → 16/49 | 21/49 → 21/49 | 91 → 91 | 46/50 → 46/50 |

All four arms attempted 100 queries, retained the same primary eligibility counts,
scored the same 92 queries, and recorded the same eight minimum-peak processing
drops and one prespecified potential-leakage exclusion. Per-query top-1, top-5 order,
true-structure rank, best score, candidate structure domain, eligibility, processing
state, and legacy q-value comparisons report **zero changes**. The complete
machine-readable per-query comparison, before/after metric records, evidence paths,
and artifact hashes are in
[`integrated-candidate-comparison.json`](integrated-candidate-comparison.json).

## Confidence and candidate-domain findings

| Arm | Target / surviving decoy query competitions | Explicit candidate confidence state |
| --- | --- | --- |
| Present cosine | 55 / 0 | `unavailable` |
| Present modified cosine | 92 / 1 | `estimated_uncalibrated` |
| Mixed cosine | 29 / 0 | `unavailable` |
| Mixed modified cosine | 91 / 1 | `estimated_uncalibrated` |

The integrated exports now include explicit confidence states and nullable
`confidence_q_value`; zero or one decoy remains insufficient to establish
calibration. Legacy numerical q values did not change on this cohort. The candidate
prefilter was disabled in the frozen parameters both before and after, so this
benchmark does **not** evaluate the mixed-frame Numba correction. That correction
has separate synthetic numerical regressions. No benchmark result here validates
chemical identity, calibrated FDR, or a deployment population.

## Reproduction

From a clean checkout of the candidate SHA with Python 3.13 and the committed lock:

```sh
uv run python -m scripts.validation_benchmark.runner run \
  --cache /Users/ericjanusson/.cache/massflow-validation \
  --output /Users/ericjanusson/.cache/massflow-validation/runs/ae-v1-final-dd41813 \
  --manifest benchmarks/massbank-v1/manifest.json \
  --manifest-sha256 66e27364488feb3e4dc74a452d964d0240d2763e1d1dd40c94ff33c400410b49 \
  --parameters benchmarks/massbank-v1/parameters.json \
  --parameters-sha256 096a6ea137ec010d28c5b657a7225bf57c274cc15fa0e3125166b0b5dd446738
```

Large source, native databases and complete candidate/reference ledgers remain outside
Git; their hashes and the local run path are retained in the comparison JSON. These
results are bounded reproducibility evidence for this deposited cohort only.
