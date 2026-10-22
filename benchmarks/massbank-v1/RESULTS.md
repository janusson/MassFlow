# Measured MassBank diagnostic results — 2026-10-22

**Verdict: reproducible retrieval diagnostics, no validated identification or FDR
control.** The unchanged production pipeline ranks incorrect exact structures,
particularly when modified cosine is used for an exact-structure endpoint. Do not
promote these estimates to calibrated confidence or tune thresholds to this test
cohort. These findings support keeping 1.0 scoped to transparent candidate review.

## What was executed

Clean application/harness commit `4dab3c2` (full SHA in artifacts.json), on macOS
arm64 / Python 3.13 with the committed uv lock. All four run provenances report
`dirty=false`. Existing production `run_annotation_pipeline` executes ingestion,
processing, library preparation, actual matching/decoys and native exports; no
replacement scorer or synthetic fragmentation is used. Frozen source and selection
are documented in [README](README.md), [manifest](manifest.json) and
[parameters](parameters.json). Two independent fresh archive retrievals produced
SHA256 `cb30a4aab9d06dc5978c08e73397106a104955f1c31a57671ee5c15025985529`.

The broad [validation specification](../../docs/validation/program/README.md)
remains a proposed program. This smaller study explicitly amends source scope,
raw-lineage verification and release-qualification prerequisites to permit a
**diagnostic** run. Source and group selection were frozen before initial scoring.
The parameter file subsequently recorded the unchanged settings used in that
initial diagnostic. The clean rerun is a reproducibility check of the same cohort,
not an independent holdout or new calibration experiment.

The first execution reached real scoring but failed JSON serialization on a NumPy
boolean. That failed attempt is retained outside Git at `ae-v1-raw-attempt`. After
a strict serializer regression/correction, `ae-v1-complete` completed. The final
clean run `ae-v1-clean-4dab3c2` produced **identical complete metric dictionaries**
for all four arms. Earlier failed/successful runs were not overwritten.

## Selection, failures and leakage

The pinned archive contains 139,240 text records. Source scope excludes 112,890;
15,435 records within Athens/Eawag fail prespecified eligibility; 10,915 qualify
(3,385 Athens, 7,530 Eawag). There are 281 shared connectivity groups; 20 fixed
groups are reserved/unused development and 100 disjoint test groups are selected.
Truth is the depositor-declared standard-compound full InChIKey, not independent
identity authentication. Only positive ESI [M+H]+ MS2 is evaluated.

Every arm attempts **100** queries. Eight are dropped by minimum-peak processing;
92 reach search. A prespecified similarity screen flags one query/reference pair:
`MSBNK-Athens_Univ-AU241404` / `MSBNK-Eawag-EA016602`. Both have a single matching
peak (cosine ~1), so this is **potential** leakage, not proof of shared acquisition.
The flagged query is still attempted and reported, and is independently rejected
by minimum-three-peaks processing. Primary evaluation excludes it, leaving **99**
queries. This removes one difficult present query from the primary denominator;
the sensitivity analysis below shows the full selected denominator as well.

There are zero exact cross-source duplicates under the declared hash policy,
740 repeated full-key identities and 33 exact peak-list duplicate groups within
the reference library. Those reference records remain present; scores are collapsed
by structure only for ranking metrics. The full library loses 613/7,530 records at
the minimum-peak filter, leaving 6,917. In the mixed arm, removal of 50 whole
connectivity groups leaves 6,904 raw references; 559 fail the minimum-peak filter,
leaving 6,345. Individual reference reasons/rights/metadata remain in hashed external
`references.jsonl` ledgers. IO/metadata/physical-integrity rejections were zero in
these final selected arms; source-inventory exclusions remain recorded separately.

## Exact-structure retrieval results

Rates use all eligible present queries, including processing failures. Top five
counts distinct full keys, not acquisition rows. A call means a surviving candidate,
not a correct identity. False matches count incorrect top-one calls; missed matches
count present truths without a correct top-one (the two sets can overlap).

| Arm | Present/absent eligible | Top-1 | Top-5 | Truth anywhere / retrieval coverage | Calls | False / missed |
| --- | --- | --- | --- | --- | --- | --- |
| Present cosine | 99 / 0 | 52/99 (52.5%) | 53/99 (53.5%) | 53/99 (53.5%) | 55 | 3 / 47 |
| Present modified cosine | 99 / 0 | 27/99 (27.3%) | 46/99 (46.5%) | 53/99 (53.5%) | 92 | 65 / 72 |
| Mixed cosine | 49 / 50 | 26/49 (53.1%) | 27/49 (55.1%) | 27/49 (55.1%) | 29 | 3 / 23 |
| Mixed modified cosine | 49 / 50 | 16/49 (32.7%) | 21/49 (42.9%) | 27/49 (55.1%) | 91 | 75 / 33 |

Mixed cosine assigns a candidate to **2/50** absent truths; modified cosine assigns
**46/50**. Among 47 searched absent queries, correct no-candidate rejection is
45/47 for cosine and 1/47 for modified cosine. Three absent processing failures
are not credited as successful open-set rejection.

Retaining the potential-leakage query gives top-1 **52/100**, **27/100**, **26/50**
and **16/50**, respectively. See
[all-selected sensitivity](measured-2026-10-22/all-selected-sensitivity.json).
This analysis only sets ledger eligibility to true and re-evaluates the same
captured candidates; it does not rescore or change data selection.

Secondary connectivity top-one results equal the full-key results on this cohort.
This does not prove stereoisomer discrimination. Candidate score distributions and
33 source CE-text strata are in the machine reports. Queries share LC-ESI-QTOF
instrument class; many CE strata have very few examples, and ramp text is retained
without inventing eV/NCE conversions. No general instrument/polarity/adduct effect
can be estimated from this limited cohort.

The two engines have **different production candidate domains**: cosine uses a
precursor window, while modified cosine permits precursor-shifted analogue matching.
Their comparison is end-to-end and does not isolate scorer quality. An analogue
can be useful scientifically and is still wrong for this exact-structure endpoint.
No post hoc relabelling of analogue matches as correct was performed.

## Confidence diagnostics, not calibration evidence

| Arm | Best target / decoy observations | State | Observed wrong top-one fraction at legacy q <=0.05 |
| --- | --- | --- | --- |
| Present cosine | 55 / 0 | unavailable | 3/55 (5.5%) — legacy rank diagnostic only |
| Present modified cosine | 92 / 1 | estimated_uncalibrated | 65/92 (70.7%) |
| Mixed cosine | 29 / 0 | unavailable | 3/29 (10.3%) — legacy rank diagnostic only |
| Mixed modified cosine | 91 / 1 | estimated_uncalibrated | 75/91 (82.4%) |

These are descriptive **observed FDPs in this frozen labelled cohort**, not expected
population FDR, individual error probabilities, a formal independent-batch control
test or calibrated uncertainty. One surviving decoy supplies arithmetic, not
representative false-match evidence. Query-best q is inherited by lower-ranked
candidate rows. The legacy threshold diagnostics at 0.01/0.05/0.1/1.0 are retained
machine-readably; the benchmark does not reinterpret empty-null values as estimates.
The new confidence PR is separate and was not present in the measured baseline.
Its tie correction and explicit states need a rerun on the eventual combined version.

Do not extrapolate binomial intervals from this deterministic convenience cohort
to deployment calibration. Independent batches, source-authenticated identities,
query/library population controls and clustered uncertainty remain pending under
the proposed protocol. These data justify rejecting a calibrated-FDR claim here;
they do not measure a universal error rate for MassFlow.

## Resources, artifacts and verification

Pipeline wall times were approximately 14.8 s / 33.6 s / 10.8 s / 39.4 s in the
above order. Peak process RSS was about 405 MB in the first arm and 621 MB thereafter.
RSS is a whole-process high-water mark and includes earlier arms. Tests ran
concurrently; these are execution observations, not controlled acceleration benchmarks.
Parsing/leakage auditing is outside the per-arm pipeline timer.

[Artifact index](measured-2026-10-22/artifacts.json) identifies 51 external native,
input, candidate and reference-ledger files by byte count/SHA256. Source archive and
native databases/peak files are not committed. Each arm has a small committed JSON
report and JSONL ledger retaining all 100 query outcomes, including exclusions.
Per-row source accession/hash plus the frozen archive resolve ground-truth records.
All installed library versions, effective defaults, input hashes, harness hashes,
commit/dirty state, OS and configuration are preserved.

Final full command:
`uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v`:
**1,369 passed, 7 skipped, 35 deselected; 82.39% coverage**. All 13 new tests pass,
including hand-counted denominators/ranks, schema/manifest/config drift, strict JSON,
real-file present/absent workflows, low-peak failure accounting and failed-family
non-success. Ruff, formatting, typing, strict MkDocs and commit hooks pass.
Existing SQLite resource warnings remain; passing tests are engineering evidence.

## Left unresolved and next steps

1. Integrate and qualify the independent correctness/provenance/confidence PRs on
   one clean human-reviewed candidate; reconcile processing/cache revisions and
   rebuild stores into new destinations. Repeat this exact frozen benchmark.
2. Add observable per-pair gates: current production search exports qualified
   hits, so precursor/adduct/score-only/peak-only losses cannot all be separated.
   Reports identify this limit; missing per-gate counts are not inferred from no-hit
   rows. Minimum-peak and other observable processing losses are accounted for.
3. Authenticate raw acquisition lineage and additional independent standards,
   batches, instruments, negative mode, adducts and charge states; audit mirror and
   acquisition reuse beyond the declared similarity screen.
4. Run planned mass-only baselines, noise/preprocessing controls, decoy alternatives,
   batch-level calibration and uncertainty on new held-out data. No empirical fix is
   substituted by changing the software formula or tuning this test cohort.
5. Full independent review of the completed harness/results remains required. A
   prior read-only reviewer contributed dataset/spec checks; a later audit found
   the separate tied-score defect but exhausted its quota before a final combined
   review. This draft is not a completed independent scientific validation.

No merge, release, package publication or protected-branch mutation was performed.
