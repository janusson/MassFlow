# Scientific validation program for core MS/MS annotation

**Specification v1.0 — proposed experiments, not a validation certificate.**
Audit: 2026-10-21 America/Vancouver (2026-10-22 UTC). Implementation baseline:
`ffef50bff913fcf400cb3de72595dab50e4f3fd8`, matching remote `main` when checked.
No new experimental spectra were acquired and no new annotation benchmark was run
for this specification. Existing saved outputs were recounted independently.

MassFlow has useful software correctness evidence and limited retrospective
reference-library retrieval evidence. **Chemical identification performance and
empirical FDR control are not established for a declared deployment domain.**
A score, a passing test suite, or a nonempty decoy null cannot establish them.

Read the [implementation/evidence audit](implementation-audit.md),
[source assessment](datasets.md), [manifest contract](manifest-contract.md), and
[ordered GitHub-ready issue drafts](issue-backlog.md) alongside this protocol.
The [audit snapshot](evidence/audit-snapshot.json) records source hashes and counts.

## 1. Intended decision and claim boundaries

The initial intended use is **retrospective retrieval of a small-molecule library
structure from centroided, experimental ESI-MS/MS reference spectra**, with explicit
abstention. This does not validate an LC/matrix workflow. Inlet types (LC, flow
injection or direct infusion) are retained and reported separately for queries
and references. The initial benchmark is positive-mode, singly protonated MS2;
expansion to
negative ions, other adducts, multiple charges, instruments, or matrices requires
separate evidence. No quantitative concentration, recovery, LOD/LOQ, diagnostic,
regulatory, peptide-search, or de novo structure-elucidation claim follows.

Three experiments answer different questions:

| Experiment | Library truth | Decision being assessed |
| --- | --- | --- |
| Closed set | At least one independent source acquisition of each query identity is present | Can the true structure be retrieved and ranked? |
| Open set | Every reference equivalent to the query truth is removed | Does the procedure abstain instead of assigning an incorrect structure? |
| Confidence calibration | A prespecified mixture of present and absent known truths | Does the reported acceptance procedure achieve its claimed error behavior? |

Modified cosine can retrieve analogues through precursor-shifted fragments.
An analogue is **incorrect for exact-structure retrieval** even if chemically
interesting; score analogue retrieval separately using independently curated
relationships. Do not relabel incorrect identities as successes after inspection.

A library match supports a candidate annotation under tested conditions. It does
not automatically confer a confirmed structure or a confidence level from an
external framework. Confirmation needs independent evidence appropriate to that
framework and use, including authentic standards and applicable orthogonal
measurements [S1].

## 2. Freeze the study before scoring

1. Record the scientific question, intended domain, identity rules, error costs,
   acceptance criteria, primary endpoints, exclusions, candidate policy, configuration
   grid, and uncertainty method in a dated protocol. Freeze its SHA-256.
2. Resolve the known affected-path software defects listed in the audit before a
   release-validation run. A run on baseline main is a diagnostic experiment with
   known limitations. Do not silently combine PRs or substitute their test results.
3. Freeze the source release, asset hashes, record-level labels, acquisition groups,
   duplicate graph, all exclusions, development/test partitions, and two libraries
   (truth present / truth removed). Publish manifests where rights permit.
4. Select settings on development structures only. Freeze the complete effective
   config, environment lock, clean application commit and harness commit. Any
   scientific change after unblinding creates a new version and a new independent
   confirmatory set; the old result stays in the record.
5. Run the independent scorer baseline and MassFlow on identical frozen data. Save
   all candidates before q filtering, stage counters, best target/decoy scores,
   source-record mappings, failures, logs, final calls, and checksums.
6. Unblind once, compute the prespecified tables, and retain all failed and negative
   results. All selected queries remain in the end-to-end denominator.

A repository change may be diagnostic and useful while still unsuitable for
release validation. Capture `git status`, `git diff` digest, Python/OS/CPU, installed
package versions using distribution names, lock hash, command line, seeds, backend,
worker count, cache construction/reuse state, source hashes and run timestamps.
A dirty SHA alone cannot reproduce uncommitted behavior. Existing provenance
version fields containing requirement expressions must be checked against the
actual installed versions; see the audit.

## 3. Identity, eligibility and independence

### Chemical identity

Primary endpoint: **full standardized InChIKey agreement**, with an archived
structure, standardization policy/version, and independently reviewed mapping.
This is a strict label comparison, not proof that MS/MS resolves stereochemistry.
For stereochemically undefined references or insufficiently verified labels,
report primary-endpoint ineligibility and a separate connectivity endpoint; do not
invent stereochemical truth. Report stereo/isotope failures separately.

Secondary endpoint: first 14 InChIKey characters (connectivity block), explicitly
named connectivity retrieval. Freeze salt/solvate stripping, neutralization,
tautomer and isotope treatment before scoring; retain original structures and
all alias mappings. Do not collapse unrelated ions by nominal mass, formula or
name. Charged structures and mixtures require manual adjudication before inclusion.
Evaluate correct **ion/adduct/charge assignment** separately from structure rank.
A protonated and a sodiated spectrum may share chemical identity while being
separate experimental records with different matching eligibility.

### Experimental record identity

Use a source namespace + immutable accession/version + source-file SHA-256 and
loader ordinal/native scan locator. A full InChIKey identifies a chemical label;
it cannot identify an acquisition. Keep collision energy and its original units,
instrument model/analyzer, polarity, adduct, charge, contributing laboratory,
source experiment, raw file, processing lineage, and library accession per record.
Preserve missing values. Every exported candidate must join to exactly one source
spectrum. Aggregate structure ranks without discarding the contributing record IDs.

### Query eligibility

Select queries using label evidence and acquisition metadata **before seeing
MassFlow scores or whether its gates accept them**. Primary reference-standard
cohort: measured ESI MS2, verified centroided representation (or a documented,
validated centroiding conversion), documented known standard identity, finite
positive precursor, identified polarity/adduct/charge, interpretable acquisition lineage, and legal
access. Records only labelled by another spectral search are a separate cohort.
Exclude MS1, EI, MS3/pseudo-MS2/merged-stage spectra, simulated spectra, unverified
suspects and ambiguous mixtures from the primary cohort. Record every exclusion.

Valid but sparse/noisy spectra remain selected; pipeline rejection is an outcome.
Known metadata defects enter a labelled challenge cohort rather than quietly
shrinking the primary denominator. Unknown truth cannot be counted correct or
incorrect: report it separately, never treat it as a negative.

Keep the evaluator's structure/name/formula truth in a separate label table.
Provide the search process only deployment-available query peaks, precursor,
polarity, charge/adduct if measured/declared, instrument and CE. Remove query names,
formula, SMILES, InChI/InChIKey, answer-bearing accessions and lookup routes from
scoring input. Reference metadata may retain its legitimate library chemistry.
Do not let query truth drive formula-derived filtering or metadata repair.

### Independence and leakage

Maintain separate IDs for chemical identity, vendor standard/sample lot,
experiment/raw acquisition, scan, preprocessing derivative and spectral duplicate
component. All scans, collision energies, consensus spectra, exports, and mirrored
copies of one acquisition belong to the same group. Unknown lineage is **unknown
independence**, not an independent replicate.

Development and test query **structures and connectivity groups** are disjoint;
their acquisition and near-duplicate groups are disjoint too. This also prevents
tuning exposure for the secondary connectivity endpoint. In a closed-set test the test structure
must nevertheless exist in the searchable reference library: this is necessary,
and differs from training/tuning exposure. Its query acquisition must be separate
from reference acquisitions. For optional learned methods, additionally audit the
actual model training structures/spectra; unavailable training data prevents an
unseen-structure claim.

Audit across repositories using original accessions, raw-file/scan links, SPLASH,
exact normalized peak hashes and near-duplicate comparisons. SPLASH alone does
not certify independent acquisitions. Proposed conservative screening rule: flag
pairs with equal polarity/adduct, precursor within 5 ppm, and direct cosine
>=0.99 with >=90% of each spectrum's peaks matched within 0.005 Da. Also flag
same-lineage pairs irrespective of score. These numbers are a **study screening
rule**, not a chemical identity rule. Cluster linked derivatives; if independent
lineage cannot resolve a flagged query/reference pair, exclude it from the strict
independence cohort and retain it in the leakage sensitivity cohort. Audit the
consequence of this exclusion; do not hide the loss of easy legitimate matches.
Freeze the deduplication implementation and rule before evaluating retrieval.

## 4. Protocols

### P0 — Software and provenance qualification

Run the required full coverage gate on the exact candidate release:

```sh
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v
uv run pytest -m core -v
uv run ruff check .
uv run mypy .
uv lock --check
```

These establish software evidence only. Add independent known-answer comparisons
for greedy cosine, shifted matching, mixed frames, ppm boundaries, signed charge,
missing metadata, ties, record collisions, source round-trips and rejected spectra.
Compare the enabled prefilter with exhaustive exact scoring on every small-study
query/reference pair. Compare in-memory, freshly built SQLite, warm managed cache,
iterator/chunk and single/multiple-worker paths. Require identical eligible
candidate identities, matched counts, rankings and q-values within declared float64
roundoff; numerical tolerance must never excuse changed threshold decisions.
Inspect CSV/mzTab-M joins and failed/rejected-query accounting. Legacy stores must
be rebuilt into new paths; their lineage must identify the processing semantics.

### P1 — Closed-set acquisition-independent retrieval

Use one prespecified query acquisition per test structure for the primary analysis.
Retain additional acquisitions for repeated-measurement/stratum analyses. Search
against the entire frozen eligible background library plus independent references
for the true structures, excluding all query acquisition groups and unresolved
cross-set duplicates. Freeze the actual library size and spectrum multiplicity
per structure; do not create a toy library containing only the correct answers.

Run cosine and modified cosine separately. Initial fixed operating configuration:
`ms1_tolerance=0.02 Da`, `resolution_ppm=null`, `ms2_tolerance=0.02 Da`,
`min_score=0.6`, `min_matched_peaks=3`, `rt_tolerance=null`, routing/HNSW off,
`top_n=None`. Declare the modified-cosine default prefilter state and perform its
exhaustive agreement control. Preserve processing defaults from the pinned version
as complete JSON, with `filter_by_mz=false`; do not rely on changing defaults.
These settings are a baseline, not an assertion of instrument suitability.

Use an analysis collection pass with `fdr_threshold=1.0` to retain all
score/match-gate-qualified targets and decoys; capture pre-q internal results as
well, since normal exports omit decoys. Separately run the unchanged deployment
config including its default q threshold. The collection pass is a benchmark
instrumentation mode, not a recommendation to relax production confidence.
For exhaustive candidate-loss diagnosis use the same processed spectra and scorer
with score/match pruning disabled in the **harness**, and label that counterfactual.

Collapse eligible reference acquisitions to structures using maximum score, keeping
all supporting record IDs. Rank distinct structures, not five repeated spectra of
one compound. Deterministic tie order: descending score then standardized identity;
also report pessimistic/optimistic rank bounds for exact-score ties, independent
of correctness. Do not select the correct isomer to break ties. Report both strict
and connectivity metrics, and record-level rank as a separate reproducibility check.

### P2 — Open-set rejection and paired removal

Starting from the same known-truth query panel, remove **all** reference records
sharing the query connectivity block, including full-key variants, salts mapped
by the identity policy, acquisition derivatives and mirrored copies. Verify removal
against the independent label ledger, not only metadata parsed by MassFlow. This
makes the open-set challenge unambiguously absent under both identity endpoints.
Keep near-isobaric and related but different structures to make plausible false
matches possible. List removal counts and changed library composition.

Use both (a) a mixed present/absent batch and (b) an all-absent challenge. Pair
queries across present/removal arms; do not regard the two outputs as independent.
At fixed score/match and q acceptance policies, count an accepted exact-structure
assignment for any absent truth as a false positive. Correct rejection means
**searched and abstained**. A preprocessing failure is an operational failure;
report it separately, with an additional end-to-end no-false-call rate if useful.
No-hit does not establish chemical absence in a sample. Top-k identity success is
undefined as a retrieval objective for the absent set; all forced identity calls
are wrong. Evaluate analogue usefulness only under its separate endpoint.

### P3 — Confidence calibration against experimental truth

Freeze test structure groups into acquisition-balanced batches before scoring.
Create explicit present/absent fractions 0, 0.25, 0.50, 0.75 and 1 as **proposed
stress scenarios**, not estimates of real sample prevalence. Keep total query
count, query quality strata and library construction policy documented. Repeat
batch allocations using stored seeds; reuse of compounds creates dependence, so
repeated allocations quantify design sensitivity rather than new independent
experiments. Include a separate laboratory/source holdout for transfer claims.

Evaluate the **current procedure as implemented**, including score/match truncation,
best-target/best-decoy pooling, ties, per-file calibration, q lookup and target
row inheritance. Do not replace it with winner-take-all TDC or a published decoy
algorithm and claim to have validated MassFlow. Compare alternate procedures in
separately named research arms only. The primary discovery unit is one top-ranked
structure per query. Assess the risk of treating every exported row as a discovery
as a separate endpoint; a query q-value is not candidate-specific evidence.

For each batch, retain best target and best decoy **before and after each gate**,
including no-hit sentinels. Tabulate how often known incorrect target maxima and
decoy maxima survive. On absent queries, compare their score distributions,
matched counts, precursor eligibility, peak density and score tails within strata.
This probes exchangeability; matching entropy alone cannot establish it. The
published fragmentation-tree decoy work [S2] is relevant precedent, not validation
of MassFlow's jitter/permutation generator. Li et al. [S3] also caution that
computational decoys can underestimate errors.

Evaluate fixed q thresholds `{0.01, 0.05, 0.10, 0.20, 1.0}` and the deployment
threshold, without choosing a favorable one after test inspection. Plot reported
threshold versus observed FDP and number of accepted queries, including zero-call
points. Plot raw decoy/target tail counts and empty-null frequency. A q-value is a
set-level error measure, not the posterior probability that a particular structure
is incorrect [S4]. Do not apply reliability-bin/Brier-score interpretation to it.

Every empty-null batch is labelled uncalibrated. Current code can give every target
query `q=1/N_target` when decoys are absent; increasing query count can therefore
make rows pass a fixed cutoff without adding null evidence. Test this explicitly
by partitioning/recombining the same known-truth queries. This is a design
sensitivity test, not extra independent sample size. Nonempty nulls still require
empirical calibration. Never tune jitter, gates, or query batch size to manufacture
nominally significant calls.

### P4 — Candidate and filter loss decomposition

For every selected query, retain truth-presence indicators at ordered checkpoints:
raw frozen library; post-ingestion/physical-validation library; post-metadata
eligibility; precursor candidacy; accelerator candidacy; exact scoring; score and
matched-peak gates; adduct/RT gates; distinct-structure top-1/top-5; q acceptance.
Match actual execution order in the pinned implementation and label counterfactual
reordering as such. Assign a first-loss reason; retain overlapping reasons too.

Compare each accelerator to exhaustive scoring on the same input and eligibility
rules. Precursor filtering loss is not synonymous with 5-ppm model validation:
report separately query metadata rejection, reference chemistry rejection, and
query-reference search window loss. Modified cosine lacks cosine's hard precursor
window, so candidate expansion is expected and must be evaluated for both true
retrieval and false calls. If raw-library validation aborts the run, record the
abort before curation; any curated rerun is a separately versioned library.

### P5 — Strata and robustness

Prespecify instrument analyzer/model, ionization method, inlet type, source
laboratory, acquisition family,
polarity, adduct/charge, CE value/unit, precursor mass, usable peak count, entropy,
structural class and library multiplicity. Report counts and uncertainty for every
stratum, including unknown metadata and zero-call strata. Report same-instrument,
independent acquisition and different-instrument results separately. Never convert
NCE to eV or compare numeric energies across instruments without a validated mapping.
The pilot does not support these expanded claims until adequate eligible records
exist; empty strata remain unsupported.

Use development data to select instrument-appropriate MS1 and MS2 tolerance ranges
independently. Then freeze a 2x2 factorial sensitivity comparison (for the pilot,
MS1 `{0.01,0.02}` Da x MS2 `{0.01,0.02}` Da); do not attribute a paired change in
both tolerances to either alone. The repository's 5-ppm metadata rule stays fixed;
its empirical appropriateness is evaluated through rejection accounting, not
silently weakened.

On the unchanged held-out queries, compare default preprocessing with a locked
identity-preserving normalization-only analysis, 1% base-peak noise removal and
top-50 peak reduction, each one factor at a time. Document exact operation order,
renormalization and resulting empty spectra. Additional synthetic perturbations
of real query spectra: fragment shifts `{-10,-5,0,5,10}` ppm; independent precursor
shifts at the same values; remove `{0,10,30,50}%` of fragments; add `{0,10,30}%`
background peaks sampled from a disjoint measured blank pool. Use five fixed
seeds per stochastic condition (42–46), peak-count rounding down, no label-dependent
selection, float64 arrays and stored original/perturbed hashes. Declare this grid
as sensitivity analysis rather than a physical instrument-error model. If no
measured blanks are available, defer that arm. Synthetic two-spectrum chimeras
(10% and 50% intensity contribution, normalized component TIC) are stress controls,
not measured co-isolation validation. Preserve the source groups for both parts.
Confirm unexpected effects later using independent experimental acquisitions.

## 5. Metrics with explicit denominators

Let `N` be all prespecified labelled test queries, `N_present` those with truth in
the frozen raw reference library, `S` successfully searched, `C` queries with any
qualified target candidate, and `A(t)` queries making one accepted top-1 structure
call under policy `t`. Let `TP(t)` be correct accepted calls and `FP(t)` incorrect
accepted calls; an abstention is never a TP. Let `H_k` indicate that truth is among
the first `k` **distinct structures** before confidence filtering. No numerical
rate is reported with denominator zero: use null plus raw counts.

| Metric | Definition and interpretation |
| --- | --- |
| Closed-set end-to-end top-k accuracy | `sum(H_k) / N_present`, k=1,5; selected queries rejected or without candidates contribute zero |
| Conditional top-k | `sum(H_k) / searched_present`; report alongside the end-to-end result |
| Candidate retrieval coverage | Queries with any true structure among candidates / `N_present`; specify checkpoint |
| Call coverage | `A(t)/N`, also `A(t)/S`; candidate coverage `C/N` is a different measure |
| Accepted-call precision | `TP/(TP+FP)` for the declared strict/connectivity endpoint and mixed cohort |
| Identification recall | Correct accepted present-query calls / `N_present`; wrong or missing call is a miss |
| False negatives | `N_present - correct accepted present calls`; distinguish wrong assignment, abstention and operational failure |
| Observed FDP | `FP/max(A,1)` by convention 0 for A=0; show `A=0` and precision undefined so no-call runs cannot pass utility criteria |
| Estimated FDR at score cutoff | Recompute MassFlow's `(1+D_ge)/(T_ge)`, clipped to 1; show raw counts, gates and q closure |
| Open-set false assignment rate | Incorrect accepted absent-query calls / `N_absent`; additionally condition on successfully searched absent queries |
| Open-set searched rejection rate | Successfully searched absent queries with no accepted call / `searched_absent`; failures reported separately |
| Specificity | Same searched-absent rejection fraction, using the explicit detection-task convention |
| Stage survival | Queries retaining >=1 true reference at stage j / `N_present`; conditional loss at j = first lost at j / truth-surviving prior stage |
| Ranking loss | Truth retrievable but not top-k / truth retrievable; retain total selected denominator too |
| Procedure FDR | Expectation of FDP over the declared sampling/batch-generating process; estimated only with appropriately independent batches |

A wrong call on a present query is both an FP assignment and an FN for the correct
identity; explain this paired interpretation. Do not construct invented
`WRONG:<name>` classes or use weighted multiclass accuracy as identification
precision. Candidate-row counts measure computational output, not discoveries.
Report acquisition-weighted metrics and a structure-macro average (mean of each
structure's rate), with unique-compound counts; multiple CEs are not independent
compounds. Report per-structure discovery FDR separately if deduplicating calls
across queries; query-level q-values do not establish that error rate.

## 6. Uncertainty and statistical decisions

Report counts and 95% uncertainty intervals with each primary rate where the
sampling model supports them. Wilson intervals for retrieval/coverage and exact
binomial one-sided bounds for rare errors apply only to a **fixed per-query
decision rule under an explicitly justified independent Bernoulli sampling model**,
conditional on the fixed library. Unique structures alone do not justify this
model, and it does not apply to file-level q-based acceptance: those decisions
share a data-dependent threshold. If sampling assumptions are unsupported, report
descriptive counts and design sensitivity rather than an inferential binomial bound.

For q-based pilot analyses as well as expanded repeated acquisitions, resample
whole connectivity groups (all full-key variants and conditions together),
2,000 bootstrap replicates, seed 42. Where stratified resampling is used, assign
each group to its prespecified primary query's source/polarity stratum; all paired
acquisitions follow that group's draw. Record the anchor query and weights, so
cross-stratum acquisitions cannot be independently resampled. Recompute
acceptance sets and q-values from resampled best-target/decoy maxima, preserving
resampling multiplicities as weights; resampling final accepted rows alone misses
selection uncertainty. Within-query ranks come from the unchanged frozen library.
Identical resampled acquisitions are weights, not new measurements or collapsed
query IDs. Pilot bootstrap results are sensitivity analyses, not proof of control.
Use batch/laboratory-level resampling for transfer and distinguish sensitivity
intervals on reused data from independent validation intervals.

Observed FDP for one fixed batch is known conditional on its adjudicated labels.
An interval concerns generalization to other sampled structures/batches, not
uncertainty in counting that batch. Record disputed-label adjudications and give
best/worst-case sensitivity bounds; a statistical interval does not repair bad
truth. At zero accepted calls FDP=0 is bookkeeping, not evidence of useful control.

For a claimed operating error bound, set a scientifically justified allowable error
and minimum useful retrieval/coverage **before** testing, based on the intended
cost of false calls versus misses. Compare a one-sided upper uncertainty bound
with that limit; also require a minimum useful number of calls. If precision is
insufficient, conclude inconclusive rather than passing or moving the threshold.
As a sample-size illustration under an ideal independent fixed-rule Bernoulli
model only, zero errors among 50 independent accepted calls has a 95% one-sided
binomial upper bound about 5.82%; 299 independent accepted
calls with zero errors are needed to put that bound below 1%
(`1 - 0.05**(1/n)`). This is not proof of FDR control or a universal sample-size
requirement. Correlation, repeated compounds, label errors, batches and strata
can require substantially more evidence.

Evaluate q calibration across all fixed thresholds with intervals and explicit
multiplicity scope. Use a single prespecified primary operating point for a
confirmatory claim; use a simultaneous band or prespecified multiplicity correction
if claiming multiple strata/thresholds simultaneously. Do not declare calibration
from a nonsignificant deviation or overlap of wide intervals. Report unresolved
regions as unsupported. Paired algorithm comparisons use compound-level paired
bootstrap differences, not independent-spectrum tests. Baselines are compared on
identical eligible denominators and operating policies.

## 7. Required controls and baselines

| Control/baseline | Purpose and allowed interpretation |
| --- | --- |
| Exact self-match, kept outside test accuracy | Plumbing/serialization positive control; disclose inevitable leakage |
| Independent standard acquisition, same structure | Real retrieval positive control |
| Truth removal plus isobaric/isomeric decoys from measured compounds | Real open-set false-call challenge |
| Wrong polarity, contradictory charge/adduct, missing precursor and metadata | Failure/rejection/provenance behavior; constructed controls cannot establish prevalence |
| Label permutation after scoring | Evaluator sanity check only; not a physically realistic null or FDR estimate |
| Measured blanks and held-out unrelated standards | Operational false calls; no exhaustive chemical-negative truth inferred from a blank |
| Library multiplicity: natural, one reference/structure, controlled duplicates | Detect score/opportunity inflation and traceability failures; do not tune multiplicity on test outcomes |
| Independent matchms 0.33.1 greedy exhaustive search | Detect wrapper/prefilter/ranking differences with identical preprocessing and gates |
| Maximum-weight one-to-one matching on a small subset | Distinguish greedy assignment approximation from an implementation error; not assumed to be the production algorithm |
| Precursor-only nearest eligible reference | Measures information added by MS/MS beyond mass, with deterministic ties and no chemical label access |
| Fixed-seed random eligible structure rank | Checks candidate density and evaluator behavior; retain all failures |
| Published decoy method, if fully reproduced | Research comparator for error estimation [S2]; never borrowed calibration for the current generator |

Fixed-count/shuffled-intensity/jittered spectra are software stress inputs, not
real negative chemicals. No decoy generator is accepted solely because its
entropy, peak count or precursor distribution matches the targets.

## 8. Smallest useful benchmark: MVB-1

**Proposed, not executed:** a MassBank frozen-release pilot with **20 development
and 100 test structures from 120 distinct connectivity groups**, one query per
structure, each with a traceable
independent reference acquisition. Select positive-mode `[M+H]+` experimental MS2
with reference-standard label evidence, CC0/CC BY record rights, full-key identity,
verified centroided representation, ESI ionization, known analyzer and CE units for
both query and reference. Record inlet types and report cross-inlet retrieval
separately; do not claim LC validation from flow-injection/direct-infusion data.
Prefer different laboratories/instruments; a same-lab
independent acquisition cohort must be reported separately. CE variants of one
injection alone do not meet strict acquisition independence.

Freeze MassBank release `2026.03`, resolved commit
`705afb7bccc3b2c42410a744eef73674716a60ef` (verified tag), plus every asset SHA-256.
Inventory before asserting that the requested panel exists. First choose one
eligible full-key truth per connectivity group by ascending
`SHA256("massflow-mvb1-truth-v1|" + full_inchikey)`. Then sort connectivity groups
by `SHA256("massflow-mvb1-v1|" + connectivity_key)`; choose the first 20 for
development and next 100 for test, retaining the entire ordered eligible/reserve
ledger and the within-group choices. Never split variants of one connectivity
group across development/test or present/absent assignments.
Within each selected structure choose query/reference groups by sorted immutable accession
among qualifying independent pairs, using a frozen instrument-priority list
(QTOF query / Orbitrap reference, then the reverse, then explicitly separate
same-analyzer acquisitions). No fallback based on matching success.

Library L0 contains all rights/metadata-eligible positive protonated experimental
MS2 records from that release except test/development query acquisition groups and
unresolved cross-set duplicate components. Deduplicate mirrored acquisitions,
retaining independent repeated references and their multiplicity. Preserve all
background structures; report the resulting library size.

Library L50 removes all connectivity-equivalent records for the first 50 test
groups in the frozen order. Because each test query has a different connectivity
group, the other 50 remain present. Search the same 100 queries against L0 (all
present) and L50 (50 present/50 absent), with both classical engines: **four primary
search runs**. Construct development libraries using only development absence
assignments so test labels do not tune settings. Save P4 funnels, top-1/top-5,
precision/recall/coverage, paired truth-removal behavior, all target/decoy maxima,
q/FDP tables and strata. A separate all-absent library is a required P2 extension,
not one of these four minimal runs. P0 controls and the deployment-policy pass
remain mandatory qualification checks.

This pilot is sufficient to exercise an honest retrieval/rejection evaluation and
expose gross gate/calibration problems. It cannot validate a 1% FDR claim,
comprehensive instrument performance or matrix generalization. If 120 qualifying
structures/pairs do not exist, publish the inventory and the shortfall; reduce
scope by a new protocol amendment or use a rights-cleared independent source,
not synthetic replacement or relaxed undocumented independence. Expand calibration
sample size only after observing pilot precision and predeclaring a new design.

## 9. Release evidence categories

| Category | Evidence required | Current audit recommendation |
| --- | --- | --- |
| Validated stable functionality | Exact version/domain/config/data identified; software qualification; independent held-out experimental results with adequate uncertainty; prespecified error and utility criteria met; provenance complete | No broad chemical-identification or calibrated-FDR claim qualifies from the inspected evidence. Narrow arithmetic/I/O invariants can be called software-verified |
| Functional but scientifically unvalidated | Qualified software behavior, transparent limitations, no unsupported identity/FDR claims | Classical library retrieval and numerical q/p outputs, subject to affected-path correctness fixes and explicit calibration warnings |
| Experimental | Explicit opt-in, isolated outputs, model/training or method evidence incomplete | ML/meta-engine/routing/network/analogue inference and confidence transfer across untested strata |
| Correction required before release of affected claims/paths | Reproduced correctness/provenance defect, unreconciled metadata/units, materially misleading scientific claim, or inability to reproduce | Review/integrate affected-path fixes #101/#103–#107; reconcile cache semantics; correct confidence/identity wording and RT units; repair benchmark/provenance deficiencies before evidentiary use |

Release managers must decide whether an affected path is corrected or explicitly
excluded/disabled and documented. A draft PR is not a correction on main. Green CI
is necessary software evidence under repository policy, not a release approval.
Do not advertise `Matched` at score >=0.9 as a confirmed identity, entropy
preservation as calibrated FDR, or a 2,000-entry library as sufficient statistical
power. Scope any future validation statement to the actual domain and frozen
procedure. Human review is required for any merge or release; this program does
neither.

## 10. Limitations and next work

Public library labels may be wrong or incomplete; source mirroring and undocumented
replicates limit independence. Full-key truth does not guarantee stereochemical
selectivity. A standard-library pilot omits extraction losses, chromatographic
coelution, ion suppression, endogenous concentration effects and real chimeras.
Prospective blinded standard mixtures in relevant matrices, including blanks,
spikes and independently acquired confirmatory spectra, are the next evidence
layer after the retrospective pilot. Their composition, practical replicates and
error limits must be designed for a stated application; they are not supplied by
unit tests or this specification.

### Primary references

- **S1:** Schymanski et al. (2014), *Identifying Small Molecules via High Resolution
  Mass Spectrometry: Communicating Confidence*, DOI
  [10.1021/es5002105](https://doi.org/10.1021/es5002105). Bibliographic identity
  verified through [PubMed](https://pubmed.ncbi.nlm.nih.gov/24476540/); publisher
  full text was inaccessible during this audit, so no detailed level assignment
  is inferred here.
- **S2:** Scheubert et al. (2017), *Significance estimation for large scale
  metabolomics annotations by spectral matching*, DOI
  [10.1038/s41467-017-01318-5](https://doi.org/10.1038/s41467-017-01318-5).
  Read through [Europe PMC full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5684233/fullTextXML),
  Results and Methods. Its algorithms and evaluations differ from MassFlow's.
- **S3:** Li et al. (2021), *Spectral entropy outperforms MS/MS dot product
  similarity for small-molecule compound identification*, DOI
  [10.1038/s41592-021-01331-z](https://doi.org/10.1038/s41592-021-01331-z),
  [author-hosted full text](https://yuanyueli.github.io/project/spectral_entropy/Yuanyue-spectral_entropy.pdf),
  Discussion pp. 6–7. Similarity evidence does not validate entropy-preserving decoys.
- **S4:** Storey and Tibshirani (2003), *Statistical significance for genomewide
  studies*, DOI [10.1073/pnas.1530509100](https://doi.org/10.1073/pnas.1530509100),
  [author-hosted paper](https://genomics.princeton.edu/storeylab/papers/Storey_Tibs_PNAS_2003.pdf).
  General q-value interpretation, not the estimator implemented by MassFlow.
- **S5:** Huber et al. (2020), *matchms — processing and similarity evaluation of
  mass spectrometry data*, DOI [10.21105/joss.02411](https://joss.theoj.org/papers/10.21105/joss.02411).
  Software reference; benchmark baseline must additionally pin the exact version.
