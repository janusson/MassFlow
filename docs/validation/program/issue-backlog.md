# Ordered GitHub-ready validation issues

These are local drafts, **not newly posted issues or claims of completion**.
Use existing issues when scopes overlap; no automatic merge/release is authorized.
Dependencies are scientific gates, not permission to combine unrelated patches.

## V01 — Qualify one exact baseline and reconcile scientific claims

Priority: release-critical prerequisite. Reuse #95 and PR #101/#103–#107 for their
existing fixes; link rather than duplicate their implementations. Related #92.

**Problem:** Unmerged correctness/provenance changes and overbroad claims prevent a
stable interpretation of new benchmark results.

**Scope:** Choose a human-reviewed release candidate; resolve affected charge,
prefilter, fragment-window, missing-ID and capped-search defects; reconcile combined
processing revision/cache behavior. Audit RT unit contract. Replace claims of
confirmed identity, universal 5-ppm soundness, isotope tie-breaking and entropy-decoy
calibration with precise supported behavior. No new scoring model.

**Acceptance criteria:**

- [ ] Record clean application SHA, lock hash and installed dependency versions.
- [ ] Run full mandatory coverage, core, lint and typing gates on that exact SHA.
- [ ] Show prior failing cases and passing corrected controls; do not regenerate
      goldens to hide numerical disagreements.
- [ ] Prove source-record traceability and rebuild stores into new paths.
- [ ] Resolve RT seconds/minutes or exclude RT claims and disable that arm.
- [ ] Publish an evidence/claim table; software verification remains distinct from
      experimental accuracy/calibration. No merge is performed by this issue draft.

Deliverables: qualification report, check logs and bounded residual-risk list.
Depends on: human review/integration decisions. Blocks: confirmatory validation;
diagnostic runs may proceed with explicit known-defect labels.

## V02 — Freeze a rights-cleared reference-standard corpus and identity ledger

Priority: first data work. Depends on V01 for software runs, but inventory can start
independently. Primary source: MassBank 2026.03; alternatives per datasets.md.

**Problem:** Repository names and InChIKeys do not establish measured-standard truth,
acquisition independence or redistribution permission.

**Acceptance criteria:**

- [ ] Freeze source tag/commit, byte hashes, original records, rights and attribution.
- [ ] Inventory measured/simulated/merged status, ionization method, inlet type,
      centroid/profile representation and required metadata before scoring.
- [ ] Keep chemical identity separate from source-record/acquisition/duplicate IDs.
- [ ] Review full-key/connectivity mappings, stereo/salt/isotope limitations and labels.
- [ ] Audit mirrors, same-acquisition CE variants and near duplicates across sources.
- [ ] Produce protocol-schema manifests, full exclusion ledger and deterministic
      development/test/reference splits; report whether 120 independent pairs exist.
- [ ] Verify two fresh retrievals produce identical frozen bytes; unresolved rights
      block redistribution, not silently convert to an assumed open license.

Deliverables: immutable manifest/record ledger, rights report and feasibility counts.
No fabricated replacement spectra if the requested cohort cannot be assembled.

## V03 — Add an independent query-accounting and metric evaluator

Priority: required harness work. Reuse/reconcile existing #97 and uncommitted
`benchmark_reporting.py` work before implementing another evaluator. Depends on V02.

**Problem:** Candidate rows, accepted calls, searched queries and selected queries
have different denominators; repeated structure records can fill top five slots.

**Acceptance criteria:**

- [ ] One terminal state per selected query including whole-file failures.
- [ ] Distinct-structure top-1/top-5 with deterministic ties plus tie bounds.
- [ ] Strict full-key and explicitly secondary connectivity endpoints.
- [ ] Correct precision/recall/coverage/FDP/open-set definitions with raw counts;
      zero denominators serialize as null; no invented WRONG labels.
- [ ] Freeze evaluator fixtures with hand-calculated ranks, missed truth, false
      calls, rejected queries, duplicate structures and zero-call batches.
- [ ] Preserve all source-record IDs and private label separation.
- [ ] Recount historical outputs independently; retain unresolved LTQ behavior and
      environment/label limitations under #97 instead of silently replacing results.

Deliverables: versioned standalone evaluator, meaningful tests and accounting tables.
No production scoring changes are part of this issue.

## V04 — Measure candidate and preprocessing losses against exhaustive search

Priority: mechanism before optimization. Related #70/#71/#95. Depends on V01–V03.

**Acceptance criteria:**

- [ ] Retain truth presence and first-loss reasons at every P4 checkpoint.
- [ ] Compare accelerator on/off to exhaustive exact greedy scoring on identical
      processed inputs; no dropped eligible true pairs allowed by a claimed exact
      accelerator. Document floating-point threshold handling.
- [ ] Separate 5-ppm metadata rejection from precursor search-window losses.
- [ ] Compare MS1/MS2 tolerances using a locked 2x2 design, not a two-knob causal claim.
- [ ] Report losses by source/analyzer/adduct/CE/missing-metadata group, including zeros.
- [ ] Retain failed raw-library builds and version any curated rerun.

Deliverables: candidate funnel tables, paired diffs and bounded hypotheses. A
physics/notation mismatch is investigated before proposing tolerance changes.

## V05 — Execute MVB-1 closed-set and paired open-set benchmark

Priority: smallest useful experimental retrieval evidence. Depends on V01–V04.

**Acceptance criteria:**

- [ ] Preregister 20 development / 100 test structures from distinct connectivity
      groups, with group-level split/absence assignment, or document a prospective
      amendment if feasible independent pairs are fewer.
- [ ] Freeze L0 and L50 with realistic background and verified absence assignments.
- [ ] Run four primary searches (two engines x two libraries), plus P0 controls and
      separately labelled deployment-policy checks; archive all candidates/maxima.
- [ ] Report all primary metrics, uncertainty and per-source counts; label analogues
      incorrect for exact-identity endpoints.
- [ ] Compare exhaustive matchms, precursor-only and random eligible baselines.
- [ ] State that this pilot is not adequate to certify 1% FDR or matrix performance.

Deliverables: frozen run bundle and report distinguishing observed/proposed/unknown.
Stop if leakage or truth adjudication fails; retain the negative audit result.

## V06 — Validate confidence semantics and empirically challenge the decoy null

Priority: blocks calibrated-FDR claims. Reuse #68/#91; #91 remains analysis-only.
Depends on V05 and a separately frozen precision/sample-size plan.

**Problem:** Best-target/best-decoy pooling, gate truncation, empty-null rank values,
per-file families and candidate-row inheritance have untested empirical meaning.

**Acceptance criteria:**

- [ ] Independently recompute current q/p formula and lookup on saved maxima.
- [ ] Compare estimated and observed FDP at locked thresholds with counts and
      uncertainty; distinguish a fixed-batch FDP from expected procedure FDR.
      Recompute q during group resampling; do not use binomial exact intervals for
      dependent q-based calls merely because query structures are unique.
- [ ] Evaluate mixed/all-absent batches, actual wrong target tails, gate attrition,
      empty-null frequency, file partitioning and seed sensitivity.
- [ ] Evaluate one top structure/query separately from all exported candidate rows.
- [ ] Treat current decoys as unvalidated until empirical evidence supports the null;
      entropy or score-scale similarity alone is not acceptance.
- [ ] Size independent cohorts for a declared error/utility decision; no universal
      library-size threshold and no declaration that nominal 1% implies actual 1%.
- [ ] Preserve failures and unsupported strata. Alternative estimators/decoy methods
      require a separate research protocol and implementation review.

Deliverables: calibration/uncertainty report or a precise negative/inconclusive
finding. Do not patch thresholds to force significance or substitute new software
for missing empirical evidence.

## V07 — Expand acquisition strata and noise/preprocessing transfer tests

Priority: after pilot interpretation. Depends on V05; calibration claims depend V06.

**Acceptance criteria:**

- [ ] Independently held-out source/laboratory cohort; both polarities and supported
      adduct/charge combinations only where real labelled records exist.
- [ ] CE units and acquisition derivatives audited; no NCE/eV equivalence assumed.
- [ ] Run prespecified noise/preprocessing grids with frozen seeds and paired
      structure-level uncertainty; label synthetic perturbations as such.
- [ ] Preserve missing strata and insufficient precision as limitations.
- [ ] If ML/meta-engines are assessed, inventory training overlap and score semantics
      first; no borrowed calibration from the classical engines.

Deliverables: stratum evidence matrix and scope-limited claims, not a pooled
universal accuracy number.

## V08 — Prospective blinded standard-mixture validation and release evidence review

Priority: application-specific confirmation after retrospective design is useful.
Depends on V05–V07 and an explicitly stated analytical use case.

**Acceptance criteria:**

- [ ] Design independent standard identities, blinded mixture composition, blanks,
      matrix spikes and acquisition days/laboratories for the intended workflow.
- [ ] Declare identification truth, relevant orthogonal confirmation, false-call
      cost and missed-call cost before acquisition/unblinding.
- [ ] Choose sample/replicate counts and useful precision from those decisions;
      distinguish technical replicates from independent compounds/batches.
- [ ] Publish accepted, wrong, absent, rejected and inconclusive outcomes with
      provenance and uncertainty; no concentration or LOD claim without its own study.
- [ ] Classify each exact-version/domain feature using the four release categories
      in the program; require human review, with no automatic merge/release.

Deliverables: prospective study protocol/results and a human-reviewed evidence
package, or a documented reason why the intended claim remains unvalidated.
