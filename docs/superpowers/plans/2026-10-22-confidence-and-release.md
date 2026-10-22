# Conservative Confidence and Release Preparation Implementation Plan

> For agentic workers: execute inline with test-first changes and one independent final review. The user has explicitly authorized implementation and dedicated PRs; do not request another scope approval.

**Goal:** Make confidence reporting explicit and empirically honest, execute a real-data benchmark through the existing pipeline, and prepare a narrow 1.0 release decision.

**Architecture:** Independent benchmark scripts do not change scientific production scoring. Confidence semantics wrap the existing numerical estimator with nullable estimate/state/method/limitations; invalid numerical inputs fail explicitly. Preserve legacy numerical columns/filtering for compatibility and identify them as diagnostics when unavailable. Release documentation separates supported engineering behavior from empirical scientific validation.

**Spec:** The scientific validation program in the attached scientific-validation-program worktree; current user benchmark, FDR-audit and release-management instructions.

## Constraints

- Preserve primary dirty/concurrent work; use managed worktrees and codex branches.
- No synthetic data presented as experimental evidence; real fixtures carry original source/license/hash/accession.
- No automatic merge, protected-branch mutation, package publication or official release.
- Float64, fixed seeds, missing confidence distinct from zero, real chemistry validators and existing annotation pipeline.
- Full mandatory coverage >=80%, lint/typing and meaningful adversarial tests.

## Task 1 — Confidence semantics

- [x] Add failing tests for explicit unavailable/estimated_uncalibrated/invalid states and independent numerical invariants.
- [x] Reject nonfinite/invalid confidence inputs before ranking; keep legitimate numerical zero distinct from missing.
- [x] Add per-file and per-row machine-readable method, scope, nullable estimate and limitations without asserting calibration or changing valid score ordering.
- [x] Correct README/scoring/results claims, document legacy fields and prove output serialization.
- [ ] Run full gates, review, commit and open dedicated draft PR.

## Task 2 — Experimental benchmark

- [ ] Inventory frozen MassBank release and predeclare data eligibility/selection before scoring.
- [ ] Add failing metrics/manifests/integration tests; implement isolated runner and transparent pipeline observation.
- [ ] Fetch/hash outside Git; document original ground truth and acquisition/duplicate limitations.
- [ ] Execute present/absent libraries and both classical engines, saving every query and failure.
- [ ] Publish small machine-readable results/report, verify gates and open dedicated draft PR.

## Task 3 — Release scope and gates

- [ ] Refresh exact PR heads/CI, packaging/optional-dependency and installation evidence.
- [ ] Classify major features A–E, rank blockers and specify manual PR integration/migration sequence.
- [ ] Make narrowly scoped documentation corrections and create only nonduplicate implementation issues.
- [ ] Verify docs/artifacts, review and open release-preparation draft PR without version/release actions.

## Review focus

Empty decoy null is not empirical calibration; few observations are reported as counts, not a magic sufficiency threshold. Query maxima are not candidate-specific confidence. Invalid/nonfinite inputs cannot silently disappear. Query/file partition and library redundancy can affect estimates. Real-data labels and source independence must not be upgraded by code or unit tests.

## Execution record

A partial independent review demonstrated a valid finite tied-decoy undercount.
The complete equal-score block correction is justified by an independent
1,156-vector-pair oracle; threshold selection may change for affected ties.
Final frozen full suite: 1,393 passed, coverage 82.53%. The real-data benchmark
executes in its separate worktree on clean commit 4dab3c2; all four metric
dictionaries reproduce the earlier diagnostic. PR creation and release handoff
remain the final delivery tasks, with no merging or publication.
