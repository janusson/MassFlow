# FDR and confidence-reporting audit

Baseline: main `ffef50bff913fcf400cb3de72595dab50e4f3fd8`. Review date:
2026-10-22. This change corrects inspectability and proven numerical-input/
serialization defects; it does not supply missing empirical calibration.

## Findings and correction

1. Main computes a pooled best-target/best-decoy query-maxima ratio with a +1
   numerator and reverse-minimum closure. Both sides may include the same query;
   it is not paired winner-take-all TDC. Independent review reproduced a tied-score
   undercount: targets [0.6,0.9,0.9,0.9], decoys [0.6,0.6] gave q(0.6)=2/3;
   the complete-threshold formula gives 3/4. Partial ranks inside a tie were
   allowed into reverse-minimum closure and the per-score lookup selected the
   first descending occurrence. Counts now include each entire tie block before
   closure. Both q fields and threshold selection can change for affected ties.
   An independent enumeration covers all 1,156 pairs of nonempty sorted vectors
   of length 1–4 over three score levels, in addition to named counterexamples.
2. Nonfinite values could silently disappear from query maxima or enter array
   ranking. Arrays now require one-dimensional finite real scores; invalid
   candidate families get an explicit invalid assessment and workflow failure.
3. Empty-null values `q=1/N_target`, `p=1` were described as conservative bounds.
   They are retained only as historical numerical diagnostics; the statistical
   estimate is null with state unavailable. No empirical bound follows from them.
4. A nonempty decoy null was being described as calibrated. A new assessment
   reports estimated_uncalibrated, never calibrated; it records method/version,
   family, unit, sample counts, nullable estimate and unvalidated assumptions.
   All newly assessed unvalidated hits use Annotation_Status=Uncalibrated.
5. Candidate rows inherit the best-target query statistic. The new metadata says
   candidate_specific=false and query_best_target; extra rows are not discoveries
   independently supported by that q value.
6. Streaming serialization converted a numerical q of 0 to NaN through truthiness,
   and an absent response q defaulted to zero. Explicit presence checks preserve
   numerical zero and use NaN for missing experimental streaming confidence.
   This does not make per-packet FDR available or validate the experimental API.
7. README and scoring/results guidance overstated identity, exchangeability and
   calibration, and encouraged threshold relaxation as a remedy. Those claims are
   corrected. Numerical ranking, chemical plausibility, putative annotation and
   empirical confirmation remain distinct.

## Assumptions not established

The source retains target precursor/adduct/other metadata, permutes selected
intensities and jitters fragments uniformly by +/-1 Da by default. The jitter
includes displacements smaller than the scoring tolerance. Entropy preservation
alone does not establish a chemically credible false-match population. Degenerate
intensities can be tapered, so entropy preservation is not universal either.

Candidate eligibility, precursor and fragment tolerances, shifted matching,
minimum-score/peak gates and adduct/RT filters condition which target and decoy
maxima survive. Duplicate acquisitions, repeated structures, query dependence,
per-file partitioning, library composition and routed engine scales can change
numerical estimates. None is repaired by a universal library-size threshold or
by adding a pseudo-count. The new method metadata preserves these limitations.

## Evidence and tests

The initial new assessment/input/serialization test run on unchanged baseline
failed 27 cases and passed one valid numerical-array control. After correction,
the new suite passes; separate red/green tests reproduce zero-to-missing streaming
coercion and absent-to-zero response coercion. Tests include empty/no-target/null
cases, one surviving decoy, score ties, nonmonotonic raw ratios, redundant candidate
rows, target-absent labels and NaN/infinity/non-vector/complex/boolean inputs.

The independent oracle enumerates complete score cutoffs and computes tail counts;
it does not reproduce the implementation's sorting/cumulative-sum steps. Constructed
caffeine integration inputs exercise real files, formula/adduct/5-ppm validation,
SQLite preparation, workflow estimation and native CSV/YAML output. These are
software tests, not measured fragmentation validation.

Golden changes are deliberate **reporting semantics**: annotation_status
values change from Matched/Putative to Uncalibrated and unsupported warning
claims are corrected in the scientific manifest.
Scores, q/p numbers, matched peaks, ranks, FDR counts and input spectra are unchanged.
The older multiprocessing CSV anchor is still compared on all its original
numerical columns, with an explicit five-column confidence extension and the
specified status migration. The independent fixture generator now checks the
same reporting policy; it was not used to regenerate numerical expectations.

Final frozen verification: **1,393 passed, 7 skipped, 35 deselected; 82.53%
coverage** with the mandatory full command. The focused FDR/workflow selection has
98 passing cases, including all 37 new confidence tests. Ruff, formatting, mypy,
strict MkDocs and wheel/sdist build pass. The independent review found the tie,
no-hit export and warning defects but exhausted quota before its final full review;
that review remains required before merging. [Verification metadata](confidence-verification.json)
records log hashes and the excluded intermediate coverage measurement.

## Experimental calibration remains pending

Historical 600-query cross-instrument MoNA outputs have empty-decoy regimes,
zero LTQ calls and unresolved dirty/environment/source-lineage limitations. They
do not validate this null generator or a nominal error bound. A separate frozen
MassBank benchmark is being prepared using measured standard records and known
present/absent structures; its retrieval results must not be treated as FDR control.
A single surviving decoy permits the arithmetic but is sparse evidence, explicitly
recorded as single_decoy_observation. Known-truth independent-batch calibration
and uncertainty remain required before a calibrated state can be justified.

## Compatibility / remaining risks

- Historical empty-null q_value/p_value and fdr_threshold behavior remain for compatibility; tied-score estimates are corrected.
  Consumers must check confidence_state and confidence_q_value. Empty-null legacy
  filtering remains query-count dependent and must not be used to claim error control.
- Direct similarity APIs and older manually exported rows can lack assessments;
  their legacy score labels are not evidence of calibration.
- Invalid input now fails explicitly; callers relying on silent NaN/inf omission
  must correct their input rather than infer confidence from the remaining rows.
- Other source-record, charge, fragment-filter and prefilter corrections are separate
  PRs #101/#103–#107. This change does not integrate them or repair legacy libraries.
- Streaming is experimental, has no per-packet calibration, and uses NaN for
  unavailable protobuf confidence. Full explicit-state protocol migration is deferred.

See [confidence semantics](../user-guide/confidence-semantics.md) for the exact
machine-readable contract and primary literature references. No merge or release.
