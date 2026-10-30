# Implementation and evidence audit

Status: inspected source at `ffef50bff913fcf400cb3de72595dab50e4f3fd8` on
2026-10-21 America/Vancouver; remote `main` matched. This is an audit, not a new
experimental performance result. See the [program](README.md) for proposed tests.
The main working checkout contains concurrent uncommitted benchmark/processing
work; it was read but not modified. This specification uses an isolated worktree.

## Supported annotation types

| Surface | Actual annotation/evidence | Boundary |
| --- | --- | --- |
| Stable `annotate`, cosine | Reference-spectrum candidate rows with score/matched peaks, precursor metadata, reference identity and per-query q/p fields | Small-molecule library retrieval; no de novo elucidation, concentration, or peptide identification |
| Stable `modified_cosine` | Direct and precursor-shift-aligned spectral similarity; can retrieve different-mass analogues | A shifted high score does not establish the same structure or a calibrated analogue relationship |
| CSV / experimental mzTab-shaped | Multiple rows per query; `Matched` for score >=0.9, otherwise `Putative`; empty-null hits `Uncalibrated`, no-hit processed queries `Unknown` | Labels are implementation categories, not validated external identification-confidence levels |
| Experimental consensus/cascade | Weighted score aggregation / staged candidate reduction; optional routing and ML engines | Different score meanings and training exposure; no independent calibration demonstrated |
| Experimental network branch (#80) | Direct evidence plus graph/context hypotheses and tiers on a separate branch | Not present as a stable main capability; never count propagated hypotheses as independent confirmed identifications |
| Formula/adduct/isotope model helpers | Chemical plausibility calculations and theoretical isotope envelopes | Plausibility is not empirical identity verification; no observed isotope tie-break in classical/consensus search was found |

`annotation_tier` is initialized null in classical results. `structural_similarity`
in consensus is populated with a best individual engine score; its name does not
make it an independently measured molecular-structure similarity. `analog_search`
is a configuration field with no active use found outside configuration on the
pinned source; modified-cosine shift behavior occurs independently of that flag.
Do not advertise an additional validated analogue workflow on this basis.

## Scoring and filtering

Source locators refer to the pinned baseline; links will remain stable:
[similarity.py](https://github.com/janusson/MassFlow/blob/ffef50bff913fcf400cb3de72595dab50e4f3fd8/src/MassFlow/similarity.py),
[processing.py](https://github.com/janusson/MassFlow/blob/ffef50bff913fcf400cb3de72595dab50e4f3fd8/src/MassFlow/processing.py),
[models.py](https://github.com/janusson/MassFlow/blob/ffef50bff913fcf400cb3de72595dab50e4f3fd8/src/MassFlow/models.py),
[config.py](https://github.com/janusson/MassFlow/blob/ffef50bff913fcf400cb3de72595dab50e4f3fd8/src/MassFlow/config.py).

- Core engines instantiate matchms `CosineGreedy` and its modified-cosine greedy
  implementation (installed lock version 0.33.1), fragment tolerance in Da. With
  default matchms exponents (m/z power 0, intensity power 1), score is the sum of
  products of greedily paired intensities divided by the product of the full
  spectral L2 norms. Greedy one-to-one pairing need not be globally optimal.
  Modified cosine allows direct and precursor-mass-difference-shifted matches.
  Validate mixed frames, shared peaks and charge/adduct dependence explicitly.
- Cosine's sparse-array path applies a query/reference precursor window: fixed Da
  or query-scaled `resolution_ppm`. Standard reported mass error uses reference
  m/z in its ppm denominator. Missing precursor handling differs at the direct
  scorer boundary versus validated file ingestion. Modified cosine does not use
  cosine's hard MS1 window; a larger candidate pool is expected, not necessarily
  increased identification sensitivity.
- Default-on Numba modified-cosine prefilter checks peak/neutral-loss match counts
  before exact scoring. PR #104 demonstrates mixed-frame false pruning in main.
  Candidate accelerators require comparison to exhaustive scoring, not just speed.
- Ingestion and processing perform metadata repair, optional intensity/fragment
  filtering, peak reduction and normalization. Structural claims trigger
  formula/structure/adduct-based precursor checks (repository policy: 5 ppm).
  Missing structural context skips parts of this check; a query's unknown true
  chemistry is not tested against every candidate by the model validator.
  Raw reference failures can abort `annotate`; `db build` instead curates/rejects.
  Explicit persistent stores are trusted and do not automatically revalidate.
- Search applies score and matched-peak thresholds; main truncates `top_n` before
  the adduct/RT checks (PR #106). Default workflow requests unlimited candidates,
  so do not claim that capped-API defect explains every benchmark failure.
- Adduct compatibility checks ion-mode sign; it does not require identical adduct
  species or compatible charge magnitudes for every direct scorer pair. Structural
  ingestion validation has the separate signed-charge defect demonstrated in #103.
- RT filtering directly compares stored retention-time values with `rt_tolerance`.
  Its config describes minutes while repository spectrum governance specifies
  seconds, and this code does not convert units. This is a source-level contract
  discrepancy; no new empirical RT failure was reproduced in this audit. Disable
  RT in the initial benchmark pending a documented end-to-end unit resolution.
- Native SQLite stores full spectrum metadata and float64 peak arrays. Chemical
  identity is not inherently a unique experimental-record ID. Main can emit
  `reference_id="None"` or repeated IDs; #107 contains synthetic reproductions,
  record identity assignment and serialization/provenance safeguards. Legacy
  stores and general append/merge collisions still need explicit handling.

## Confidence calculations: precise current procedure

`generate_decoys` (similarity.py around lines 649–842) makes one decoy per target:
copy precursor and metadata; retain all peaks; permute intensities among peaks
above a 1%-of-base-peak floor; jitter fragment positions uniformly by +/-1 Da
(default); clip positive; sort. Uniform/degenerate intensities may be tapered,
so entropy is not preserved for every case. RNG derives from seed and peak-array
bytes, invariant to chunk boundaries. The jitter includes arbitrarily small
shifts: source comments claiming that no fragment can remain within scoring
tolerance are stronger than the implemented uniform distribution guarantees.

`calibrate_query_level_fdr` (around lines 1000–1099) retains the maximum qualifying
target score and maximum qualifying decoy score per query. Queries without a
qualified hit on one side do not enter that side's finite-score array. Both maxima
can enter: this is **not a paired winner-take-all competition**. Both arrays are
pooled and ranked. At threshold `t`, the implementation uses:

```text
T(t) = count(best target score >= t)
D(t) = count(best decoy score >= t)
FDR_hat(t) = min(1, (1 + D(t)) / T(t))    [1 if T(t) = 0]
q(s) = min over thresholds t <= s of FDR_hat(t)
p(s) = (1 + count(finite best decoy scores >= s)) / (1 + N_finite_decoy)
```

Decoys precede targets at tied scores; cumulative raw ratios are reverse-monotone
closed. A per-score lookup makes equal best-target scores share q. Empty decoy
arrays give p=1 and all target q-values `1/N_target` after closure (when targets
exist). The code and sidecar flag this as `fdr_uncalibrated`; **it is an arithmetic
rank output, not a demonstrated conservative statistical error bound**.

Workflow `_process_single_file` calls calibration after assembling all chunks for
that input file. It attaches the query's best-target q and p to **every target
candidate row** and accepts all such rows if q <= `fdr_threshold`; p is diagnostic
only. Decoys are removed from normal output. Thus the calibration family is an
input file, the statistic is a best-target query score, and exported candidate
rows are not independent discoveries. A low-ranked incorrect row can inherit a
q-value determined by a different, higher-ranked target for the same query.
Changing file partitioning can change q outputs. The study must test the actual
one-call/query decision separately from all-row exports.

The source claims pooling heterogeneous engines is valid because of null
exchangeability. Using the same engine within a target/decoy pair does not itself
establish this assumption; routed score distributions and gate asymmetry require
empirical examination. A warning that fewer than about 2,000 library entries is
statistically weak does not supply a validated sufficiency threshold. Statistical
power depends on independent query/error counts, null quality, library composition
and the decision procedure, not one universal library size.

## Evidence that exists

| Evidence inspected | What it supports | What it does not support |
| --- | --- | --- |
| `test_mathematical_proof.py`, precursor/adduct/isotope boundary suites | Hand-crafted numerical scoring and chemistry contracts; optional-dependency behavior | Measured selectivity in real unknown mixtures |
| `tests/scientific_validation/ground_truth_*.msp/json`, generator and golden suite | Deterministic end-to-end integration and known-answer arithmetic, mostly constructed spectra | An independent experimental cohort or acquisition-independent chemical truth |
| `test_fdr_statistics.py`, FDR invariants and golden q values | Formula/tie/empty-null/chunk contracts, some fake-engine fixtures | Exchangeability of false target hits and synthetic decoys, or observed error control |
| Existing saved MoNA VF-NPL cross-instrument outputs | Limited retrospective connectivity retrieval under specified configurations | Strict full-key accuracy, matrix identification, new-version reproducibility, or calibrated FDR |
| PR #103–#107 evidence reports | Reproduced synthetic correctness/provenance defects and focused corrections on their branches | Empirical defect prevalence; merged corrections; chemical identification validation |

Saved output recount, **not a fresh rerun**:

| Quantity | 0.01/0.01 Da MS1/MS2 | 0.02/0.02 Da MS1/MS2 |
| --- | ---: | ---: |
| Selected queries (label ledger) | 600 | 600 |
| Loaded / rejected / searched (saved report) | 600 / 100 / 500 | 600 / 100 / 500 |
| Qualified target candidate rows (CSV recount) | 472 | 508 |
| Queries with a target call (CSV recount) | 148 | 155 |
| Correct top-1 connectivity (saved truth-ID mapping + CSV recount) | 116 | 116 |
| End-to-end connectivity retrieval, correct / selected | 19.33% | 19.33% |
| Correct among calls | 78.38% (116/148) | 74.84% (116/155) |
| True structure represented among candidates | 134 | 136 |
| LTQ-source calls | 0 | 0 |

No top-score ties occurred in either saved output. Both runs have an empty decoy
null and `Uncalibrated` hits. Library: 28,260 records / 2,226 connectivity labels;
queries: 300 LTQ and 300 QTOF against QExactive. Both use cosine, score 0.4, three
matched peaks, q threshold 1.0 and dirty checkout at the baseline SHA. Both MS1
and MS2 tolerances changed, so this is not a causal comparison of either parameter.

The recount confirms arithmetic from candidate rows and the existing label map;
it does **not** independently authenticate purchased-standard identities, audit
all near duplicates, reconstruct missing raw acquisitions, or rerun the software.
Exact-peak exclusion in the harness does not establish cross-acquisition lineage.
The LTQ zero-call discrepancy versus earlier reported behavior remains unresolved.
Saved provenance lists several requirement strings as package keys with
`<not installed>` despite the executed workflow; dependency versions need repair
before this bundle supports exact computational reproduction. Original MoNA
licensing and frozen-download provenance are not established by these outputs.

## Claims that exceed the inspected evidence

- `docs/user-guide/scientific_validation.md` calls the golden suite authoritative
  for scientific meaning. Its authority can govern intended software semantics;
  it cannot substitute for experimental identification accuracy or null calibration.
  The cited Li paper develops entropy similarity, not this entropy-preserving
  jitter-decoy procedure. Its discussion explicitly raises risks of computational
  decoys; the citation does not validate this generator.
- `docs/user-guide/validation.md` says every spectrum/molecule is scientifically
  sound and that the 5-ppm gate prevents unsupported lucky matches. Missing-context
  exemptions, trusted stores, unknown query structure and reproduced defects make
  those universal statements untenable. Five ppm is repository policy for a
  declared domain, not an instrument-independent law of measurement.
- The same guide describes isotope-based consensus tie-breaking. Classical search
  and current consensus scoring do not consume measured isotope agreement as that
  claimed tie-break; model-envelope calculations alone do not implement it.
- Score-based `Matched` and query-q inheritance must not be interpreted as confirmed
  identity or candidate-specific error probabilities. Header/docstrings calling
  benchmark labels objective do not audit label correctness or stereo equivalence.
- The claim that the default accelerator is numerically identical to full scoring
  is contradicted on main by the mixed-frame reproduction in PR #104.

## Open work, verified live

Issues [#68](https://github.com/janusson/MassFlow/issues/68) and
[#91](https://github.com/janusson/MassFlow/issues/91): zero-decoy investigation;
[#70](https://github.com/janusson/MassFlow/issues/70): impact of the 5-ppm metadata
gate; [#71](https://github.com/janusson/MassFlow/issues/71): modified-cosine
signal/noise; [#97](https://github.com/janusson/MassFlow/issues/97): benchmark
metrics/provenance; [#95](https://github.com/janusson/MassFlow/issues/95): fragment
filter contract. These are unresolved validation/correctness work, not completed
scientific evidence. Performance issues #61/#65–#67 do not validate identification.

| PR | Open head at audit | Scope; release-validation implication |
| --- | --- | --- |
| [#101](https://github.com/janusson/MassFlow/pull/101) | `db774aa` | Missing precursor as NaN; reconcile with fragment-window contract |
| [#103](https://github.com/janusson/MassFlow/pull/103) | `113df60` | Charge/adduct contradiction; declared chemistry and cache semantics |
| [#104](https://github.com/janusson/MassFlow/pull/104) | `134a204` | Mixed-frame prefilter recall |
| [#105](https://github.com/janusson/MassFlow/pull/105) | `93fc25b` | Fragment filtering and cache semantics |
| [#106](https://github.com/janusson/MassFlow/pull/106) | `f377ef3` | Capped search eligibility order |
| [#107](https://github.com/janusson/MassFlow/pull/107) | `2f8d408` | Acquisition record identity/provenance; legacy-store limitations |
| [#92](https://github.com/janusson/MassFlow/pull/92) | `e5b62e5` | Advisory diagnosis only; does not resolve FDR calibration |
| [#80](https://github.com/janusson/MassFlow/pull/80) | `73deedf` | Separate experimental network branch; unmerged, not stable evidence |

None were merged when checked. Multiple standalone PRs change processing-fingerprint
semantics; eventual integration needs a new coherent revision, fresh libraries,
full qualification and empirical reruns. This specification changes no algorithms,
confidence formulas, fixtures, dependencies or scientific thresholds, and creates
no GitHub issues or releases.
