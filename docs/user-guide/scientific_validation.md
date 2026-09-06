# Scientific-Validation Suite (Golden Tests)

> **STATUS: authoritative.** This is the "golden" known-answer test suite that
> detects changes in *scientific meaning*, not merely changes in code output.
> It is part of the default release test selection (marker: `scientific`) and
> runs in CI on every PR.

## What the suite asserts

The suite (`tests/test_scientific_validation.py`, fixtures in
`tests/scientific_validation/`) locks down the scientific contract end-to-end:

| Fixture category | What is asserted |
| --- | --- |
| Perfect spectral match | cosine / modified-cosine score = 1.0, all peaks matched, `Matched` status |
| Near match | hand-computed cosine (3 of 4 fragments), matched-peak count, ranking |
| Precursor-mass violation | cosine MS1 gate (0.0 sentinel); modified cosine still matches in the exact frame |
| Adduct violation | `[M-H]-` vs `[M+H]+` rejected by both engines |
| Missing precursor (query) | rejected by the I/O validation layer, counted in `spectra_rejected`, never silently analyzed |
| Missing RT (query) | bypasses the RT filter when it is configured |
| RT filtering | `rt_tolerance` removes exactly the RT-violating reference |
| Modified-cosine behavior | precursor-shift alignment (Watrous 2012), both-frame matching, `|shift| <= tolerance` → plain cosine |
| Decoy generation | determinism, entropy preservation (Li et al. 2021), precursor preservation, decoy ≠ source, decoy matches tagged `is_decoy` for TDC |
| FDR / q-values | per-query competition unit, q traceable to the query's best target score, 1/N rank bound on an empty decoy null, p-value is diagnostic only, `q <= fdr_threshold` is the only filter |
| Unmatched queries | exported as `Unknown`, never calibrated |
| Duplicate scores | identical scores → identical q-values; equal-score references both exported |
| Large peak counts | 60-peak round trip, no peak loss, self-match counts all 60 |
| Multiple engines | cosine / modified_cosine / consensus / cascade known answers, including consensus's 0.5 dilution of MS1-gated pairs and cascade's stage semantics |
| Storage equivalence | SQLite and Zarr runs produce byte-identical result CSVs |

## How expectations are derived

Every expected value is a **known answer** computed from the published
formulas — cosine / modified cosine (Watrous et al., *PNAS* 2012), target-decoy
competition (Elias & Gygi, *Nat. Methods* 2007), spectral-entropy decoys
(Li et al., *Nat. Methods* 2021) — not captured from the implementation.

The fixture spectra are anchored to the well-documented caffeine `[M+H]+`
fragment series (m/z 195.0877 → 138.0662, loss of C2H3NO; → 110.0717, loss of
CO; → 83.0608, loss of HCN; plus the characteristic 42.0344 fragment), as
reported in public MS/MS libraries (MassBank / GNPS) and the caffeine
metabolism literature. The remaining fixture spectra are explicitly synthetic
with hand-computed expectations.

## The ground-truth manifest

`tests/scientific_validation/ground_truth_results.json` records, for every
engine run (cosine, modified_cosine, consensus, cascade, cosine+RT, and the
Zarr-backend equivalence run):

* the exact configuration used;
* per-query best target/decoy scores, q-values, and p-values;
* every exported row (reference, score, matched peaks, q, p, annotation
  status, score breakdown);
* the FDR summary (competition counts, true library size), warnings, degraded
  flags, and the SHA-256 of the result CSV.

The suite re-runs the pipeline and asserts the recorded bytes and rows are
reproduced exactly.

## Regeneration policy

The fixtures change **only when the scientific contract intentionally
changes**. Regenerate with:

```sh
uv run python tests/scientific_validation/generate_ground_truth.py
```

The generator re-verifies the current pipeline against its independent
reference implementation of the published formulas *before* writing the
manifest — if the pipeline diverges from the formulas, it fails loudly and
the recorded ground truth is left untouched. A divergence that "passes" by
updating the manifest is a deliberate scientific decision and must be
reviewed as such.

## Relationship to the other suites

* `tests/test_fdr_statistics.py` — synthetic statistical tests of the FDR
  contract (degenerate inputs, ties, duplicates, execution-mode equivalence).
* `tests/test_similarity.py`, `tests/test_mathematical_proof.py` — unit-level
  scoring correctness.
* `tests/test_library.py::TestGoldenDeterminism` — byte-determinism of the
  worker-path outputs against the pre-refactor in-memory design.
* The suite here is the *integration* layer: real engines, real files, real
  store round-trips, real exports — the whole pipeline against known answers.
