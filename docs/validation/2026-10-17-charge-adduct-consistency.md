# Scientific red-team experiment: charge/adduct contradictions

> **Integration note:** this report documents standalone processing-semantics revision 1. The combined candidate uses revision 4 for missing-precursor normalization, fragment-only windows, charge/adduct consistency, and acquisition-record identity; the historical standalone revision is not the current managed-cache contract.


Run date: 2026-10-17 (America/Vancouver). Baseline: `5b847c0c45af32f2a2286371b4893c2af5481903`.

## Finding

**Demonstrated implementation defect:** a structurally annotated spectrum could
pass precursor validation with a charge that contradicted its registered adduct.
For synthetic caffeine, `[M+2H]2+` plus charge `+1` passed at m/z
196.094928513, twice the correct m/z 98.047464256. That is a 1,000,000 ppm
error relative to the declared adduct, despite the 5 ppm validation contract.
The retained row could then yield cosine 1.0 with three matched fragments.

This is evidence of an invalid library claim entering scoring, not a measured
false-identification rate or evidence of how frequently real libraries contain
such metadata. No FDR calibration claim follows from this experiment.

## Target selection and hypothesis

Reviewed current issues, open PRs, recent commits, existing physics/adduct tests,
and the golden validation manifest/documentation before choosing this target.
[Issue #95](https://github.com/janusson/MassFlow/issues/95) already records the
fragment-range bug; [issue #91](https://github.com/janusson/MassFlow/issues/91)
records the decoy-null investigation. Neither is this experiment's target.
[PR #76](https://github.com/janusson/MassFlow/pull/76) added canonical adduct
handling and signed registry charges, but the precursor validator did not compare
that charge with the separate `charge` metadata field. Existing adduct tests
reject contradictory notation within one string; they did not exercise a
contradiction between those two metadata fields.

**Hypothesis:** the validator uses the adduct's atom/electron offset but divides
by the separately declared `abs(charge)`. A mass calculated with a contradictory
charge can therefore pass; `abs()` also hides polarity contradictions.

Recent baseline changes were the prepared-cache replacement (#99), SciPy
compatibility (#98), and build-only library workflow (#96). No claim is made
about the historical introducing commit; a bisect was not performed.

## Discriminating experiment and data

All spectra are explicitly synthetic. No experimental acquisition, instrument
mass accuracy, real caffeine fragmentation abundance, or public reference-spectrum
identity is asserted.

- Neutral formula: caffeine, `C8H10N4O2`; matching SMILES:
  `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`.
- Neutral/proton masses come directly from pyteomics, independently of MassFlow's
  adduct-offset and theoretical-mass helpers. Expected m/z is
  `(neutral_mass + n * proton_mass) / abs(z)`, with `n` fixed by the adduct.
  See the [Pyteomics mass API](https://pyteomics.readthedocs.io/en/stable/api/mass.html).
- Synthetic fragment m/z values: `[42.033826, 67.029075, 69.044725]`;
  intensities `[0.4, 1.0, 0.6]`, all `float64`. These are rounded ion masses
  for the caffeine-compatible subformulas `C2H4N+`, `C3H3N2+`, `C3H5N2+`,
  including subtraction of the electron mass from each neutral formula mass.
  They are chosen to give an exact, hand-computable cosine of 1 for identical
  peak vectors, not to simulate a measured spectrum.
- Change only the metadata charge/precursor relationship. Test both formula and
  SMILES resolution, correct-charge controls, and 4.75/5.25 ppm boundaries.
- Exercise the real Pydantic validators, matchms harmonisation, batch quarantine,
  ordinary cosine MS1 filtering, and actual SQLite/Zarr/hybrid prepared stores.
  Chemistry, mass validation, and scoring are not mocked.

| Adduct | Declared charge | Input m/z | Baseline verdict | Corrected verdict |
| --- | ---: | ---: | --- | --- |
| `[M+H]+` | +1 | 195.087652046 | valid | valid |
| `[M+2H]2+` | +2 | 98.047464256 | valid | valid |
| `[M-H]-` | -1 | 193.073099112 | valid | valid |
| `[M+2H]2+` | +1 | 196.094928513 | **valid** | invalid |
| `[M+H]+` | +2 | 97.543826023 | **valid** | invalid |
| `[M+H]+` | -1 | 195.087652046 | **valid** | invalid |
| `[M-H]-` | +1 | 193.073099112 | **valid** | invalid |

For the scoring experiment, the unknown synthetic query has precursor m/z
196.094928513, `[M+H]+`, charge +1, no structural claim, and the three fragments
above. Search uses cosine, MS1/MS2 tolerance 0.02 Da, minimum score 0.7, minimum
matched peaks 3, decoys disabled. The malformed caffeine row survived processing
and produced score 1.0 / three matched peaks on the baseline. After correction it
is quarantined and produces no hit. The correctly labelled double-charge control
survives processing at 98.047464256 but produces no hit against that query because
it is outside the cosine MS1 window. This isolates the metadata defect from
intended modified-cosine analogue matching and from statistical calibration.

## Root cause and correction

`SpectrumMetadata.validate_precursor_mass_logic()` combined the registered adduct
offset with the independent, unchecked charge denominator. It now requires exact
agreement with the registry's signed charge before applying the existing m/z
formula and unchanged 5 ppm threshold. The processing rejection reason identifies
the contradictory charge/adduct rather than reporting a misleading zero ppm error.
The ingestion guarantee applies to metadata **after matchms harmonisation**.
Existing matchms behaviour can repair an original charge-sign contradiction when
explicit `ionmode` supplies the intended polarity: `[M+H]+`/charge -1/positive
becomes charge +1, and `[M-H]-`/charge +1/negative becomes charge -1. Those repaired
records still pass if their masses are consistent; original-sign auditing is not
implemented here. The model itself rejects either original contradiction.

Managed prepared stores previously fingerprinted configuration alone. Correcting
the validator without changing that fingerprint left old invalid entries reusable;
seven additional tests demonstrated this bypass before the fingerprint correction.
The fingerprint now includes canonical configuration plus processing-semantics
revision 1. Old managed caches are reprocessed once through the existing staged
replacement code. Tests cover all three backends, no old/new mixing, reuse after
successful rebuilding, and aborting a raw library with contradictory entries.
This introduces the revision mechanism anticipated by #95; its fragment-filter
correction remains outstanding and must increment the revision when implemented.

**Migration limit:** externally supplied `.db`, `.sqlite`, and `.zarr` libraries
continue to be trusted direct inputs. They are not retrospectively revalidated or
modified. Rebuild them from original raw sources with `massflow db build` into a
**new output path that does not already exist**. Building into an existing store
appends spectra and would preserve its invalid entries. Inspect the new store and
review quarantine messages/counts before switching `input.library_path` to it.
Missing structural context/charge keeps its documented exemption.
Direct similarity calls that bypass preprocessing also bypass this ingestion gate.
The experimental streaming transport is outside this test's validated scope.

## Reproduction and verification

From the review branch, run:

```sh
uv sync --locked --extra chem
uv run pytest tests/test_charge_adduct_consistency.py -v
uv run --isolated --locked --no-extra chem pytest tests/test_charge_adduct_consistency.py -v
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

Before the validator fix, the initial 30-case experiment produced **11 failures,
19 passes**, including formula and SMILES contradictions; valid-ion/boundary
controls passed. After fixing validation, all 30 passed. The seven cache/revision
tests then failed for the expected stale-cache reuse or unchanged-fingerprint
reason, and all **37 cases passed** after adding the revision.

After correcting the synthetic fragments for electron mass during review, the
final fixture was copied to a separate, pristine checkout of the exact baseline
commit and retested: **11 failed, 19 passed, 7 deselected**. The standalone scoring
probe again retained the contradictory row and returned **cosine 1.0 / three
matches**; the valid double-charge control remained unmatched. Import provenance
confirmed that this repetition used the baseline checkout's source. Final quality
gates below use the corrected fixture.

The untouched baseline full suite passed: **1,355 passed, 7 skipped,
35 deselected; 82.39% coverage**. Baseline lint, formatting and mypy also passed.

Corrected-tree quality gates on the same macOS environment:

| Gate | Result |
| --- | --- |
| Full mandatory coverage command above | **1,392 passed, 7 skipped, 35 deselected; 82.42% coverage**, exit 0 |
| Focused charge/adduct and cache regressions | **37 passed**, exit 0 |
| Lockfile consistency | passed |
| Repository lint and formatting | passed |
| Repository mypy | passed, 146 source files |
| Golden scientific fixtures | unchanged; included in passing full suite |

The full suite emitted 81 non-fatal warnings (the baseline emitted 79). Hosted
Linux/macOS PR checks are separate from these local results.

The isolated run without the `chem` extra passed all 37 cases. Its environment
still installed RDKit transitively through matchms; this run does **not** verify
an environment in which RDKit cannot be imported. Formula-based cases always run
and use pyteomics; the SMILES cases have the established optional-dependency skip.

Environment: macOS 27.0.1 arm64, CPython 3.13.12, matchms 0.33.1,
SciPy 1.16.3, NumPy 2.2.6, pyteomics 4.7.5, Pydantic 2.13.4,
RDKit 2026.3.1, committed lockfile. The imported `MassFlow.__file__` was verified
inside the isolated review worktree. Golden scientific fixtures were not regenerated.

Repository governance mismatch: `AGENTS.md` requests a `core` pytest marker,
but the strict marker registry currently has no `core` entry and existing tests
do not use it. These regressions follow the registered `scientific` convention
and run in the full mandatory release suite; no separate comprehensive core-marker
gate is claimed.

Passing regression tests establishes the specified software correction. It does
not establish identification specificity, real-library prevalence, exchangeable
decoys, empirical FDR control, or independent scientific validation.
