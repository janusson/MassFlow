# Top-N eligibility: scientific red-team evidence

## Finding

On main `ffef50bff913fcf400cb3de72595dab50e4f3fd8`, a capped classical
search can lose an eligible near-match because an ineligible higher-scoring
reference consumes the result budget. The same search with `top_n=None`
returns the eligible result. Both cosine and modified cosine are affected.

This is a reproducible candidate-recall defect in the search API. It does not
establish how often this happens in measured data, compound identification
accuracy, or FDR calibration. Default `annotate` and normal consensus
sub-engine searches request unlimited candidates and do not exercise this
defect. Consensus fallback can forward a caller's cap to the classical engine.
That call site establishes reachability, not an end-to-end reproduction of every
consuming workflow. Classical `batch_score` supplies only one reference per
search, so it has no competing reference to trigger this defect.

## Hypothesis and decision rule

Before executing the new tests, the run protocol fixed this hypothesis:
`top_n` is applied before polarity and retention-time eligibility checks.
For each query, capped search must return the first N eligible, score-ordered
results of unlimited search with the same parameters. A higher-scoring
ineligible reference must not change that result.

The experiment varies the eligibility gate, algorithm, result cap, and reference
delivery mode. No scoring algorithm, scientific threshold, confidence estimator,
or golden fixture is changed to obtain a result.

## Explicit synthetic data

All spectra are deliberately constructed caffeine ions (`C8H10N4O2`). Neutral
formula masses come from real `pyteomics.mass.calculate_mass`; precursor m/z
uses registered `[M+H]+` or `[M-H]-` offsets. Every constructed spectrum
exercises the real `MolecularStructure` and `SpectrumMetadata` formula and
5-ppm validators. Nothing in those validators is mocked. RDKit is unnecessary.

The three ascending fragment atom inventories are `C2H4N`, `C4H7N2`, and
`C6H8N3O`. Their masses are computed with pyteomics and corrected by minus
one electron mass for positive fragments, plus one electron mass for negative
fragments. Arrays are float64. These are hypothetical fragments, not measured
caffeine MS/MS or a claim about a validated fragmentation pathway. Opposite
fragment charges differ by two electron masses, within the 0.02 Da fragment
window; the real polarity gate must still reject their pairing.

| Spectrum | Intensities | RT in seconds | Polarity |
| --- | --- | --- | --- |
| Query | `(1, 1, 1)` | 60 | positive |
| Ineligible reference, RT arm | `(1, 1, 1)` | 600 | positive |
| Ineligible reference, polarity arm | `(1, 1, 1)` | 60 | negative |
| Eligible first | `(1, 1, 0.5)` | 61 | positive |
| Eligible second | `(1, 0.5, 0.5)` | 64 | positive |

RT tolerance is 5 seconds. Fragment tolerance is 0.02 Da; minimum matched
peaks is 3; minimum score is 0.9. The RT arm uses a 0.02 Da precursor window.
The polarity arm deliberately widens cosine's precursor window to 3 Da so
the downstream polarity gate, rather than precursor separation, discriminates
the negative reference. Modified cosine does not impose that precursor window.
The experiment assumes comparable RT units and conditions; it does not validate
cross-method use of retention time.

With isolated one-to-one fragment edges, both algorithms reduce to normalized
dot products, independently derived as:

- Ineligible reference: `3 / (sqrt(3) * sqrt(3)) = 1`.
- Eligible first: `2.5 / (sqrt(3) * sqrt(2.25)) = 0.9622504486493763`.
- Eligible second: `2 / (sqrt(3) * sqrt(1.5)) = 0.9428090415820634`.

Each has three matched peaks. Tests assert those values using actual matchms
pair scoring before testing MassFlow's result selection. See the
[matchms CosineGreedy definition](https://matchms.readthedocs.io/en/latest/api/matchms.similarity.CosineGreedy.html)
for the matching tolerance and default intensity/mass powers. No ambiguous
peak assignment or greedy-versus-optimal comparison is involved here.

## Reproduction and results

Environment: macOS arm64, Python 3.13.12, matchms 0.33.1, dependencies from
the unchanged `uv.lock`.

```sh
uv sync --frozen
uv run pytest tests/test_top_n_eligibility.py -v
```

To reproduce the pre-fix failure, use a clean checkout of the base revision
and copy only `tests/test_top_n_eligibility.py` into it. Verify `MassFlow.__file__`
points into that checkout before running. Do not revert a shared checkout.

| Experiment | Unchanged base | Corrected |
| --- | --- | --- |
| Unlimited search, each arm/algorithm | eligible first, eligible second | same |
| `top_n=1`, each arm/algorithm | empty | eligible first |
| `top_n=2`, each arm/algorithm | eligible first only | both eligible references |
| List / iterator / cached precursor-array modes | same defect in all three | eligible results preserved |
| Two queries with mutually ineligible highest scorers | both empty | each gets its own eligible reference |
| Missing RT, exact 5-second boundary, just outside boundary | expected bypass/inclusion/rejection | same |
| Zero result budget | empty | empty |

Before correction, the regression file yields **26 failed, 8 passed**. Failures
are assertions on missing eligible results; direct scoring, real chemistry
validation, and unlimited-search assertions pass. After correction, all 34
regressions pass; the combined focused search/physics suite passes 120 tests.
Iterator regressions use one library chunk; the existing wrapper remains
responsible for combining and limiting results across chunks.

The mandatory full-suite command passes on both revisions:

```sh
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v
```

- Unchanged base: 1,356 passed, 7 skipped, 35 deselected; 82.39% coverage.
- Corrected: 1,390 passed, 7 skipped, 35 deselected; 82.39% coverage.
- Ruff lint/format, mypy, `uv lock --check`, and strict MkDocs build pass.

Both full runs emit warnings, including unclosed SQLite resource warnings;
passing the suite is not a claim of warning-free execution. Golden scientific
fixtures and dependencies are unchanged.

## Root cause and minimal correction

`SimilarityEngine.search` sorted score/matched-peak survivors and sliced their
indices to N. Only then did it evaluate adduct polarity and retention time.
Rejecting a sliced candidate left a vacant slot without considering the next
eligible reference.

The correction collects eligible results in the existing score order for each
query, then slices that query's result list to N. `None` retains all results;
zero remains empty. Scorers, physical validation, metadata gates, ordering,
decoy generation and confidence formulas are unchanged. More metadata can be
visited in capped searches because eligibility must precede truncation.

Decoys still share the existing result budget with targets. Changing that
competition policy or calibrating top-N confidence is outside this correction.
Software tests alone do not validate target-decoy assumptions. Existing active
drafts #103, #104 and #105 address separate issues and are not incorporated here.
