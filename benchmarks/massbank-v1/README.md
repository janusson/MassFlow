# Frozen measured-spectrum diagnostic benchmark

See [measured results and limitations](RESULTS.md) for the completed clean run.

This is an executable, small **diagnostic**, not a validated identification method
or FDR-control certificate. Production algorithms are unchanged on main
`ffef50bff913fcf400cb3de72595dab50e4f3fd8`. Correctness PRs #101/#103–#107 and the
confidence correction are not integrated. Re-run this identical cohort on a clean
combined candidate before a release decision. No thresholds were tuned to results.

## Source, truth and independence

MassBank-data release 2026.03, commit
`705afb7bccc3b2c42410a744eef73674716a60ef`, is downloaded outside Git. The complete
archive SHA256 is `cb30a4aab9d06dc5978c08e73397106a104955f1c31a57671ee5c15025985529`.
The manifest contains source URL, full query labels/metadata/hashes, reference
accessions, eligibility and deterministic selection. Each execution writes a
`references.jsonl` with each reference's immutable URL, hash, original license,
authors, chemical label, instrument, collision energy and terminal processing state.
See [original data](https://github.com/MassBank/MassBank-data/tree/705afb7bccc3b2c42410a744eef73674716a60ef)
and [MassBank record format](https://github.com/MassBank/MassBank-web/blob/dev/Documentation/MassBankRecordFormat.md).

Only records declaring `CONFIDENCE standard compound`, experimental positive ESI
MS2, [M+H]+, full InChIKey, finite positive precursor, collision-energy text,
ordered finite/nonnegative peaks and CC BY or CC0 enter selection. **Truth is the
depositor's standard-compound label**, not independent re-authentication of identity
or raw-scan verification. Athens queries and Eawag references give distinct
contributor/instrument contexts. Mirrors, shared standards and unknown raw-scan
lineage remain possible. Deposited centroid peak lists are used; raw centroiding
and matrix performance are not validated. Original record licenses govern data;
the software MIT license does not replace them. Four original CC BY text fixtures
retain authors/copyright/license, with attribution in their fixture README.

One full key shared between sources is selected per connectivity group using
SHA256(`massflow-mvb1-truth-v1|fullkey`). Groups are ordered by
SHA256(`massflow-mvb1-v1|connectivity`): first 20 are development (unused), next 100
are test; first query accession per group is fixed. All eligible Eawag records
form the library. In the mixed arm, every connectivity-equivalent reference is
removed for the first 50 test groups. No learned model/training or tuning occurs.
Queries are blinded: no name, formula, SMILES, InChIKey or source accession is
passed to production. Reference metadata remains available to real validators.

Cross exact duplicates use float64 m/z and max-normalized intensities rounded to
12 decimals. Near duplicates require precursor agreement within 5 ppm, real
matchms cosine >=0.99 at 0.005 Da and >=90% of peaks matched in both spectra.
Flagged queries remain attempted and scored; they are excluded from primary
accuracy denominators with explicit reasons. Repeated library identities and
exact reference duplicates are reported, retained and not silently deduplicated.
This is a screening definition, not proof of acquisition independence.

## Reproduce

Use Python 3.13 and the committed lock in a clean checkout. From repository root:

```bash
uv sync --locked
uv run python -m scripts.validation_benchmark.runner run \
  --cache "$HOME/.cache/massflow-validation" \
  --output "$HOME/.cache/massflow-validation/runs/ae-v1-independent" \
  --manifest benchmarks/massbank-v1/manifest.json \
  --manifest-sha256 66e27364488feb3e4dc74a452d964d0240d2763e1d1dd40c94ff33c400410b49 \
  --parameters benchmarks/massbank-v1/parameters.json \
  --parameters-sha256 096a6ea137ec010d28c5b657a7225bf57c274cc15fa0e3125166b0b5dd446738
```

New output paths are required; existing runs are never overwritten. The independent
pins are canonical JSON digests, not pretty-printed file hashes. Archive bytes are
verified before parsing. Manifest/config schema changes or selection drift fail.
`freeze` is for a new prospectively versioned study, not regenerating this manifest.
Do not commit downloaded corpora, native databases or peak files. Raw source is
~60 MB; expanded native outputs and environment require more disk. The runner uses
only existing locked dependencies. Network is required only for an uncached archive.

The existing `workflow.run_annotation_pipeline` executes normal ingestion,
processing, SQLite preparation, target/decoy matching, legacy q filtering and native
CSV/YAML export. Observer wrappers delegate original functions without replacing
science. Captured candidate rows precede q filtering and contain actual qualified
scores. No benchmark-only scorer is substituted. Settings are frozen: 0.02 Da
precursor window (cosine), 0.02 Da fragment tolerance, score >=0.6, >=3 matched
peaks, no RT gate, q threshold 1.0, seed 42, minimum three processed peaks,
`filter_by_mz=False`, chemical-string repair disabled, Numba prefilter disabled
because its known mixed-frame correctness fix is separate. Real formula/adduct/
5-ppm physical validation still runs. Modified cosine retains its production
shift/analogue candidate semantics; it is not precursor-gated like cosine.

Every run records effective defaults, lock/source/input/harness hashes, commit and
dirty state, installed library versions, Python/OS, seed, wall time and whole-process
high-water RSS. Absolute paths describe the actual run; reproduction uses new paths.
RSS can include earlier arms, and concurrent system load means timing is descriptive.
A failed family is saved and makes CLI exit nonzero; individual drops remain in the
query ledger. Success with zero decoys does not become a calibration success.

## Metrics and denominators

`queries.jsonl` contains every attempted query, including exclusions and failures.
`candidates.jsonl` preserves qualified target/decoy rows. `result.json` includes
per-query structural outcomes, strict/full-key and secondary connectivity metrics,
counts, score distributions, CE/instrument strata and q-threshold diagnostics.
`summary.json` collects all four arms; `inventory.json`, `leakage.json` and reference
ledgers preserve selection/rejection details. Small machine-readable outcomes and
an artifact-hash index accompany the scientific report; larger native outputs stay
outside Git and can be reproduced by the command above.

- Attempted = all 100 selected test queries per arm, including potential leakage.
- Eligible = attempted minus predeclared duplicate/leakage flags. Operational failures
  remain eligible and are missed matches when truth is present.
- Successfully scored = eligible queries reaching the normal search, even if no
  qualified hit survives. It does not mean a correct structure was retrieved.
- Distinct structures are ranked by maximum actual score per full key, then
  lexicographic key on ties; redundant acquisitions do not fill top-five slots.
- Top-1/top-5 = correct true full key in first one/five divided by **all eligible
  library-present queries**, including processing failures. Full keys do not prove
  stereo/isotope discrimination. Connectivity is a separate secondary endpoint.
- Retrieval coverage = truth anywhere among qualified candidates / eligible present.
  This is post-gate retrieval, not ungated precursor-candidate recall.
- Precision = correct top-one calls / all eligible top-one calls. Recall = correct
  top-one / eligible present. False matches = called wrong top-one; missed matches
  = eligible present without correct top-one (the sets may overlap).
- Absent false-assignment rate = eligible absent with a call / all eligible absent.
  Searched rejection rate uses only searched absent queries; failed processing is
  not credited as correct open-set rejection.
- Observed FDP = wrong top-one calls / calls, a descriptive cohort fraction.
  Threshold diagnostics apply actual legacy query q values to the same rows. No
  zero-decoy rank diagnostic or nominal q level establishes error control.
- Undefined rates serialize as null. Score histograms describe actual gated rows.

The current API observes IO/metadata/peak/physics/minimum-peak rejection and final
qualified-candidate presence. It does **not** expose every precursor/adduct pair
or separately count score-only and peak-only rejection. These losses remain
explicitly unresolved, never invented. Calibration uncertainty across independent
batches is pending; one deposited spectrum per structure is not a representative
random sample of unknown samples. Cosine and modified cosine are the implemented
classical baselines. A mass-only baseline, noise/preprocessing sweeps, decoy-method
controls and independent-batch calibration remain proposed in the
[full scientific protocol](../../docs/validation/program/README.md).

The original broad protocol and this minimal diagnostic are distinct: the frozen
manifest records amendments that permit limited diagnostic execution despite
unresolved release qualification, limited source coverage and unverified raw
lineage. This benchmark does not complete all proposed protocol experiments.
