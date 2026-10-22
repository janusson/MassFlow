# Spectral-record provenance review

## What was reproduced

Reviewed main `ffef50bff913fcf400cb3de72595dab50e4f3fd8`, active worktrees,
recent PRs/issues, the experimental network branch `73deedf`, matchms 0.33.1
metadata repair, and existing storage/lineage/search/export tests. No dedicated
InChIKey metadata-cache implementation was found in those inspected paths.

Two constructed caffeine records with the same InChIKey and **distinct record
IDs** retained their different source files, library accessions, collision
energies, adducts, instruments and instrument types. Reversing processing order
did not produce crossover. The reported InChIKey-keyed metadata inheritance was
therefore **not reproduced** on current main.

Two related provenance failures were reproduced:

1. **Stable library ingestion/search/export:** raw libraries without `id` keep
   that absence through preparation. Two different acquisitions then both
   produce `reference_id="None"`, including two modified-cosine hits scoring
   1.0 with three matched peaks. Repeated explicit IDs similarly fail to
   distinguish records. The store retains acquisition metadata, but annotation
   IDs cannot unambiguously link back to it.
2. **Experimental consensus:** results aggregate by `(query_id, reference_id)`.
   Colliding IDs combine evidence from different acquisitions under a single
   metadata template. In the discriminating fixture, the protonated record
   scores `0.9622504486493763` in both engines; the sodiated record scores 0 in
   precursor-gated cosine and 1 in modified cosine. They must produce two
   consensus rows scoring approximately 0.96225 and 0.5, not one blended row.
   This collision does not require matching InChIKeys; sharing a structure does
   not make the acquisitions interchangeable.

Explicit `None` structural identifiers also caused matchms repair to raise and
the batch pipeline to discard otherwise usable records. An absent InChIKey is
not a reason to discard a valid measured spectrum or copy another record's key.

## Identity contract and correction

InChIKey represents chemical identity. `id` represents a spectral record in the
library being searched. `library_id` names a library and is not a record key.

Raw library ingestion now opts into `load_spectra(..., assign_record_ids=True)`:

- Clone each loader record before adding provenance; acquisition metadata and
  float64 peak arrays stay with that record.
- Preserve the first nonempty supplied `id`. Replace missing, empty or repeated
  IDs with `massflow:sha256:<source-file-digest>:spectrum:<loader-index>`.
- Preserve the supplied ID in `massflow_original_id`, including explicit
  missingness. Record `massflow_source_path`, `massflow_source_sha256` and
  `massflow_source_index` independently of chemical identifiers.
- Count zero-based **loader records before validation/filtering**. A rejected
  record does not renumber subsequent acquisitions. This is the importer's
  record ordinal, not an assertion about vendor scan numbering or text lines.
- Fail rather than reuse a generated ID if it collides with an earlier ID.

Managed annotation preparation, the direct raw-library adapter and `db build`
enable this policy. Generic/query loading retains its existing behavior. Native
store inputs are read without assigning new IDs or rewriting stored provenance.
The source digest is content-based: moving identical bytes preserves generated
IDs; changing source bytes changes them. The path/ordinal fields describe the
file ingested on this call; declared upstream acquisition metadata is retained.

Consensus and cascade now reject missing, empty or duplicate reference IDs at
their public search boundary. They also reject a supplied target ID that would
collide with another record's generated `<id>_decoy` ID when decoys are enabled.
Direct API callers and legacy-store users get an
actionable error instead of silent record aggregation/routing. Classical
scorers and confidence formulas are unchanged. Missing primary structural
strings are normalized on a clone before repair, without supplying chemistry.

Annotations retain the existing `reference_id` column. Its value can be used
to retrieve the complete stored acquisition record, then its source digest and
ordinal. CSV and mzTab-M do not add collision-energy or instrument columns;
those fields remain in the referenced native-store record. MSP/MGF spectrum
exports retain the acquisition fields and IDs as well.

## Cache and compatibility

Processing fingerprint revision **3** invalidates config-only managed caches.
They rebuild using the existing staged replacement, then are reused. Tests
demonstrate replacement rather than append for SQLite, Zarr and hybrid caches.
This standalone revision does not incorporate PR #103 or #105. Combining their
processing changes requires another revision, rather than aliasing revision 3.

Three historical multiprocessing CSV fixtures previously expected `None` as
every reference ID. Only their `reference_id` cells are intentionally updated,
using the fixture file's SHA-256 and independently enumerated `NAME` records.
Every other cell, including scores, matched peaks, q-values and p-values, is
unchanged. The scientific-validation manifest is not regenerated. An existing
mock loader accepts the added keyword; its FDR assertions are unchanged.

## Reproducible tests

Fixtures are deliberately synthetic caffeine acquisitions, not measured spectra
or confirmed identities. They use formula `C8H10N4O2`, real pyteomics adduct
arithmetic and the actual formula/5-ppm validators. Three hypothetical fragments
are at m/z 42.033826, 83.060374 and 138.066188. Acquisition A declares `[M+H]+`,
`collision_energy=20`, QTOF-A, LIBRARY-A and ACCESSION-A; B declares `[M+Na]+`,
`collision_energy=45`,
Orbitrap-B, LIBRARY-B and ACCESSION-B. `accession` is canonicalized to
`spectrum_id` by matchms; its value remains record-specific. No physical
validator, isotope calculation or scoring algorithm is mocked.

```sh
uv sync --frozen
uv run pytest tests/test_spectral_record_provenance.py tests/test_record_identity_consensus.py -v
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v
uv run ruff check .
uv run mypy .
uv lock --check
```

To reproduce the failures, copy only the two regression files into a separate
clean checkout at the base revision and verify `MassFlow.__file__` points there.
On the unchanged base the final 38 regressions produce **31 failed, 7 passed**.
The passing cases include the shared-structure preservation controls: this is
not evidence of a structure-keyed metadata cache defect.

Independent review reproduced a further target/decoy collision: IDs `x` and
`x_decoy` produced a consensus row with decoy-of-x precursor metadata but the
sodiated target's score. Two additional regressions failed before adding the
generated-namespace check. Decoy generation and confidence formulas are unchanged.

Coverage includes duplicate structures, absent/empty/repeated IDs (including
InChIKey strings misused as record IDs), conflicting and missing acquisition
fields, reversed input order, exact source ordinals after quarantine, generated
ID stability, SQLite/Zarr/hybrid reopen and individual lookup, MSP/MGF spectrum
round-trips, both classical scorers, CSV/mzTab-M reference traceability, legacy
cache replacement/reuse, experimental meta-engine ambiguity rejection, and
target/generated-decoy ID collisions.
Consensus/cascade tests are not marked `core`.

Validation on macOS arm64 / Python 3.13.12 with the unchanged lockfile:

- Unchanged main full suite: **1,356 passed, 7 skipped, 35 deselected;
  82.39% coverage**.
- Corrected final full suite: **1,394 passed, 7 skipped, 35 deselected;
  82.44% coverage**, including all 38 new regressions.
- Focused ingestion/search/FDR/equivalence tests: 110 passed before adding the
  final three duplicate-InChIKey-ID variants; the full suite includes those.
- Ruff lint, mypy, `uv lock --check` and strict MkDocs build pass. Full-suite
  warnings include unclosed SQLite resources; execution is not warning-free.
- Independent read-only review found the target/decoy collision, verified its
  correction, independently passed all initial 36 cases and the final nine
  meta-engine cases, and found no remaining blocking code issues. It verified
  that all 12 changed golden rows differ only in `reference_id`.

## Remaining limits

- Existing external stores are not repaired automatically. Rebuild from raw
  sources into a new destination when IDs are absent or nonunique. Old exports
  containing `None` cannot be repaired from the export alone.
- Supplied IDs remain scoped to a library. Appending/merging independently
  curated stores can introduce duplicate supplied IDs; the meta engines now
  reject these, but classical direct API callers must ensure uniqueness.
  This change does not migrate all store lookup or merge APIs.
- Direct classical calls with manually constructed, unidentified Spectrum
  objects retain their old scoring behavior; callers must supply record IDs
  for traceable annotations. Use the library ingestion policy for raw files.
- Source files must stay unchanged while hashing/parsing. The extra digest
  pass and ID set have not been performance-benchmarked. Parser-level records
  skipped before yielding cannot receive a loader ordinal from MassFlow.
- Explicit configuration overrides (for example `processing.instrument`) still
  follow the existing contract; this patch does not reinterpret those settings.
- Synthetic software tests do not measure real-library prevalence,
  identification accuracy, chemical identity validity or FDR calibration.
