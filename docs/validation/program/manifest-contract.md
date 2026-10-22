# Frozen dataset and run manifest contract

Status: proposed schema v1. JSON schemas and an **empty draft template** accompany
this document. The template contains no experimental records or invented labels;
it deliberately cannot qualify as a frozen benchmark.

- [Dataset schema](manifest.schema.json)
- [Record-row schema](record.schema.json), one JSON object per line in `records.jsonl`
- [Empty draft template](manifest.template.json)

## Files in a dataset freeze

| File | Required content |
| --- | --- |
| `manifest.json` | Schema/version/status, protocol hash, identity policy, split policy, asset/derivative hashes, row-ledger hash and record count |
| `records.jsonl` | One immutable experimental spectral record per row, including excluded records; schema below |
| `structures.jsonl` | Original and standardized representations, full/first-block keys, salt/tautomer/isotope policy, evidence tier and adjudication |
| `acquisitions.jsonl` | Raw-file/native-scan/sample/standard-lot groups, lab/instrument/method, derivative parent links and independence evidence |
| `duplicates.jsonl` | Exact/near-duplicate links, rule/version, source evidence, adjudication and retained/excluded decision |
| `selection.jsonl` | Entire eligible ordered population, hash-based selection/reserve order, split, absence assignment, exclusion reason and timestamp |
| `licenses.jsonl` | Per-source/record license identifier/text locator, authors, required attribution, access and redistribution decision with evidence |
| `labels.private.jsonl` | Evaluator-only query truth, full/connectivity identity, truth-present flags for each frozen library and independent verification |
| `library-L0.*`, `library-L50.*`, `queries.*` | Exact search inputs; each parser record maps to one record UID; query truth fields removed |
| `conversions.jsonl` | Parent/derivative hashes, converter/version/options, field mapping, units, rounding and rejected records |

The dataset schema checks types and local constraints; it does not establish rights,
truth, independence or referential integrity. A future manifest validator must
implement the semantic checks below. A schema-valid freeze alone is not scientific
validation. The entire selection population and excluded records must be auditable,
not just the successful final inputs.

## Record row fields

| Field group | Contract |
| --- | --- |
| `record_uid`, source | Globally unique record UID; source namespace, accession nullable, source asset ID, zero-based parser ordinal, native scan nullable and source annotation revision nullable. Source asset references a frozen file SHA-256 |
| `chemical_identity` | Internal `structure_key`; original and standardized full keys nullable; first-block key nullable; standardization policy ID; truth tier and evidence links. Never use these fields as acquisition identity |
| `measurement` | Experimental/simulated/consensus kind, MS level, centroid/profile representation, ionization method and original method text, inlet type, precursor m/z, adduct, signed integer charge, polarity, instrument model/analyzer and collision energy value/unit/original text; null/unknown for missing values |
| `lineage` | Acquisition group nullable, raw-file hash nullable, laboratory, sample/standard-lot group nullable, derivative parents, duplicate component, and independence verdict/evidence |
| `rights` | License identifier nullable, attribution text, evidence URLs, `redistribute` allowed/restricted/unresolved |
| `spectral_content` | Exact original peak hash, normalized peak hash nullable, peak count; hashing rule/version; intensity and m/z numeric types are float64 in derived files |
| `selection` | Eligible/excluded/challenge, reasons, development/test/reference-only/excluded split, query/reference/background roles. A query is never its own independent reference |

Original metadata must be retained as a hashed asset even when normalized fields
exist. Missing scientific values serialize as JSON `null` (JSON does not permit
NaN); loaders convert to float64 NaN where numeric missingness is required.
Missing CE, RT, adduct, formula or charge must never become scientific zero.
Known zero collision energy is distinct from missing CE. RT, if present, is in
`retention_time_seconds`; preserve source value/unit and conversion lineage.
NCE/%/eV/unknown are distinct units. Do not infer equivalence between them.

Hash source-file bytes before parsing; capture source immutability throughout
parsing. Hash peaks using a documented canonical little-endian float64 encoding,
array length, sorted m/z/intensity pairs and policy for duplicate m/z/zero intensities.
Keep raw and normalized hashes because normalization can mask copied spectra.
Hash equality is sufficient to flag duplication, not to prove chemical identity.

## Mandatory semantic validation before freezing

1. Every asset exists and matches its declared SHA-256 and byte length; every UID
   is unique. Record source asset IDs and derivative/structure/group references
   resolve; parser ordinals map to original source records, including quarantines.
2. Count totals reconcile: imported = eligible + excluded + challenge under the
   declared exclusive disposition policy. Record-ledger rows equal `record_count`.
   The separate manifest-selection population includes all labelled queries selected
   before pipeline processing; a runtime rejection cannot delete its entry.
3. Every primary eligible query/reference is measured genuine ESI MS2, has verified
   centroided representation or documented validated centroiding provenance,
   documented truth, positive finite precursor and compatible declared polarity/adduct/charge, and
   passes protocol eligibility independent of MassFlow scores. Missing/contradictory
   source data remain visible in exclusions/challenges.
4. Primary development/test query structures, connectivity groups and
   acquisition/duplicate components do not overlap. Test query acquisition groups do not occur in the reference
   library. Shared test structure identity across query and reference is required
   for closed-set retrieval, not a leak in itself.
5. Each closed-set truth has >=1 qualified independent reference. Each open-set
   truth has zero connectivity-equivalent references after all alias mappings and
   source mirrors; save removal counts. Do not inspect only the output accession.
6. Every record has a rights verdict and preserved attribution. No unresolved
   record may be included in a redistributed freeze. Verified-standard truth
   requires supporting evidence; a full InChIKey alone is insufficient.
7. Query search files contain no answer-bearing chemical labels. Stored references
   retain source record IDs/metadata. Reopen/round-trip every selected record and
   verify peak arrays, record identity, adduct/charge, CE and instrument/source fields.
8. Freeze requires completed protocol/identity-policy/split-policy hashes, immutable
   source versions and a clean harness version. An empty draft must fail promotion.

## Per-run artifacts and query accounting

Each run records application/harness commits, clean/dirty state and diff hash,
lockfile hash, resolved package versions, host details, effective config and hash,
commands, RNG seeds, library/input hashes, cache/store lineage, processing revision,
backend, worker/chunk layout and FDR family (file/batch membership).

Required tables: `query_accounting.jsonl`, `candidate_stages.jsonl`,
`all_candidates.parquet` or lossless CSV, `query_maxima.jsonl`, `calls.jsonl`,
`metrics.json`, `calibration.json`, `failures.jsonl`, `artifact_checksums.json`.
Store raw target and decoy scores, gate status and full record UID for every retained
candidate. Counterfactual diagnostic passes are separate run IDs.

Exactly one terminal accounting state per selected query:

```text
selected = processing_failed_or_rejected
         + searched_without_qualified_target
         + searched_with_qualified_target
```

A separate status indicates q-based abstention, accepted correct/incorrect call,
truth presence, first-loss stage, source/stratum and reason. A whole-file failure
must still generate a terminal state for each selected query through the harness.
Only processed queries may appear as Unknown rows in native exports, so do not use
export row count as the selected-query denominator. Native CSV can have several
rows per query; left join the independent selection ledger to account for all
queries. Null metrics must retain numerator and denominator, never silently zero.
