# Experimental dataset assessment and freeze plan

Source documentation checked 2026-10-21 America/Vancouver. **Accessible documentation
is verified; a complete experimental corpus has not been downloaded or curated
for this program.** All counts and eligible-pair feasibility must be established
from frozen files. Proposed experiments are in the [program](README.md).

## Candidate resources

| Resource | Ground truth and experimental diversity | Rights and reproducibility | Recommendation |
| --- | --- | --- | --- |
| MassBank official records | Record structures, accession, instrument, precursor/adduct, CE and provenance fields; record assertion is not independent identity verification. Select measured MS2 with standard evidence, preserving missing fields. Contributor/instrument diversity is useful; same-lab CE series may be one acquisition | Record-level `LICENSE` can be CC0, CC BY, CC BY-SA, CC BY-NC or CC BY-NC-SA. Check each record and its attribution; no blanket CC0 assumption. Versioned GitHub/Zenodo releases support freezes [D1,D2] | Primary MVB-1 source: release 2026.03, tag resolves to `705afb7bccc3b2c42410a744eef73674716a60ef`. Restrict initial redistributable panel to verified CC0/CC BY records; audit independent pairs before promising sample size |
| MoNA / VF-NPL | Hosts both measured and in-silico records; LC-MS/MS, polarity, instrument and named-library downloads. VF-NPL QTOF/LTQ/QExactive offers a practical cross-instrument panel already used locally, but shared source standards and repository mirrors need lineage checks [D3] | Download endpoints are mutable. Freeze exact request/response, JSON plus MSP, timestamps, SHA-256, record IDs and original library links. The inspected license page concerns software; it does not establish blanket spectral-data redistribution rights [D4] | Secondary/reconciliation cohort. Existing saved results are useful diagnostics. Do not redistribute VF-NPL derivatives until its original data rights and identity provenance are resolved |
| GNPS directly contributed reference spectra | Standards and community records with annotation histories and variable curation quality; separate measured standards from propagated/suspect/analogue hypotheses. Preserve accession, USI/raw scan where present and annotation revision [D5] | Direct contributions default CC0; imported third-party libraries need their own rights checks. MGF/JSON download is possible; freeze content because live libraries change [D5] | Independent source/laboratory holdout after cross-repository duplicate audit. Exclude propagated/suspect libraries as exact truth |
| CASMI 2016 | Known challenge solutions; category 1 and categories 2/3 are different cohorts. Original automated-method paper describes 208 category 2/3 peak lists and separate training data, and documents training overlap; historical challenges are not necessarily unseen today [D6] | Official category-1 MassIVE MSV000079573 landing page identifies challenges 1–19 and CC0, with “Partial Public” status. This does not establish identical availability/license for categories 2/3, solution tables or every supplemental asset [D7] | Small external challenge and label/provenance check. Resolve precise category, file access and identity solution provenance before freezing. Missing library truth belongs in open set, not closed-set failure inflation |
| MSnLib (comparable reference resource) | Measured Orbitrap ID-X flow-injection standards, both polarities and energy series, with raw/mzML and processing lineage. Export includes individual, merged-energy and pseudo-MS2 spectra; use only genuine single-stage MS2 with declared SPECTYPE for primary testing [D8,D9] | Zenodo version record 21105617 inspected; API reports `cc-by-4.0`. Earlier concept record redirects to latest, so use a version DOI plus file hashes. Source lists raw-data records and checksum-bearing MGF/JSON assets [D8] | Traceable expanded standard cohort, especially for pairing independent external references. Do not treat different energies, merged products or MSn derivatives as independent experiments |
| Commercial NIST tandem libraries | Potential high-quality comparator if properly licensed | No open redistribution permission established here; published performance is not permission to ship data | Optional locally licensed external comparison, excluded from an openly redistributed minimal benchmark |

These are resource assessments, not declarations that every deposited chemical
identity is correct. Require original contributor/standard evidence or classify
the record as repository-asserted truth. Where an authentic-standard certificate,
lot identity or orthogonal confirmation is unavailable, record that limitation.
Evidence tier must never be upgraded merely because several repositories agree.

## Exact acquisition/freeze procedure

1. Archive release metadata and record-format documentation, capture access date,
   original publisher/laboratory, release DOI/tag and resolved commit. For MassBank,
   verify `git rev-parse 2026.03^{commit}` equals the pinned SHA after a clean clone;
   use that commit, not a moving default branch. Download the corresponding record
   archive and retain its byte length and SHA-256. Select source records directly
   from that commit; export to MSP/MGF using a separately pinned conversion step.
2. Inventory records **before selection**. Extract native identity fields, source
   license/attribution, experimental/simulated/merged status, MS level, polarity,
   adduct/charge, instrument/analyzer, CE text/value/unit, precursor, peak count,
   sample and acquisition lineage. Keep original metadata JSON/text unmodified.
3. Resolve structure labels in an evaluator-only table; record inconsistencies,
   missing stereo/isotope information and adjudication. Compute exact peak hashes
   and cross-source duplicate components. Do not use MassFlow acceptance or score
   to decide label eligibility. Retain the entire exclusion ledger.
4. Apply the frozen eligibility and deterministic split/selection rules in MVB-1.
   Report the number of eligible structures, independent acquisition pairs, records
   and missing metadata by source/stratum. If sample-size/independence needs cannot
   be met, stop the confirmatory run and amend the design before scoring.
5. Write an immutable manifest conforming to the [contract](manifest-contract.md).
   Store original assets, labels, scripts, mapping tables, input exports, checksums
   and the exact licensing/attribution evidence. For assets without redistribution
   permission, publish retrieval instructions and hashes only where permitted;
   access alone does not authorize repackaging. Resolve ambiguity before publishing.
6. Fetch each frozen file twice to distinct local locations and verify identical
   SHA-256. Changing content requires a new dataset version. Rebuild search stores
   from these frozen files into new paths, save lineage, and verify record-count,
   structure-count and metadata round-trips. A live endpoint alone is not a freeze.

A possible MassBank source acquisition command (not executed as a corpus download
in this task):

```sh
git clone --branch 2026.03 --single-branch https://github.com/MassBank/MassBank-data.git massbank-2026.03
git -C massbank-2026.03 rev-parse HEAD
```

Raw-source SHA-256 and lineage must supplement repository commit IDs. Do not use
a mutable download filename as the dataset version. Collision energy numeric
values with missing units stay unknown; polarity and adduct must not be inferred
from the desired answer. Exclude computational libraries (e.g. LipidBlast) from
the measured-reference primary cohort even when hosted alongside measured spectra.

## Original documentation and literature

- **D1:** [MassBank official data repository](https://github.com/MassBank/MassBank-data)
  and [release 2026.03](https://github.com/MassBank/MassBank-data/releases/tag/2026.03).
  Release metadata and remote tag checked; full data inventory deferred.
- **D2:** [MassBank Record Format 2.6.0](https://raw.githubusercontent.com/MassBank/MassBank-web/dev/Documentation/MassBankRecordFormat.md),
  sections LICENSE, acquisition, focused-ion and peak metadata. Record-level licenses
  must be rechecked in the frozen dataset; this documentation URL can change.
- **D3:** [MoNA downloads](https://mona.fiehnlab.ucdavis.edu/downloads) and
  [original query/API documentation](https://mona.fiehnlab.ucdavis.edu/documentation/query).
- **D4:** [MoNA license documentation](https://mona.fiehnlab.ucdavis.edu/documentation/license).
  Search-index text describes software licensing; rendered page did not expose a
  blanket data license in this audit. Dataset rights remain unresolved.
- **D5:** [GNPS reference-library documentation](https://ccms-ucsd.github.io/GNPSDocumentation/browselibraries/),
  annotation history, suspect libraries and License; [GNPS2 library inventory](https://library.gnps2.org/libraries).
- **D6:** Schymanski et al. (2017), *Critical Assessment of Small Molecule
  Identification 2016: automated methods*, DOI
  [10.1186/s13321-017-0207-1](https://doi.org/10.1186/s13321-017-0207-1),
  [full text](https://d-nb.info/1129985008/34), Methods and training-overlap evaluation.
- **D7:** [MassIVE MSV000079573, original CASMI category-1 deposit](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=2acf8b2e56d441ffa1dba8ad83ca99ad).
- **D8:** [MSnLib versioned library deposit 21105617](https://zenodo.org/records/21105617),
  SPECTYPE descriptions and raw-data links; [API metadata](https://zenodo.org/api/records/21105617)
  inspected for `metadata.license.id` (`cc-by-4.0`). Do not rely on concept DOI redirects.
- **D9:** Brungs et al. (2025), *MSnLib: efficient generation of open multi-stage
  fragmentation mass spectral libraries*, DOI
  [10.1038/s41592-025-02813-0](https://doi.org/10.1038/s41592-025-02813-0).
  Dataset-owner documentation, rather than the paper's aggregate performance claims,
  supplies the acquisition/export facts used here.
