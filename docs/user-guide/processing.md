# Processing & Filtering

The `processing` section of the MassFlow YAML configuration defines how raw spectral data is transformed before it is scored by a similarity engine. MassFlow acts as a declarative facade for the `matchms` library, applying a standardized two-stage pipeline: **Metadata Processing** and **Peak Processing**.

To ensure that query spectra and reference libraries are scored fairly, MassFlow strictly applies the exact same processing configuration to both sets of data.

---

## Metadata Processing

Metadata filters repair, harmonize, and standardize the annotations attached to a `matchms.Spectrum` object (e.g., InChIKeys, SMILES, charge state, adducts).

```yaml
processing:
  clean_metadata: true
  add_retention_time: true
  repair_inchi_inchikey_smiles: true
  derive_adduct_from_name: true
  derive_formula_from_name: true
  clean_compound_name: true
  derive_ionmode: true
  make_charge_int: true
```

*   `clean_metadata`: Applies the `default_filters` from `matchms` to correct common metadata formatting issues (e.g., stripping whitespace, harmonizing keys).
*   `add_retention_time`: Extracts supported retention-time aliases into canonical seconds, preserving the original value and unit in `retention_time_source`. Explicit minute/second MGF fields, unit-bearing MSP values, and mzML/mzXML unit metadata are converted before storage. Disabling alias extraction still normalizes an existing canonical `retention_time` field. Unlabelled canonical/MSP numeric times follow the seconds contract; an unlabelled mzML scan time is left missing. Conflicting alias values are left missing. Metadata repairs operate on a clone of the source spectrum.
*   `repair_inchi_inchikey_smiles`: Attempts to fix broken or malformed structural identifiers. Crucial if you plan to use structure-aware experimental engines later.
*   `derive_adduct_from_name` & `derive_formula_from_name`: Parses the compound name string to extract adduct and chemical formula information if it is not explicitly provided in the metadata fields.
*   `clean_compound_name`: Standardizes compound names for consistent downstream reporting.
*   `derive_ionmode`: Infers the ionization mode (positive/negative) from other metadata fields if it is missing.
*   `make_charge_int`: Ensures the charge state is represented as a clean integer.

### Contextual Metadata Injection

You can also inject project-specific or instrument-specific metadata into all processed spectra by defining them in the configuration:

```yaml
processing:
  instrument: "Orbitrap"
  mode: "positive"
```

---

## Peak Processing

Peak filters physically alter the mass-to-charge (m/z) and intensity arrays of the spectra. They are critical for removing noise, ensuring comparability between different instruments, and optimizing the speed of the similarity search.

These filters are applied sequentially in a specific order:

### 1. Intensity Filtering (Noise Removal)

Removes all peaks below a certain absolute intensity threshold.

```yaml
processing:
  filter_by_intensity: true
  noise_threshold: 1000.0
```
*   `filter_by_intensity`: Master toggle for noise removal.
*   `noise_threshold`: The absolute minimum intensity a peak must have to be retained. (If set to `0.0`, MassFlow will fall back to checking the `min_intensity` field).

### 2. Minimum Peak Count

Drops the entire spectrum from the analysis if it contains too few peaks to be statistically meaningful.

```yaml
processing:
  filter_min_peaks: true
  min_peaks: 5
```
*   `filter_min_peaks`: Master toggle for peak count enforcement.
*   `min_peaks`: The minimum number of peaks a spectrum must contain *after* noise filtering to survive. If a spectrum has fewer peaks than this, it is silently dropped.

### 3. M/Z Range Truncation

Discards peaks that fall outside a biologically or chemically relevant mass window.

```yaml
processing:
  filter_by_mz: true
  mz_min: 0.0
  mz_max: 1000.0
```
*   `filter_by_mz`: Master toggle for **fragment-peak** m/z truncation. It does not filter the precursor m/z or decide precursor candidacy.
*   `mz_min` & `mz_max`: The allowed m/z window. Peaks outside this range are deleted. (Note: `mz_max` must be strictly greater than `mz_min`).

### 4. Top-N Peak Reduction

A highly aggressive filter that retains only the most intense peaks in the spectrum, discarding the rest. This can drastically speed up classical similarity scoring on noisy data.

```yaml
processing:
  reduce_to_top_n_peaks: true
  n_max: 100
```
*   `reduce_to_top_n_peaks`: Master toggle for Top-N reduction.
*   `n_max`: The maximum number of peaks to retain.

### 5. Intensity Normalization

Scales the intensity array of the spectrum so the maximum peak has an intensity of `1.0`.

```yaml
processing:
  normalize_intensity: true
```
*   `normalize_intensity`: Required for almost all classical similarity engines (like `cosine`) to function correctly, as they expect intensities to be on a consistent scale.

---

## Filter order and empty spectra

The batch path may reject an input already below the enabled minimum peak
count. Metadata repair and the strict 5 ppm physical-integrity gate then run
before peak processing. Intensity filtering precedes minimum-count filtering;
fragment truncation, top-N reduction and normalization follow in that order.
The minimum count is not reapplied after truncation. An empty spectrum created
by truncation can therefore survive processing; this correction preserves that
existing policy rather than introducing a new empty-spectrum filter.

For example, a spectrum with precursor m/z 800 and fragments at 49, 50, 100,
500 and 501 survives a 50–500 fragment window with peaks 50, 100 and 500.
Precursor candidacy is a separate scoring setting. The physical-integrity gate
still rejects a declared formula/adduct whose precursor exceeds the fixed 5 ppm
tolerance.

## Processing fingerprints and cache migration

`MassFlow.library.processing_fingerprint()` hashes canonical JSON containing
`processing_semantics_revision` and the JSON processing configuration, with
sorted keys and compact separators. The explicit revision changes when an
algorithm change alters processing even if the configuration stays identical;
it is independent of package versions and Git SHAs. It is also recorded in
library-build provenance.

The integrated candidate uses processing-semantics revision 4. It covers
missing-precursor normalization, fragment-only m/z windows, charge/adduct
consistency and per-acquisition source identity. Earlier config-only and
standalone revisions invalidate managed SQLite, Zarr and hybrid caches built
from raw files. Staged replacement rebuilds without mixing old/new rows, then
reuses a compatible rebuilt cache.

Explicit `.db`, `.sqlite` and `.zarr` input libraries are opened directly and
are **not** rebuilt or rewritten automatically. Previously generated persistent
stores may retain rows discarded or altered under old processing. Rebuild from
raw source into a **new destination** when corrected processing is required;
appending to an existing store does not remove its old rows.

Do not reuse the standalone revision numbers described in individual PR
evidence reports; the combined revision is the cache contract for this candidate.
