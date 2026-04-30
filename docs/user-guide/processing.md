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
*   `add_retention_time`: Attempts to extract and format retention time data from the raw file into a standardized numeric format.
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
*   `filter_by_mz`: Master toggle for m/z truncation.
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

## Fail-Fast Behavior

MassFlow's processing pipeline is designed to "fail fast".

If a spectrum becomes invalid during processing—for example, if noise filtering removes so many peaks that the spectrum falls below the `min_peaks` threshold—the `processing.process_spectra` pipeline will silently drop that spectrum from the iterator rather than passing empty arrays to the similarity engine.
