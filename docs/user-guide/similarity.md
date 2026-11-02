# Classical Similarity Scoring

The restricted candidate-retrieval scope centers on `cosine` and `modified_cosine` from `matchms`. These engines score spectra under configured peak and precursor domains; their outputs do not confirm identities.

These engines compare experimental and reference m/z/intensity arrays to retrieve and rank candidate library records. Similarity alone does not confirm chemical identity.

---

## Supported Stable Engines

The `similarity.algorithm` field in your `massflow_config.yaml` determines which scoring engine is used for the annotation workflow.

### `cosine`
The classic Cosine Greedy algorithm. It computes the normalized dot product of the aligned fragment peaks between two spectra.

```yaml
similarity:
  algorithm: "cosine"
  ms1_tolerance: 0.02
  ms2_tolerance: 0.02
  tolerance_unit: "Da"
  min_score: 0.6
  min_matched_peaks: 3
  fdr_threshold: 0.05
```

**When to use:** This is the standard choice for comparing identical MS/MS spectra (e.g., standardizing against an authenticated reference library). The precursor m/z of the query and reference must match within the configured `ms1_tolerance`.

### `modified_cosine`
An extension of the classic cosine algorithm designed to account for mass shifts (neutral losses) between a query and a reference. It aligns peaks by calculating the difference between the precursor masses ($\Delta M$) and shifting the query fragments by that amount.

```yaml
similarity:
  algorithm: "modified_cosine"
  ms2_tolerance: 0.02
  tolerance_unit: "Da"
  min_score: 0.6
  min_matched_peaks: 3
  fdr_threshold: 0.05
```

**When to use:** Search for spectra with precursor-shifted fragment patterns, such as possible structural analogues. A modified-cosine match alone does not confirm an analogue or molecular identity.

---

## Configuration Parameters

The legacy `fdr_threshold` is a numerical candidate filter, not a calibrated
false-discovery guarantee. Consult [confidence semantics](confidence-semantics.md)
before interpreting q-like values.

The `similarity` section controls candidate scoring and retention. Structural metadata checks are described separately; a retained candidate is not a validated identification.

### `ms1_tolerance`
*   **Type:** `float`
*   **Default:** `10.0`
*   **Description:** The maximum allowable difference between the query precursor m/z and the reference precursor m/z.
*   **Note:** This is only strictly enforced by the standard `cosine` engine. The `modified_cosine` engine intentionally ignores this to allow for mass-shifted analog searching.

### `ms2_tolerance`
*   **Type:** `float`
*   **Default:** `0.02`
*   **Description:** The maximum allowable m/z difference for two fragment peaks to be considered an alignment match during scoring.

### `tolerance_unit`
*   **Type:** `string` (`"Da"` or `"ppm"`)
*   **Default:** `"Da"`
*   **Description:** The unit of measurement applied to the tolerances.

### `min_score`
*   **Type:** `float`
*   **Default:** `0.6`
*   **Description:** The absolute minimum similarity score (0.0 to 1.0) required to retain a candidate before configured confidence filtering.

### `min_matched_peaks`
*   **Type:** `int`
*   **Default:** `3`
*   **Description:** The minimum number of aligned fragment peaks required. Even if two spectra share a single extremely intense peak that yields a high cosine score, the hit will be rejected if it does not meet this physical evidence threshold.

### `fdr_threshold`
*   **Type:** `float`
*   **Default:** `0.05`
*   **Description:** Legacy threshold applied to the numerical query-level target/decoy estimate. Estimates are explicitly uncalibrated; changing this value changes candidate selection, not the evidence for FDR control. See [confidence semantics](confidence-semantics.md).

---

## Experimental Engines

MassFlow contains several advanced machine learning models and orchestration logic (like `spec2vec`, `ms2deepscore`, `consensus`, and `cascade`).

Because these engines require external model weights, complex validation, or specific tie-breaking logic, they are not currently part of the stable v0.1 pipeline contract.

For documentation on how to configure and run these engines at your own risk, see the [Similarity Engine API](../api/similarity.md) reference.
