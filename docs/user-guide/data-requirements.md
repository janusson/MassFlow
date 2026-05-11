# MassFlow Data Requirements & Metadata Contract

MassFlow employs a rigid Pydantic validation layer to ensure scientific accuracy during spectral annotation. This document outlines the exact metadata requirements for user-provided spectral files (e.g., `.msp`, `.mgf`) and explains how MassFlow processes, repairs, and validates these fields.

## Supported File Formats

MassFlow requires open, vendor-neutral data formats for both query spectra and reference libraries:

*   **Supported:** `.mzML`, `.mzXML`, `.mgf`, `.msp`
*   **Unsupported:** Proprietary vendor formats (`.raw`, `.d`, `.wiff`, `.lcd`, `.t2d`). These must be converted to an open format (like `.mzML`) using tools like ProteoWizard MSConvert prior to ingestion.

## Required Metadata Fields

To successfully pass the `SpectrumMetadata` and `MolecularStructure` validation contracts, the following fields are strictly required for every spectrum:

| Field | Description | Type / Format |
| :--- | :--- | :--- |
| `precursor_mz` | The measured *m/z* of the precursor ion. | Float (e.g., `195.088`) |
| `charge` | The integer charge state of the ion. | Integer (e.g., `1`, `2`, `-1`). *Note: If missing, MassFlow will attempt to impute it.* |
| `mz_array` | Array of fragment *m/z* values. | List of floats. Must be monotonically increasing. |
| `intensity_array` | Array of fragment intensities. | List of floats. Must match the length of `mz_array`. |

### Charge Imputation
If the `charge` is missing, the `matchms` filtering sequence (`make_charge_int`) will attempt to derive it. However, if the charge cannot be determined and the pipeline reaches the strict mass validation layer with a missing charge, the spectrum will be rejected.

## Optional & Conditionally Required Metadata

While not strictly required for basic ingestion, the following fields unlock advanced validation, structural verification, and Machine Learning capabilities.

### 1. `adduct` and `ionmode`
MassFlow rigorously validates the relationship between `precursor_mz`, `charge`, exact mass, and the ionization adduct.

*   **`adduct`:** Must be a standard notation string matching MassFlow's internal registry (e.g., `[M+H]+`, `[M-H]-`, `[M+Na]+`). Non-standard names (e.g., "sodium adduct") will cause strict validation to fail.
*   **`ionmode`:** Must be exactly `"positive"`, `"negative"`, or `"neutral"`.

**Imputation Behavior:**
*   If `adduct` is missing but `ionmode` is provided, MassFlow defaults to `[M+H]+` for positive mode and `[M-H]-` for negative mode.
*   The `matchms` filter `derive_adduct_from_name` will attempt to parse the adduct from the compound name if possible.

### 2. Structural Identifiers (`smiles`, `inchi`, `inchikey`)
Structural identifiers are optional, but their presence fundamentally alters how MassFlow validates the spectrum.

#### The 5 ppm Strict Mass Validation
If a **`smiles`** or **`inchi`** is provided, MassFlow triggers rigorous structural validation:

1.  **Parsing:** The structure is parsed using RDKit. If the SMILES/InChI is syntactically invalid, the spectrum is flagged as physically invalid.
2.  **Theoretical Calculation:** RDKit calculates the theoretical monoisotopic exact mass of the molecule.
3.  **Conflict Checking:** If your library provides an `exact_mass` field alongside the SMILES, MassFlow checks for conflicts. If the provided mass deviates from the RDKit-calculated mass by **> 5.0 ppm**, the pipeline flags the spectrum as physically invalid, assuming a corrupted library entry.
4.  **Adduct Validation:** If the exact mass is known (calculated from SMILES) and the `adduct` is standard, MassFlow calculates the theoretical *m/z* of the precursor ion. If the experimental `precursor_mz` deviates from this theoretical *m/z* by **> 5.0 ppm**, the spectrum is flagged as physically invalid.

*Note: In future updates, "physically invalid" spectra will gracefully fallback to classical Cosine scoring, bypassing advanced structural checks rather than crashing the pipeline.*

#### Isotopic Envelope Generation
When a valid **`smiles`** is present, MassFlow automatically calculates a theoretical MS1 isotopic envelope (M, M+1, M+2, etc., normalized to the base peak). This theoretical envelope acts as a ground-truth signature, used by advanced ML routing and the `ConsensusEngine` to break ties between competing MS2 fragmentation annotations. If `smiles` is missing, this advanced credibility check cannot be performed.

### 3. General Metadata
| Field | Description | Type / Format |
| :--- | :--- | :--- |
| `retention_time` | Chromatographic retention time. | Float (seconds). Extracted/formatted automatically if possible. |
| `exact_mass` | Provided monoisotopic mass. | Float. Will be auto-calculated if SMILES is present. |
| `formula` | Chemical formula. | String (e.g., `C8H10N4O2`). Will be auto-calculated if SMILES is present. |

## Metadata Harmonization Pipeline

During ingestion, MassFlow runs a series of `matchms` filters (configurable via `ProcessingConfig`) designed to repair common library issues before strict Pydantic validation:

1.  `default_filters`: Normalizes common keys (e.g., `mz` -> `precursor_mz`).
2.  `repair_inchi_inchikey_smiles`: Attempts to fix broken formatting in structural identifiers.
3.  `harmonize_undefined_*`: Cleans up undefined strings (e.g., "N/A", "null") in structural fields.
4.  `derive_formula_from_name` & `derive_adduct_from_name`: Attempts to extract missing data from the `compound_name` string.

If these filters cannot repair a non-standard entry, and it lacks the critical fields (or violates the 5 ppm physics check when SMILES are present), the spectrum will be flagged or rejected depending on the active validation mode.
