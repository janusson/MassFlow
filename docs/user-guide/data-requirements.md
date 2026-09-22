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
If the `charge` is missing, the `matchms` filtering sequence (`make_charge_int`) will attempt to derive it. If the charge cannot be determined, the strict 5 ppm mass comparison is skipped for that spectrum (an unknown charge disables the check per the `SpectrumMetadata` contract); the spectrum remains analyzable through classical cosine scoring.

## Optional & Conditionally Required Metadata

While not strictly required for basic ingestion, the following fields unlock advanced validation, structural verification, and Machine Learning capabilities.

### 1. `adduct` and `ionmode`
MassFlow rigorously validates the relationship between `precursor_mz`, `charge`, exact mass, and the ionization adduct.

*   **`adduct`:** Must be an ionization adduct whose chemistry MassFlow recognises. The registry covers the common LC-MS adducts (`[M+H]+`, `[M-H]-`, `[M+Na]+`, `[M+NH4]+`, `[M+K]+`, `[M+Cl]-`, `[M+HCOO]-`, `[M+CH3COO]-`, multiply-charged and solvent/water-loss variants — see `docs/user-guide/validation.md` for the full table and offsets).
    *   **Notation is canonicalised, so library spellings are accepted:** `M+H`, `[M+H]1+`, `[M+H]+1`, `[m+h]+`, interior whitespace, and unbalanced brackets all resolve to `[M+H]+`; alias chemistry such as `M+FA-H` / `M+HCOOH-H` (formate), `M+OAc` / `M+Ac-H` (acetate), `M+ACN+H`, and `M-H2O+H` resolves too. The stored value is rewritten to the canonical spelling.
    *   Prose descriptions (e.g. "sodium adduct") and adducts outside the registry cannot be resolved, so strict validation fails for those spectra; normalisation never guesses, and a resolved adduct must still clear the 5 ppm tolerance.
*   **`ionmode`:** Must be exactly `"positive"`, `"negative"`, or `"neutral"`.

**Imputation Behavior:**
*   If `adduct` is missing but `ionmode` is provided, MassFlow defaults to `[M+H]+` for positive mode and `[M-H]-` for negative mode.
*   The `matchms` filter `derive_adduct_from_name` will attempt to parse the adduct from the compound name if possible.

### 2. Structural Identifiers (`smiles`, `inchi`, `inchikey`)
Structural identifiers are optional, but their presence fundamentally alters how MassFlow validates the spectrum.

#### The 5 ppm Strict Mass Validation
If a **`smiles`**, **`inchi`**, or **`formula`** is provided, MassFlow triggers rigorous structural validation:

1.  **Parsing:** The structure is parsed using RDKit (when a formula is not already declared). If the SMILES/InChI is syntactically invalid and the `matchms` repair filters could not fix it, the spectrum is flagged as physically invalid.
2.  **Theoretical Calculation:** The theoretical monoisotopic exact mass of the molecule is calculated (pyteomics from the declared formula; RDKit + pyteomics from the structure otherwise).
3.  **Conflict Checking:** If your library provides an `exact_mass` field alongside the structure, MassFlow checks for conflicts. If the provided mass deviates from the calculated mass by **> 5.0 ppm**, the spectrum is flagged as physically invalid, assuming a corrupted library entry.
4.  **Adduct Validation:** If the exact mass is known and the `adduct` is standard, MassFlow calculates the theoretical *m/z* of the precursor ion. If the experimental `precursor_mz` deviates from this theoretical *m/z* by **> 5.0 ppm**, the spectrum is flagged as physically invalid.

**Enforcement in the pipeline** — "physically invalid" spectra are rejected by the processing gate in the classical `annotate`/`db build` paths (see `docs/user-guide/validation.md`): query spectra are rejected and counted per file (an all-rejected file is an explicit failure), raw reference-library entries abort the annotate run, and `db build` quarantines them. Spectra without structural claims are exempt from the gate.

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

If these filters cannot repair a non-standard entry, and it lacks the critical fields (or violates the 5 ppm physics check when structural identifiers are present), the spectrum is rejected by the strict processing gate: counted per query file (see the [failure model](results.md)), fatal for raw reference libraries during `annotate`, and quarantined during `db build`.
