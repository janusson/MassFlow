# Scientific Validation & Data Integrity

MassFlow applies explicit checks to declared molecular metadata and reports processing rejections. These checks can detect inconsistencies under the registered chemistry and fixed tolerances; they do not prove a structure or establish universal instrument, matrix, or adduct validity.

---

## Precursor Mass Validation (5 ppm)

A cornerstone of MassFlow's data integrity is the **5.0 ppm precursor tolerance check**.

In high-resolution mass spectrometry, the experimental precursor m/z must align with the theoretical monoisotopic mass of the candidate molecule, adjusted for its ionization adduct and charge state.

When the `MassFlow.models` layer processes a candidate structure (via a chemical formula, SMILES, or InChI), it automatically:

1. Calculates the **exact monoisotopic mass** of the neutral molecule (formula-first, pyteomics; structure parsing via RDKit when no formula is declared).
2. Identifies the **adduct offset** (e.g., +1.007276 Da for `[M+H]+`).
3. Requires the declared **signed charge** to agree with the registered adduct (for example, `[M+2H]2+` requires `+2`). Contradictory charge magnitude or sign sets `is_physically_valid = False`.
4. Computes the **theoretical m/z**: $(ExactMass + AdductOffset) / |Charge|$.
5. Compares this to your experimental `precursor_mz`.

If the deviation is greater than **5.0 ppm**, the declared metadata fails this fixed software check. This prevents a candidate row from passing on contradictory declared precursor/structure metadata; it does not confirm candidates that pass.

### Where the check is enforced

* **Model layer** — the two Pydantic contracts above always compute the verdict (`is_physically_valid`); this is the stable core implementation, and both streaming and classical paths consume it.
* **Streaming ingestion gate** (experimental `stream-server`) — incoming packets that violate the contracts are rejected with a structured `StreamingValidationError` before scoring.
* **Classical `annotate` / `db build` processing gate** — during metadata processing (`MassFlow.processing`), explicitly declared recognized adduct/charge pairs are checked after `matchms` harmonization; spectra with structural claims also receive structural and 5 ppm mass checks:
  * a *query* spectrum that fails the gate (malformed claim, unsupported adduct with a complete context, or a **>5 ppm** precursor deviation) is **rejected and counted** (`spectra_rejected`) with a human-readable reason in the quarantine log; a file whose spectra are all rejected is an explicit `failed` result with the reasons in `<stem>_failed.report.yaml`;
  * a *raw reference library* entry that fails the gate **aborts the annotate run** with a `PhysicalIntegrityError` listing the offending spectra, because silently changing the candidate library would change every query's search results and legacy numerical estimates. `massflow db build` is the curation workflow: it quarantines such entries (they are never stored) and reports the stored count.
  * spectra **without** structural claims skip the structural/5 ppm mass checks, but an explicitly declared supported adduct/charge pair is still validated.

An adduct is required for the precursor-mass comparison: it may be declared explicitly or imputed from `ionmode` (`positive` → `[M+H]+`, `negative` → `[M-H]-`). When a recognized adduct and charge are both explicitly declared, their signed-charge consistency is checked independently of precursor availability. The 5 ppm mass comparison still requires a structural claim and usable precursor mass. Missing adduct/charge context skips the corresponding check.

### Applying validation changes to stored libraries

Managed prepared caches are fingerprinted with canonical processing configuration plus processing-semantics revision 4, which covers missing-precursor normalization, fragment-only m/z windows, charge/adduct consistency, and source-record identity. Older managed caches rebuild once from raw sources using staged replacement. A raw library with contradictory structural metadata aborts preparation with the rejection reason.

User-supplied `.db`, `.sqlite`, and `.zarr` libraries are used directly and are not revalidated or rewritten automatically. Rebuild from the original raw sources with `massflow db build` into a **new output path that does not already exist**: the command appends to an existing store, so reusing the old output preserves invalid entries. Inspect the new store and review quarantine messages/counts, then update `input.library_path` to the new store. See the [charge/adduct experiment](../validation/2026-10-17-charge-adduct-consistency.md) for evidence and limitations.

### Supported Adducts & Offsets

MassFlow maintains a high-precision internal registry of monoisotopic offsets for common LC-MS adducts:

| Adduct | Mode | Offset (Da) | Description |
| :--- | :--- | :--- | :--- |
| `[M+H]+` | Pos | +1.007276 | Protonated |
| `[M+NH4]+` | Pos | +18.033826 | Ammonium |
| `[M+Na]+` | Pos | +22.989221 | Sodium |
| `[M+K]+` | Pos | +38.963158 | Potassium |
| `[M]+` | Pos | -0.000549 | Radical Cation |
| `[M+2H]2+` | Pos | +2.014553 | Doubly protonated |
| `[M+3H]3+` | Pos | +3.021829 | Triply protonated |
| `[M+2Na-H]+` | Pos | +44.971165 | Disodium salt |
| `[M+H-H2O]+` | Pos | -17.003288 | Protonation with water loss |
| `[M+CH3CN+H]+` | Pos | +42.033826 | Acetonitrile adduct |
| `[M+CH3OH+H]+` | Pos | +33.033491 | Methanol adduct |
| `[M-H]-` | Neg | -1.007276 | Deprotonated |
| `[M-2H]2-` | Neg | -2.014553 | Doubly deprotonated |
| `[M+Cl]-` | Neg | +34.969401 | Chlorine |
| `[M+Br]-` | Neg | +78.918886 | Bromine |
| `[M+I]-` | Neg | +126.905022 | Iodine |
| `[M+HCOO]-` | Neg | +44.998203 | Formate |
| `[M+CH3COO]-` | Neg | +59.013853 | Acetate |
| `[M+TFA-H]-` | Neg | +112.985587 | Trifluoroacetate |
| `[M+HCO3]-` | Neg | +60.993117 | Bicarbonate |
| `[M+NO3]-` | Neg | +61.988366 | Nitrate |
| `[M]-` | Neg | +0.000549 | Radical Anion |

#### Notation is normalised, not guessed

Library files spell the same ion in many ways, and a spelling the registry could not resolve used to quarantine an otherwise correct spectrum. Adduct notation is therefore canonicalised before lookup (see `MassFlow.cheminformatics.normalize_adduct`):

* **Brackets and charge position:** `M+H`, `[M+H]1+`, `[M+H]+1`, and `[M+H]+` are equivalent; interior whitespace and unbalanced brackets are tolerated.
* **Letter case:** `[m+h]+` and `M+NA` resolve to `[M+H]+` and `[M+Na]+`.
* **Alias chemistry:** formate written as `M+FA-H`/`M+HCOOH-H`, acetate as `M+OAc`/`M+Ac-H`, trifluoroacetate as `M+CF3COO`, acetonitrile as `M+ACN+H`, and `M-H2O+H` all resolve to their canonical keys.

Normalisation fails **closed**: notation that declares a charge contradicting the chemistry (e.g. `[M+H]2+`), a charge-less ambiguous body (`M`), or chemistry outside the registry (e.g. `[M+Weird]+`) still resolves to nothing and is still rejected by the gate. Crucially, normalisation cannot manufacture a false positive — the resolved adduct must still clear the **5 ppm** precursor tolerance.

---

## Theoretical Isotopic Envelopes

For advanced structural verification, MassFlow automatically generates **Theoretical Isotopic Envelopes** for every reference compound with a valid SMILES string.

Using high-precision calculations (via `pyteomics` and `RDKit`), the pipeline determines the relative abundance and centroid masses of the M, M+1, M+2, and M+3 isotopologues.

* **MS1 Verification:** This envelope establishes a ground-truth signature that can be compared against experimental MS1 data.
* **Tie-Breaking:** If two candidates share similar MS2 fragmentation scores, the `ConsensusEngine` can use the isotopic pattern fit as an orthogonal tie-breaker to select the most likely structure.

---

## Spectral Integrity Checks

Beyond chemical structures, MassFlow validates the physical properties of the mass spectra themselves:

1. **Monotonicity:** The `mz_array` must be strictly increasing. Out-of-order peaks (common in malformed open formats) are detected and blocked.
2. **Array Parity:** The m/z and intensity arrays must have identical lengths.
3. **Positive Intensity:** All intensities are validated to ensure they are non-negative.
4. **Minimum Peak Counts:** Configurable thresholds (default: 5 peaks) ensure that "empty" or noise-only spectra do not consume CPU cycles in the similarity engine.

---

## Triage Bitmasking

During the library ingestion phase (`massflow db build`), MassFlow performs a fast NumPy-based scan of every spectrum to identify **diagnostic fragments**.

These fragments (e.g., the Tyrosine immonium ion at `136.076` Da) are stored in a `triage_flags` bitmask in the SQLite database, so downstream stages can route specific spectra toward specialized ML models without re-scanning the raw peak data.
