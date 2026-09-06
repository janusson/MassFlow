# Scientific Validation & Data Integrity

MassFlow v0.1 enforces strict physical and chemical boundaries at the point of ingestion. Unlike many pipelines that silently propagate low-quality or chemically impossible matches, MassFlow uses `Pydantic`-backed validation to ensure that every spectrum and molecule in your pipeline is mathematically and scientifically sound.

---

## Precursor Mass Validation (5 ppm)

A cornerstone of MassFlow's data integrity is the **5.0 ppm precursor tolerance check**.

In high-resolution mass spectrometry, the experimental precursor m/z must align with the theoretical monoisotopic mass of the candidate molecule, adjusted for its ionization adduct and charge state.

When the `MassFlow.models` layer processes a candidate structure (via SMILES or InChI), it automatically:

1. Calculates the **exact monoisotopic mass** of the neutral molecule.
2. Identifies the **adduct offset** (e.g., +1.007276 Da for `[M+H]+`).
3. Computes the **theoretical m/z**: $(ExactMass + AdductOffset) / |Charge|$.
4. Compares this to your experimental `precursor_mz`.

If the deviation is greater than **5.0 ppm**, the record is rejected with a `ValidationError`. This prevents "lucky" MS2 matches from being reported if the parent mass doesn't physically support the identification.

### Supported Adducts & Offsets

MassFlow maintains a high-precision internal registry of monoisotopic offsets for common LC-MS adducts:

| Adduct | Mode | Offset (Da) | Description |
| :--- | :--- | :--- | :--- |
| `[M+H]+` | Pos | +1.007276 | Protonated |
| `[M+NH4]+` | Pos | +18.033826 | Ammonium |
| `[M+Na]+` | Pos | +22.989221 | Sodium |
| `[M+K]+` | Pos | +38.963158 | Potassium |
| `[M]+` | Pos | -0.000549 | Radical Cation |
| `[M-H]-` | Neg | -1.007276 | Deprotonated |
| `[M+Cl]-` | Neg | +34.969401 | Chlorine |
| `[M+HCOO]-` | Neg | +44.998203 | Formate |
| `[M+CH3COO]-` | Neg | +59.013853 | Acetate |
| `[M]-` | Neg | +0.000549 | Radical Anion |

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
