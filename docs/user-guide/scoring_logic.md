# MassFlow Scoring Logic

This document provides transparency into the mathematical operations and filtering logic used by MassFlow to generate similarity annotations.

## 1. Precursor Matching & Strict 5.0 ppm Validation

MassFlow employs high-precision precursor matching. When structural identifiers (SMILES) are provided, the workflow performs a rigorous physics-informed mass validation.

### Equation: Precursor Mass Error (ppm)
The difference between the experimental precursor $m/z$ ($m_{\text{exp}}$) and the theoretical $m/z$ ($m_{\text{theo}}$) is calculated as:

$$ \text{Error (ppm)} = \frac{|m_{\text{exp}} - m_{\text{theo}}|}{m_{\text{theo}}} \times 10^6 $$

*   **$m_{\text{theo}}$** is computed by parsing the SMILES string in RDKit to determine the monoisotopic exact mass, then adding the exact mass offset of the ionization adduct.
*   **Threshold:** If this Error > 5.0 ppm, the match (or library entry) is flagged as physically invalid.

## 2. Cosine Scoring

The classical `cosine` and `modified_cosine` algorithms compute similarity between the $m/z$ and intensity arrays of the query and reference spectra.

*   **Peak Matching:** MassFlow bins and matches fragment peaks between spectra using the user-defined `ms2_tolerance` (e.g., 0.02 Da).
*   **Cosine Calculation:** The normalized dot product (cosine angle) of the matched intensity arrays is computed. Both algorithms support intensity weighting powers.

## 3. False Discovery Rate (FDR) vs. Empirical P-Value

To estimate statistical confidence, MassFlow generates a null distribution of *decoy spectra* (by shuffling fragment intensities to break structural correlations).

### Large Libraries ($\ge 2000$ spectra)
MassFlow uses a standard Target-Decoy approach to calculate the $q$-value (FDR) of a match:

$$ \text{FDR} = \frac{N_{\text{decoys}} + 1}{N_{\text{targets}}} $$

### Small Libraries ($< 2000$ spectra)
Target-Decoy FDR generates overly conservative (often 1.0) $q$-values for small specialized in-house libraries. When MassFlow detects a small library, it automatically switches to an **Empirical P-Value** calculation:

$$ p = \frac{\sum (S_{\text{decoy}} \ge S_{\text{target}}) + 1}{N_{\text{total\_decoys}} + 1} $$

This provides a valid statistical metric (the probability of observing a score $S$ by chance) suitable for custom libraries.
