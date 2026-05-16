# MassFlow Tutorial: Steroid Annotation & Networking

This tutorial demonstrates how to use MassFlow to annotate experimental MS/MS spectra using a reference library of steroids. We will cover configuration, processing, and interpreting similarity results.

---

## 1. High-Level Workflow

The diagram below illustrates how inputs flow through the MassFlow pipeline:

```mermaid
graph TD
    subgraph Inputs
        E[tutorial_experimental.mgf<br/>4 Scans]
        L[tutorial_library.msp<br/>3 Steroids]
    end

    C[tutorial_config.yaml] -.->|Configures| M

    E --> M{MassFlow CLI}
    L --> M

    M --> P[Processing Pipeline<br/>Clean Metadata & Filter Peaks]
    P --> S[Similarity Engine<br/>Score & Rank]
    S --> R[results/ directory<br/>CSV Results & YAML Report]

    style C fill:#f9f,stroke:#333,stroke-width:2px
    style M fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 2. Tutorial Dataset

The `/tutorial` folder contains:
- `tutorial_library.msp`: A reference library containing 3 steroids (Testosterone, Progesterone, and Cortisol) with standardized metadata (InChIKey, SMILES).
- `tutorial_experimental.mgf`: Simulated experimental data with 4 scans.

Here is what MassFlow will do with each scan:

```mermaid
flowchart LR
    subgraph Experimental Data
        S101[Scan 101<br/>Testosterone Match]
        S102[Scan 102<br/>Noisy Progesterone]
        S103[Scan 103<br/>Shifted Cortisol +2Da]
        S104[Scan 104<br/>Random Noise]
    end

    subgraph Processing Pipeline
        PF[min_peaks filter]
    end

    subgraph Expected Annotation
        O1[High Score: Testosterone]
        O2[Good Score: Progesterone]
        O3[Low Cosine Score /<br/>High Modified Cosine Score]
        O4[Dropped / No Result]
    end

    S101 --> PF --> O1
    S102 --> PF --> O2
    S103 --> PF --> O3
    S104 -.->|Filtered out by min_peaks| PF -.-> O4
```

---

## 3. Configuration Walkthrough

Open `tutorial/tutorial_config.yaml`. The key sections are:

### Input
Points to our tutorial files:
```yaml
input:
  file_path: "tutorial/tutorial_experimental.mgf"
  library_path: "tutorial/tutorial_library.msp"
```

### Processing
Ensures spectra are cleaned and normalized before scoring:
```yaml
processing:
  clean_metadata: true
  normalize_intensity: true
  filter_min_peaks: true
  min_peaks: 3
```

### Similarity
Uses the standard `cosine` algorithm. We've set `min_score: 0.1` very low for this tutorial to ensure we see the scores for all candidates.
```yaml
similarity:
  algorithm: "cosine"
  ms1_tolerance: 0.02
  ms2_tolerance: 0.02
  tolerance_unit: "Da"
```

---

## 4. Running the Analysis

Execute the annotation workflow from the project root:

```bash
uv run massflow annotate --config tutorial/tutorial_config.yaml
```

Check the `tutorial/results/` directory after running.

### Annotation Report (`tutorial_experimental_results.csv`)
- **Scan 101** should show a high score (>0.9) against Testosterone.
- **Scan 102** should show a high score against Progesterone.
- **Scan 103** (Modified Cortisol) will likely have a **low** score with standard `cosine` because the peaks are shifted.

---

## 5. Experimenting with Molecular Networking (Modified Cosine)

To find related molecules like the shifted Cortisol in **Scan 103**, we need an algorithm that forgives consistent mass shifts. Change the algorithm in `tutorial_config.yaml` to `modified_cosine`:

```yaml
similarity:
  algorithm: "modified_cosine"
```

Run the annotation again. You will notice that the score for Scan 103 against Cortisol increases significantly.

### How Modified Cosine Works

The diagram below shows why standard cosine fails on Scan 103 and why modified cosine succeeds:

```mermaid
sequenceDiagram
    participant E as Scan 103 (Shifted +2Da)
    participant M as Modified Cosine Engine
    participant L as Library: Cortisol

    Note over E, L: 1. Calculate Precursor Difference
    E->>M: Query Precursor (365.217)
    L->>M: Ref Precursor (363.217)
    Note over M: Delta = +2.000 Da

    Note over E, L: 2. Attempt Standard Match
    E-->>M: Query Fragment (e.g., 123.065)
    L-->>M: Ref Fragment (121.065)
    Note over M: 123 != 121 (Standard Cosine Misses)

    Note over E, L: 3. Attempt Shifted Match
    Note over M: Shift Ref Fragment by Delta (+2.000)<br/>121.065 + 2.000 = 123.065
    M->>M: 123.065 == 123.065 (Modified Cosine Matches!)
    M-->>E: Returns High Similarity Score
```

## 6. Summary
This small dataset showcases:
1. **Spectral Matching**: High-confidence identification of knowns.
2. **Noise Resilience**: Filtering out low-quality scans.
3. **Modified Cosine**: Identifying structurally related molecules with mass offsets, which is the foundation of molecular networking.
