# Architecture Diagrams

This document contains simplified, high-level visual representations of the MassFlow system.

## 1. System Summary
This diagram shows the 10,000-foot view of MassFlow.

```mermaid
graph LR
    Config[YAML Config] --> Engine{MassFlow}
    Experiments[(Experiments)] --> Engine
    Library[(Library)] --> Engine
    Engine --> CSV[CSV Results]
    Engine --> YAML[YAML Provenance]
```

## 2. Core Data Flow (The Scientific Pipeline)
This sequence illustrates the clean-score-filter-export pipeline.

```mermaid
graph TD
    Exp[Query Spectrum] --> Clean[Processing Filters]
    Ref[Reference Library] --> Clean
    Clean --> Sim{Similarity Scoring}
    Sim --> FDR[FDR Filtering]
    FDR --> Results[Filtered Results]
    Results --> CSV[CSV Export]
```

## 3. Multiprocessing & Performance
MassFlow uses parallel workers and shared memory to accelerate processing.

```mermaid
graph TD
    Parent[Parent Process] --> Shared[(Shared Library Memory)]
    Shared --> W1[Worker 1]
    Shared --> W2[Worker 2]
    Shared --> W3[Worker 3]
    W1 --> R1[Result 1]
    W2 --> R2[Result 2]
    W3 --> R3[Result 3]
```

## 4. The Orchestrator API & Consensus
How the Orchestrator routes spectra to multiple engines and calculates a consensus.

```mermaid
graph TD
    Spectrum[Spectrum] --> Triage{Triage Scan}
    Triage --> Easy[Classical Engine]
    Triage --> Hard[ML Engine]
    Easy --> Consensus{Consensus}
    Hard --> Consensus
    Consensus --> Winner[Winning Hit]
    Winner --> Report[Final Report]
```
