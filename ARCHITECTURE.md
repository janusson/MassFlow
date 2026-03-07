# MassFlow Architecture

## System Overview

MassFlow is a tandem mass spectrometry (MS/MS) data analysis pipeline designed for spectral cleaning, filtering, and similarity searching. The system is built using a modular architecture that emphasizes type safety (via Pydantic), extensibility (via the Strategy Pattern), and robustness (via the Facade Pattern over `matchms`).

The architecture strictly separates I/O side-effects from stateless processing logic, ensuring that the pipeline remains reproducible, testable, and highly stable even when encountering malformed vendor data.

---

## Component Architecture Diagram

This diagram outlines the core modules of MassFlow and their structural relationships.

```mermaid
graph TD
    subgraph User Interfaces
        CLI["<b>CLI</b><br>(src/MassFlow/cli.py)"]
        GUI["<b>GUI</b><br>(src/MassFlow/ui/main.py)"]
        API["<b>Python API</b>"]
    end

    subgraph Orchestration
        Workflow["<b>Workflow Orchestrator</b><br>(src/MassFlow/workflow.py)"]
        Config["<b>Configuration Manager</b><br>(src/MassFlow/config.py)<br><i>Pydantic</i>"]
    end

    subgraph Data Ingestion & Storage
        IO["<b>I/O Layer</b><br>(src/MassFlow/io.py)"]
        DB["<b>Spectral Database</b><br>(src/MassFlow/database.py)<br><i>SQLite</i>"]
    end

    subgraph Data Processing & Validation
        Validation["<b>Data Validation</b><br>(src/MassFlow/validation.py)<br><i>Pydantic</i>"]
        Process["<b>Processing Facade</b><br>(src/MassFlow/processing.py)<br><i>matchms filtering</i>"]
    end

    subgraph Similarity Engine
        Sim["<b>Similarity Engine</b><br>(src/MassFlow/similarity.py)<br><i>Strategy Pattern</i>"]
        Algorithms["CosineGreedy<br>ModifiedCosine<br>Spec2Vec<br>MS2DeepScore"]
    end

    %% Flow Connections
    CLI --> Workflow
    GUI --> Workflow
    API --> Workflow
    
    Workflow --> Config
    Workflow --> IO
    Workflow --> Validation
    Workflow --> Process
    Workflow --> Sim
    
    IO <--> DB
    Validation --> Process
    Sim --> Algorithms
```

---

## Annotation Pipeline Data Flow (Sequence Diagram)

This diagram illustrates the sequence of operations and the flow of data when executing a typical annotation run via the CLI or GUI.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant CLI as CLI / GUI
    participant Workflow as workflow.py
    participant Config as config.py
    participant IO as io.py
    participant DB as database.py
    participant Process as processing.py
    participant Validation as validation.py
    participant Sim as similarity.py

    User->>CLI: Start Annotation (Config Path)
    CLI->>Workflow: run_annotation_pipeline(config_path)
    
    %% Config Phase
    Workflow->>Config: Parse & Validate YAML Config
    Config-->>Workflow: MassFlowConfig Object
    
    %% Ingestion Phase
    Workflow->>IO: Load Reference Library
    IO->>DB: Store/Retrieve SQLite Cache
    DB-->>IO: Cached Spectra
    IO-->>Workflow: Reference Spectra Generator
    
    Workflow->>IO: Load Query Spectra
    IO-->>Workflow: Query Spectra Generator
    
    %% Processing Phase
    loop For each Reference and Query Spectrum
        Workflow->>Validation: Validate Metadata (SpectrumSchema)
        Validation-->>Workflow: Validated Spectrum (Pydantic Fail-Fast)
        Workflow->>Process: Apply Filters (matchms Facade)
        Process-->>Workflow: Cleaned & Normalized Spectrum
    end
    
    %% Computation Phase
    Workflow->>Sim: Initialize SimilarityEngine(algorithm_choice)
    Workflow->>Sim: search(processed_queries, processed_references)
    Sim->>Sim: Calculate Scores Matrix (Vectorized Numpy)
    Sim-->>Workflow: List[SearchResult] (Thresholded)
    
    %% Export Phase
    Workflow->>IO: save_match_results(results, output_path)
    IO-->>Workflow: CSV Export Complete
    Workflow-->>CLI: Pipeline Finished
    CLI-->>User: Execution Summary / Success
```

---

## Module Responsibilities

### `MassFlow.cli` & `MassFlow.ui`
The entry points for users. The CLI (`cli.py`) handles command-line arguments and logging, while the GUI (`ui/main.py` using `CustomTkinter`) provides an interactive interface for configuring and running the pipeline without touching the terminal. Both ultimately dispatch to the `workflow` module.

### `MassFlow.workflow`
The central orchestrator (`workflow.py`). It manages the lifecycle of a mass spectrometry analysis:
1. Loading and validating the configuration via `config.py`.
2. Ingesting reference and query data via `io.py`.
3. Coordinating processing and validation steps (`processing.py`, `validation.py`).
4. Executing similarity searches (`similarity.py`).
5. Saving final results to disk.

### `MassFlow.config`
Defines the schema for the entire system using Pydantic (`config.py`). It includes custom validators for mass spectrometry parameters (e.g., ensuring non-negative $m/z$ and retention times, algorithm choices) and provides a unified interface for YAML-based configuration.

### `MassFlow.io`
A unified I/O layer (`io.py`) that abstracts the complexity of different spectral formats (MGF, MSP, mzML). It handles both the streaming of input data, automated format conversions (e.g., triggering `msconvert` for vendor formats), and the structured export of results to CSV or other formats.

### `MassFlow.database`
Implements a local SQLite storage backend (`database.py`) for efficient, batched caching and retrieval of massive spectral reference libraries, drastically reducing memory overhead and load times for repeated runs.

### `MassFlow.validation`
A Pydantic-powered validation layer (`validation.py`) that strictly enforces data integrity. It catches and cleans malformed inputs (like string values like "N/A" in numeric fields) early, acting as a fail-fast mechanism before dirty data hits downstream processing and causes silent calculation errors.

### `MassFlow.processing`
Acts as a facade for the `matchms` filtering library (`processing.py`). It implements a strict, sequential cleaning pipeline:
1. **Metadata Cleaning**: Standardization of names, adducts, formulas, and IDs.
2. **Intensity Filtering**: Removal of noise peaks below strict baselines.
3. **Peak Count Filtering**: Ensuring spectra meet minimum quality requirements or restricting to Top-N peaks.
4. **Normalization**: Scaling intensities to a consistent 0.0 to 1.0 range.

### `MassFlow.similarity`
Implements spectral similarity calculations (`similarity.py`) using the **Strategy Design Pattern**. This provides a unified API to swap between multiple backend algorithms at runtime:
*   **CosineGreedy:** Bipartite peak matching heuristic for exact fragment overlaps.
*   **ModifiedCosine:** Mass-shift incorporated peak matching for structurally related analogues.
*   **Spec2Vec / MS2DeepScore:** Advanced machine learning embeddings for contextual structural relationships.

It leverages vectorized `numpy` operations and sparse matrices for efficient math, strictly avoiding standard Python loops for peak iteration to scale to hundreds of thousands of spectral comparisons.