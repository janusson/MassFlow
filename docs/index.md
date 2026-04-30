# MassFlow

**MassFlow** is a config-first Python toolkit for local, reproducible tandem mass spectrometry (MS/MS) annotation workflows. It is designed to be **dead easy to run** locally, producing highly reproducible scientific outputs.

!!! example "TL;DR - Dead easy to run"
    Run your entire annotation pipeline in just two commands:
    ```shell
    # 1. Generate a default config file
    uv run massflow init

    # 2. Run your annotation pipeline
    uv run massflow annotate --config massflow_config.yaml
    ```

## Mission

MassFlow v1.0 delivers a reliable and easily configurable pipeline that takes open-format experimental spectra and reference libraries, applies standard `matchms` processing, performs classical similarity searching, and produces predictable tabular results.

We prioritize stability, predictability, and correct data handling over cutting-edge features.

---

## Stable vs. Experimental

| Surface | Status | Notes |
| --- | --- | --- |
| `massflow annotate --config ...` | **Stable** | Main documented workflow |
| YAML configuration | **Stable** | Standardized execution parameters |
| Open-format ingestion (`mzML`, `mzXML`, `MGF`, `MSP`) | **Stable** | Vendor raw conversion is explicitly out of scope |
| SQLite library workflows (`massflow db ...`) | **Stable** | Recommended for reusable local libraries |
| `cosine` and `modified_cosine` | **Stable** | Best-supported classical scoring paths |
| CSV result export | **Stable** | Main reporting surface (w/ YAML provenance reports) |
| Orchestrator API (`ConsensusEngine`, etc.) | *Experimental* | Engine-agnostic data contracts for v1.1 ML integration |
| Advanced Engines (`spec2vec`, `ms2deepscore`, `cascade`) | *Experimental* | Higher setup and complex scientific validation |
| GraphML networking | *Experimental* | Optional and non-core |

---

## Why MassFlow?

MassFlow is designed specifically for domain experts (Analytical Chemists, Metabolomics Researchers) who just want their annotations to work, predictably and reliably.

1.  **Stop writing ad-hoc scripts:** No more copy-pasting code from old projects. Keep all your noise thresholds and match tolerances in one easy-to-read, version-controlled YAML file.
2.  **Use open formats directly:** Stop worrying about converters. Feed your `mzML`, `mzXML`, `MGF`, and `MSP` files directly into the pipeline.
3.  **Say goodbye to memory crashes:** Instead of loading massive 10GB reference libraries into RAM every time, use MassFlow to compress them into lightning-fast local SQLite databases.
4.  **Protect your science:** We strictly enforce scientific constraints. MassFlow protects critical metadata like precursor m/z and retention time so your alignments are always mathematically sound.
5.  **Easy Excel-ready exports:** Review your final annotations via standard, clean CSV files. We even provide a "receipt" alongside every CSV so you know exactly which settings produced it for your paper.

---

## Getting Started

Ready to dive in? Head over to the [Installation](getting-started/installation.md) and [Quickstart](getting-started/quickstart.md) guides to configure your first MS/MS annotation run.
