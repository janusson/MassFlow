# MassFlow

**MassFlow** is a config-first Python toolkit for local, reproducible tandem mass spectrometry (MS/MS) annotation workflows. It is designed to be **dead easy to run** locally, producing highly reproducible scientific outputs.

!!! example "TL;DR - Dead easy to run"
    Run your entire annotation pipeline in just two commands:
    ```shell
    # 1. Generate a default config file
    uv run massflow init

    # 2. Run your annotation pipeline
    uv run massflow annotate --config massflow_config.yaml

    # 3. Or, run interactively and watch for file changes
    uv run massflow watch --config massflow_config.yaml
    ```

## Mission

MassFlow v0.1 delivers a reliable and easily configurable pipeline that takes open-format experimental spectra and reference libraries, applies standard `matchms` processing, performs classical similarity searching, and produces predictable tabular results.

We prioritize stability, predictability, and correct data handling over cutting-edge features.

---

## Stable vs. Experimental

| Surface | Status | Notes |
| --- | --- | --- |
| `massflow annotate --config ...` | **Stable** | Main documented workflow |
| YAML configuration | **Stable** | Standardized execution parameters |
| Open-format ingestion (`mzML`, `mzXML`, `MGF`, `MSP`) | **Stable** | Vendor raw formats rejected with an actionable error; `massflow convert` (experimental wrapper) can convert them externally |
| SQLite library workflows (`massflow db ...`) | **Stable** | Recommended for reusable local libraries |
| `cosine` and `modified_cosine` | **Stable** | Best-supported classical scoring paths |
| CSV and mzTab-M export | **Stable** | Main reporting surfaces (w/ YAML provenance reports). FBMN export is **not shipped** |
| Model-layer scientific validation (5 ppm precursor check, isotopic envelopes) | **Stable (model layer)** | Implemented and tested in `MassFlow.models`/`cheminformatics`; enforced as an ingestion gate in the streaming path. Not enforced as a gate on library/query spectra in the classical `annotate` path |
| `massflow watch --config ...` | *Experimental* | Interactive live-reloading workflow (`[watch]` extra) |
| `massflow stream-server` | *Experimental* | Real-time gRPC streaming (loopback default, TLS/auth required for remote) |
| `massflow tui` | *Experimental* | Interactive terminal console (`[tui]` extra) |
| Advanced Engines (`spec2vec`, `ms2deepscore`, `consensus`, `cascade`) | *Experimental* | Higher setup and complex scientific validation; outside the stable support promise |
| GraphML networking & Visualization | *Planned* | Documented in places, **not implemented** — do not rely on it |
| Language Server (LSP) | Removed | The standalone LSP module is not part of the codebase; `docs/api/server.md` documents the removal |

---

## Why MassFlow?

MassFlow is designed specifically for domain experts (Analytical Chemists, Metabolomics Researchers) who just want their annotations to work, predictably and reliably.

1. **Stop writing ad-hoc scripts:** No more copy-pasting code from old projects. Keep all your noise thresholds and match tolerances in one easy-to-read, version-controlled YAML file.
2. **Use open formats directly:** Stop worrying about converters. Feed your `mzML`, `mzXML`, `MGF`, and `MSP` files directly into the pipeline, or use our `massflow convert` wrapper for vendor files.
3. **Say goodbye to memory crashes:** Instead of loading massive 10GB reference libraries into RAM every time, use MassFlow to compress them into lightning-fast local SQLite databases.
4. **Protect your science:** We strictly enforce scientific constraints. MassFlow physically validates precursor masses against their theoretical exact mass and adduct to within a strict 5.0 ppm tolerance, preventing impossible matches.
5. **Easy Excel-ready exports:** Review your final annotations via standard, clean CSV files. We even provide a "receipt" alongside every CSV so you know exactly which settings produced it for your paper.

---

## Getting Started

Ready to dive in? Head over to the [Installation](getting-started/installation.md) and [Quickstart](getting-started/quickstart.md) guides to configure your first MS/MS annotation run.
