# Yogimass API reference (developer-focused)

This document summarizes the major public functions and classes, grouped by module. Use it as a quick map alongside `ARCHITECTURE.md`.

## Config-first workflow

- `yogimass.config.load_config(config)` — parse + validate YAML/JSON into a `WorkflowConfig` with defaults applied; raises `ConfigError(path, message)` on unknown keys, missing outputs, or invalid values.
- `ConfigError` — includes a dotted path (e.g., `network.threshold`) and a concise message; surfaced by both the library and CLI.
- `yogimass.workflow.run_from_config(config)` — main entrypoint to execute end-to-end pipelines from a validated config or path.
- CLI: `python -m yogimass.cli config run --config <file>` uses the same validation layer before running.
- Config fields: `similarity.backend` (``naive`` default; ``annoy``/``faiss`` optional), `similarity.vectorizer` (``spec2vec`` or ``embedding``), `input.msdial_output` for directing cleaned MS-DIAL exports.
- Spectrum processor: `similarity.processor` validates normalization (`tic`/`basepeak`/`none`), min relative/absolute intensities (non-negative), max_peaks (positive int/None), optional `mz_dedup_tolerance`, and `float_dtype` (`float32`/`float64`); defaults match `SpectrumProcessor`.
- Library storage formats are documented in `docs/library_schema.md` (JSON/SQLite layouts, invariants, compatibility expectations).

## yogimass.workflow (primary API)

- `load_data(inputs, input_format="mgf", recursive=False, msdial_output_dir=None)` — load spectra or MS-DIAL tables; returns list of `Spectrum` or dataframe.
- `build_library(sources, library_path, input_format="mgf", recursive=False, msdial_output_dir=None, storage=None, processor=None, overwrite=True, vectorizer=None)` — process spectra and add to `LocalSpectralLibrary`; returns library.
- `search_library(queries, library_path, input_format="mgf", recursive=False, msdial_output_dir=None, top_n=5, min_score=0.0, backend="naive", vectorizer=None, processor=None)` — run similarity search; returns list of `SearchMatch`.
- `curate_library(input_library_path, output_library_path, config=None, qc_report_path=None)` — QC + de-duplication, writes curated library/report; returns `CurationResult`.
- `build_network(input_dir=None, library_path=None, metric="cosine", threshold=None, knn=None, processor=None, reference_mz=None, undirected=True, output_path=None)` — construct similarity network; returns `(nodes, edges)`.

## yogimass.config

- `WorkflowConfig` — strongly typed view of the YAML/JSON config sections (input, library, similarity, network, outputs, curation).
- `SimilarityConfig` — includes backend/vectorizer options for search and shared defaults for query handling.
- `NetworkConfig` — enforces mutual exclusivity of `threshold` vs `knn` and requires an output path when networking is enabled.

## yogimass.io

- MGF/MSP helpers: `list_mgf_libraries`, `list_msp_libraries`, `fetch_mgflib_spectrum`, `save_spectra_to_mgf/msp/json/pickle`.
- MS-DIAL helpers: `list_msdial_txt`, `load_msdial_data`, `clean_and_combine_msdial`, `extract_msms_peaks`, `process_msdial`, `summarize_by_experiment`.

## yogimass.similarity

- `SpectrumProcessor` — filters/aligns spectra (`process`, `filter_noise`, `align_to_reference`).
- Metrics: `cosine_similarity`, `modified_cosine_similarity`, `spec2vec_vectorize`, `cosine_from_vectors`.
- Library/search: `LocalSpectralLibrary`, `LibraryEntry`, `SearchHit`, `LibrarySearcher`, `SearchResult`, `batch_process_folder`.
- Backends: `create_index_backend` builds a search backend (`naive` by default, optional `annoy` placeholder for ANN, `faiss` falls back to naive with a warning).
- Embeddings: `embedding_vectorizer` and `build_embeddings` provide hashed embedding vectors without external models (placeholder for future model-backed embeddings).

## yogimass.networking

- `SpectrumNode`, `SimilarityEdge` — data classes representing graph nodes/edges.
- `build_network_from_spectra`, `build_network_from_library`, `build_network_from_folder` — construct networks via threshold or k-NN.
- `export_network` — export to CSV/GraphML/GEXF/GML.

## yogimass.curation and reporting

- `score_quality`, `detect_duplicate_groups`, `curate_entries`, and the `CurationResult` data model.
- Reporting helpers: `write_search_results`, `write_network_summary`, `write_curation_report`, `summarize_network`, `summarize_quality_distributions`.

## CLI surface

- `yogimass.cli.main` with subcommands:
  - `config run --config <file>` — canonical entrypoint for full workflows.
  - `library build/search/curate` — library operations.
  - `network build` — build/export similarity networks.
  - `clean` — legacy batch cleaning wrapper.

## Legacy / dev utilities

- `scripts/example_workflow.py` is kept for ad-hoc experimentation; it is not the recommended entrypoint now that configs + CLI are canonical.
