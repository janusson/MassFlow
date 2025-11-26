# Yogimass API reference (developer-focused)

This document summarizes the major public functions and classes, grouped by module. Use it as a quick map alongside `ARCHITECTURE.md`.

## yogimass.io
- `list_mgf_libraries(directory)` — discover `.mgf` files; returns list of paths. Use before batch processing.
- `list_msp_libraries(directory)` — discover `.msp` files; returns list of paths.
- `list_available_libraries(mgf_libraries_list, msp_libraries_list)` — log/return names of found libraries.
- `fetch_mgflib_spectrum(library_filepath, spectrum_number)` — load a single spectrum from an MGF by index; returns `(peaks_df, metadata, name)`.
- `save_spectra_to_mgf|msp|json|pickle(spectra_list, export_filepath, export_name)` — export processed spectra to common formats; returns `None`.
- `clean_and_combine_msdial(input_dir, output_dir)` (in `io.msdial_clean_combine`) — aggregate MS-DIAL alignment outputs into a cleaned dataframe; returns `(dataframe, export_path)`.
- `list_msdial_txt`, `load_msdial_data` (in `io.msdial_clean_combine`) — helpers to enumerate and read MS-DIAL TSVs.

## yogimass.similarity
- `SpectrumProcessor` (processing) — filters/aligns spectra (`process`, `filter_noise`, `align_to_reference`); depends on `matchms`. Use before vectorization/search.
- Metrics (`similarity.metrics`) — `cosine_similarity`, `modified_cosine_similarity`, `spec2vec_vectorize`, `cosine_from_vectors`; compute scores/vectors for spectra or sparse maps.
- `LocalSpectralLibrary` (storage) — lightweight JSON/SQLite store; `add_spectrum`, `iter_entries`, `search(query_vector, top_n, min_score)`, `write_entries`; powers search/network build.
- `LibraryEntry`, `SearchHit` — simple data holders for stored spectra and search results.
- `LibrarySearcher` (search) — wraps processing + library search; `search_spectrum(spectrum, top_n, min_score, reference_mz)` returns `SearchResult` list.
- `SearchResult` — flattened hit with identifier/score/metadata/precursor.
- `batch_process_folder` — process spectra in a directory and optionally add to a library; returns processed spectra mapping.

Fits into workflow: processors/vectorizers → libraries/search → networking/curation/reporting via `workflow`.

## yogimass.networking.network
- `SpectrumNode`, `SimilarityEdge` — data classes representing graph nodes/edges.
- `build_network_from_spectra(spectra, metric, threshold|knn, processor, reference_mz, undirected)` — process spectra and build similarity edges.
- `build_network_from_library(library_path, metric, threshold|knn, undirected)` — build a network using stored vectors from a library.
- `build_network_from_folder(input_dir, metric, threshold|knn, processor, reference_mz, undirected)` — load MGF spectra then build a network.
- `export_network(nodes, edges, output_path, undirected)` — export to CSV/GraphML/GEXF/GML.

Fits into workflow: called by `workflow.build_network` after library build or direct folder input.

## yogimass.curation
- `QualityThresholds` — configuration for QC (min peaks/TIC, max single-peak fraction, precursor requirement).
- `QualityResult` — per-spectrum QC stats/issues/score.
- `CurationDecision` — action per spectrum (`keep`, `drop`, `merge`) with reasons/representative/merged IDs.
- `DuplicateGroup` — grouped duplicate IDs with representative.
- `CurationResult` — bundle of curated entries, decisions, quality map, duplicate groups, summary counts.
- `score_quality(entry, thresholds)` — compute `QualityResult`.
- `detect_duplicate_groups(entries, precursor_tolerance, similarity_threshold)` — return lists of near-duplicate IDs.
- `curate_entries(entries, config)` — apply QC + de-duplication; returns `CurationResult` (entries are `LibraryEntry` objects).

Fits into workflow: `workflow.curate_library` uses these to write curated libraries and QC reports.

## yogimass.reporting
- `SearchMatch` — flattened hit for reporting.
- `write_search_results(results, output_path)` — CSV/JSON writer for search hits; returns output `Path`.
- `summarize_network(nodes, edges)` — basic counts/degree stats; returns dict.
- `write_network_summary(summary, output_path)` — write summary as JSON/text; returns `Path`.
- `summarize_quality_distributions(quality_map)` — coarse histograms for QC fields; returns dict of bucket counts.
- `write_curation_report(result, output_path)` — JSON/CSV QC report from `CurationResult`; returns `Path`.

Fits into workflow: called after search/network/curation to persist results.

## yogimass.workflow
- `load_data(inputs, input_format="mgf", recursive=False, msdial_output_dir=None)` — load spectra or MS-DIAL tables; returns list of `Spectrum` or dataframe.
- `build_library(sources, library_path, input_format="mgf", recursive=False, storage=None, processor=None, overwrite=True)` — process spectra and add to `LocalSpectralLibrary`; returns library.
- `search_library(queries, library_path, input_format="mgf", recursive=False, top_n=5, min_score=0.0, processor=None)` — run similarity search; returns list of `SearchMatch`.
- `curate_library(input_library_path, output_library_path, config=None, qc_report_path=None)` — QC + de-duplication, writes curated library/report; returns `CurationResult`.
- `build_network(input_dir=None, library_path=None, metric="cosine", threshold=None, knn=None, processor=None, reference_mz=None, undirected=True, output_path=None)` — construct similarity network; returns `(nodes, edges)`.
- `run_from_config(config_path)` — orchestrate end-to-end pipeline from YAML/JSON config.

Fits into workflow: central orchestrator used by CLI and configs; glues I/O, similarity, networking, curation, reporting.

## yogimass.cli
- `main(argv=None)` — entrypoint for the `yogimass` CLI (also runnable via `python -m yogimass.cli`).
- Subcommands:
  - `config run --config <file>` — run a config-driven workflow (golden configs: `examples/simple_workflow.yaml`, `examples/curation_workflow.yaml`).
  - `library build/search/curate` — library operations.
  - `network build` — build/export similarity networks.
  - `clean` — legacy batch cleaning wrapper.

Fits into workflow: user-facing shell around `yogimass.workflow` and `yogimass.reporting`.
