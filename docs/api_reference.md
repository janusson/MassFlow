# MassFlow API Reference

This document provides a concise reference for all public classes, attributes, methods, and functions in the MassFlow package.

## `MassFlow.acceleration`

- `build_flat_peak_arrays(spectra: Sequence[Spectrum]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
- `prefilter_candidate_pairs(query_spectra, reference_spectra, min_matched_peaks, ms2_tolerance, ...) -> np.ndarray`

Numba-accelerated peak/neutral-loss prefilter used by `SimilarityEngine` to skip query–reference pairs below `min_matched_peaks` before exact modified-cosine scoring, with a pure-NumPy fallback when `numba` is not installed.

## `MassFlow.cheminformatics`

- `calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> Optional[float]`
- `get_isotopic_distribution(smiles: str, threshold: float) -> list[tuple[float, float]]`
- `calculate_isotopic_envelope(smiles: str, max_isopeaks: int) -> list[tuple[float, float]]`
- `calculate_theoretical_mass(smiles: str, adduct: str) -> Optional[float]`

## `MassFlow.cli`

- `setup_logging() -> None`
- `version_callback(value: bool) -> Any`
- `main_callback(version: Optional[bool]) -> Any`
- `run_init(output: str, force: bool) -> Any`
- `run_tutorial(output: str, force: bool) -> Any`
- `run_annotate(config: str) -> Any`
- `run_convert(input: str, output: str) -> Any`
- `run_db_build(input: str, output: str, config: str, category: str, backend: str) -> Any`
- `run_db_inspect(file: str) -> Any`
- `run_db_merge(inputs: List[str], output: str) -> Any`
- `run_stream_server(config: str, host: str, port: int, ...) -> Any`
- `run_serve(config: str, host: str, port: int, ...) -> Any`
- `run_watch(config: str) -> Any`
- `run_tui(input: Optional[str], library: Optional[str], workspace: Optional[str]) -> Any`
- `main() -> Any`

## `MassFlow.config`

- **`class LineNumberLoader`**
  - `construct_mapping(node: Any, deep: Any) -> Any`
- **`class ProjectConfig`**
  - `name: str`
  - `output_directory: Path`
- **`class InputConfig`**
  - `input_path: Path`
  - `format: Optional[Literal['mgf', 'msp', 'mzml', 'mzxml', 'db', 'sqlite']]`
  - `library_path: Optional[Path]`
  - `reference_library() -> Optional[Path]`
  - `reference_library(value: Optional[Path]) -> None`
- **`class SolventConfig`**
  - `name: str`
  - `formula: Optional[str]`
  - `mz: float`
  - `validate_mz(v: float) -> float`
- **`class ProcessingConfig`**
  - `min_peaks: int`
  - `min_intensity: float`
  - `normalize_intensity: bool`
  - `clean_metadata: bool`
  - `add_retention_time: bool`
  - `repair_inchi_inchikey_smiles: bool`
  - `derive_adduct_from_name: bool`
  - `derive_formula_from_name: bool`
  - `clean_compound_name: bool`
  - `derive_ionmode: bool`
  - `make_charge_int: bool`
  - `filter_by_intensity: bool`
  - `filter_min_peaks: bool`
  - `filter_by_mz: bool`
  - `reduce_to_top_n_peaks: bool`
  - `mz_min: float`
  - `mz_max: float`
  - `n_max: int | None`
  - `validate_mz_range(v: float, info: ValidationInfo) -> float`
  - `noise_threshold: float`
  - `validate_non_negative_intensities(v: float, info: ValidationInfo) -> float`
  - `validate_non_negative_peaks(v: int, info: ValidationInfo) -> int`
  - `instrument: Optional[str]`
  - `mode: Literal['positive', 'negative', '']`
  - `solvents: List[SolventConfig]`
  - `precursor_mz: float`
  - `retention_time: float`
  - `decoy_min_relative_intensity: float`
  - `decoy_mz_shift_da: float`
  - `validate_precursor_mz(v: float) -> float`
- **`class SimilarityConfig`**
  - `algorithm: Literal['cosine', 'modified_cosine', 'spec2vec', 'ms2deepscore', 'consensus', 'cascade']`
  - `consensus_weights: Optional[dict[str, float]]`
  - `allow_consensus_fallback: bool`
  - `cascade_tier1: Literal['cosine', 'modified_cosine']`
  - `cascade_tier2: Literal['spec2vec', 'ms2deepscore']`
  - `cascade_lower_bound: float`
  - `cascade_upper_bound: float`
  - `cascade_stages: list[str]`
  - `consensus_min_engines: int`
  - `hnsw_enabled: bool`
  - `hnsw_m: int`
  - `hnsw_ef_construction: int`
  - `hnsw_ef_search: int`
  - `hnsw_candidates_per_query: int`
  - `hnsw_bin_width: float`
  - `hnsw_mz_min: float`
  - `hnsw_mz_max: float`
  - `hnsw_random_seed: int`
  - `ml_endpoints: Optional[dict[str, str]]`
  - `ml_request_timeout_seconds: float`
  - `ml_circuit_breaker_threshold: int`
  - `ml_circuit_breaker_cooldown_seconds: float`
  - `model_path: Optional[Path]`
  - `ms1_tolerance: float`
  - `resolution_ppm: Optional[float]`
  - `ms2_tolerance: float`
  - `min_score: float`
  - `analog_search: bool`
  - `min_matched_peaks: int`
  - `fdr_threshold: float`
  - `validate_mass_tolerances(v: float, info: ValidationInfo) -> float`
  - `validate_score_ranges(v: float, info: ValidationInfo) -> float`
  - `validate_cascade_range(v: float, info: ValidationInfo) -> float`
  - `validate_min_matched_peaks(v: int, info: ValidationInfo) -> int`
- **`class WorkflowConfig`**
  - (no active fields; reserved for future pipeline stages)
- **`class ExportConfig`**
  - `format: Literal['csv', 'mztab']` (only these two are wired into the workflow)
- **`class MassFlowConfig`**
  - `project: ProjectConfig`
  - `input: InputConfig`
  - `processing: ProcessingConfig`
  - `similarity: SimilarityConfig`
  - `workflow: WorkflowConfig`
  - `export: ExportConfig`
  - `output_directory() -> Path`
  - `from_yaml(path: Union[str, Path]) -> 'MassFlowConfig'`

## `MassFlow.convert`

- **`class MSConvertNotFoundError`**
- **`class ConversionError`**
- `check_msconvert_installed() -> bool`
- `get_vendor_files(input_dir: Path) -> List[Path]`
- `run_conversion(input_path: Path, output_dir: Path, output_format: str) -> None`
- `convert_directory(input_dir: Path, output_dir: Path, output_format: str) -> int`

## `MassFlow.database`

- **`class LegacyDatabaseSchemaError`**
- `get_spectra_table_columns(connection: sqlite3.Connection) -> list[str]`
- `has_table(connection: sqlite3.Connection, table_name: str) -> bool`
- `is_legacy_spectra_schema(connection: sqlite3.Connection) -> bool`
- `is_current_spectra_schema(connection: sqlite3.Connection) -> bool`
- `legacy_migration_error_message(db_path: Union[str, Path]) -> str`
- `create_current_spectra_table(connection: sqlite3.Connection) -> None`
- `create_legacy_backup_table(connection: sqlite3.Connection, backup_table_name: Optional[str]) -> str`
- `create_migrated_spectra_table(connection: sqlite3.Connection) -> None`
- `migrate_legacy_peaks_database(db_path: Union[str, Path]) -> dict[str, Any]`
- `migrate_legacy_peaks_to_arrays(db_path: Union[str, Path]) -> dict[str, Any]`
- `migrate_blobs_to_zarr(db_path: Union[str, Path], ...) -> dict[str, Any]`
- **`class SpectralDatabase`**
  - `__init__(db_path: Union[str, Path], allow_destructive_upgrade: bool) -> Any`
  - `add_spectra(spectra: Iterator[Spectrum], category: str, batch_size: int) -> int`
  - `get_spectra(category: Optional[str], name_pattern: Optional[str]) -> Iterator[Spectrum]`
  - `get_total_spectra_count() -> int`
  - `get_category_counts() -> dict[str, int]`
  - `get_precursor_mz_range() -> tuple[float, float]`
  - `close() -> None`

`SpectralDatabase` also supports hybrid mode: when opened with a `zarr_path` (or built with `--backend hybrid`), peak arrays live in a chunked Zarr store referenced by `zarr_ref`/`zarr_index` columns.

## `MassFlow.hnsw`

- `spectrum_to_binned_vector(spectrum: Spectrum, bin_width: float, mz_min: float, mz_max: float) -> np.ndarray`
- `bin_spectra(spectra: Iterable[Spectrum], bin_width: float, mz_min: float, mz_max: float) -> np.ndarray`
- **`class HNSWSpectralIndex`**
  - `__init__(dim: int, m: int, ef_construction: int, seed: int, ...) -> Any`
  - `add_spectra(vectors: np.ndarray, ids: list[str]) -> None`
  - `search(query_vector: np.ndarray, k: int, ef_search: int) -> tuple[np.ndarray, np.ndarray]`
  - `save(path: Path) -> None`
  - `load(path: Path, ...) -> 'HNSWSpectralIndex'`

hnswlib-backed two-channel candidate index (`[binned exact m/z, binned neutral losses]`) used by `CascadeEngine` for sub-linear approximate candidate retrieval. Requires the `hnsw` extra.

## `MassFlow.io`

- **`class UnsupportedVendorFormatError`**
- `load_spectra(file_path: Path, file_format: Optional[str]) -> Iterator[Spectrum]`
- `save_match_results(results: list[dict[str, Any]], output_path: Path, query_spectra: Optional[Iterable[Spectrum]]) -> None`
- `save_match_results_to_json(results: list[dict[str, Any]], output_path: Path, query_spectra: Optional[Iterable[Spectrum]]) -> None`
- `save_match_results_to_xlsx(results: list[dict[str, Any]], output_path: Path, query_spectra: Optional[Iterable[Spectrum]]) -> None`
- `save_match_results_to_parquet(results: list[dict[str, Any]], output_path: Path, query_spectra: Optional[Iterable[Spectrum]]) -> None`
- `save_analysis_report(output_path: Path, report_data: dict[str, Any]) -> None`
- `save_spectra_to_msp(spectra: Iterable[Spectrum], export_path: Path) -> None`
- `save_spectra_to_pickle(spectra: Iterable[Spectrum], export_path: Path) -> None`
- `save_spectra_to_mgf(spectra: Iterable[Spectrum], export_path: Path) -> None`
- `save_match_results_to_mztab(results: list[dict[str, Any]], output_path: Path, query_spectra: Optional[Iterable[Spectrum]]) -> None`

## `MassFlow.library`

- **`class LibrarySpec`** — compact, pickle-safe store reference (path + backend) crossed between processes.
- **`class RawFileLibraryStore`** — read-only `SpectralStore` adapter over raw spectral files (mzML/mzXML/MGF/MSP); writes are rejected explicitly.
- `prepare_library(config: MassFlowConfig) -> LibrarySpec` — normalize a raw spectral library into a store in the configured backend (`sqlite`/`zarr`/`hybrid`); store inputs (`.db`/`.zarr`) are used directly.
- `open_library(spec: LibrarySpec, config: MassFlowConfig) -> SpectralStore`
- `library_spec_for_config(config: MassFlowConfig) -> LibrarySpec`

Worker-owned backend model: the parent builds the library once, workers open it themselves, so RAM scales with chunk size, not library size.

## `MassFlow.log_config`

- **`class StructuredFormatter`**
  - `format(record: Any) -> Any`
- `setup_structured_logging(level: Any) -> Any`

## `MassFlow.ml_client`

- **`class CircuitBreaker`** — fail-fast wrapper (open/half-open/closed) protecting remote scoring.
- **`class CircuitOpenError`** (exception)
- **`class RemoteMLEngine`** (implements `MLEngineProtocol`) — REST (`http(s)://`) or gRPC (`grpc://`) client for remote Spec2Vec/MS2DeepScore scoring via the `massflow.v1.ml` service; used when `SimilarityConfig.ml_endpoints` is configured. Core MassFlow stays free of PyTorch/Gensim.

## `MassFlow.models`

- **`class AnnotationHit`**
  - `engine_id: str`
  - `reference_id: str`
  - `score: float`
  - `rank: int`
  - `inchikey: Optional[str]`
  - `smiles: Optional[str]`
- **`class ConsensusInput`**
  - `query_id: str`
  - `hits: List[AnnotationHit]`
- **`class AggregatedCandidate`**
  - `reference_id: str`
  - `inchikey: Optional[str]`
  - `smiles: Optional[str]`
  - `consensus_score: float`
  - `engine_scores: Dict[str, float]`
  - `engine_ranks: Dict[str, int]`
- **`class ConsensusResult`**
  - `query_id: str`
  - `best_reference_id: Optional[str]`
  - `best_consensus_score: Optional[float]`
  - `flagged_for_review: bool`
  - `review_reason: Optional[str]`
  - `candidates: List[AggregatedCandidate]`
- **`class ConsensusConfig`**
  - `engine_weights: Dict[str, float]`
  - `tie_breaker_strategy: Literal['highest_rank', 'average_score', 'validator_engine']`
  - `validator_engine: Optional[str]`
  - `flag_rank_discrepancy_threshold: int`
- **`class IsotopicDistribution`**
  - `peaks: List[tuple[float, float]]`
- **`class MolecularStructure`**
  - `smiles: Optional[str]`
  - `inchi: Optional[str]`
  - `formula: Optional[str]`
  - `exact_mass: Optional[float]`
  - `isotopic_distribution: Optional[IsotopicDistribution]`
  - `isotopic_envelope: Optional[List[tuple[float, float]]]`
  - `is_physically_valid: bool`
  - `validate_and_compute_mass() -> 'MolecularStructure'`
- **`class SpectrumMetadata`**
  - `spectrum_id: str`
  - `precursor_mz: float`
  - `retention_time: Optional[float]`
  - `charge: Optional[int]`
  - `ion_mode: Optional[Literal['positive', 'negative', 'neutral']]`
  - `collision_energy: Optional[float]`
  - `adduct: Optional[str]`
  - `molecule: Optional[MolecularStructure]`
  - `is_physically_valid: bool`
  - `validate_precursor_mass_logic() -> 'SpectrumMetadata'`
- **`class SpectralPeaks`**
  - `mz_array: List[float]`
  - `intensity_array: List[float]`
  - `validate_arrays() -> 'SpectralPeaks'`
- **`class MassFlowSpectrum`**
  - `metadata: SpectrumMetadata`
  - `peaks: SpectralPeaks`

## `MassFlow.processing`

- `compute_spectral_metrics(mz_array: np.ndarray, precursor_mz: float) -> Tuple[np.ndarray, np.ndarray]`
- `metadata_processing(spectrum: Spectrum, config: Optional[ProcessingConfig]) -> Optional[Spectrum]`
- `calculate_triage_flags(spectrum: Spectrum) -> Spectrum`
- `peak_processing(spectrum: Spectrum, config: ProcessingConfig) -> Optional[Spectrum]`
- `process_spectra_batch(spectra: List[Spectrum], config: ProcessingConfig) -> List[Spectrum]`
- `process_spectra(spectra: Iterator[Spectrum], config: ProcessingConfig) -> Iterator[Spectrum]`

## `MassFlow.protocols`

- **`class MLEngineProtocol`** (abstract base)
  - `score(query_spectra, reference_spectra) -> list[list[float]]`

Engine-agnostic contract implemented by local ML engines and by `RemoteMLEngine` (see `MassFlow.ml_client`) so core MassFlow never depends on PyTorch/Gensim at import time.

## `MassFlow.similarity`

- **`class SearchResult`**
  - `query_id: str`
  - `query_precursor_mz: float | None`
  - `reference_id: str`
  - `reference_name: str | None`
  - `reference_precursor_mz: float | None`
  - `score: float`
  - `matched_peaks: int`
  - `smiles: str | None`
  - `inchikey: str | None`
  - `is_decoy: bool`
  - `q_value: float`
  - `p_value: float | None`
  - `annotation_tier: str | None`
  - `structural_similarity: float | None`
- `generate_decoys(spectra: List[Spectrum], random_seed: int, min_relative_intensity: float, mz_shift_da: float) -> List[Spectrum]`
- `calculate_empirical_p_values(target_scores: np.ndarray, decoy_scores: np.ndarray) -> np.ndarray`
- `calculate_fdr(target_scores: np.ndarray, decoy_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- **`class SimilarityEngine`**
  - `__init__(config: SimilarityConfig) -> Any`
  - `search(query_spectra: List[Spectrum], reference_spectra: List[Spectrum], min_score: float | None, top_n: int | None, include_decoys: bool) -> List[SearchResult]`
- **`class CascadeEngine`**
  - `__init__(config: SimilarityConfig) -> Any`
  - `search(query_spectra: List[Spectrum], reference_spectra: List[Spectrum], min_score: float | None, top_n: int | None, include_decoys: bool) -> List[SearchResult]`
- **`class ConsensusEngine`**
  - `__init__(config: SimilarityConfig) -> Any`
  - `search(query_spectra: List[Spectrum], reference_spectra: List[Spectrum], min_score: float | None, top_n: int | None, include_decoys: bool) -> List[SearchResult]`
- `get_similarity_engine(config: SimilarityConfig) -> SimilarityEngine | ConsensusEngine | CascadeEngine`

## `MassFlow.tui` (interactive terminal console, requires the `tui` extra)

- `MassFlow.tui.state` — `SpectrumSummary`, `SearchHit`, `IdentificationRequest`,
  `IdentificationOutcome`, `QueryLoadResult`, `LibraryInfo` dataclasses
- `MassFlow.tui.spectrum_data` — `downsample_peaks`, `display_entropy`,
  `summarize_spectrum`, `peak_bounds`, `mirror_align`, `format_mz`,
  `format_retention_time`, `annotation_status`
- `MassFlow.tui.plot` — `render_stick_plot`, `render_mirror_plot`,
  `render_score_gauge`, `render_axis_labels`
- `MassFlow.tui.files` — `classify_file`, `discover_spectral_files`,
  `copy_into_workspace`, `human_size`, `guess_backend`
- `MassFlow.tui.diagnostics` — `TuiError`, `Problem`, `suggest_fix`,
  `parse_quarantine_log`, `QuarantineEntry`
- `MassFlow.tui.pipeline` — `load_query_preview`, `inspect_library`,
  `run_identification`, `capture_quarantine_records`
- `MassFlow.tui.app` — `MassFlowApp` (Textual application)

## `MassFlow.storage`

- **`class SpectralStore`** (abstract base)
  - `add_spectra(spectra, category, batch_size) -> int`
  - `get_spectra(category, name_pattern) -> Iterator[Spectrum]`
  - `get_total_spectra_count() -> int`
  - `get_category_counts() -> dict[str, int]`
  - `get_precursor_mz_range() -> tuple[float, float]`
  - `close() -> None`
- `create_spectral_store(path: Union[str, Path], mode: str, category: str, ...) -> SpectralStore`

Factory for the SQLite (`SpectralDatabase`), Zarr, and hybrid backends.

## `MassFlow.streaming.queue`

- **`class BoundedQueue`**
  - `put(packet: QueuedPacket) -> None`
  - `get() -> QueuedPacket`
  - `stats() -> QueueStats`
- `compute_packet_quality(packet: QueuedPacket) -> float`
- **`class QueueStats`**
- **`class QueuedPacket`**
- **`class OverflowPolicy`** (`drop_oldest`, `block`, `drop_newest`)
- **`class QueueFull`** (exception)

Quality-gated bounded queue used by the gRPC streaming server for backpressure and low-quality packet shedding.

## `MassFlow.streaming.server`

- **`class MassFlowStreamingServicer`**
- `serve(config: MassFlowConfig, host: str, port: int, queue_capacity: int, ...) -> Any`
- `run_server(config_path: str, host: str, port: int, ...) -> Any`

## `MassFlow.workflow`

- `run_annotation_pipeline(config: MassFlowConfig, config_path: Path | str | None) -> None`

## `MassFlow.zarr_store`

- **`class ZarrSpectralStore`** (implements `SpectralStore`)
  - `add_spectra(spectra, category, batch_size) -> int`
  - `get_spectra(category, name_pattern) -> Iterator[Spectrum]`
  - `get_spectrum_by_id(spectrum_id: str) -> Optional[Spectrum]`
  - `batch_get_arrays(indices) -> tuple[list[np.ndarray], list[np.ndarray]]`
  - `get_total_spectra_count() -> int`
  - `get_category_counts() -> dict[str, int]`
  - `get_precursor_mz_range() -> tuple[float, float]`
  - `cache_stats() -> dict[str, int]`
  - `is_remote() -> bool`
  - `close() -> None`
- **`class ZarrPeakArrayStore`**
  - Chunked, compressed storage of fragment `mz`/`intensity` arrays with lock-free concurrent reads for multiprocessing.
