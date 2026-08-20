# MassFlow API Reference

This document provides a concise reference for all public classes, attributes, methods, and functions in the MassFlow package.

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
- `run_annotate(config: str) -> Any`
- `run_convert(input: str, output: str) -> Any`
- `run_db_build(input: str, output: str, config: str, category: str) -> Any`
- `run_db_inspect(file: str) -> Any`
- `run_db_merge(inputs: List[str], output: str) -> Any`
- `run_visualize(graphml_path: str, output: str) -> Any`
- `run_watch(config: str) -> Any`
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
  - `validate_precursor_mz(v: float) -> float`
- **`class SimilarityConfig`**
  - `algorithm: Literal['cosine', 'modified_cosine', 'spec2vec', 'ms2deepscore', 'consensus', 'cascade']`
  - `consensus_weights: Optional[dict[str, float]]`
  - `allow_consensus_fallback: bool`
  - `cascade_tier1: Literal['cosine', 'modified_cosine']`
  - `cascade_tier2: Literal['spec2vec', 'ms2deepscore']`
  - `cascade_lower_bound: float`
  - `cascade_upper_bound: float`
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
  - `perform_peak_picking: bool`
  - `perform_alignment: bool`
  - `perform_networking: bool`
  - `export_consensus: bool`
- **`class ExportConfig`**
  - `format: Literal['csv', 'pickle', 'msp', 'mgf', 'json', 'xlsx', 'parquet', 'fbmn', 'mztab']`
- **`class MassFlowConfig`**
  - `project: ProjectConfig`
  - `input: InputConfig`
  - `processing: ProcessingConfig`
  - `similarity: SimilarityConfig`
  - `workflow: WorkflowConfig`
  - `export: ExportConfig`
  - `output_directory() -> Path`
  - `from_yaml(path: Union[str, Path]) -> 'MassFlowConfig'`

## `MassFlow.consensus`

- `generate_consensus(input_data: ConsensusInput, config: ConsensusConfig) -> ConsensusResult`
- **`class ConsensusEngine`**
  - `__init__(config: ConsensusConfig) -> None`
  - `resolve(consensus_input: ConsensusInput) -> ConsensusResult`

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
- **`class SpectralDatabase`**
  - `__init__(db_path: Union[str, Path], allow_destructive_upgrade: bool) -> Any`
  - `add_spectra(spectra: Iterator[Spectrum], category: str, batch_size: int) -> int`
  - `get_spectra(category: Optional[str], name_pattern: Optional[str]) -> Iterator[Spectrum]`
  - `get_total_spectra_count() -> int`
  - `get_category_counts() -> dict[str, int]`
  - `get_precursor_mz_range() -> tuple[float, float]`
  - `close() -> None`

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

## `MassFlow.log_config`

- **`class StructuredFormatter`**
  - `format(record: Any) -> Any`
- `setup_structured_logging(level: Any) -> Any`

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

## `MassFlow.networking`

- `generate_molecular_network(all_queries: List[Spectrum], all_references: List[Spectrum], all_results: List[SearchResult], config: MassFlowConfig, output_path: Path) -> None`

## `MassFlow.processing`

- `compute_spectral_metrics(mz_array: np.ndarray, precursor_mz: float) -> Tuple[np.ndarray, np.ndarray]`
- `metadata_processing(spectrum: Spectrum, config: Optional[ProcessingConfig]) -> Optional[Spectrum]`
- `calculate_triage_flags(spectrum: Spectrum) -> Spectrum`
- `peak_processing(spectrum: Spectrum, config: ProcessingConfig) -> Optional[Spectrum]`
- `process_spectra_batch(spectra: List[Spectrum], config: ProcessingConfig) -> List[Spectrum]`
- `process_spectra(spectra: Iterator[Spectrum], config: ProcessingConfig) -> Iterator[Spectrum]`

## `MassFlow.server`

- `is_plausible_formula(text: str) -> bool`
- `check_smiles_validity(smiles: str) -> Optional[str]`
- `validate_document(ls: LanguageServer, params: Any) -> Any`
- `hover(ls: LanguageServer, params: HoverParams) -> Optional[Hover]`

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
  - `__init__(engines: list[tuple[SimilarityEngine, float]], min_score: float) -> Any`
  - `search(query_spectra: List[Spectrum], reference_spectra: List[Spectrum], min_score: float | None, top_n: int | None, include_decoys: bool) -> List[SearchResult]`
- `get_similarity_engine(config: SimilarityConfig) -> SimilarityEngine | ConsensusEngine | CascadeEngine`

## `MassFlow.visualization.network`

- `visualize_graphml(graphml_path: str | Path, output_html: str | Path, notebook: bool) -> None`

## `MassFlow.workflow`

- `run_annotation_pipeline(config: MassFlowConfig, config_path: Path | str | None) -> None`
