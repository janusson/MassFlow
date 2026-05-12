"""
Pydantic configuration models for YAML-driven MassFlow execution.

This module defines the nested schema used by the CLI and workflow layers to
validate pipeline configuration loaded from YAML. The models capture project
paths, input sources, processing toggles, similarity-engine settings, optional
workflow features, and declared export preferences.

Validation in this layer is intentionally focused on local structural
correctness such as field types and simple physical constraints. Broader runtime
assumptions, such as whether a reference library is present for annotation or
whether certain workflow toggles are currently implemented, are enforced in the
orchestrating modules.
"""

from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)


class LineNumberLoader(yaml.SafeLoader):
    """Custom YAML loader that extracts line numbers for configuration keys."""

    def construct_mapping(self, node, deep=False):
        mapping = {}
        lines = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
            lines[key] = key_node.start_mark.line + 1
        mapping["__lines__"] = lines
        return mapping


class ProjectConfig(BaseModel):
    """Project metadata and output locations shared across a MassFlow run."""

    name: str = "MassFlow_Project"
    output_directory: Path = Path("results")


class InputConfig(BaseModel):
    """
    Input path and format hints for annotation.

    The ``input_path`` can point to either a single spectral file or a
    directory containing multiple files.
    """

    model_config = ConfigDict(populate_by_name=True)

    input_path: Path = Field(
        ...,
        description="Path to the input spectral file or directory of files.",
    )
    format: Optional[Literal["mgf", "msp", "mzml", "mzxml", "db", "sqlite"]] = Field(
        default=None,
        description="Optional explicit format hint. If omitted, MassFlow infers the format from the file extension.",
    )
    library_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("library_path", "reference_library"),
    )

    @property
    def reference_library(self) -> Optional[Path]:
        return self.library_path

    @reference_library.setter
    def reference_library(self, value: Optional[Path]) -> None:
        self.library_path = value


class SolventConfig(BaseModel):
    """
    Named solvent/adduct mass used as optional contextual processing metadata.

    These values are stored in the configuration schema for downstream
    extensions and user metadata, but they are not currently consumed directly
    by the core annotation workflow in this repository snapshot.
    """

    name: str
    formula: Optional[str] = None
    mz: float

    @field_validator("mz")
    @classmethod
    def validate_mz(cls, v: float) -> float:
        """Ensure solvent m/z is not negative."""
        if v < 0:
            raise ValueError(f"Solvent m/z cannot be negative. Received: {v}")
        return v


class ProcessingConfig(BaseModel):
    """
    Parameters for metadata harmonization and peak-level processing.

    The fields in this model map onto the operations implemented in
    :mod:`MassFlow.processing`, including optional ``matchms`` metadata repairs,
    m/z truncation, intensity filtering, top-N peak reduction, normalization,
    and injection of instrument context into spectra.
    """

    # Standard peak filters
    min_peaks: int = 5
    min_intensity: float = 0.0
    normalize_intensity: bool = True

    # Metadata filtering toggles
    clean_metadata: bool = Field(
        default=True, description="Apply matchms default metadata cleaning."
    )
    add_retention_time: bool = Field(
        default=True, description="Extract and format retention time."
    )
    repair_inchi_inchikey_smiles: bool = Field(
        default=True, description="Repair structural identifiers."
    )
    derive_adduct_from_name: bool = Field(
        default=True, description="Derive adducts from compound names."
    )
    derive_formula_from_name: bool = Field(
        default=True, description="Derive formulas from compound names."
    )
    clean_compound_name: bool = Field(
        default=True, description="Standardize compound names."
    )
    derive_ionmode: bool = Field(
        default=True, description="Derive ion mode from metadata."
    )
    make_charge_int: bool = Field(
        default=True, description="Ensure charge is an integer."
    )

    # Peak filtering toggles
    filter_by_intensity: bool = Field(
        default=False, description="Filter peaks by minimum intensity/noise threshold."
    )
    filter_min_peaks: bool = Field(
        default=False, description="Require a minimum number of peaks."
    )
    filter_by_mz: bool = Field(
        default=False, description="Truncate peaks outside of an m/z range."
    )
    reduce_to_top_n_peaks: bool = Field(
        default=False, description="Reduce spectrum to top N most intense peaks."
    )

    # HLD v0.9 Pre-Processing
    mz_min: float = Field(default=0.0, ge=0.0, description="Minimum m/z")
    mz_max: float = Field(default=1000.0, description="Maximum m/z")
    n_max: int | None = Field(default=None, gt=0, description="Top N peaks")

    @field_validator("mz_max")
    @classmethod
    def validate_mz_range(cls, v: float, info: ValidationInfo) -> float:
        """
        Validate that the maximum m/z value is greater than the minimum m/z value.

        This validator ensures logical consistency in the m/z range settings by
        comparing ``mz_max`` against ``mz_min`` (if present in the validation context).

        Parameters
        ----------
        v : float
            The value of ``mz_max`` being validated.
        info : ValidationInfo
            The validation context containing other field values, specifically ``mz_min``.

        Returns
        -------
        float
            The validated ``mz_max`` value.

        Raises
        ------
        ValueError
            If ``mz_max`` is less than or equal to ``mz_min``.
        """
        if "mz_min" in info.data and v <= info.data["mz_min"]:
            raise ValueError(
                f"mz_max ({v}) must be greater than mz_min ({info.data['mz_min']})"
            )
        return v

    noise_threshold: float = Field(
        default=1000.0, description="Minimum intensity threshold"
    )

    @field_validator("min_intensity", "noise_threshold")
    @classmethod
    def validate_non_negative_intensities(cls, v: float, info: ValidationInfo) -> float:
        """Ensure intensity thresholds are non-negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative. Received: {v}")
        return v

    @field_validator("min_peaks")
    @classmethod
    def validate_non_negative_peaks(cls, v: int, info: ValidationInfo) -> int:
        """Ensure min_peaks is non-negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative. Received: {v}")
        return v

    # Metadata context
    instrument: Optional[str] = None
    mode: Literal["positive", "negative", ""] = ""
    solvents: List[SolventConfig] = Field(default_factory=list)

    # Legacy support (keeping for backward compatibility if needed)
    precursor_mz: float = Field(default=0.0)
    retention_time: float = Field(default=0.0)

    @field_validator("precursor_mz")
    @classmethod
    def validate_precursor_mz(cls, v: float) -> float:
        """
        Validate that the precursor m/z value is non-negative.

        Parameters
        ----------
        v : float
            The precursor m/z value to validate.

        Returns
        -------
        float
            The validated precursor m/z value.

        Raises
        ------
        ValueError
            If ``precursor_mz`` is negative.
        """
        if v < 0:
            raise ValueError(f"precursor_mz must be non-negative. Received: {v}")
        return v


class SimilarityConfig(BaseModel):
    """
    Settings for similarity scoring, ML model loading, and advanced engines.

    Not every field applies to every algorithm. For example,
    ``consensus_weights`` is only meaningful for the ``consensus`` engine,
    cascade tier settings are only used for ``cascade``, and ``model_path`` is
    required by ML-backed engines such as ``spec2vec`` and ``ms2deepscore``.
    Legacy ``tolerance`` is retained for compatibility with existing configs.

    New in this version:
    - ``allow_consensus_fallback``: when True, the system will gracefully fall
      back to a single 'cosine' engine if ``algorithm == "consensus"`` but no
      ``consensus_weights`` are provided. When False, the factory will raise a
      ValueError to enforce strict configuration.
    """

    algorithm: Literal[
        "cosine", "modified_cosine", "spec2vec", "ms2deepscore", "consensus", "cascade"
    ] = Field(
        default="cosine",
        description="Core: 'cosine', 'modified_cosine'. Experimental: 'spec2vec', 'ms2deepscore', 'consensus', 'cascade'.",
    )
    consensus_weights: Optional[dict[str, float]] = Field(
        default=None,
        description="Experimental: Dictionary mapping algorithm names to their weights for consensus search.",
    )

    # Allow graceful fallback when consensus_weights is omitted. If set to False,
    # requesting 'consensus' without weights will surface an explicit error.
    allow_consensus_fallback: bool = Field(
        default=True,
        description="If True, fallback to a single 'cosine' engine when consensus_weights is None. If False, require explicit consensus_weights.",
    )

    # Cascade Routing Parameters (Experimental)
    cascade_tier1: Literal["cosine", "modified_cosine"] = "cosine"
    cascade_tier2: Literal["spec2vec", "ms2deepscore"] = "ms2deepscore"
    cascade_lower_bound: float = 0.4
    cascade_upper_bound: float = 0.85
    model_path: Optional[Path] = Field(
        default=None,
        description="Experimental: Path to ML model file (e.g., gensim Word2Vec for Spec2Vec or PyTorch model for MS2DeepScore).",
    )

    # Evidence-Based Consensus Parameters
    isotopic_credibility_weight: float = Field(
        default=0.0,
        description="Weight applied to the MS1 Isotopic Credibility Factor in consensus engines.",
    )
    penalize_impossible_neutral_losses: bool = Field(
        default=False,
        description="If True, candidates with physically impossible major neutral losses receive a severe score penalty.",
    )
    neutral_loss_penalty_factor: float = Field(
        default=0.1,
        description="Score multiplier applied when an impossible neutral loss is detected.",
    )

    # Fixed-unit Tolerances
    ms1_tolerance: float = Field(
        default=0.02, description="Precursor mass tolerance in Da"
    )
    resolution_ppm: Optional[float] = Field(
        default=None,
        description="Optional: Precursor mass resolution in ppm. If set, this overrides ms1_tolerance for MS1 filtering.",
    )
    ms2_tolerance: float = Field(
        default=0.02, description="Fragment mass tolerance in Da"
    )

    min_score: float = 0.6
    analog_search: bool = False
    min_matched_peaks: int = 3
    fdr_threshold: float = 0.01

    @field_validator("ms1_tolerance", "ms2_tolerance")
    @classmethod
    def validate_mass_tolerances(cls, v: float, info: ValidationInfo) -> float:
        """Ensure mass tolerances are not negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative. Received: {v}")
        return v

    @field_validator(
        "min_score", "fdr_threshold", "cascade_lower_bound", "cascade_upper_bound"
    )
    @classmethod
    def validate_score_ranges(cls, v: float, info: ValidationInfo) -> float:
        """Ensure scores and thresholds are within [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"{info.field_name} must be between 0.0 and 1.0. Received: {v}"
            )
        return v

    @field_validator("cascade_upper_bound")
    @classmethod
    def validate_cascade_range(cls, v: float, info: ValidationInfo) -> float:
        """Ensure cascade upper bound is greater than lower bound."""
        if "cascade_lower_bound" in info.data and v <= info.data["cascade_lower_bound"]:
            raise ValueError(
                f"cascade_upper_bound ({v}) must be greater than cascade_lower_bound ({info.data['cascade_lower_bound']})"
            )
        return v

    @field_validator("min_matched_peaks")
    @classmethod
    def validate_min_matched_peaks(cls, v: int, info: ValidationInfo) -> int:
        """Ensure minimum matched peaks is not negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative. Received: {v}")
        return v


class WorkflowConfig(BaseModel):
    """
    High-level workflow feature flags (All current fields are Experimental).

    In the current workflow implementation, ``perform_networking`` is the main
    toggle consumed directly by :mod:`MassFlow.workflow`. The remaining fields
    are schema-level placeholders for adjacent orchestration features.
    """

    perform_peak_picking: bool = Field(
        default=True, description="Experimental placeholder."
    )
    perform_alignment: bool = Field(
        default=True, description="Experimental placeholder."
    )
    perform_networking: bool = Field(
        default=False, description="Experimental: Generate GraphML molecular network."
    )
    export_consensus: bool = Field(
        default=True, description="Experimental placeholder."
    )


class ExportConfig(BaseModel):
    """
    Declared export preferences for downstream result handling.

    The current annotation workflow writes per-file CSV result reports and, when
    enabled, a GraphML molecular network. This model preserves a broader output
    schema for future exporters and configuration compatibility. Note that only 'csv'
    is currently part of the stable v1.0 contract; other formats are experimental.
    """

    format: Literal[
        "csv", "pickle", "msp", "mgf", "json", "xlsx", "parquet", "fbmn", "mztab"
    ] = Field(
        default="csv",
        description="Stable: 'csv'. Experimental: 'pickle', 'msp', 'mgf', 'json', 'xlsx', 'parquet', 'fbmn', 'mztab'.",
    )


class MassFlowConfig(BaseModel):
    """
    Root configuration object loaded from MassFlow YAML files.

    This model is the contract passed between the CLI, workflow, processing,
    similarity, and optional networking layers.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    input: InputConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    # Legacy alias for root output_directory to map to project.output_directory
    @property
    def output_directory(self) -> Path:
        """
        Retrieve the output directory from the project configuration.

        This property serves as an alias to ``self.project.output_directory``,
        providing backward compatibility or convenient access to the output path.

        Returns
        -------
        Path
            The path to the configured output directory.
        """
        return self.project.output_directory

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MassFlowConfig":
        """
        Load and validate a ``MassFlowConfig`` from a YAML file.

        The file is parsed using a custom YAML loader that tracks line numbers,
        and then validated by Pydantic against the nested configuration models
        defined in this module. Human-readable error messages with exact line
        numbers are provided when validation fails.

        Parameters
        ----------
        path : str or Path
            The file system path to the YAML configuration file.

        Returns
        -------
        MassFlowConfig
            Validated configuration populated from the YAML document.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.
        ValueError
            If the YAML content does not conform to the MassFlow schema.
        yaml.YAMLError
            If the YAML document cannot be parsed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.load(f, Loader=LineNumberLoader)

        if data is None:
            data = {}

        try:
            config_instance = cls(**data)
        except ValidationError as e:
            error_messages = []
            for err in e.errors():
                loc = err["loc"]
                msg = err["msg"].replace("Value error, ", "")

                current = data
                line_num = "Unknown"
                for i, part in enumerate(loc):
                    if (
                        isinstance(current, dict)
                        and "__lines__" in current
                        and part in current["__lines__"]
                    ):
                        if i == len(loc) - 1:
                            line_num = current["__lines__"][part]
                        current = current.get(part)
                    elif isinstance(current, list) and isinstance(part, int):
                        current = current[part]
                    else:
                        break

                key_path = " -> ".join(str(k) for k in loc)
                error_messages.append(f"Line {line_num}, Key '{key_path}': {msg}")

            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(error_messages)
            ) from e

        # Expand user for relevant Path fields in InputConfig
        if config_instance.input.input_path:
            config_instance.input.input_path = (
                config_instance.input.input_path.expanduser()
            )
        if config_instance.input.library_path:
            config_instance.input.library_path = (
                config_instance.input.library_path.expanduser()
            )

        # Expand user for relevant Path fields in SimilarityConfig
        if config_instance.similarity.model_path:
            config_instance.similarity.model_path = (
                config_instance.similarity.model_path.expanduser()
            )

        return config_instance
