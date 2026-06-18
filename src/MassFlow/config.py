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
    streaming_threshold_mb: int = Field(
        default=500,
        description="Threshold (in MB) above which the library is streamed from disk instead of loaded into memory.",
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
    Settings for similarity scoring with classical spectral matching algorithms.

    Supports cosine and modified cosine similarity backed by matchms.
    Legacy ``tolerance`` is retained for compatibility with existing configs.
    """

    algorithm: Literal["cosine", "modified_cosine"] = Field(
        default="cosine",
        description="Similarity algorithm: 'cosine' or 'modified_cosine'.",
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

    rt_tolerance: Optional[float] = Field(
        default=None, description="Retention time tolerance in minutes"
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

    @field_validator("min_score", "fdr_threshold")
    @classmethod
    def validate_score_ranges(cls, v: float, info: ValidationInfo) -> float:
        """Ensure scores and thresholds are within [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"{info.field_name} must be between 0.0 and 1.0. Received: {v}"
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
    High-level workflow feature flags.

    These fields are schema-level placeholders for orchestration features.
    """

    perform_peak_picking: bool = Field(default=True, description="Placeholder.")
    perform_alignment: bool = Field(default=True, description="Placeholder.")


class ExportConfig(BaseModel):
    """
    Declared export preferences for result output.

    The annotation workflow writes per-file result reports in the configured
    format. Only 'csv' and 'mztab' are part of the stable v1.0 contract.
    """

    format: Literal["csv", "mztab"] = Field(
        default="csv",
        description="Output format: 'csv' or 'mztab'.",
    )


class MassFlowConfig(BaseModel):
    """
    Root configuration object loaded from MassFlow YAML files.

    This model is the contract passed between the CLI, workflow, processing,
    and similarity layers.
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

        return config_instance
