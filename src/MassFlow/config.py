"""
Configuration schema definitions for MassFlow.

This module employs Pydantic models to validate and structure configuration data
loaded from YAML files. It defines a hierarchy of configuration classes including
``ProjectConfig``, ``InputConfig``, ``ProcessingConfig``, ``SimilarityConfig``,
``WorkflowConfig``, and ``ExportConfig``, culminating in the root ``MassFlowConfig``
object. These classes enforce strict type hints, default values, and logical
validations (e.g., m/z ranges, tolerance units) to ensure the integrity of the
analysis pipeline settings.
"""

from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ProjectConfig(BaseModel):
    """General project metadata."""

    name: str = "MassFlow_Project"
    output_directory: Path = Path("results")


class InputConfig(BaseModel):
    """Configuration for input data."""

    file_path: Optional[Path] = None
    data_directory: Optional[Path] = None
    format: Optional[Literal["mgf", "msp", "mzml", "mzxml", "db", "sqlite"]] = None
    reference_library: Optional[Path] = None

    @field_validator("data_directory")
    @classmethod
    def validate_data_directory(cls, v: Optional[Path]) -> Optional[Path]:
        """
        Validate that the data directory exists if provided.

        This validator checks if the provided path for ``data_directory`` exists.
        Currently, it allows non-existent directories to pass without raising an
        error, permitting creation at a later stage or tolerating user error
        during configuration loading.

        Parameters
        ----------
        v : Path or None
            The path to the data directory to validate.

        Returns
        -------
        Path or None
            The validated path object, or None if not provided.
        """
        if v and not v.exists():
            # Start warning but allow it (could be created later or user error)
            pass
        return v


class SolventConfig(BaseModel):
    """Configuration for a specific solvent."""

    name: str
    formula: Optional[str] = None
    mz: float


class ProcessingConfig(BaseModel):
    """Configuration for spectral processing."""

    # Standard filters
    min_peaks: int = 5
    min_intensity: float = 0.0
    normalize_intensity: bool = True
    clean_metadata: bool = True

    # HLD v1.0 Pre-Processing
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

    # New instrument-specific parameters
    ms1_tolerance: float = Field(
        default=10.0, description="Precursor mass tolerance in ppm"
    )
    ms2_tolerance: float = Field(
        default=0.02, description="Fragment mass tolerance in Da"
    )
    noise_threshold: float = Field(
        default=1000.0, description="Minimum intensity threshold"
    )

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
    """Configuration for similarity search."""

    algorithm: Literal["cosine", "modified_cosine", "spec2vec", "ms2deepscore"] = (
        "cosine"
    )
    model_path: Optional[Path] = None
    tolerance: float = 0.005
    tolerance_unit: Literal["Da", "ppm"] = "Da"
    min_score: float = 0.6
    analog_search: bool = False
    min_matched_peaks: int = 3
    fdr_threshold: float = 0.01

    @field_validator("tolerance_unit")
    @classmethod
    def validate_tolerance_unit(cls, v: str) -> str:
        """
        Validate that the tolerance unit is either 'Da' or 'ppm'.

        Parameters
        ----------
        v : str
            The tolerance unit string to validate.

        Returns
        -------
        str
            The validated tolerance unit.

        Raises
        ------
        ValueError
            If ``tolerance_unit`` is not 'Da' or 'ppm'.
        """
        valid_units = {"Da", "ppm"}
        if v not in valid_units:
            raise ValueError(
                f"tolerance_unit must be one of {valid_units}. Received: {v}"
            )
        return v


class WorkflowConfig(BaseModel):
    """Configuration for workflow steps."""

    perform_peak_picking: bool = True
    perform_alignment: bool = True
    perform_networking: bool = False
    export_consensus: bool = True


class ExportConfig(BaseModel):
    """Configuration for data export."""

    format: Literal["csv", "pickle", "msp", "mgf", "json", "xlsx", "parquet"] = "csv"


class MassFlowConfig(BaseModel):
    """Root configuration object."""

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
        Load configuration from a YAML file.

        This factory method reads a YAML file from the specified path, parses
        its content, and instantiates a ``MassFlowConfig`` object using the
        loaded data.

        Parameters
        ----------
        path : str or Path
            The file system path to the YAML configuration file.

        Returns
        -------
        MassFlowConfig
            An instance of the configuration object populated with data from the YAML file.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)
