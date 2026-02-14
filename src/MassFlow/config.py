"""
Configuration schema for MassFlow.
Uses Pydantic to validate the configuration YAML.
"""

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ProjectConfig(BaseModel):
    """General project metadata."""

    name: str = "MassFlow_Project"
    output_directory: Path = Path("results")


class InputConfig(BaseModel):
    """Configuration for input data."""

    file_path: Optional[Path] = None
    data_directory: Optional[Path] = None
    format: Literal["mgf", "msp", "mzml", "mzxml", "db"] = "mgf"
    reference_library: Optional[Path] = None

    @field_validator("data_directory")
    @classmethod
    def validate_data_directory(cls, v: Optional[Path]) -> Optional[Path]:
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
        if v < 0:
            raise ValueError(f"precursor_mz must be non-negative. Received: {v}")
        return v


class SimilarityConfig(BaseModel):
    """Configuration for similarity search."""

    algorithm: Literal["cosine", "modified_cosine"] = "cosine"
    tolerance: float = 0.005
    tolerance_unit: Literal["Da", "ppm"] = "Da"
    min_score: float = 0.6
    analog_search: bool = False
    min_matched_peaks: int = 3

    @field_validator("tolerance_unit")
    @classmethod
    def validate_tolerance_unit(cls, v: str) -> str:
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
        return self.project.output_directory

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MassFlowConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)
