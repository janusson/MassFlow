"""
Configuration schema for MassFlow.
Uses Pydantic to validate the configuration YAML.
"""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class InputConfig(BaseModel):
    """Configuration for input data."""

    file_path: Path
    format: Literal["mgf", "msp", "mzml"] = "mgf"
    reference_library: Optional[Path] = None


class ProcessingConfig(BaseModel):
    """Configuration for spectral processing."""

    min_peaks: int = 5
    min_intensity: float = 0.0
    normalize_intensity: bool = True
    clean_metadata: bool = True

    # New MS parameters with validation
    precursor_mz: float = Field(default=0.0)
    retention_time: float = Field(default=0.0)

    @field_validator("precursor_mz")
    @classmethod
    def validate_precursor_mz(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"precursor_mz must be non-negative. Received: {v}")
        return v

    @field_validator("retention_time")
    @classmethod
    def validate_retention_time(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"retention_time must be non-negative. Received: {v}")
        return v


class SimilarityConfig(BaseModel):
    """Configuration for similarity search."""

    algorithm: Literal["cosine", "modified_cosine"] = "cosine"
    tolerance: float = 0.005
    tolerance_unit: Literal["Da", "ppm"] = "Da"
    min_score: float = 0.6
    analog_search: bool = False
    # Only used if analog_search is True
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


class MassFlowConfig(BaseModel):
    """Root configuration object."""

    input: InputConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    output_directory: Path = Path("results")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MassFlowConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)
