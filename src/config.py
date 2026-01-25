"""
Configuration schema for MassFlow.
Uses Pydantic to validate the configuration YAML.
"""
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
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

class SimilarityConfig(BaseModel):
    """Configuration for similarity search."""
    algorithm: Literal["cosine", "modified_cosine"] = "cosine"
    tolerance: float = 0.005
    min_score: float = 0.6
    analog_search: bool = False
    # Only used if analog_search is True
    min_matched_peaks: int = 3


class MassFlowConfig(BaseModel):
    """Root configuration object."""
    input: InputConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
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
