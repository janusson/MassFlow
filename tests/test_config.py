"""
Tests for MassFlow configuration.
"""
import pytest
import yaml
from pathlib import Path
from MassFlow.config import MassFlowConfig, InputConfig, ProcessingConfig, SimilarityConfig

def test_default_config():
    """Test default configuration values."""
    config = MassFlowConfig(
        input=InputConfig(file_path=Path("test.mgf"))
    )
    assert config.processing.min_intensity == 0.0  # Actually it is min_intensity in model, need to check
    # Checking the model definition:
    # min_intensity: float = 0.0
    assert config.processing.min_intensity == 0.0
    assert config.similarity.algorithm == "cosine"

def test_load_from_yaml(tmp_path):
    """Test loading configuration from a YAML file."""
    config_data = {
        "input": {
            "file_path": "/path/to/data.mgf",
            "format": "mgf"
        },
        "processing": {
            "min_intensity": 100.0,
            "clean_metadata": False
        },
        "similarity": {
            "algorithm": "modified_cosine",
            "min_score": 0.8
        },
        "database": {
            "url": "sqlite:///:memory:"
        }
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
        
    config = MassFlowConfig.from_yaml(config_file)
    
    assert str(config.input.file_path) == "/path/to/data.mgf"
    assert config.processing.min_intensity == 100.0
    assert config.processing.clean_metadata is False
    assert config.similarity.algorithm == "modified_cosine"
    assert config.database.url == "sqlite:///:memory:"

def test_load_file_not_found():
    """Test that FileNotFoundError is raised for missing config file."""
    with pytest.raises(FileNotFoundError):
        MassFlowConfig.from_yaml("nonexistent_config.yaml")

def test_invalid_yaml(tmp_path):
    """Test validation error for invalid YAML."""
    # Missing required 'input' field
    config_data = {
        "processing": {}
    }
    config_file = tmp_path / "invalid.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
        
    with pytest.raises(Exception): # Pydantic ValidationError
        MassFlowConfig.from_yaml(config_file)
