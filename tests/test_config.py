"""
Tests for MassFlow configuration.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
)


def test_default_config():
    """Test default configuration values."""
    config = MassFlowConfig(input=InputConfig(file_path=Path("test.mgf")))
    # Check Project defaults
    assert config.project.name == "MassFlow_Project"
    assert config.project.output_directory == Path("results")

    # Check Processing defaults
    assert config.processing.min_intensity == 0.0
    assert config.processing.min_peaks == 5

    # Check Similarity defaults
    assert config.similarity.ms1_tolerance == 10.0
    assert config.similarity.ms2_tolerance == 0.02

    assert config.processing.noise_threshold == 1000.0
    assert config.processing.instrument is None

    # Check Similarity defaults
    assert config.similarity.algorithm == "cosine"

    # Check Workflow defaults
    assert config.workflow.perform_peak_picking is True

    # Check Export defaults
    assert config.export.format == "csv"

    # Check Property mapping
    assert config.output_directory == Path("results")


def test_load_from_yaml(tmp_path):
    """Test loading configuration from a YAML file."""
    config_data = {
        "project": {"name": "Test_Project", "output_directory": "/tmp/results"},
        "input": {"file_path": "/path/to/data.mgf", "format": "mgf"},
        "processing": {
            "min_intensity": 100.0,
            "noise_threshold": 500.0,
            "instrument": "QTOF",
            "mode": "positive",
            "solvents": [
                {"name": "Water", "mz": 18.01},
                {"name": "Methanol", "mz": 32.04},
            ],
        },
        "similarity": {"algorithm": "modified_cosine", "min_score": 0.8},
        "export": {"format": "json"},
    }

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = MassFlowConfig.from_yaml(config_file)

    assert config.project.name == "Test_Project"
    assert str(config.project.output_directory) == "/tmp/results"
    assert str(config.input.file_path) == "/path/to/data.mgf"

    assert config.processing.min_intensity == 100.0
    assert config.processing.noise_threshold == 500.0
    assert config.processing.instrument == "QTOF"
    assert config.processing.mode == "positive"
    assert len(config.processing.solvents) == 2
    assert config.processing.solvents[0].name == "Water"
    assert config.processing.solvents[0].mz == 18.01

    assert config.similarity.algorithm == "modified_cosine"
    assert config.export.format == "json"


def test_load_file_not_found():
    """Test that FileNotFoundError is raised for missing config file."""
    with pytest.raises(FileNotFoundError):
        MassFlowConfig.from_yaml("nonexistent_config.yaml")


def test_invalid_yaml(tmp_path):
    """Test validation error for invalid YAML."""
    # Missing required 'input' field
    config_data = {"processing": {}}
    config_file = tmp_path / "invalid.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValidationError):
        MassFlowConfig.from_yaml(config_file)


def test_data_directory_validation():
    """Test data_directory validation logic."""
    # Valid existing directory
    config = MassFlowConfig(input=InputConfig(data_directory=Path(".")))
    assert config.input.data_directory == Path(".")

    # Non-existing directory (should pass but might log warning in real usage, validator currently allows it)
    config = MassFlowConfig(input=InputConfig(data_directory=Path("nonexistent")))
    assert config.input.data_directory == Path("nonexistent")


def test_input_config_accepts_library_path_keyword():
    """Preferred library_path keyword should populate the library field."""
    config = InputConfig(
        file_path=Path("query.mgf"),
        library_path=Path("library.msp"),
        format="mgf",
    )

    assert config.library_path == Path("library.msp")


def test_input_config_accepts_reference_library_keyword_for_backward_compatibility():
    """Legacy reference_library keyword should still populate the library field."""
    config = InputConfig.model_validate(
        {
            "file_path": Path("query.mgf"),
            "reference_library": Path("library.msp"),
            "format": "mgf",
        }
    )

    assert config.library_path == Path("library.msp")


def test_load_from_yaml_accepts_library_path_alias(tmp_path):
    """YAML using library_path should load into the same input field."""
    config_data = {
        "project": {"name": "Alias_Test", "output_directory": "/tmp/results"},
        "input": {
            "file_path": "/path/to/data.mgf",
            "library_path": "/path/to/library.msp",
            "format": "mgf",
        },
    }

    config_file = tmp_path / "config_with_library_path.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = MassFlowConfig.from_yaml(config_file)

    assert str(config.input.file_path) == "/path/to/data.mgf"
    assert str(config.input.library_path) == "/path/to/library.msp"
