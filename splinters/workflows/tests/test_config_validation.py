from pathlib import Path

import pytest

from yogimass.config import ConfigError, load_config


def test_missing_inputs_raise_error():
    config = {}
    with pytest.raises(ConfigError) as excinfo:
        load_config(config)
    assert excinfo.value.path == "input.paths"


def test_network_threshold_and_knn_mutually_exclusive():
    config = {
        "input": {"path": "tests/data/simple.mgf"},
        "library": {"path": "out/library.json"},
        "network": {
            "enabled": True,
            "threshold": 0.5,
            "knn": 5,
            "output": "out/network.csv",
        },
    }
    with pytest.raises(ConfigError) as excinfo:
        load_config(config)
    assert excinfo.value.path == "network.threshold"
    assert "threshold or knn" in excinfo.value.message


def test_invalid_enum_value_reports_field():
    config = {
        "input": {"path": "data/example.txt", "format": "txt"},
        "library": {"path": "out/library.json"},
    }
    with pytest.raises(ConfigError) as excinfo:
        load_config(config)
    assert excinfo.value.path == "input.format"


def test_unknown_top_level_key_reported():
    with pytest.raises(ConfigError) as excinfo:
        load_config({"unexpected": {"value": 1}})
    assert excinfo.value.path == "unexpected"


def test_processor_config_parses_and_validates():
    config = {
        "input": {"path": "tests/data/simple.mgf"},
        "library": {"path": "out/library.json"},
        "similarity": {
            "processor": {
                "normalization": "basepeak",
                "min_relative_intensity": 0.05,
                "mz_dedup_tolerance": 0.01,
            }
        },
    }
    cfg = load_config(config)
    proc = cfg.similarity.processor
    assert proc.normalization == "basepeak"
    assert proc.min_relative_intensity == 0.05
    assert proc.mz_dedup_tolerance == 0.01


def test_processor_config_rejects_negative_values():
    config = {
        "input": {"path": "tests/data/simple.mgf"},
        "library": {"path": "out/library.json"},
        "similarity": {
            "processor": {
                "min_absolute_intensity": -0.1,
            }
        },
    }
    with pytest.raises(ConfigError) as excinfo:
        load_config(config)
    assert excinfo.value.path == "similarity.processor.min_absolute_intensity"


def test_msdial_example_config_is_valid():
    config = load_config(Path("examples/msdial_workflow.yaml"))
    assert config.input.format == "msdial"
    assert config.network.enabled
