import numpy as np
import pytest

from yogimass.config import ConfigError, ProcessorConfig


def test_processor_config_defaults():
    cfg = ProcessorConfig.from_mapping({})
    assert cfg.normalization is None
    assert cfg.min_relative_intensity == 0.01
    assert cfg.min_absolute_intensity == 0.0
    assert cfg.max_peaks == 500
    assert cfg.mz_dedup_tolerance is None
    assert cfg.float_dtype == np.float64


def test_processor_config_rejects_unknown_keys():
    with pytest.raises(ConfigError) as excinfo:
        ProcessorConfig.from_mapping({"unexpected": 1})
    assert excinfo.value.path == "processor.unexpected"


def test_processor_config_rejects_negative_values():
    with pytest.raises(ConfigError) as excinfo:
        ProcessorConfig.from_mapping({"min_absolute_intensity": -0.1})
    assert excinfo.value.path == "processor.min_absolute_intensity"


def test_processor_config_parses_values():
    cfg = ProcessorConfig.from_mapping(
        {
            "normalization": "basepeak",
            "min_relative_intensity": 0.05,
            "mz_dedup_tolerance": 0.01,
            "max_peaks": 200,
            "float_dtype": "float32",
        }
    )
    assert cfg.normalization == "basepeak"
    assert cfg.min_relative_intensity == 0.05
    assert cfg.mz_dedup_tolerance == 0.01
    assert cfg.max_peaks == 200
    assert cfg.float_dtype == np.float32


def test_processor_config_rejects_invalid_normalization():
    with pytest.raises(ConfigError) as excinfo:
        ProcessorConfig.from_mapping({"normalization": "unit"})
    assert excinfo.value.path == "processor.normalization"


def test_processor_config_rejects_invalid_float_dtype():
    with pytest.raises(ConfigError) as excinfo:
        ProcessorConfig.from_mapping({"float_dtype": "float128"})
    assert excinfo.value.path == "processor.float_dtype"
