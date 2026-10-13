"""
Configuration tests for the networking subsystem.

Verifies that ``NetworkConfig`` follows the strict-MassFlow conventions and that
the optional ``network:`` YAML section integrates with ``MassFlowConfig``
(disabled by default, unknown keys rejected, recorded in normalized provenance).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from MassFlow.config import MassFlowConfig, NetworkConfig

pytestmark = pytest.mark.unit


def _write(tmp_path: Path, text: str) -> Path:
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(text)
    return config_file


def test_network_config_defaults_are_disabled() -> None:
    cfg = NetworkConfig()
    assert cfg.enabled is False
    assert cfg.algorithm == "modified_cosine"
    assert cfg.min_score == pytest.approx(0.7)
    assert cfg.top_k_per_node == 10
    assert cfg.build_features is True
    assert cfg.feature_precursor_tolerance == pytest.approx(0.01)
    assert cfg.feature_rt_tolerance == pytest.approx(30.0)


def test_network_config_forbids_extra_and_bounds_values() -> None:
    with pytest.raises(ValidationError):
        NetworkConfig(bogus=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        NetworkConfig(min_score=1.5)
    with pytest.raises(ValidationError):
        NetworkConfig(top_k_per_node=0)
    with pytest.raises(ValidationError):
        NetworkConfig(ms1_tolerance=0.0)


def test_massflow_config_defaults_network_off(tmp_path: Path) -> None:
    config_file = _write(tmp_path, "input:\n  input_path: data.mgf\n")
    cfg = MassFlowConfig.from_yaml(config_file)
    assert cfg.network.enabled is False


def test_massflow_config_parses_network_section(tmp_path: Path) -> None:
    config_file = _write(
        tmp_path,
        "input:\n"
        "  input_path: data.mgf\n"
        "network:\n"
        "  enabled: true\n"
        "  algorithm: cosine\n"
        "  min_score: 0.5\n"
        "  top_k_per_node: 3\n",
    )
    cfg = MassFlowConfig.from_yaml(config_file)
    assert cfg.network.enabled is True
    assert cfg.network.algorithm == "cosine"
    assert cfg.network.min_score == pytest.approx(0.5)
    assert cfg.network.top_k_per_node == 3


def test_network_section_rejects_unknown_key(tmp_path: Path) -> None:
    config_file = _write(
        tmp_path,
        "input:\n  input_path: data.mgf\nnetwork:\n  bogus: 1\n",
    )
    with pytest.raises(ValueError) as excinfo:
        MassFlowConfig.from_yaml(config_file)
    assert "bogus" in str(excinfo.value)


def test_normalized_config_records_network_section(tmp_path: Path) -> None:
    config_file = _write(
        tmp_path,
        "input:\n  input_path: data.mgf\nnetwork:\n  enabled: true\n",
    )
    cfg = MassFlowConfig.from_yaml(config_file)
    normalized = cfg.normalized_config()
    assert normalized["effective_config"]["network"]["enabled"] is True
    assert normalized["config_digest_sha256"]
