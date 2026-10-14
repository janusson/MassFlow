"""
The configuration wizard must emit a configuration that validates against the
real schema, and wire the processing toggles so the values it collects actually
take effect.
"""

from __future__ import annotations

from unittest.mock import patch

from MassFlow.config import MassFlowConfig
from MassFlow.wizard import run_config_wizard

_QUERY_MSP = "NAME: Q\nPRECURSOR_MZ: 100.0\nCHARGE: 1\nNUM PEAKS: 1\n50.0\t1.0\n"
_LIB_MSP = "NAME: L\nPRECURSOR_MZ: 100.0\nCHARGE: 1\nNUM PEAKS: 1\n50.0\t1.0\n"


def _answers(query_path, lib_path):
    """Return a fake prompt that fills paths and otherwise accepts defaults."""

    def ask(prompt, *args, **kwargs):
        text = str(prompt).lower()
        if "experimental" in text:
            return str(query_path)
        if "reference library" in text:
            return str(lib_path)
        return kwargs.get("default")

    return ask


def _run_wizard(tmp_path):
    query = tmp_path / "query.msp"
    query.write_text(_QUERY_MSP)
    library = tmp_path / "library.msp"
    library.write_text(_LIB_MSP)
    out = tmp_path / "config.yaml"
    ask = _answers(query, library)
    with (
        patch("MassFlow.wizard.Prompt.ask", side_effect=ask),
        patch("MassFlow.wizard.IntPrompt.ask", side_effect=ask),
        patch("MassFlow.wizard.FloatPrompt.ask", side_effect=ask),
    ):
        run_config_wizard(out)
    return out


def test_wizard_output_validates_against_schema(tmp_path):
    out = _run_wizard(tmp_path)
    cfg = MassFlowConfig.from_yaml(out)
    assert cfg.input.input_path.exists()
    assert cfg.input.library_path is not None


def test_wizard_enables_processing_toggles_for_collected_values(tmp_path):
    cfg = MassFlowConfig.from_yaml(_run_wizard(tmp_path))
    # The values the wizard collects are only effective with their toggles on.
    assert cfg.processing.filter_min_peaks is True
    assert cfg.processing.filter_by_intensity is True
    assert cfg.processing.min_peaks == 5
    assert cfg.processing.noise_threshold == 1000.0


def test_wizard_sets_scientific_thresholds(tmp_path):
    cfg = MassFlowConfig.from_yaml(_run_wizard(tmp_path))
    assert cfg.similarity.min_score == 0.7
    assert cfg.similarity.min_matched_peaks == 3
    assert cfg.similarity.fdr_threshold == 0.05
