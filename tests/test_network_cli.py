"""
CLI tests for ``massflow network build`` (P1, EXPERIMENTAL).

These are integration tests: they run the real command against a tiny MGF and a
real YAML config, exercising loading, processing, graph construction, and export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from matchms import Spectrum
from typer.testing import CliRunner

from MassFlow import cli, io
from MassFlow.config import NetworkConfig
from MassFlow.network.build import build_spectral_graph
from MassFlow.workflow import FileExecutionResult

pytestmark = pytest.mark.integration

_BASE_PEAKS = [
    (100.0, 1.0),
    (150.0, 0.8),
    (200.0, 0.5),
    (250.0, 0.4),
    (300.0, 0.3),
    (350.0, 0.2),
    (400.0, 0.15),
    (450.0, 0.1),
]


def _ion(title: str, precursor_mz: str, shift: float) -> str:
    rows = "".join(
        f"{mz + shift:.2f} {intensity:.3f}\n" for mz, intensity in _BASE_PEAKS
    )
    return (
        f"BEGIN IONS\nTITLE={title}\nPEPMASS={precursor_mz}\nCHARGE=1\n{rows}END IONS\n"
    )


def _write_fixtures(tmp_path: Path, network_block: str) -> Path:
    mgf = tmp_path / "data.mgf"
    mgf.write_text(
        _ion("x1", "500.0", 0.0) + _ion("x2", "500.01", 0.0) + _ion("x3", "600.0", 20.0)
    )
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(
        f"input:\n  input_path: data.mgf\n  format: mgf\n{network_block}"
    )
    return config_file


def test_network_build_writes_graph(tmp_path: Path) -> None:
    config_file = _write_fixtures(
        tmp_path,
        "network:\n  enabled: true\n  min_score: 0.7\n  min_matched_peaks: 3\n",
    )
    output_path = tmp_path / "graph.json"

    result = CliRunner().invoke(
        cli.app,
        [
            "network",
            "build",
            "--config",
            str(config_file),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Network built" in result.stdout
    assert output_path.exists()

    document = json.loads(output_path.read_text())
    assert len(document["nodes"]) == 3
    assert len(document["relationships"]) == 1
    assert document["fdr_assessments"] == {}


def test_network_build_created_at_pins_provenance_for_reproducible_output(
    tmp_path: Path,
) -> None:
    config_file = _write_fixtures(
        tmp_path,
        "network:\n  enabled: true\n  min_score: 0.7\n  min_matched_peaks: 3\n",
    )
    pinned = "2026-01-01T00:00:00+00:00"
    runner = CliRunner()

    documents: list[bytes] = []
    for label in ("first", "second"):
        output_path = tmp_path / f"{label}.json"
        result = runner.invoke(
            cli.app,
            [
                "network",
                "build",
                "--config",
                str(config_file),
                "--output",
                str(output_path),
                "--created-at",
                pinned,
            ],
        )
        assert result.exit_code == 0, result.stdout
        document = json.loads(output_path.read_text())
        assert document["provenance"]["created_at"] == pinned
        for relationship in document["relationships"]:
            assert relationship["provenance"]["created_at"] == pinned
        documents.append(output_path.read_bytes())

    assert documents[0] == documents[1]


def test_network_build_is_disabled_by_default(tmp_path: Path) -> None:
    config_file = _write_fixtures(tmp_path, "")

    result = CliRunner().invoke(
        cli.app, ["network", "build", "--config", str(config_file)]
    )

    assert result.exit_code == 1
    assert "disabled" in result.stdout.lower()


def test_network_build_analyse_export_round_trip(tmp_path: Path) -> None:
    config_file = _write_fixtures(
        tmp_path,
        "network:\n  enabled: true\n  min_score: 0.7\n  min_matched_peaks: 3\n",
    )
    runner = CliRunner()

    graph_path = tmp_path / "graph.json"
    build = runner.invoke(
        cli.app,
        ["network", "build", "--config", str(config_file), "--output", str(graph_path)],
    )
    assert build.exit_code == 0, build.stdout
    assert json.loads(graph_path.read_text())["families"]

    analysed_path = tmp_path / "analysed.json"
    analyse = runner.invoke(
        cli.app,
        [
            "network",
            "analyse",
            "--input",
            str(graph_path),
            "--output",
            str(analysed_path),
        ],
    )
    assert analyse.exit_code == 0, analyse.stdout
    assert "Analysed" in analyse.stdout
    assert json.loads(analysed_path.read_text())["families"]

    families_path = tmp_path / "families.jsonl"
    export = runner.invoke(
        cli.app,
        [
            "network",
            "export",
            "--input",
            str(analysed_path),
            "--output",
            str(families_path),
        ],
    )
    assert export.exit_code == 0, export.stdout
    assert "Exported" in export.stdout
    assert families_path.read_text().strip()


def _spectra() -> list[Spectrum]:
    mz = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0])
    intensities = np.array([1.0, 0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])

    def build(spec_id: str, precursor_mz: float) -> Spectrum:
        return Spectrum(
            mz=mz.copy(),
            intensities=intensities.copy(),
            metadata={
                "id": spec_id,
                "precursor_mz": precursor_mz,
                "retention_time": 100.0,
                "adduct": "[M+H]+",
                "charge": 1,
            },
        )

    return [build("a", 500.0), build("b", 500.005), build("c", 700.0)]


def test_network_analyse_with_config_seeds_context(tmp_path: Path) -> None:
    spectra = _spectra()
    graph = build_spectral_graph(
        spectra,
        NetworkConfig(
            enabled=True, min_score=0.7, min_matched_peaks=3, top_k_per_node=None
        ),
    )
    graph_path = tmp_path / "graph.json"
    io.save_molecular_graph(graph, graph_path)

    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(
        "input:\n  input_path: data.mgf\nnetwork:\n  enabled: true\n"
    )

    row: Any = {
        "query_id": "a",
        "reference_id": "lib1",
        "reference_name": "Caffeine",
        "q_value": 0.001,
        "score": 0.99,
    }
    result = FileExecutionResult(
        status="success",
        input_path=tmp_path / "data.mgf",
        query_spectra=spectra,
        results=[row],
    )
    output_path = tmp_path / "analysed.json"
    with patch("MassFlow.workflow.run_annotation_pipeline", return_value=[result]):
        invoked = CliRunner().invoke(
            cli.app,
            [
                "network",
                "analyse",
                "--input",
                str(graph_path),
                "--config",
                str(config_file),
                "--output",
                str(output_path),
            ],
        )

    assert invoked.exit_code == 0, invoked.stdout
    document = json.loads(output_path.read_text())
    assert document["network_contexts"]
    assert document["inferences"]
    assert any(family["label"] == "Caffeine" for family in document["families"])
