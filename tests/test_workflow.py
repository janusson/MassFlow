import csv
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("matchms", reason="Workflow tests require matchms.")

from yogimass import workflow


def test_run_from_config_builds_outputs(tmp_path):
    fixture = Path(__file__).parent / "data" / "simple.mgf"
    library_path = tmp_path / "library.json"
    search_output = tmp_path / "search.csv"
    network_output = tmp_path / "network.csv"
    summary_output = tmp_path / "network_summary.json"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
input:
  path: {fixture}
  format: mgf
library:
  path: {library_path}
  build: true
similarity:
  search: true
  queries:
    - {fixture}
  top_n: 1
outputs:
  search_results: {search_output}
  network: {network_output}
  network_summary: {summary_output}
network:
  enabled: true
  metric: spec2vec
  threshold: 0.1
        """,
        encoding="utf-8",
    )

    workflow.run_from_config(config_path)

    assert library_path.exists()
    assert search_output.exists()
    assert network_output.exists()
    summary = json.loads(summary_output.read_text())
    assert summary["nodes"] >= 1


def test_run_from_config_msdial_pipeline(tmp_path):
    fixture_dir = Path(__file__).parent / "data" / "msdial_small"
    msdial_output = tmp_path / "msdial_clean"
    library_path = tmp_path / "msdial_library.json"
    curated_library = tmp_path / "msdial_library_curated.json"
    qc_report = tmp_path / "msdial_qc.json"
    search_output = tmp_path / "msdial_search.csv"
    network_output = tmp_path / "msdial_network.csv"
    summary_output = tmp_path / "msdial_network_summary.json"

    config_path = Path("examples/msdial_workflow.yaml")
    config = yaml.safe_load(config_path.read_text())
    config["input"]["path"] = str(fixture_dir)
    config["input"]["msdial_output"] = str(msdial_output)
    config["library"]["path"] = str(library_path)
    config["curation"]["output"] = str(curated_library)
    config["curation"]["qc_report"] = str(qc_report)
    config["similarity"]["queries"] = [str(fixture_dir)]
    config["network"]["output"] = str(network_output)
    config["outputs"]["search_results"] = str(search_output)
    config["outputs"]["network_summary"] = str(summary_output)

    workflow.run_from_config(config)

    assert curated_library.exists()
    assert qc_report.exists()
    assert search_output.exists()
    with search_output.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
        assert len(lines) > 1
    assert network_output.exists()
    with network_output.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert rows
    summary = json.loads(summary_output.read_text())
    assert summary["nodes"] == 3
    assert summary["edges"] >= 2
    combined_dir = msdial_output / "combined_results"
    assert combined_dir.exists()
    assert any(combined_dir.glob("*.csv"))
