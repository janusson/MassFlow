import json
from pathlib import Path

import pytest

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
