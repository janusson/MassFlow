import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matchms", reason="CLI commands require matchms.")


def test_cli_library_build_and_search(tmp_path):
    fixture = Path(__file__).parent / "data" / "simple.mgf"
    library_path = tmp_path / "library.json"
    results_path = tmp_path / "results.json"

    build_cmd = [
        sys.executable,
        "-m",
        "yogimass.cli",
        "library",
        "build",
        "--input",
        str(fixture),
        "--library",
        str(library_path),
    ]
    build_result = subprocess.run(build_cmd, capture_output=True, text=True)
    assert build_result.returncode == 0
    assert library_path.exists()

    search_cmd = [
        sys.executable,
        "-m",
        "yogimass.cli",
        "library",
        "search",
        "--queries",
        str(fixture),
        "--library",
        str(library_path),
        "--output",
        str(results_path),
        "--top-n",
        "1",
    ]
    search_result = subprocess.run(search_cmd, capture_output=True, text=True)
    assert search_result.returncode == 0
    assert results_path.exists()


def test_cli_config_run(tmp_path):
    fixture = Path(__file__).parent / "data" / "simple.mgf"
    library_path = tmp_path / "cli_library.json"
    search_output = tmp_path / "search.csv"
    network_output = tmp_path / "network.csv"
    summary_output = tmp_path / "network_summary.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
input:
  path: {fixture}
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

    cmd = [
        sys.executable,
        "-m",
        "yogimass.cli",
        "config",
        "run",
        "--config",
        str(config_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert library_path.exists()
    assert search_output.exists()
    assert network_output.exists()
