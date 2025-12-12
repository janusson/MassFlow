import csv
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("matchms", reason="Workflow tests require matchms.")

from yogimass import cli
from yogimass import workflow


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def test_processing_workflow_runs_with_processor_block(tmp_path):
    fixture = Path(__file__).parent / "data" / "simple.mgf"
    example_cfg_path = Path("examples/processing_workflow.yaml")
    config = yaml.safe_load(example_cfg_path.read_text())
    cfg_library = tmp_path / "cfg_library.json"
    cfg_search = tmp_path / "cfg_search.csv"
    cfg_network = tmp_path / "cfg_network.csv"
    cfg_summary = tmp_path / "cfg_network_summary.json"

    config["input"]["path"] = str(fixture)
    config["library"]["path"] = str(cfg_library)
    config["similarity"]["queries"] = [str(fixture)]
    config["network"]["output"] = str(cfg_network)
    config["outputs"]["search_results"] = str(cfg_search)
    config["outputs"]["network_summary"] = str(cfg_summary)

    cfg_path = tmp_path / "processing_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    workflow.run_from_config(cfg_path)

    assert cfg_library.exists()
    assert cfg_search.exists()
    assert cfg_network.exists()
    cfg_search_rows = _read_csv_dicts(cfg_search)
    cfg_summary_data = json.loads(cfg_summary.read_text())

    cli_library = tmp_path / "cli_library.json"
    cli_search = tmp_path / "cli_search.csv"
    cli_network = tmp_path / "cli_network.csv"
    cli_summary = tmp_path / "cli_network_summary.json"

    processor_flags = [
        "--similarity.processor.normalization",
        "basepeak",
        "--similarity.processor.min-relative-intensity",
        "0.02",
        "--similarity.processor.min-absolute-intensity",
        "0.01",
        "--similarity.processor.max-peaks",
        "500",
        "--similarity.processor.mz-dedup-tolerance",
        "0.01",
        "--similarity.processor.float-dtype",
        "float64",
    ]

    rc_build = cli.main(
        [
            "library",
            "build",
            "--input",
            str(fixture),
            "--library",
            str(cli_library),
            "--format",
            "mgf",
        ]
        + processor_flags
    )
    assert rc_build == 0

    rc_search = cli.main(
        [
            "library",
            "search",
            "--queries",
            str(fixture),
            "--library",
            str(cli_library),
            "--format",
            "mgf",
            "--top-n",
            "3",
            "--min-score",
            "0.0",
            "--output",
            str(cli_search),
        ]
        + processor_flags
    )
    assert rc_search == 0

    rc_network = cli.main(
        [
            "network",
            "build",
            "--library",
            str(cli_library),
            "--metric",
            "spec2vec",
            "--knn",
            "5",
            "--output",
            str(cli_network),
            "--summary",
            str(cli_summary),
        ]
        + processor_flags
    )
    assert rc_network == 0

    cli_search_rows = _read_csv_dicts(cli_search)
    cli_summary_data = json.loads(cli_summary.read_text())

    assert cli_search_rows == cfg_search_rows
    assert cli_summary_data["nodes"] == cfg_summary_data["nodes"]
    assert cli_summary_data["edges"] == cfg_summary_data["edges"]


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
