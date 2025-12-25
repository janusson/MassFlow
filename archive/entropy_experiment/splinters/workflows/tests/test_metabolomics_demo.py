import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("matchms", reason="Metabolomics demo requires matchms.")

from scripts import run_metabolomics_demo


def test_metabolomics_demo_runs_and_reports(capsys, tmp_path):
    base_cfg = yaml.safe_load(Path("examples/metabolomics_workflow.yaml").read_text())
    fixture = Path(__file__).parent / "data" / "simple.mgf"
    base_cfg["input"]["path"] = str(fixture)
    base_cfg["library"]["path"] = str(tmp_path / "library.json")
    base_cfg["similarity"]["queries"] = [str(fixture)]
    base_cfg["network"]["output"] = str(tmp_path / "network.csv")
    base_cfg["outputs"]["search_results"] = str(tmp_path / "search.csv")
    base_cfg["outputs"]["network_summary"] = str(tmp_path / "network_summary.json")

    cfg_path = tmp_path / "metabolomics_config.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")

    rc = run_metabolomics_demo.main(["--config", str(cfg_path)])
    assert rc == 0
    captured = capsys.readouterr().out.strip()
    assert captured.startswith("inputs=")
    assert "library_entries=" in captured
    summary_path = Path(base_cfg["outputs"]["network_summary"])
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary.get("nodes", 0) >= 1
