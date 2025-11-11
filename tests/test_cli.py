import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matchms", reason="CLI cleaning requires matchms to load libraries.")


def test_cli_clean_command(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixture = Path(__file__).parent / "data" / "simple.mgf"
    shutil.copy(fixture, data_dir / "example.mgf")
    output_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        "-m",
        "yogimass.cli",
        "clean",
        str(data_dir),
        str(output_dir),
        "--type",
        "mgf",
        "--formats",
        "mgf",
        "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0

    assert (output_dir / "example_cleaned.mgf").exists()
    assert (output_dir / "example_cleaned.json").exists()
