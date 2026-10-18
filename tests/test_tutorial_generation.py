"""Reproducible synthetic onboarding; no experimental validation is claimed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from MassFlow import cli


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "tutorial_generator",
        Path(__file__).parents[1] / "scripts/generate_tutorial_data.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tutorial_generation_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    generator = _generator()
    first = generator.main()
    initial = {key: path.read_bytes() for key, path in first.items()}
    second = generator.main(clean_first=True)
    assert {key: path.read_bytes() for key, path in second.items()} == initial


def test_tutorial_does_not_overwrite_existing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "tutorial/tutorial_library.msp"
    target.parent.mkdir()
    target.write_text("user library")
    result = CliRunner().invoke(cli.app, ["tutorial"])
    assert result.exit_code != 0
    assert target.read_text() == "user library"
    assert not (tmp_path / "tutorial/tutorial_experimental.mgf").exists()


def test_tutorial_cli_creates_usable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["tutorial"])
    assert result.exit_code == 0, result.output
    config_path = tmp_path / "tutorial/tutorial_config.yaml"
    result = runner.invoke(cli.app, ["annotate", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tutorial/results/tutorial_experimental_results.csv").exists()
    assert (
        tmp_path / "tutorial/results/tutorial_experimental_results.report.yaml"
    ).exists()
