"""
Tests for the ``massflow tui`` CLI command (typer wiring and import guard).
"""

import builtins
from unittest.mock import patch

from typer.testing import CliRunner

from MassFlow import cli

runner = CliRunner()


def test_tui_missing_textual_exits_with_install_hint(monkeypatch):
    """The command degrades to an install hint when Textual is unavailable."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "MassFlow.tui.app":
            raise ImportError("No module named 'textual'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(cli.app, ["tui"])
    assert result.exit_code == 1
    assert "massflow[tui]" in result.output
    assert "textual" in result.output


def test_tui_launches_app_with_defaults(monkeypatch):
    with patch("MassFlow.tui.app.MassFlowApp") as mock_app:
        result = runner.invoke(cli.app, ["tui"])
    assert result.exit_code == 0
    assert mock_app.called
    kwargs = mock_app.call_args.kwargs
    assert kwargs["initial_query"] is None
    assert kwargs["initial_library"] is None
    assert kwargs["workspace"] is None
    mock_app.return_value.run.assert_called_once()


def test_tui_launches_app_with_paths(monkeypatch, tmp_path):
    query = tmp_path / "query.mgf"
    library = tmp_path / "library.msp"
    workspace = tmp_path / "workspace"
    with patch("MassFlow.tui.app.MassFlowApp") as mock_app:
        result = runner.invoke(
            cli.app,
            [
                "tui",
                "--input",
                str(query),
                "--library",
                str(library),
                "--workspace",
                str(workspace),
            ],
        )
    assert result.exit_code == 0
    kwargs = mock_app.call_args.kwargs
    assert kwargs["initial_query"] == query
    assert kwargs["initial_library"] == library
    assert kwargs["workspace"] == workspace


def test_tui_expands_tilde_paths(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    with patch("MassFlow.tui.app.MassFlowApp") as mock_app:
        result = runner.invoke(cli.app, ["tui", "--workspace", "~/workspace"])
    assert result.exit_code == 0
    assert mock_app.call_args.kwargs["workspace"] == home / "workspace"
