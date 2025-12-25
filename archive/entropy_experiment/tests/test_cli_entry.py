import runpy
import pytest


def test_cli_main_callable():
    import yogimass.cli as cli

    assert callable(cli.main)


def test_run_module_raises_systemexit():
    # Running `python -m yogimass` with no args should trigger argparse and exit
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("yogimass", run_name="__main__")
    assert isinstance(exc.value.code, int)


def test_help_exits_zero():
    import yogimass.cli as cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
