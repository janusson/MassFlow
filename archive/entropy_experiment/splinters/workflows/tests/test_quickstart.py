from pathlib import Path

import yaml

from yogimass import quickstart
from yogimass.config import load_config


def test_quickstart_generates_valid_config(tmp_path, monkeypatch):
    inputs = iter(
        [
            "untargeted_metabolomics",
            "mgf",
            str(tmp_path / "input.mgf"),
            str(tmp_path / "outdir"),
            "y",  # overwrite confirmation (file absent, still accepts)
        ]
    )

    output_file = tmp_path / "yogimass-quickstart.yaml"
    monkeypatch.setenv("PWD", str(tmp_path))

    # run in tmp dir by adjusting cwd via chdir context
    cwd = Path.cwd()
    try:
        Path(tmp_path).mkdir(parents=True, exist_ok=True)
        # temporarily chdir to tmp_path so default filename lands there
        import os

        os.chdir(tmp_path)
        rc = quickstart.run_quickstart(input_func=lambda _: next(inputs))
    finally:
        import os

        os.chdir(cwd)

    assert rc == 0
    assert output_file.exists()
    data = yaml.safe_load(output_file.read_text())
    cfg = load_config(data)
    assert cfg.input.format == "mgf"
    assert cfg.similarity.enabled
    assert cfg.library.build
