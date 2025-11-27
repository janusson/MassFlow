import importlib

import pytest

pytest.importorskip("matchms", reason="Core modules depend on matchms.")


def test_imports_do_not_create_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    modules = [
        "yogimass",
        "yogimass.cli",
        "yogimass.workflow",
        "yogimass.pipeline",
        "yogimass.config",
        "yogimass.io.msdial_clean_combine",
        "yogimass.io.msdial_process",
        "yogimass.similarity.library",
        "yogimass.similarity.backends",
        "yogimass.similarity.embeddings",
        "yogimass.networking.network",
    ]
    for module in modules:
        importlib.invalidate_caches()
        importlib.import_module(module)
    after = set(tmp_path.iterdir())
    assert before == after
