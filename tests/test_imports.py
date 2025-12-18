import importlib

import pytest

pytest.importorskip("matchms", reason="Core modules depend on matchms.")


def test_imports_do_not_create_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    modules = [
        "yogimass",
        "yogimass.config",
        "yogimass.similarity",
        "yogimass.similarity.metrics",
        "yogimass.similarity.processing",
        "yogimass.scoring",
        "yogimass.scoring.cosine",
        "yogimass.filters",
        "yogimass.filters.metadata",
        "yogimass.utils",
        "yogimass.utils.logging",
    ]
    for module in modules:
        importlib.invalidate_caches()
        importlib.import_module(module)
    after = set(tmp_path.iterdir())
    assert before == after
