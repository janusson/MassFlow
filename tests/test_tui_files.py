"""
Tests for MassFlow.tui.files — discovery, classification, workspace upload.
"""

from pathlib import Path

import pytest

from MassFlow.tui.files import (
    classify_file,
    copy_into_workspace,
    discover_spectral_files,
    guess_backend,
    human_size,
)


class TestClassifyFile:
    def test_query_formats(self, tmp_path: Path):
        for name in ("run.mzml", "run.mzxml"):
            entry = classify_file(tmp_path / name)
            assert entry.kind == "query"
            assert entry.format_hint == name.split(".")[1]

    def test_text_library_formats(self, tmp_path: Path):
        for name in ("library.msp", "library.mgf"):
            entry = classify_file(tmp_path / name)
            assert entry.kind == "library"

    def test_databases(self, tmp_path: Path):
        for name in ("lib.db", "lib.sqlite"):
            entry = classify_file(tmp_path / name)
            assert entry.kind == "database"

    def test_zarr_directory(self, tmp_path: Path):
        store = tmp_path / "store.zarr"
        store.mkdir()
        (store / ".zgroup").write_text("{}")
        entry = classify_file(store)
        assert entry.kind == "database"
        assert entry.format_hint == "zarr"

    def test_vendor_formats(self, tmp_path: Path):
        for name in ("run.raw", "run.wiff", "run.d"):
            entry = classify_file(tmp_path / name)
            assert entry.kind == "vendor"

    def test_unsupported(self, tmp_path: Path):
        entry = classify_file(tmp_path / "notes.txt")
        assert entry.kind == "unsupported"

    def test_plain_directory_is_unsupported(self, tmp_path: Path):
        entry = classify_file(tmp_path)
        assert entry.kind == "unsupported"


class TestDiscoverSpectralFiles:
    def _tree(self, tmp_path: Path) -> Path:
        data = tmp_path / "data"
        (data / "nested").mkdir(parents=True)
        (data / "run.mzml").write_text("x")
        (data / "nested" / "run2.mgf").write_text("x")
        (data / "library.msp").write_text("x")
        (data / "vendor.raw").write_text("x")
        (data / "notes.txt").write_text("x")
        (data / ".hidden.mgf").write_text("x")
        store = data / "store.zarr"
        store.mkdir()
        (store / ".zgroup").write_text("{}")
        return data

    def test_discovers_and_classifies(self, tmp_path: Path):
        data = self._tree(tmp_path)
        entries = discover_spectral_files(data)
        paths = {entry.path.name for entry in entries}
        assert "run.mzml" in paths
        assert "run2.mgf" in paths
        assert "library.msp" in paths
        assert "vendor.raw" in paths
        assert "store.zarr" in paths
        assert "notes.txt" not in paths
        assert ".hidden.mgf" not in paths

    def test_hidden_included_when_requested(self, tmp_path: Path):
        data = self._tree(tmp_path)
        entries = discover_spectral_files(data, include_hidden=True)
        assert any(entry.path.name == ".hidden.mgf" for entry in entries)

    def test_max_depth(self, tmp_path: Path):
        data = self._tree(tmp_path)
        entries = discover_spectral_files(data, max_depth=0)
        # Depth 0 = the root only; nested/run2.mgf is at depth 1.
        assert not any(entry.path.name == "run2.mgf" for entry in entries)
        assert any(entry.path.name == "run.mzml" for entry in entries)

    def test_missing_directory(self, tmp_path: Path):
        assert discover_spectral_files(tmp_path / "nope") == []

    def test_directories_first_in_sort_order(self, tmp_path: Path):
        data = self._tree(tmp_path)
        entries = discover_spectral_files(data)
        kinds = [entry.kind for entry in entries]
        assert kinds[0] == "database"  # zarr store sorts before files
        assert kinds == sorted(
            kinds,
            key={
                "database": 0,
                "library": 1,
                "query": 2,
                "vendor": 3,
                "unsupported": 4,
            }.__getitem__,
        )


class TestCopyIntoWorkspace:
    def test_basic_copy(self, tmp_path: Path):
        source = tmp_path / "run.mgf"
        source.write_text("spectrum")
        workspace = tmp_path / "workspace"
        destination = copy_into_workspace(source, workspace)
        assert destination == workspace / "run.mgf"
        assert destination.read_text() == "spectrum"

    def test_collision_safe(self, tmp_path: Path):
        source = tmp_path / "run.mgf"
        source.write_text("new")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "run.mgf").write_text("old")
        first = copy_into_workspace(source, workspace)
        second = copy_into_workspace(source, workspace)
        assert first.name == "run_2.mgf"
        assert second.name == "run_3.mgf"

    def test_missing_source_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            copy_into_workspace(tmp_path / "nope.mgf", tmp_path / "workspace")


class TestHumanSize:
    def test_units(self):
        assert human_size(512) == "512 B"
        assert human_size(1536) == "1.5 KB"
        assert human_size(5 * 1024 * 1024) == "5.0 MB"
        assert human_size(None) == "n/a"


class TestGuessBackend:
    def test_text(self, tmp_path: Path):
        assert guess_backend(tmp_path / "lib.msp") == "text"

    def test_sqlite(self, tmp_path: Path):
        assert guess_backend(tmp_path / "lib.db") == "sqlite"

    def test_zarr_directory(self, tmp_path: Path):
        store = tmp_path / "store.zarr"
        store.mkdir()
        (store / ".zgroup").write_text("{}")
        assert guess_backend(store) == "zarr"

    def test_hybrid(self, tmp_path: Path):
        db = tmp_path / "lib.db"
        db.write_text("")
        (tmp_path / "lib.zarr").mkdir()
        assert guess_backend(db) == "hybrid"


class TestVendorExtensionSync:
    """The console's vendor set must stay in sync with the core loader."""

    def test_matches_core_proprietary_formats(self):
        from MassFlow.io import PROPRIETARY_FORMATS

        from MassFlow.tui.files import VENDOR_EXTENSIONS

        assert VENDOR_EXTENSIONS == PROPRIETARY_FORMATS
