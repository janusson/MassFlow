"""
Tests for MassFlow io module.
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import io


@pytest.fixture
def mock_spectra():
    return [
        Spectrum(
            mz=np.array([100.0, 200.0], dtype="float"),
            intensities=np.array([0.5, 1.0], dtype="float"),
            metadata={"name": "C1", "spectrum_id": "1"},
        )
    ]


# --- load_spectra tests ---


def test_load_spectra_mgf():
    with patch("MassFlow.io.load_from_mgf") as mock_load:
        mock_load.return_value = iter(["spec"])
        result = io.load_spectra(Path("test.mgf"), "mgf")
        assert list(result) == ["spec"]
        mock_load.assert_called_once_with("test.mgf")


def test_load_spectra_msp():
    with patch("MassFlow.io.load_from_msp") as mock_load:
        mock_load.return_value = iter(["spec"])
        result = io.load_spectra(Path("test.msp"), "msp")
        assert list(result) == ["spec"]
        mock_load.assert_called_once_with("test.msp")


def test_load_spectra_mzml():
    with patch("MassFlow.io.load_from_mzml") as mock_load:
        mock_load.return_value = iter(["spec"])
        result = io.load_spectra(Path("test.mzml"), "mzml")
        assert list(result) == ["spec"]
        mock_load.assert_called_once_with("test.mzml")


def test_load_spectra_mzxml():
    with patch("MassFlow.io.load_from_mzxml") as mock_load:
        mock_load.return_value = iter(["spec"])
        result = io.load_spectra(Path("test.mzxml"), "mzxml")
        assert list(result) == ["spec"]
        mock_load.assert_called_once_with("test.mzxml")


def test_load_spectra_db():
    with patch("MassFlow.io.SpectralDatabase") as MockDB:
        mock_instance = MockDB.return_value
        mock_instance.get_spectra.return_value = iter(["spec"])

        result = io.load_spectra(Path("test.db"), "db")
        assert list(result) == ["spec"]
        MockDB.assert_called_once_with(Path("test.db"))
        mock_instance.get_spectra.assert_called_once()


def test_load_spectra_raw_error():
    with pytest.raises(
        ValueError, match="Direct loading of raw files is not supported"
    ):
        io.load_spectra(Path("test.raw"), "raw")


def test_load_spectra_invalid_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        io.load_spectra(Path("test.txt"), "txt")


# --- save_match_results tests ---


def test_save_match_results(tmp_path):
    results = [{"id": "1", "score": 0.9}]
    out_path = tmp_path / "results.csv"

    io.save_match_results(results, out_path)

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "id,score" in content
    assert "1,0.9" in content


def test_save_match_results_excel(tmp_path):
    import pandas as pd

    results = [{"id": 1, "score": 0.9}]
    out_path = tmp_path / "results.xlsx"

    io.save_match_results(results, out_path)
    assert out_path.exists()

    df = pd.read_excel(out_path)
    assert len(df) == 1
    assert df.iloc[0]["id"] == 1
    assert df.iloc[0]["score"] == 0.9


def test_save_match_results_empty(tmp_path):
    out_path = tmp_path / "results.csv"
    io.save_match_results([], out_path)
    assert not out_path.exists()


def test_save_match_results_io_error(tmp_path):
    results = [{"id": "1"}]
    # directory that can't be written to or invalid path logic is hard to force with tmp_path fixture
    # without permissions manipulation, so we mock open.

    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = IOError("Boom")
        with pytest.raises(IOError):
            io.save_match_results(results, Path("dummy.csv"))


# --- save_spectra_* tests ---


def test_save_spectra_to_mgf(mock_spectra):
    with patch("MassFlow.io.save_as_mgf") as mock_save:
        path = Path("/out/test.mgf")
        with patch.object(Path, "mkdir"):
            io.save_spectra_to_mgf(mock_spectra, path)
        mock_save.assert_called_once_with(mock_spectra, str(path))


def test_save_spectra_to_msp(mock_spectra):
    with patch("MassFlow.io.save_as_msp") as mock_save:
        path = Path("/out/test.msp")
        with patch.object(Path, "mkdir"):
            io.save_spectra_to_msp(mock_spectra, path)
        mock_save.assert_called_once_with(mock_spectra, str(path))


def test_save_spectra_to_json(mock_spectra):
    with patch("MassFlow.io.save_as_json") as mock_save:
        path = Path("/out/test.json")
        with patch.object(Path, "mkdir"):
            io.save_spectra_to_json(mock_spectra, path)
        mock_save.assert_called_once_with(mock_spectra, str(path))


def test_save_spectra_to_pickle(mock_spectra):
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.dump") as mock_dump:
            path = Path("/out/test.pickle")
            with patch.object(Path, "mkdir"):
                io.save_spectra_to_pickle(mock_spectra, path)

            mock_file.assert_called_with(path, "wb")
            mock_dump.assert_called_once()
            args, _ = mock_dump.call_args
            # Verify list conversion happened if iterable passed
            assert args[0] == mock_spectra


def test_save_spectra_to_mzml(mock_spectra, tmp_path):
    # Ensure spectrum has an ID for this test (matchms Spectrum is immutable, so replace in list)
    mock_spectra[0] = mock_spectra[0].set("id", "1")
    mock_spectra[0] = mock_spectra[0].set("precursor_mz", 150.0)
    mock_spectra[0] = mock_spectra[0].set("charge", 1)
    mock_spectra[0] = mock_spectra[0].set("precursor_intensity", 1000.0)

    path = tmp_path / "test.mzml"
    io.save_spectra_to_mzml(mock_spectra, path)

    assert path.exists()
    content = path.read_text(encoding="utf-8")

    assert "mzML" in content
    assert 'id="1"' in content
    assert 'defaultArrayLength="2"' in content
    assert "binaryDataArray" in content
    # Check for precursor info presence
    assert 'name="selected ion m/z"' in content
    assert 'value="150.0"' in content
    assert 'name="charge state"' in content
    assert 'value="1"' in content
    assert 'name="peak intensity"' in content
    assert 'value="1000.0"' in content


def test_save_spectra_to_mzml_empty_peaks(tmp_path):
    empty_spec = Spectrum(
        mz=np.array([], dtype="float"),
        intensities=np.array([], dtype="float"),
        metadata={"id": "empty"},
    )
    path = tmp_path / "empty.mzml"
    io.save_spectra_to_mzml([empty_spec], path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert 'encodedLength="0"' in content


# --- list_files_by_extension tests ---


def test_list_files_by_extension(tmp_path):
    (tmp_path / "test1.mgf").touch()
    (tmp_path / "test2.mgf").touch()
    (tmp_path / "other.txt").touch()

    files = io.list_files_by_extension(tmp_path, "mgf")
    assert len(files) == 2
    names = [f.name for f in files]
    assert "test1.mgf" in names
    assert "test2.mgf" in names


def test_list_files_by_extension_not_dir():
    files = io.list_files_by_extension(Path("nonexistent"), "mgf")
    assert files == []
