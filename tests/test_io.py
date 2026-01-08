
import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
from matchms import Spectrum
from MassFlow import io

@pytest.fixture
def mock_spectrum_list():
    return [
        Spectrum(
            mz=np.array([100.0, 200.0], dtype="float"),
            intensities=np.array([0.5, 1.0], dtype="float"),
            metadata={"name": "C1", "spectrum_id": "1"}
        ),
        Spectrum(
            mz=np.array([300.0], dtype="float"),
            intensities=np.array([1.0], dtype="float"),
            metadata={"name": "C2", "spectrum_id": "2"}
        )
    ]

def test_list_msp_libraries():
    with patch("glob.glob") as mock_glob:
        mock_glob.return_value = ["/a/b/lib1.msp", "/a/b/lib2.msp"]
        result = io.list_msp_libraries("/a/b")
        assert len(result) == 2
        assert "/a/b/lib1.msp" in result

def test_list_mgf_libraries():
    with patch("glob.glob") as mock_glob:
        mock_glob.return_value = ["/a/b/lib1.mgf"]
        result = io.list_mgf_libraries("/a/b")
        assert len(result) == 1

def test_list_available_libraries():
    mgf = ["/path/test.mgf"]
    msp = ["/path/test.msp"]
    summary = io.list_available_libraries(mgf, msp)
    assert summary["mgf"] == ["test.mgf"]
    assert summary["msp"] == ["test.msp"]

def test_fetch_mgflib_spectrum(mock_spectrum_list):
    with patch("MassFlow.io.load_from_mgf") as mock_load:
        mock_load.return_value = mock_spectrum_list
        
        # Test valid fetch
        xy_data, meta, chem_name = io.fetch_mgflib_spectrum("dummy_path", 0)
        
        assert isinstance(xy_data, pd.DataFrame)
        # matchms might convert name to compound_name
        assert meta.get("name") == "C1" or meta.get("compound_name") == "C1"
        assert chem_name == "C1"

        # Check normalization (max is 1.0, so 0.5 becomes 50%)
        # Note: function logic: normalized_counts * 100
        # 0.5 / 1.0 * 100 = 50.0
        assert pytest.approx(xy_data.loc[xy_data["m/z"] == 100.0, "Abundance (%)"].values[0]) == 50.0

def test_fetch_mgflib_spectrum_out_of_range(mock_spectrum_list):
    with patch("MassFlow.io.load_from_mgf") as mock_load:
        mock_load.return_value = mock_spectrum_list
        with pytest.raises(IndexError):
            io.fetch_mgflib_spectrum("dummy_path", 99)

def test_save_spectra_to_pickle(mock_spectrum_list):
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.dump") as mock_dump:
            io.save_spectra_to_pickle(mock_spectrum_list, "/out", "testlib")
            
            # Check file open
            mock_file.assert_called_with("/out/testlib.pickle", "wb")
            # Check dump called
            mock_dump.assert_called_once()

def test_save_spectra_to_mgf(mock_spectrum_list):
    with patch("MassFlow.io.save_as_mgf") as mock_save:
        io.save_spectra_to_mgf(mock_spectrum_list, "/out", "testlib")
        mock_save.assert_called_once_with(mock_spectrum_list, "/out/testlib.mgf")
