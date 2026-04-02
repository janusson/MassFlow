"""
Tests for MassFlow I/O module.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import io


@pytest.fixture
def mock_spectrum():
    return Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([0.5, 1.0], dtype="float"),
        metadata={"precursor_mz": 150.0, "id": "spec1"},
    )


def test_load_spectra_mgf(mock_spectrum):
    with patch("MassFlow.io.load_from_mgf") as mock_load:
        # Mock load_from_mgf to return an iterator of spectra
        mock_load.return_value = iter([mock_spectrum])

        # io.load_spectra returns a generator yielding processed spectra
        # load_spectra calls _apply_filters which calls _sanitize_metadata
        # _sanitize_metadata expects a Spectrum object (has .get())
        result = list(io.load_spectra(Path("test.mgf"), "mgf"))

        assert len(result) == 1
        assert result[0].get("id") == "spec1"


def test_load_spectra_msp(mock_spectrum):
    with patch("MassFlow.io.load_from_msp") as mock_load:
        mock_load.return_value = iter([mock_spectrum])
        result = list(io.load_spectra(Path("test.msp"), "msp"))
        assert len(result) == 1
        assert result[0].get("id") == "spec1"


def test_load_spectra_mzml(mock_spectrum):
    with patch("MassFlow.io.load_from_mzml") as mock_load:
        mock_load.return_value = iter([mock_spectrum])
        result = list(io.load_spectra(Path("test.mzml"), "mzml"))
        assert len(result) == 1
        assert result[0].get("id") == "spec1"


def test_load_spectra_mzxml(mock_spectrum):
    with patch("MassFlow.io.load_from_mzxml") as mock_load:
        mock_load.return_value = iter([mock_spectrum])
        result = list(io.load_spectra(Path("test.mzxml"), "mzxml"))
        assert len(result) == 1
        assert result[0].get("id") == "spec1"


def test_load_spectra_db(mock_spectrum):
    # SpectralDatabase is imported inside the function, so we patch where it is defined
    with patch("MassFlow.database.SpectralDatabase") as MockDB:
        mock_instance = MockDB.return_value
        mock_instance.get_spectra.return_value = iter([mock_spectrum])

        result = list(io.load_spectra(Path("test.db"), "db"))

        assert len(result) == 1
        assert result[0].get("id") == "spec1"
        MockDB.assert_called_with(Path("test.db"))


def test_load_spectra_proprietary_error():
    # Test that it raises error if vendor format is passed
    with pytest.raises(
        io.UnsupportedVendorFormatError,
        match="MassFlow requires open data formats. Please convert vendor files to .mzML or .mgf using ProteoWizard or MS-DIAL prior to pipeline ingestion.",
    ):
        list(io.load_spectra(Path("test.raw")))


def test_load_spectra_unsupported_format():
    with pytest.raises(ValueError, match="is not supported"):
        list(io.load_spectra(Path("test.xyz"), "xyz"))


@pytest.mark.parametrize(
    ("file_name", "loader_name"),
    [
        ("bad.mgf", "load_from_mgf"),
        ("bad.msp", "load_from_msp"),
        ("bad.mzml", "load_from_mzml"),
        ("bad.mzxml", "load_from_mzxml"),
    ],
)
def test_load_spectra_propagates_malformed_loader_errors(file_name, loader_name):
    with patch(
        f"MassFlow.io.{loader_name}",
        side_effect=ValueError("Malformed spectral file"),
    ):
        with pytest.raises(ValueError, match="Malformed spectral file"):
            list(io.load_spectra(Path(file_name)))


def test_save_match_results(tmp_path):
    results = [{"query_id": "q1", "reference_id": "r1", "score": 0.95}]
    out_path = tmp_path / "results.csv"

    io.save_match_results(results, out_path)

    assert out_path.exists()
    with open(out_path, "r") as f:
        content = f.read()
        assert "query_id,reference_id,score" in content
        assert "q1,r1,0.95" in content


def test_save_spectra_to_msp(tmp_path, mock_spectrum):
    out_path = tmp_path / "test.msp"

    with patch("matchms.exporting.save_as_msp") as mock_save:
        io.save_spectra_to_msp([mock_spectrum], out_path)

        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert args[0] == [mock_spectrum]
        assert args[1] == str(out_path)
