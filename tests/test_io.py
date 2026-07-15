"""
Tests for MassFlow I/O module.
"""

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


def test_load_spectra_mgf(mock_spectrum, tmp_path):
    with patch("MassFlow.io.load_from_mgf") as mock_load:
        # Mock load_from_mgf to return an iterator of spectra
        mock_load.return_value = iter([mock_spectrum])

        p = tmp_path / "test.mgf"
        p.touch()

        # io.load_spectra returns a generator yielding processed spectra
        # load_spectra calls _apply_filters which calls _sanitize_metadata
        # _sanitize_metadata expects a Spectrum object (has .get())
        result = list(io.load_spectra(p, "mgf"))

        assert len(result) == 1
        assert result[0].get("id") == "spec1"


def test_load_spectra_msp(mock_spectrum, tmp_path):
    with patch("MassFlow.io.load_from_msp") as mock_load:
        mock_load.return_value = iter([mock_spectrum])
        p = tmp_path / "test.msp"
        p.touch()
        result = list(io.load_spectra(p, "msp"))
        assert len(result) == 1
        assert result[0].get("id") == "spec1"


def test_load_spectra_mzml(mock_spectrum, tmp_path):
    with patch("MassFlow.io.load_from_mzml") as mock_load:
        mock_load.return_value = iter([mock_spectrum])
        p = tmp_path / "test.mzml"
        p.touch()
        result = list(io.load_spectra(p, "mzml"))
        assert len(result) == 1
        assert result[0].get("id") == "spec1"


def test_load_spectra_mzxml(mock_spectrum, tmp_path):
    with patch("MassFlow.io.load_from_mzxml") as mock_load:
        mock_load.return_value = iter([mock_spectrum])
        p = tmp_path / "test.mzxml"
        p.touch()
        result = list(io.load_spectra(p, "mzxml"))
        assert len(result) == 1
        assert result[0].get("id") == "spec1"


def test_load_spectra_db(mock_spectrum, tmp_path):
    # io.load_spectra now uses create_spectral_store factory.
    # Patch the factory to return a mock store.
    with patch("MassFlow.storage.create_spectral_store") as mock_factory:
        mock_store = mock_factory.return_value
        mock_store.get_spectra.return_value = iter([mock_spectrum])

        p = tmp_path / "test.db"
        p.touch()

        result = list(io.load_spectra(p, "db"))

        assert len(result) == 1
        assert result[0].get("id") == "spec1"
        mock_factory.assert_called_with(p, backend="sqlite")


def test_load_spectra_proprietary_error(tmp_path):
    # Test that it raises error if vendor format is passed
    p = tmp_path / "test.raw"
    p.touch()
    with pytest.raises(
        io.UnsupportedVendorFormatError,
        match="MassFlow requires open data formats. Please convert vendor files to .mzML or .mgf using ProteoWizard or MS-DIAL prior to pipeline ingestion.",
    ):
        list(io.load_spectra(p))


def test_load_spectra_unsupported_format(tmp_path):
    p = tmp_path / "test.xyz"
    p.touch()
    with pytest.raises(ValueError, match="is not supported"):
        list(io.load_spectra(p, "xyz"))


@pytest.mark.parametrize(
    ("file_name", "loader_name"),
    [
        ("bad.mgf", "load_from_mgf"),
        ("bad.msp", "load_from_msp"),
        ("bad.mzml", "load_from_mzml"),
        ("bad.mzxml", "load_from_mzxml"),
    ],
)
def test_load_spectra_propagates_malformed_loader_errors(
    file_name, loader_name, tmp_path
):
    p = tmp_path / file_name
    p.touch()
    with patch(
        f"MassFlow.io.{loader_name}",
        side_effect=ValueError("Malformed spectral file"),
    ):
        with pytest.raises(ValueError, match="Malformed spectral file"):
            list(io.load_spectra(p))


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


def test_save_spectra_to_mgf(tmp_path, mock_spectrum):
    out_path = tmp_path / "test.mgf"

    with patch("matchms.exporting.save_as_mgf") as mock_save:
        io.save_spectra_to_mgf([mock_spectrum], out_path)

        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert args[0] == [mock_spectrum]
        assert args[1] == str(out_path)


def test_save_match_results_to_mztab(tmp_path):
    results = [{"query_id": "q1", "reference_id": "r1", "score": 0.95}]
    out_path = tmp_path / "results.mztab"

    io.save_match_results_to_mztab(results, out_path)

    assert out_path.exists()
    with open(out_path, "r") as f:
        content = f.read()
        assert "MTD\tmzTab-version\t2.0.0-M" in content
        assert "SMH\tquery_id\treference_id\tscore\tAnnotation_Status" in content
        assert "SML\tq1\tr1\t0.95\tMatched" in content
