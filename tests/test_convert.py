import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from MassFlow.convert import (
    ConversionError,
    MSConvertNotFoundError,
    convert_directory,
    get_vendor_files,
    run_conversion,
)


def test_get_vendor_files(tmp_path):
    (tmp_path / "test1.raw").touch()
    (tmp_path / "test2.raw").touch()
    (tmp_path / "not_vendor.txt").touch()

    d_dir = tmp_path / "test3.d"
    d_dir.mkdir()

    files = get_vendor_files(tmp_path)

    assert len(files) == 3
    names = [f.name for f in files]
    assert "test1.raw" in names
    assert "test2.raw" in names
    assert "test3.d" in names


@patch("MassFlow.convert.check_msconvert_installed", return_value=False)
def test_run_conversion_msconvert_not_found(mock_check, tmp_path):
    with pytest.raises(MSConvertNotFoundError):
        run_conversion(Path("test.raw"), tmp_path)


@patch("MassFlow.convert.check_msconvert_installed", return_value=True)
@patch("subprocess.run")
def test_run_conversion_success(mock_run, mock_check, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)

    in_file = tmp_path / "test.raw"
    in_file.touch()
    out_dir = tmp_path / "out"

    run_conversion(in_file, out_dir)

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "msconvert"
    assert args[1] == str(in_file)
    assert args[2] == "-o"
    assert args[3] == str(out_dir)


@patch("MassFlow.convert.check_msconvert_installed", return_value=True)
@patch("subprocess.run")
def test_run_conversion_failure(mock_run, mock_check, tmp_path):
    mock_run.side_effect = subprocess.CalledProcessError(
        1, "msconvert", stderr="Error details"
    )

    in_file = tmp_path / "test.raw"
    in_file.touch()
    out_dir = tmp_path / "out"

    with pytest.raises(ConversionError):
        run_conversion(in_file, out_dir)


@patch("MassFlow.convert.check_msconvert_installed", return_value=True)
@patch("MassFlow.convert.run_conversion")
def test_convert_directory(mock_run_conv, mock_check, tmp_path):
    (tmp_path / "test1.raw").touch()
    (tmp_path / "test2.raw").touch()

    out_dir = tmp_path / "out"

    count = convert_directory(tmp_path, out_dir)

    assert count == 2
    assert mock_run_conv.call_count == 2


def test_get_vendor_files_missing_dir(tmp_path):
    from MassFlow.convert import get_vendor_files

    assert get_vendor_files(tmp_path / "missing") == []


@patch("MassFlow.convert.check_msconvert_installed", return_value=True)
@patch("MassFlow.convert.run_conversion", side_effect=ConversionError("Failed"))
def test_convert_directory_skip_on_error(mock_run, mock_check, tmp_path):
    from MassFlow.convert import convert_directory

    (tmp_path / "test.raw").touch()

    count = convert_directory(tmp_path, tmp_path / "out")
    assert count == 0
