import shutil
from pathlib import Path

import pytest

pytest.importorskip("matchms", reason="Yogimass pipeline tests require matchms.")

from yogimass import pipeline


def test_clean_mgf_library_returns_cleaned_spectra(tmp_path):
    fixture = Path(__file__).parent / "data" / "simple.mgf"
    cleaned = pipeline.clean_mgf_library(fixture)
    assert len(cleaned) == 1
    spectrum = cleaned[0]
    assert spectrum.metadata.get("name")
    assert len(spectrum.peaks.mz) > 0


def test_clean_mgf_library_missing_file(tmp_path):
    missing = tmp_path / "missing.mgf"
    with pytest.raises(FileNotFoundError):
        pipeline.clean_mgf_library(missing)


def test_clean_msp_library_returns_cleaned_spectra():
    fixture = Path(__file__).parent / "data" / "simple.msp"
    cleaned = pipeline.clean_msp_library(fixture)
    assert len(cleaned) == 2
    assert all(spectrum.metadata.get("name") for spectrum in cleaned)


def test_batch_clean_mgf_libraries_exports_multiple_formats(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixture = Path(__file__).parent / "data" / "simple.mgf"
    target = data_dir / "example.mgf"
    shutil.copy(fixture, target)
    output_dir = tmp_path / "out"

    exported = pipeline.batch_clean_mgf_libraries(
        data_dir,
        output_dir,
        export_formats=("mgf", "json"),
    )

    expected = {
        output_dir / "example_cleaned.mgf",
        output_dir / "example_cleaned.json",
    }
    assert set(exported) == expected
    for path in expected:
        assert path.exists()


def test_batch_clean_libraries_rejects_unknown_format(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixture = Path(__file__).parent / "data" / "simple.mgf"
    shutil.copy(fixture, data_dir / "example.mgf")
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError):
        pipeline.batch_clean_mgf_libraries(
            data_dir,
            output_dir,
            export_formats=("unknown",),
        )


def test_batch_clean_msp_libraries(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixture = Path(__file__).parent / "data" / "simple.msp"
    shutil.copy(fixture, data_dir / "example.msp")
    output_dir = tmp_path / "out"

    exported = pipeline.batch_clean_msp_libraries(
        data_dir,
        output_dir,
        export_formats=("msp",),
    )

    expected = {output_dir / "example_cleaned.msp"}
    assert set(exported) == expected
    for path in expected:
        assert path.exists()
