"""Build-only CLI integration tests using real processing and storage."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from MassFlow.cli import app
from MassFlow.config import MassFlowConfig
from MassFlow.database import SpectralDatabase
from MassFlow.library import LibrarySpec, open_library


def _fixture(tmp_path: Path, suffix: str = "msp", backend: str = "sqlite") -> Path:
    """Write a library and config whose query path deliberately does not exist."""
    library = tmp_path / f"reference.{suffix}"
    if suffix == "msp":
        library.write_text(
            "Name: caffeine\nPrecursorMZ: 195.087652\n"
            "Formula: C8H10N4O2\nPrecursor_type: [M+H]+\n"
            "Ion_mode: positive\nNum Peaks: 3\n"
            "50.123456789 10\n100.123456789 20\n150.123456789 40\n\n"
        )
    else:
        library.write_text(
            "BEGIN IONS\nTITLE=caffeine\nPEPMASS=195.087652\nCHARGE=1+\n"
            "50.123456789 10\n100.123456789 20\n150.123456789 40\nEND IONS\n"
        )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"output_directory": "results"},
                "input": {
                    "input_path": "absent.mgf",
                    "library_path": library.name,
                    "storage_backend": backend,
                },
                "processing": {
                    "min_peaks": 1,
                    "filter_by_intensity": False,
                    "normalize_intensity": False,
                    "reduce_to_top_n_peaks": False,
                },
            }
        )
    )
    return config


def _invoke(config: Path, columns: int | None = None) -> Result:
    """Invoke the build-only mode through Typer."""
    environment = {"COLUMNS": str(columns)} if columns is not None else None
    return CliRunner().invoke(
        app,
        ["annotate", "--config", str(config), "--build-library-store"],
        env=environment,
    )


@pytest.mark.parametrize("backend", ["sqlite", "hybrid", "zarr"])
@pytest.mark.parametrize("suffix", ["msp", "mgf"])
def test_build_only_roundtrip_and_reuse(
    tmp_path: Path, backend: str, suffix: str
) -> None:
    config = _fixture(tmp_path, suffix, backend)
    result = _invoke(config)
    assert result.exit_code == 0, result.output
    assert "Library store ready" in result.output
    extension = "zarr" if backend == "zarr" else "db"
    store_path = tmp_path / "results" / f"reference_library.{extension}"
    cfg = MassFlowConfig.from_yaml(config)
    store = open_library(
        LibrarySpec(path=store_path, kind="store", storage_backend=backend),
        cfg.processing,
    )
    try:
        spectra = list(store.get_spectra())
        assert len(spectra) == 1
        np.testing.assert_array_equal(
            spectra[0].peaks.mz, [50.123456789, 100.123456789, 150.123456789]
        )
        np.testing.assert_array_equal(spectra[0].peaks.intensities, [10.0, 20.0, 40.0])
        assert spectra[0].peaks.mz.dtype == np.float64
        assert spectra[0].get("precursor_mz") == 195.087652
        if backend != "zarr":
            assert isinstance(store, SpectralDatabase)
            builds = store.get_library_builds()
            assert len(builds) == 1
            assert (
                builds[0]["source_sha256"]
                == hashlib.sha256(
                    (tmp_path / f"reference.{suffix}").read_bytes()
                ).hexdigest()
            )
    finally:
        store.close()
    metadata = store_path.with_suffix(store_path.suffix + ".meta.json")
    before = metadata.stat().st_mtime_ns
    assert _invoke(config).exit_code == 0
    assert metadata.stat().st_mtime_ns == before
    assert not list((tmp_path / "results").glob("*_results.*"))
    assert not list((tmp_path / "results").glob("*.report.yaml"))


@pytest.mark.parametrize(
    "failure", ["missing", "vendor", "unknown", "physics", "empty", "unconfigured"]
)
def test_build_only_fails_closed(tmp_path: Path, failure: str) -> None:
    config = _fixture(tmp_path)
    library = tmp_path / "reference.msp"
    data = yaml.safe_load(config.read_text())
    if failure == "missing":
        library.unlink()
    elif failure in {"vendor", "unknown"}:
        replacement = library.with_suffix(".raw" if failure == "vendor" else ".txt")
        library.rename(replacement)
        data["input"]["library_path"] = replacement.name
    elif failure == "physics":
        # Caffeine [M+H]+ is ~195.087652 Da: this is far outside 5 ppm.
        library.write_text(library.read_text().replace("195.087652", "210.0"))
    elif failure == "empty":
        library.write_text("")
    else:
        del data["input"]["library_path"]
    config.write_text(yaml.safe_dump(data))
    result = _invoke(config)
    assert result.exit_code == 1, result.output
    assert "Library store ready" not in result.output
    if failure != "empty":
        assert not (tmp_path / "results" / "reference_library.db").exists()
    if failure == "physics":
        assert "5 ppm" in result.output


@pytest.mark.parametrize("backend", ["sqlite", "hybrid", "zarr"])
def test_build_only_accepts_existing_store(tmp_path: Path, backend: str) -> None:
    config = _fixture(tmp_path, backend=backend)
    assert _invoke(config).exit_code == 0
    extension = "zarr" if backend == "zarr" else "db"
    data = yaml.safe_load(config.read_text())
    data["input"]["library_path"] = f"results/reference_library.{extension}"
    data["project"]["output_directory"] = "unused_output"
    # Preparing a library does not require this optional scoring engine.
    data["similarity"] = {"algorithm": "spec2vec"}
    config.write_text(yaml.safe_dump(data))
    result = _invoke(config, columns=80)
    assert result.exit_code == 0, result.output
    assert re.search(rf"\({re.escape(backend)},\s+1\s+spectra\)", result.output)
    assert not (tmp_path / "unused_output").exists()


def test_build_only_rejects_empty_existing_store(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    store_path = tmp_path / "empty.db"
    store = SpectralDatabase(store_path)
    store.close()
    data = yaml.safe_load(config.read_text())
    data["input"]["library_path"] = store_path.name
    config.write_text(yaml.safe_dump(data))
    result = _invoke(config)
    assert result.exit_code == 1
    assert "No valid spectra" in result.output
