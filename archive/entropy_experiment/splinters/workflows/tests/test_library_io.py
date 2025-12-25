import csv
import json

import numpy as np
import pytest

pytest.importorskip("matchms", reason="Library IO tests require matchms.")

from matchms import Spectrum

from yogimass.library_io import (
    assert_schema_compatible,
    export_library_summary_csv,
    read_schema_version,
)
from yogimass.similarity.library import LocalSpectralLibrary, SCHEMA_VERSION


def _spectrum(mz, intensities, **metadata):
    return Spectrum(
        mz=np.asarray(mz, dtype="float64"),
        intensities=np.asarray(intensities, dtype="float64"),
        metadata=metadata,
    )


def test_schema_version_written_and_read(tmp_path):
    lib_path = tmp_path / "lib.json"
    lib = LocalSpectralLibrary(lib_path)
    lib.add_spectrum(_spectrum([100.0], [1.0], name="s1"), overwrite=True)
    version = read_schema_version(lib_path)
    assert version == SCHEMA_VERSION
    assert_schema_compatible(lib_path, minimum_version="1.0")  # should not raise


def test_schema_version_incompatible_raises(tmp_path):
    lib_path = tmp_path / "old_lib.json"
    lib_path.write_text(
        json.dumps({"schema_version": "0.5", "entries": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        assert_schema_compatible(lib_path, minimum_version="0.9")


def test_export_library_summary_csv(tmp_path):
    lib_path = tmp_path / "lib.json"
    lib = LocalSpectralLibrary(lib_path)
    lib.add_spectrum(
        _spectrum([100.0], [1.0], name="s1", retention_time=5.0), overwrite=True
    )
    lib.add_spectrum(_spectrum([200.0], [2.0], compound_name="s2"), overwrite=True)

    csv_path = export_library_summary_csv(lib_path, tmp_path / "summary.csv")
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) == 2
    ids = {row["id"] for row in rows}
    assert ids == set(entry.identifier for entry in lib.iter_entries())
    names = {row["name"] for row in rows}
    assert "s1" in names and "s2" in names
