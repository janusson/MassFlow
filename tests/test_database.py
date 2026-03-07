import json
import sqlite3

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.database import SpectralDatabase


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database file."""
    return tmp_path / "test_spectra.db"


@pytest.fixture
def sample_spectrum():
    mz = np.array([100.0, 200.0, 300.0], dtype="float")
    intensities = np.array([0.1, 0.5, 1.0], dtype="float")
    metadata = {
        "id": "test_id_1",
        "compound_name": "Test Compound",
        "precursor_mz": 200.0,
        "charge": 1,
        "ionmode": "positive",
        "adduct": "[M+H]+",
        "extra_data": {"foo": "bar"},
    }
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)


def test_init_creates_tables(temp_db):
    db = SpectralDatabase(temp_db)
    db.close()

    assert temp_db.exists()

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='spectra'"
    )
    table = cursor.fetchone()
    conn.close()

    assert table is not None


def test_add_spectra(temp_db, sample_spectrum):
    db = SpectralDatabase(temp_db)
    count = db.add_spectra([sample_spectrum], category="test_cat")
    db.close()

    assert count == 1

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM spectra")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    # Check mapped columns
    # id, original_id, name, precursor_mz, charge, ionmode, adduct, category, metadata, peaks
    assert row[2] == "Test Compound"
    assert row[7] == "test_cat"


def test_get_spectra(temp_db, sample_spectrum):
    db = SpectralDatabase(temp_db)
    db.add_spectra([sample_spectrum], category="test_cat")

    # Retrieve all
    retrieved = list(db.get_spectra())
    assert len(retrieved) == 1
    spec = retrieved[0]

    assert spec.get("compound_name") == "Test Compound"
    assert np.allclose(spec.peaks.mz, sample_spectrum.peaks.mz)

    # Filter by category
    retrieved_cat = list(db.get_spectra(category="test_cat"))
    assert len(retrieved_cat) == 1

    retrieved_wrong = list(db.get_spectra(category="wrong"))
    assert len(retrieved_wrong) == 0

    # Filter by name pattern
    retrieved_name = list(db.get_spectra(name_pattern="%Compound%"))
    assert len(retrieved_name) == 1

    retrieved_wrong_name = list(db.get_spectra(name_pattern="%Missing%"))
    assert len(retrieved_wrong_name) == 0

    db.close()


def test_add_spectra_handling_none(temp_db, sample_spectrum):
    db = SpectralDatabase(temp_db)
    count = db.add_spectra([sample_spectrum, None], category="mixed")
    db.close()
    assert count == 1
