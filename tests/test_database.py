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


def test_operations_on_closed_db(temp_db, sample_spectrum):
    db = SpectralDatabase(temp_db)
    db.close()

    with pytest.raises(ConnectionError, match="Database not connected"):
        db.add_spectra([sample_spectrum])

    with pytest.raises(ConnectionError, match="Database not connected"):
        list(db.get_spectra())

    with pytest.raises(ConnectionError, match="Database not connected"):
        db.get_total_spectra_count()

    with pytest.raises(ConnectionError, match="Database not connected"):
        db.get_category_counts()

    with pytest.raises(ConnectionError, match="Database not connected"):
        db.get_precursor_mz_range()


def test_decode_legacy_json_peaks_variations():
    """Test the decoding of various legacy JSON peak formats."""
    import json

    import numpy as np

    from MassFlow.database import _decode_legacy_peaks_payload

    # Test dictionary with "peaks" key
    payload_1 = {"peaks": [[100.0, 50.0], [200.0, 100.0]]}
    mz, intensity = _decode_legacy_peaks_payload(json.dumps(payload_1))
    assert np.array_equal(mz, np.array([100.0, 200.0]))

    # Test dictionary with "mz" and "intensity" lists
    payload_2 = {"mz": [100.0, 200.0], "intensity": [50.0, 100.0]}
    mz, intensity = _decode_legacy_peaks_payload(json.dumps(payload_2))
    assert np.array_equal(mz, np.array([100.0, 200.0]))

    # Test dictionary with "mz_array" and "intensity_array" lists
    payload_3 = {"mz_array": [100.0, 200.0], "intensity_array": [50.0, 100.0]}
    mz, intensity = _decode_legacy_peaks_payload(json.dumps(payload_3))
    assert np.array_equal(mz, np.array([100.0, 200.0]))

    # Test list of dicts
    payload_4 = [{"mz": 100.0, "intensity": 50.0}, {"mz": 200.0, "intensity": 100.0}]
    mz, intensity = _decode_legacy_peaks_payload(json.dumps(payload_4))
    assert np.array_equal(mz, np.array([100.0, 200.0]))

    # Test list of tuples (like [[mz, int], [mz, int]])
    payload_5 = [[100.0, 50.0], [200.0, 100.0]]
    mz, intensity = _decode_legacy_peaks_payload(json.dumps(payload_5))
    assert np.array_equal(mz, np.array([100.0, 200.0]))


def test_decode_legacy_peaks_payload_edge_cases():
    from MassFlow.database import _decode_legacy_peaks_payload
    import pytest

    # Test bytes not utf-8
    with pytest.raises(
        ValueError, match="Legacy peaks payload bytes are not valid UTF-8"
    ):
        _decode_legacy_peaks_payload(b"\xff\xfe\x00")

    # Test empty string
    mz, i = _decode_legacy_peaks_payload("")
    assert len(mz) == 0
    assert len(i) == 0

    # Test invalid json and literal
    with pytest.raises(
        ValueError,
        match="Legacy peaks payload is neither valid JSON nor a Python literal",
    ):
        _decode_legacy_peaks_payload("{invalid")

    # Test dict with mz_array and intensity_array
    mz, i = _decode_legacy_peaks_payload({"mz_array": [1.0], "intensity_array": [2.0]})
    assert len(mz) == 1
    assert len(i) == 1

    # Test dict with mismatched mz_array
    with pytest.raises(
        ValueError, match="Legacy peaks dictionary has mismatched array shapes"
    ):
        _decode_legacy_peaks_payload({"mz_array": [1.0, 2.0], "intensity_array": [2.0]})

    # Test empty list
    mz, i = _decode_legacy_peaks_payload([])
    assert len(mz) == 0

    # Test list of dicts missing keys
    with pytest.raises(
        ValueError, match="Legacy peaks list-of-dicts payload must contain 'mz'"
    ):
        _decode_legacy_peaks_payload([{"mz": 1.0}])

    # Test tuple with mismatched shapes
    with pytest.raises(
        ValueError, match="Legacy peaks tuple has mismatched array shapes"
    ):
        _decode_legacy_peaks_payload(([1.0], [1.0, 2.0]))

    # Test unsupported format
    with pytest.raises(
        ValueError, match="Legacy peaks payload format is not supported"
    ):
        _decode_legacy_peaks_payload(123)
