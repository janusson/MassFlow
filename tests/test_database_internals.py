"""
Comprehensive coverage tests for MassFlow database.py:
- Migration helpers: create_legacy_backup_table, create_migrated_spectra_table,
  _fetch_legacy_rows, _validate_decoded_legacy_arrays, migrate_legacy_peaks_database
- Edge cases: triage_flags in _row_to_spectrum, add_spectra batch handling,
  _serialize_peak_arrays, _json_serialize_metadata, has_table, get_spectra_table_columns,
  is_current_spectra_schema
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.database import (
    SpectralDatabase,
    _create_sqlite_connection,
    _current_utc_timestamp_for_names,
    _decode_legacy_peaks_payload,
    _fetch_legacy_rows,
    _json_serialize_metadata,
    _serialize_peak_arrays,
    _validate_decoded_legacy_arrays,
    create_current_spectra_table,
    create_legacy_backup_table,
    create_migrated_spectra_table,
    get_spectra_table_columns,
    has_table,
    is_current_spectra_schema,
    is_legacy_spectra_schema,
    legacy_migration_error_message,
    migrate_legacy_peaks_database,
    migrate_legacy_peaks_to_arrays,
)


# ==============================================================================
# Helper fixtures
# ==============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def sample_spectrum():
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensities=np.array([0.5, 1.0, 0.3], dtype=np.float64),
        metadata={
            "id": "test_1",
            "compound_name": "Test",
            "precursor_mz": 200.0,
            "charge": 1,
            "ionmode": "positive",
            "adduct": "[M+H]+",
        },
    )


# ==============================================================================
# Timestamp helpers
# ==============================================================================


def test_current_utc_timestamp_for_names():
    ts = _current_utc_timestamp_for_names()
    assert len(ts) == 15  # YYYYMMDD_HHMMSS
    assert "_" in ts


# ==============================================================================
# SQLite connection
# ==============================================================================


def test_create_sqlite_connection(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    assert isinstance(conn, sqlite3.Connection)
    conn.close()
    assert temp_db_path.exists()


def test_create_sqlite_connection_creates_parents(tmp_path):
    nested = tmp_path / "sub" / "dir" / "test.db"
    conn = _create_sqlite_connection(nested)
    conn.close()
    assert nested.exists()


# ==============================================================================
# Schema inspection
# ==============================================================================


def test_get_spectra_table_columns_empty(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    cols = get_spectra_table_columns(conn)
    assert cols == []
    conn.close()


def test_get_spectra_table_columns_after_create(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    cols = get_spectra_table_columns(conn)
    assert "id" in cols
    assert "mz_array" in cols
    assert "intensity_array" in cols
    assert "triage_flags" in cols
    conn.close()


def test_has_table_true(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    assert has_table(conn, "spectra") is True
    conn.close()


def test_has_table_false(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    assert has_table(conn, "nonexistent") is False
    conn.close()


def test_is_legacy_spectra_schema_false(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    assert is_legacy_spectra_schema(conn) is False
    conn.close()


def test_is_current_spectra_schema_true(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    assert is_current_spectra_schema(conn) is True
    conn.close()


def test_is_current_spectra_schema_false(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    assert is_current_spectra_schema(conn) is False
    conn.close()


def test_legacy_migration_error_message():
    msg = legacy_migration_error_message("old.db")
    assert "old.db" in msg
    assert "scripts/migrations/0001_peaks_to_arrays.py" in msg


# ==============================================================================
# Migration helpers
# ==============================================================================


def test_create_current_spectra_table_adds_triage_flags(temp_db_path):
    """When creating table, triage_flags column is added if missing."""
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    cols = get_spectra_table_columns(conn)
    assert "triage_flags" in cols
    conn.close()


def test_create_legacy_backup_table(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    backup_name = create_legacy_backup_table(conn)
    assert backup_name.startswith("spectra_backup_")
    assert has_table(conn, backup_name)
    conn.close()


def test_create_legacy_backup_table_no_table(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    with pytest.raises(RuntimeError, match="does not exist"):
        create_legacy_backup_table(conn)
    conn.close()


def test_create_migrated_spectra_table(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_migrated_spectra_table(conn)
    assert has_table(conn, "spectra_migrated")
    # get_spectra_table_columns queries 'spectra' table, not 'spectra_migrated'
    # Use PRAGMA directly to check columns on spectra_migrated
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(spectra_migrated)")
    col_names = [str(row[1]) for row in cursor.fetchall()]
    assert "triage_flags" in col_names
    conn.close()


def test_json_serialize_metadata():
    data = {"id": "spec_1", "extra": [1, 2, 3]}
    result = _json_serialize_metadata(data)
    assert json.loads(result) == data


def test_serialize_peak_arrays():
    mz = np.array([100.0, 200.0], dtype=np.float64)
    intensity = np.array([0.5, 1.0], dtype=np.float64)
    mz_blob, int_blob = _serialize_peak_arrays(mz, intensity)

    # Deserialize and verify
    mz_restored = np.frombuffer(mz_blob, dtype=np.float64)
    int_restored = np.frombuffer(int_blob, dtype=np.float64)
    np.testing.assert_array_almost_equal(mz_restored, mz)
    np.testing.assert_array_almost_equal(int_restored, intensity)


def test_serialize_peak_arrays_cast_to_float64():
    """Arrays that aren't float64 get cast."""
    mz = np.array([100.0, 200.0], dtype=np.float32)
    intensity = np.array([0.5, 1.0], dtype=np.float32)
    mz_blob, int_blob = _serialize_peak_arrays(mz, intensity)
    mz_restored = np.frombuffer(mz_blob, dtype=np.float64)
    assert mz_restored.dtype == np.float64


# ==============================================================================
# _validate_decoded_legacy_arrays
# ==============================================================================


def test_validate_legacy_arrays_cast_dtype():
    mz = np.array([100.0], dtype=np.float32)
    intensity = np.array([1.0], dtype=np.float32)
    _validate_decoded_legacy_arrays(mz, intensity)
    # Should not raise; casts to float64


def test_validate_legacy_arrays_mismatched_shapes():
    mz = np.array([100.0, 200.0], dtype=np.float64)
    intensity = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="mismatched shapes"):
        _validate_decoded_legacy_arrays(mz, intensity)


def test_validate_legacy_arrays_not_1d():
    mz = np.array([[100.0], [200.0]], dtype=np.float64)
    intensity = np.array([[1.0], [2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="one-dimensional"):
        _validate_decoded_legacy_arrays(mz, intensity)


def test_validate_legacy_arrays_out_of_order():
    mz = np.array([200.0, 100.0], dtype=np.float64)
    intensity = np.array([1.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="out of order"):
        _validate_decoded_legacy_arrays(mz, intensity)


# ==============================================================================
# _decode_legacy_peaks_payload - additional edge cases
# ==============================================================================


def test_decode_legacy_none_payload():
    mz, i = _decode_legacy_peaks_payload(None)
    assert len(mz) == 0
    assert len(i) == 0


def test_decode_legacy_dict_with_peaks_key():
    mz, i = _decode_legacy_peaks_payload({"peaks": [[100.0, 1.0], [200.0, 2.0]]})
    np.testing.assert_array_almost_equal(mz, [100.0, 200.0])
    np.testing.assert_array_almost_equal(i, [1.0, 2.0])


def test_decode_legacy_unknown_dict():
    with pytest.raises(ValueError, match="not recognized"):
        _decode_legacy_peaks_payload({"unknown": "data"})


# ==============================================================================
# Triege flags / _row_to_spectrum
# ==============================================================================


def test_row_to_spectrum_with_triage_flags(temp_db_path, sample_spectrum):
    db = SpectralDatabase(temp_db_path)

    # Manually add triage_flags to the spectrum metadata
    spec = sample_spectrum
    spec.set("triage_flags", {"has_tyrosine_fragment": True})
    db.add_spectra([spec], category="test")

    retrieved = list(db.get_spectra(category="test"))
    assert len(retrieved) == 1
    r = retrieved[0]
    assert r.get("has_tyrosine_fragment") is True

    db.close()


def test_row_to_spectrum_preserves_metadata(temp_db_path, sample_spectrum):
    db = SpectralDatabase(temp_db_path)
    db.add_spectra([sample_spectrum], category="test")
    retrieved = list(db.get_spectra(category="test"))
    assert len(retrieved) == 1
    r = retrieved[0]
    assert r.get("id") == "test_1"
    assert r.get("compound_name") == "Test"

    db.close()


# ==============================================================================
# add_spectra batch handling
# ==============================================================================


def test_add_spectra_batch_commit(temp_db_path):
    """Test that large batches are committed in chunks."""
    db = SpectralDatabase(temp_db_path)
    spectra = []
    for i in range(100):
        spec = Spectrum(
            mz=np.array([100.0 + i, 200.0 + i], dtype=np.float64),
            intensities=np.array([0.5, 1.0], dtype=np.float64),
            metadata={"id": f"batch_{i}", "precursor_mz": 200.0 + i},
        )
        spectra.append(spec)

    count = db.add_spectra(iter(spectra), category="batch_test", batch_size=30)
    assert count == 100

    retrieved = list(db.get_spectra(category="batch_test"))
    assert len(retrieved) == 100

    db.close()


def test_add_spectra_none_handling(temp_db_path, sample_spectrum):
    """None spectra in the iterator are skipped."""
    db = SpectralDatabase(temp_db_path)
    count = db.add_spectra(
        iter([sample_spectrum, None, sample_spectrum.clone()]),
        category="test",
    )
    assert count == 2
    db.close()


def test_add_spectra_stores_triage_json(temp_db_path, sample_spectrum):
    db = SpectralDatabase(temp_db_path)

    spec = sample_spectrum
    spec.set("triage_flags", {"flag_a": True, "flag_b": False})
    db.add_spectra([spec], category="triage_test")

    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT triage_flags FROM spectra WHERE category = 'triage_test'")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    triage = json.loads(row[0])
    assert triage["flag_a"] is True
    assert triage["flag_b"] is False

    db.close()


# ==============================================================================
# SpectralDatabase - additional edge cases
# ==============================================================================


def test_spectral_database_with_allow_destructive_upgrade(temp_db_path):
    db = SpectralDatabase(temp_db_path, allow_destructive_upgrade=True)
    assert db.allow_destructive_upgrade is True
    db.close()


def test_get_precursor_mz_range_empty(temp_db_path):
    db = SpectralDatabase(temp_db_path)
    mz_min, mz_max = db.get_precursor_mz_range()
    assert mz_min == 0.0
    assert mz_max == 0.0
    db.close()


def test_get_total_spectra_count_empty(temp_db_path):
    db = SpectralDatabase(temp_db_path)
    assert db.get_total_spectra_count() == 0
    db.close()


def test_get_category_counts_empty(temp_db_path):
    db = SpectralDatabase(temp_db_path)
    counts = db.get_category_counts()
    assert counts == {}
    db.close()


def test_get_spectra_with_name_pattern(temp_db_path):
    db = SpectralDatabase(temp_db_path)

    s1 = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "a1", "compound_name": "Alpha", "precursor_mz": 100.0},
    )
    s2 = Spectrum(
        mz=np.array([200.0]),
        intensities=np.array([1.0]),
        metadata={"id": "b1", "compound_name": "Beta", "precursor_mz": 200.0},
    )

    db.add_spectra([s1, s2], category="test")

    # LIKE pattern matching
    matches = list(db.get_spectra(name_pattern="%Alpha%"))
    assert len(matches) == 1
    assert matches[0].get("compound_name") == "Alpha"

    db.close()


# ==============================================================================
# create_current_spectra_table - idempotent
# ==============================================================================


def test_create_table_idempotent(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    create_current_spectra_table(conn)  # Should not raise
    conn.close()


# ==============================================================================
# _fetch_legacy_rows
# ==============================================================================


def test_fetch_legacy_rows_empty(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    rows = list(_fetch_legacy_rows(conn))
    assert len(rows) == 0
    conn.close()


# ==============================================================================
# migrate_legacy_peaks_database
# ==============================================================================


def test_migrate_already_current(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    conn.close()

    summary = migrate_legacy_peaks_database(temp_db_path)
    assert summary["status"] == "already_current"


def test_migrate_unknown_schema(temp_db_path):
    """A database with an unrecognized schema raises RuntimeError."""
    conn = _create_sqlite_connection(temp_db_path)
    # Create a table that's not legacy or current
    conn.execute("CREATE TABLE spectra (foo TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="neither"):
        migrate_legacy_peaks_database(temp_db_path)


def test_migrate_legacy_peaks_to_arrays_wrapper(temp_db_path):
    conn = _create_sqlite_connection(temp_db_path)
    create_current_spectra_table(conn)
    conn.close()

    summary = migrate_legacy_peaks_to_arrays(temp_db_path)
    assert summary["status"] == "already_current"


# ==============================================================================
# SpectralDatabase with minimal DB from data/
# ==============================================================================


def test_load_from_minimal_db():
    db_path = Path(__file__).parent / "data" / "minimal_test_library.db"
    if not db_path.exists():
        pytest.skip(
            "Minimal test database not found. Run generate_minimal_db.py first."
        )

    db = SpectralDatabase(db_path)
    count = db.get_total_spectra_count()
    assert count > 0

    cats = db.get_category_counts()
    assert "test_library" in cats

    mz_min, mz_max = db.get_precursor_mz_range()
    assert mz_min > 0
    assert mz_max >= mz_min

    spectra = list(db.get_spectra())
    assert len(spectra) == count

    db.close()


def test_get_spectra_from_minimal_db():
    db_path = Path(__file__).parent / "data" / "minimal_test_library.db"
    if not db_path.exists():
        pytest.skip("Minimal test database not found.")

    db = SpectralDatabase(db_path)
    for spec in db.get_spectra():
        assert spec.peaks is not None
        assert len(spec.peaks.mz) > 0
        assert spec.get("id") is not None
    db.close()
