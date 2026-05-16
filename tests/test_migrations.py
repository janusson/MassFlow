"""
Tests for database migration safety and legacy-schema handling.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from MassFlow.database import SpectralDatabase, migrate_legacy_peaks_database


def _create_legacy_database(database_path: Path) -> None:
    """
    Create a temporary legacy SQLite database using the old ``peaks`` schema.

    Parameters
    ----------
    database_path : Path
        Path to the database file to create.

    Returns
    -------
    None
    """
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE spectra (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id TEXT,
            name TEXT,
            precursor_mz REAL,
            charge INTEGER,
            ionmode TEXT,
            adduct TEXT,
            category TEXT,
            metadata TEXT,
            peaks TEXT
        )
        """
    )

    first_mz = np.array([100.0, 150.0, 200.0], dtype=np.float64)
    first_intensities = np.array([0.1, 0.5, 1.0], dtype=np.float64)
    first_metadata = (
        '{"id": "legacy_1", "compound_name": "Legacy One", "precursor_mz": 200.0}'
    )
    first_peaks = (
        '[{"mz": 100.0, "intensity": 0.1}, {"mz": 150.0, "intensity": 0.5}, '
        '{"mz": 200.0, "intensity": 1.0}]'
    )

    second_mz = np.array([110.0, 160.0, 210.0], dtype=np.float64)
    second_intensities = np.array([0.2, 0.6, 0.9], dtype=np.float64)
    second_metadata = (
        '{"id": "legacy_2", "compound_name": "Legacy Two", "precursor_mz": 210.0}'
    )
    second_peaks = (
        '[{"mz": 110.0, "intensity": 0.2}, {"mz": 160.0, "intensity": 0.6}, '
        '{"mz": 210.0, "intensity": 0.9}]'
    )

    cursor.executemany(
        """
        INSERT INTO spectra (
            original_id, name, precursor_mz, charge, ionmode, adduct, category, metadata, peaks
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "legacy_1",
                "Legacy One",
                200.0,
                1,
                "positive",
                "[M+H]+",
                "legacy",
                first_metadata,
                first_peaks,
            ),
            (
                "legacy_2",
                "Legacy Two",
                210.0,
                1,
                "positive",
                "[M+H]+",
                "legacy",
                second_metadata,
                second_peaks,
            ),
        ],
    )

    connection.commit()
    connection.close()

    # Sanity-check the expected values used later in assertions.
    assert np.allclose(first_mz, [100.0, 150.0, 200.0])
    assert np.allclose(first_intensities, [0.1, 0.5, 1.0])
    assert np.allclose(second_mz, [110.0, 160.0, 210.0])
    assert np.allclose(second_intensities, [0.2, 0.6, 0.9])


def _load_migration_script_module():
    """
    Load the migration script module directly from its file path.

    Returns
    -------
    module
        Loaded Python module object for the migration script.
    """
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "migrations"
        / "0001_peaks_to_arrays.py"
    )
    spec = importlib.util.spec_from_file_location(
        "massflow_migration_0001", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load migration script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_schema_is_rejected_and_not_modified(tmp_path: Path) -> None:
    """
    Legacy databases should be rejected instead of being silently dropped.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
    """
    database_path = tmp_path / "legacy.db"
    _create_legacy_database(database_path)

    with pytest.raises(RuntimeError, match="0001_peaks_to_arrays.py"):
        SpectralDatabase(database_path)

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.execute("PRAGMA table_info(spectra)")
    column_names = [row[1] for row in cursor.fetchall()]
    cursor.execute("SELECT COUNT(*) FROM spectra")
    row_count = cursor.fetchone()[0]
    connection.close()

    assert "peaks" in column_names
    assert "mz_array" not in column_names
    assert "intensity_array" not in column_names
    assert row_count == 2


def test_migrate_legacy_peaks_database_preserves_data(tmp_path: Path) -> None:
    """
    Migrating a legacy database should preserve rows and reconstruct arrays.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
    """
    database_path = tmp_path / "legacy.db"
    _create_legacy_database(database_path)

    migrate_legacy_peaks_database(database_path)

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute("PRAGMA table_info(spectra)")
    column_names = [row[1] for row in cursor.fetchall()]
    assert "peaks" not in column_names
    assert "mz_array" in column_names
    assert "intensity_array" in column_names

    cursor.execute(
        """
        SELECT original_id, name, precursor_mz, category, metadata, mz_array, intensity_array
        FROM spectra
        ORDER BY original_id
        """
    )
    migrated_rows = cursor.fetchall()

    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name LIKE 'spectra_backup_%'
        ORDER BY name
        """
    )
    backup_tables = [row[0] for row in cursor.fetchall()]
    connection.close()

    assert len(migrated_rows) == 2
    assert len(backup_tables) == 1

    first_row = migrated_rows[0]
    second_row = migrated_rows[1]

    assert first_row[0] == "legacy_1"
    assert first_row[1] == "Legacy One"
    assert first_row[2] == pytest.approx(200.0)
    assert first_row[3] == "legacy"
    assert '"Legacy One"' in first_row[4]

    first_mz = np.frombuffer(first_row[5], dtype=np.float64)
    first_intensities = np.frombuffer(first_row[6], dtype=np.float64)
    assert first_mz.dtype == np.float64
    assert first_intensities.dtype == np.float64
    assert np.allclose(first_mz, [100.0, 150.0, 200.0])
    assert np.allclose(first_intensities, [0.1, 0.5, 1.0])

    assert second_row[0] == "legacy_2"
    assert second_row[1] == "Legacy Two"
    assert second_row[2] == pytest.approx(210.0)
    assert second_row[3] == "legacy"
    assert '"Legacy Two"' in second_row[4]

    second_mz = np.frombuffer(second_row[5], dtype=np.float64)
    second_intensities = np.frombuffer(second_row[6], dtype=np.float64)
    assert second_mz.dtype == np.float64
    assert second_intensities.dtype == np.float64
    assert np.allclose(second_mz, [110.0, 160.0, 210.0])
    assert np.allclose(second_intensities, [0.2, 0.6, 0.9])


def test_migrate_legacy_peaks_database_is_idempotent(tmp_path: Path) -> None:
    """
    Re-running the migration on an already migrated database should be a no-op.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
    """
    database_path = tmp_path / "legacy.db"
    _create_legacy_database(database_path)

    migrate_legacy_peaks_database(database_path)
    migrate_legacy_peaks_database(database_path)

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute("PRAGMA table_info(spectra)")
    column_names = [row[1] for row in cursor.fetchall()]
    cursor.execute("SELECT COUNT(*) FROM spectra")
    row_count = cursor.fetchone()[0]
    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name LIKE 'spectra_backup_%'
        ORDER BY name
        """
    )
    backup_tables = [row[0] for row in cursor.fetchall()]
    connection.close()

    assert "peaks" not in column_names
    assert "mz_array" in column_names
    assert "intensity_array" in column_names
    assert row_count == 2
    assert len(backup_tables) == 1


def test_migration_script_main_executes_successfully(tmp_path: Path) -> None:
    """
    The executable migration script should migrate a legacy database successfully.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
    """
    database_path = tmp_path / "legacy.db"
    _create_legacy_database(database_path)

    migration_module = _load_migration_script_module()
    exit_code = migration_module.main(["--input", str(database_path)])

    assert exit_code == 0

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.execute("PRAGMA table_info(spectra)")
    column_names = [row[1] for row in cursor.fetchall()]
    connection.close()

    assert "peaks" not in column_names
    assert "mz_array" in column_names
    assert "intensity_array" in column_names
