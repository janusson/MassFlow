"""
Spectral Database Management for MassFlow.

MIGRATION NOTE
--------------
MassFlow previously used a legacy SQLite schema in which the ``spectra`` table
stored peak data in a single ``peaks`` column. The current schema stores
fragment arrays explicitly as ``mz_array`` and ``intensity_array`` BLOB columns
encoded as ``float64`` bytes.

Legacy databases are no longer upgraded automatically. In particular, MassFlow
will not silently drop or rewrite a legacy ``spectra`` table during database
initialization. If a legacy schema is detected, :class:`SpectralDatabase`
raises a ``RuntimeError`` with instructions to run the explicit migration
workflow:

- ``scripts/migrations/0001_peaks_to_arrays.py``

That migration path is intended to be explicit, validated, and reversible. The
database module provides the schema inspection and migration helpers used by the
migration script so SQL remains centralized in this file.

Zarr array storage (hybrid SQLite + Zarr schema)
-----------------------------------------------
As of the Phase 1 storage migration, :class:`SpectralDatabase` can optionally
outsource fragment arrays to a chunked Zarr store instead of the SQLite BLOB
columns. When ``zarr_path`` is supplied the database stores only metadata plus
two reference columns:

- ``zarr_ref TEXT`` — the unique identifier (UUID) of the Zarr group that owns
  the peak arrays (:attr:`ZarrPeakArrayStore.store_uuid`).
- ``zarr_index INTEGER`` — the spectrum's index into the flat Zarr arrays
  (``peaks/mz_flat``, ``peaks/intensity_flat``, ``peaks/boundaries``).

Fresh hybrid tables are created without ``mz_array`` / ``intensity_array``
BLOB columns. Pre-existing BLOB tables keep their columns (rows written
before migration fall back to BLOB reads) and gain the two reference columns
via ``ALTER TABLE``. A BLOB-mode instance reading a Zarr-referenced row raises
``RuntimeError`` with instructions to reopen with ``zarr_path``.

Existing BLOB databases are migrated with:

- ``scripts/migrations/0002_blobs_to_zarr.py`` (wraps
  :func:`migrate_blobs_to_zarr`)

The migration is append-ordered, verified bit-for-bit before BLOBs are
NULLed, and idempotent. If a run is interrupted, rerun it; orphaned Zarr
arrays from the interrupted batch are automatically truncated. Use
``overwrite=True`` (or delete the ``.zarr`` directory) to rebuild the array
store from scratch.

Current schema version
----------------------
The active ``spectra`` table stores (BLOB mode):

- ``original_id``
- ``name``
- ``precursor_mz``
- ``charge``
- ``ionmode``
- ``adduct``
- ``category``
- ``metadata``
- ``mz_array``
- ``intensity_array``

Hybrid (Zarr) mode replaces the two BLOB columns with:

- ``zarr_ref``
- ``zarr_index``
"""

from __future__ import annotations

import ast
import json
import logging
import shutil
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from matchms import Spectrum

from MassFlow.storage import SpectralStore
from MassFlow.models import TriageProfile
from MassFlow.zarr_store import (
    ZarrPeakArrayStore,
    _DEFAULT_BOUNDARY_CHUNK_SIZE,
    _DEFAULT_PEAK_CHUNK_SIZE,
)

logger = logging.getLogger(__name__)

CURRENT_SPECTRA_COLUMNS = {
    "id",
    "original_id",
    "name",
    "precursor_mz",
    "charge",
    "ionmode",
    "adduct",
    "category",
    "metadata",
    "mz_array",
    "intensity_array",
    "triage_flags",
}
LEGACY_PEAKS_COLUMN = "peaks"


class LegacyDatabaseSchemaError(RuntimeError):
    """
    Raised when a database uses the legacy ``peaks`` schema.

    Returns
    -------
    None

    Examples
    --------
    >>> raise LegacyDatabaseSchemaError("Database requires migration.")
    """


def _current_utc_timestamp_for_names() -> str:
    """
    Build a compact UTC timestamp for SQLite backup table names.

    Parameters
    ----------
    None

    Returns
    -------
    str
        UTC timestamp in ``YYYYMMDD_HHMMSS`` format.

    Examples
    --------
    >>> _current_utc_timestamp_for_names()
    '20250101_120000'
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _create_sqlite_connection(db_path: Union[str, Path]) -> sqlite3.Connection:
    """
    Open a SQLite connection configured for MassFlow use.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        Open SQLite connection with row-factory support.

    Examples
    --------
    >>> connection = _create_sqlite_connection("library.db")
    """
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def get_spectra_table_columns(
    connection: sqlite3.Connection,
) -> list[str]:
    """
    Return the column names present in the ``spectra`` table.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.

    Returns
    -------
    list[str]
        Ordered column names reported by ``PRAGMA table_info``. Returns an empty
        list when the table does not exist.

    Examples
    --------
    >>> columns = get_spectra_table_columns(connection)
    """
    cursor = connection.cursor()
    cursor.execute("PRAGMA table_info(spectra)")
    return [str(row[1]) for row in cursor.fetchall()]


def has_table(connection: sqlite3.Connection, table_name: str) -> bool:
    """
    Check whether a SQLite table exists.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.
    table_name : str
        Table name to check.

    Returns
    -------
    bool
        True if the table exists, otherwise False.

    Examples
    --------
    >>> has_table(connection, "spectra")
    True
    """
    cursor = connection.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def is_legacy_spectra_schema(connection: sqlite3.Connection) -> bool:
    """
    Determine whether the ``spectra`` table uses the legacy schema.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.

    Returns
    -------
    bool
        True when the table exists and contains the legacy ``peaks`` column.

    Examples
    --------
    >>> is_legacy = is_legacy_spectra_schema(connection)
    """
    columns = get_spectra_table_columns(connection)
    return bool(columns) and LEGACY_PEAKS_COLUMN in columns


def is_current_spectra_schema(connection: sqlite3.Connection) -> bool:
    """
    Determine whether the ``spectra`` table matches the current schema.

    The current schema has two accepted variants:

    - BLOB mode: peak arrays stored in ``mz_array`` / ``intensity_array``.
    - Hybrid (Zarr) mode: peak arrays stored in a Zarr store referenced by
      ``zarr_ref`` / ``zarr_index``.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.

    Returns
    -------
    bool
        True when the required current-schema columns are present.

    Examples
    --------
    >>> is_current = is_current_spectra_schema(connection)
    """
    columns = set(get_spectra_table_columns(connection))
    if not columns:
        return False
    has_blob_arrays = {"mz_array", "intensity_array"}.issubset(columns)
    has_zarr_reference = {"zarr_ref", "zarr_index"}.issubset(columns)
    return has_blob_arrays or has_zarr_reference


def legacy_migration_error_message(db_path: Union[str, Path]) -> str:
    """
    Build the standard legacy-schema migration error message.

    Parameters
    ----------
    db_path : str or Path
        Path to the legacy SQLite database.

    Returns
    -------
    str
        Detailed runtime error message with remediation steps.

    Examples
    --------
    >>> message = legacy_migration_error_message("legacy.db")
    """
    return (
        f"Legacy database schema detected in '{Path(db_path)}'. "
        f"This database still contains the '{LEGACY_PEAKS_COLUMN}' column and "
        f"will not be upgraded automatically. Run the explicit migration script "
        f"'scripts/migrations/0001_peaks_to_arrays.py' before using this "
        f"database with the current MassFlow release."
    )


def create_current_spectra_table(
    connection: sqlite3.Connection,
    include_peak_blobs: bool = True,
) -> None:
    """
    Create the current ``spectra`` table if it does not exist.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.
    include_peak_blobs : bool, optional
        If ``True`` (default, BLOB mode) the table includes the
        ``mz_array`` / ``intensity_array`` BLOB columns. If ``False``
        (hybrid Zarr mode) the table includes the ``zarr_ref`` /
        ``zarr_index`` reference columns instead, so SQLite retains only
        metadata plus the Zarr reference.

    Returns
    -------
    None

    Examples
    --------
    >>> create_current_spectra_table(connection)
    """
    cursor = connection.cursor()
    if include_peak_blobs:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spectra (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id TEXT,
                name TEXT,
                precursor_mz REAL,
                charge INTEGER,
                ionmode TEXT,
                adduct TEXT,
                category TEXT,
                metadata TEXT,
                mz_array BLOB,
                intensity_array BLOB,
                triage_flags TEXT
            )
            """
        )
    else:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spectra (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id TEXT,
                name TEXT,
                precursor_mz REAL,
                charge INTEGER,
                ionmode TEXT,
                adduct TEXT,
                category TEXT,
                metadata TEXT,
                zarr_ref TEXT,
                zarr_index INTEGER,
                triage_flags TEXT
            )
            """
        )

    columns = get_spectra_table_columns(connection)
    if columns and "triage_flags" not in columns:
        cursor.execute("ALTER TABLE spectra ADD COLUMN triage_flags TEXT")

    connection.commit()


def _ensure_zarr_columns(connection: sqlite3.Connection) -> None:
    """
    Add the ``zarr_ref`` and ``zarr_index`` columns to an existing table.

    Idempotent: columns that already exist are left untouched. Used when a
    hybrid Zarr-mode :class:`SpectralDatabase` opens a table originally
    created in BLOB mode.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.

    Returns
    -------
    None

    Examples
    --------
    >>> _ensure_zarr_columns(connection)
    """
    columns = get_spectra_table_columns(connection)
    if not columns:
        return
    cursor = connection.cursor()
    if "zarr_ref" not in columns:
        cursor.execute("ALTER TABLE spectra ADD COLUMN zarr_ref TEXT")
    if "zarr_index" not in columns:
        cursor.execute("ALTER TABLE spectra ADD COLUMN zarr_index INTEGER")
    connection.commit()


def _json_serialize_metadata(metadata: dict[str, Any]) -> str:
    """
    Serialize spectrum metadata into JSON.

    Parameters
    ----------
    metadata : dict[str, Any]
        Metadata dictionary to serialize.

    Returns
    -------
    str
        JSON string.

    Examples
    --------
    >>> _json_serialize_metadata({"id": "spec_1"})
    '{"id": "spec_1"}'
    """
    return json.dumps(metadata, default=str)


def _decode_legacy_peaks_payload(
    peaks_payload: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode a legacy ``peaks`` payload into float64 m/z and intensity arrays.

    Parameters
    ----------
    peaks_payload : Any
        Legacy payload from the ``peaks`` column. Supported forms include JSON
        strings, Python literal strings, lists of ``[mz, intensity]`` pairs, or
        dictionaries containing peak arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Decoded ``mz_array`` and ``intensity_array`` as ``float64`` arrays.

    Raises
    ------
    ValueError
        If the payload cannot be decoded into a valid two-array representation.

    Examples
    --------
    >>> mz_array, intensity_array = _decode_legacy_peaks_payload("[[100.0, 1.0]]")
    """
    if peaks_payload is None:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    parsed_payload = peaks_payload

    if isinstance(parsed_payload, bytes):
        try:
            parsed_payload = parsed_payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Legacy peaks payload bytes are not valid UTF-8.") from exc

    if isinstance(parsed_payload, str):
        text_payload = parsed_payload.strip()
        if text_payload == "":
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

        try:
            parsed_payload = json.loads(text_payload)
        except json.JSONDecodeError:
            try:
                parsed_payload = ast.literal_eval(text_payload)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    "Legacy peaks payload is neither valid JSON nor a Python literal."
                ) from exc

    if isinstance(parsed_payload, dict):
        if "peaks" in parsed_payload:
            parsed_payload = parsed_payload["peaks"]
        elif "mz" in parsed_payload and "intensity" in parsed_payload:
            mz_array = np.asarray(parsed_payload["mz"], dtype=np.float64)
            intensity_array = np.asarray(parsed_payload["intensity"], dtype=np.float64)
            if mz_array.shape != intensity_array.shape:
                raise ValueError("Legacy peaks dictionary has mismatched array shapes.")
            return mz_array, intensity_array
        elif "mz_array" in parsed_payload and "intensity_array" in parsed_payload:
            mz_array = np.asarray(parsed_payload["mz_array"], dtype=np.float64)
            intensity_array = np.asarray(
                parsed_payload["intensity_array"], dtype=np.float64
            )
            if mz_array.shape != intensity_array.shape:
                raise ValueError("Legacy peaks dictionary has mismatched array shapes.")
            return mz_array, intensity_array
        else:
            raise ValueError("Legacy peaks dictionary shape is not recognized.")

    if isinstance(parsed_payload, (list, tuple)):
        if len(parsed_payload) == 0:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

        first_item = parsed_payload[0]

        if isinstance(first_item, dict):
            if not all(
                isinstance(item, dict) and "mz" in item and "intensity" in item
                for item in parsed_payload
            ):
                raise ValueError(
                    "Legacy peaks list-of-dicts payload must contain 'mz' and "
                    "'intensity' for every peak."
                )
            mz_array = np.asarray(
                [float(item["mz"]) for item in parsed_payload],
                dtype=np.float64,
            )
            intensity_array = np.asarray(
                [float(item["intensity"]) for item in parsed_payload],
                dtype=np.float64,
            )
            return mz_array, intensity_array

        if isinstance(first_item, (list, tuple)) and len(first_item) >= 2:
            mz_values = [float(item[0]) for item in parsed_payload]
            intensity_values = [float(item[1]) for item in parsed_payload]
            mz_array = np.asarray(mz_values, dtype=np.float64)
            intensity_array = np.asarray(intensity_values, dtype=np.float64)
            return mz_array, intensity_array

        if len(parsed_payload) == 2 and all(
            isinstance(item, (list, tuple)) for item in parsed_payload
        ):
            mz_array = np.asarray(parsed_payload[0], dtype=np.float64)
            intensity_array = np.asarray(parsed_payload[1], dtype=np.float64)
            if mz_array.shape != intensity_array.shape:
                raise ValueError("Legacy peaks tuple has mismatched array shapes.")
            return mz_array, intensity_array

    raise ValueError("Legacy peaks payload format is not supported.")


def _validate_decoded_legacy_arrays(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
) -> None:
    """
    Validate decoded legacy arrays before storage.

    Parameters
    ----------
    mz_array : np.ndarray
        Decoded fragment m/z values.
    intensity_array : np.ndarray
        Decoded fragment intensities.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the arrays are invalid for MassFlow storage.

    Examples
    --------
    >>> _validate_decoded_legacy_arrays(mz_array, intensity_array)
    """
    if mz_array.dtype != np.float64:
        mz_array = mz_array.astype(np.float64)

    if intensity_array.dtype != np.float64:
        intensity_array = intensity_array.astype(np.float64)

    if mz_array.shape != intensity_array.shape:
        raise ValueError("Decoded legacy arrays have mismatched shapes.")

    if mz_array.ndim != 1 or intensity_array.ndim != 1:
        raise ValueError("Decoded legacy arrays must be one-dimensional.")

    if mz_array.size > 1 and np.any(np.diff(mz_array) < 0):
        raise ValueError("Decoded legacy m/z values are out of order.")


def _serialize_peak_arrays(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
) -> tuple[bytes, bytes]:
    """
    Serialize validated fragment arrays as float64 BLOBs.

    Parameters
    ----------
    mz_array : np.ndarray
        Fragment m/z values.
    intensity_array : np.ndarray
        Fragment intensities.

    Returns
    -------
    tuple[bytes, bytes]
        Serialized ``mz_array`` and ``intensity_array`` BLOB payloads.

    Examples
    --------
    >>> mz_blob, intensity_blob = _serialize_peak_arrays(mz_array, intensity_array)
    """
    mz_float64 = np.asarray(mz_array, dtype=np.float64)
    intensity_float64 = np.asarray(intensity_array, dtype=np.float64)
    return mz_float64.tobytes(), intensity_float64.tobytes()


def _create_backup_table_name() -> str:
    """
    Build a unique backup table name for a migration run.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Backup table name such as ``spectra_backup_20250101_120000``.

    Examples
    --------
    >>> _create_backup_table_name()
    'spectra_backup_20250101_120000'
    """
    return f"spectra_backup_{_current_utc_timestamp_for_names()}"


def create_legacy_backup_table(
    connection: sqlite3.Connection,
    backup_table_name: Optional[str] = None,
) -> str:
    """
    Create a full backup copy of the legacy ``spectra`` table.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.
    backup_table_name : str or None, optional
        Explicit backup table name. If None, one is generated automatically.

    Returns
    -------
    str
        Name of the created backup table.

    Raises
    ------
    RuntimeError
        If the source ``spectra`` table does not exist.

    Examples
    --------
    >>> backup_name = create_legacy_backup_table(connection)
    """
    if not has_table(connection, "spectra"):
        raise RuntimeError(
            "Cannot back up legacy spectra table because it does not exist."
        )

    table_name = backup_table_name or _create_backup_table_name()
    cursor = connection.cursor()
    cursor.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM spectra')
    return table_name


def create_migrated_spectra_table(connection: sqlite3.Connection) -> None:
    """
    Create the temporary migrated ``spectra_migrated`` table.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.

    Returns
    -------
    None

    Examples
    --------
    >>> create_migrated_spectra_table(connection)
    """
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS spectra_migrated")
    cursor.execute(
        """
        CREATE TABLE spectra_migrated (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id TEXT,
            name TEXT,
            precursor_mz REAL,
            charge INTEGER,
            ionmode TEXT,
            adduct TEXT,
            category TEXT,
            metadata TEXT,
            mz_array BLOB,
            intensity_array BLOB,
            triage_flags TEXT
        )
        """
    )
    connection.commit()


def _fetch_legacy_rows(connection: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    """
    Fetch all legacy rows from the ``spectra`` table.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.

    Returns
    -------
    list[sqlite3.Row]
        All legacy rows.

    Examples
    --------
    >>> legacy_rows = _fetch_legacy_rows(connection)
    """
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM spectra")
    return iter(cursor.fetchall())


def migrate_legacy_peaks_database(
    db_path: Union[str, Path],
) -> dict[str, Any]:
    """
    Migrate a legacy ``spectra(peaks)`` database to the current array schema.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database to migrate.

    Returns
    -------
    dict[str, Any]
        Migration summary including backup table name and row counts.

    Raises
    ------
    RuntimeError
        If the database is not legacy or if validation fails.
    ValueError
        If legacy peak payload decoding fails for any row.

    Notes
    -----
    The migration is performed within an explicit SQLite transaction. A backup
    table is created before any schema swap occurs. On failure, the transaction
    is rolled back.

    Examples
    --------
    >>> summary = migrate_legacy_peaks_database("legacy_library.db")
    """
    connection = _create_sqlite_connection(db_path)

    try:
        if is_current_spectra_schema(connection):
            return {
                "status": "already_current",
                "database": str(Path(db_path)),
                "message": "Database already uses the current schema.",
            }

        if not is_legacy_spectra_schema(connection):
            raise RuntimeError(
                "Database schema is neither the recognized legacy schema nor the "
                "current schema."
            )

        cursor = connection.cursor()
        cursor.execute("BEGIN")

        backup_table_name = create_legacy_backup_table(connection)
        create_migrated_spectra_table(connection)

        legacy_rows = _fetch_legacy_rows(connection)

        insert_query = """
            INSERT INTO spectra_migrated (
                original_id,
                name,
                precursor_mz,
                charge,
                ionmode,
                adduct,
                category,
                metadata,
                mz_array,
                intensity_array,
                triage_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        migrated_sample_checks: list[dict[str, Any]] = []
        for row_index, row in enumerate(legacy_rows):
            mz_array, intensity_array = _decode_legacy_peaks_payload(row["peaks"])
            _validate_decoded_legacy_arrays(mz_array, intensity_array)
            mz_blob, intensity_blob = _serialize_peak_arrays(mz_array, intensity_array)

            # Fast scan for Tyrosine immonium ion (136.076 Da)
            triage_flags = {}
            target_mz = 136.076
            mz_tolerance = 0.05

            idx = np.searchsorted(mz_array, target_mz)
            has_tyrosine = False
            for i in [idx - 1, idx]:
                if 0 <= i < len(mz_array):
                    if abs(mz_array[i] - target_mz) <= mz_tolerance:
                        if intensity_array[i] > 0:
                            has_tyrosine = True
                            break

            if has_tyrosine:
                triage_flags["has_tyrosine_fragment"] = True

            triage_json = json.dumps(triage_flags)

            cursor.execute(
                insert_query,
                (
                    row["original_id"] if "original_id" in row.keys() else row["id"],
                    row["name"] if "name" in row.keys() else None,
                    row["precursor_mz"] if "precursor_mz" in row.keys() else None,
                    row["charge"] if "charge" in row.keys() else None,
                    row["ionmode"] if "ionmode" in row.keys() else None,
                    row["adduct"] if "adduct" in row.keys() else None,
                    row["category"] if "category" in row.keys() else "default",
                    row["metadata"] if "metadata" in row.keys() else "{}",
                    mz_blob,
                    intensity_blob,
                    triage_json,
                ),
            )

            if row_index < 3:
                migrated_sample_checks.append(
                    {
                        "row_index": row_index,
                        "peak_count": int(mz_array.size),
                        "first_mz": float(mz_array[0]) if mz_array.size > 0 else None,
                        "first_intensity": (
                            float(intensity_array[0])
                            if intensity_array.size > 0
                            else None
                        ),
                    }
                )

        cursor.execute("SELECT COUNT(*) FROM spectra")
        legacy_row_count = int(cursor.fetchone()[0])

        cursor.execute("SELECT COUNT(*) FROM spectra_migrated")
        migrated_row_count = int(cursor.fetchone()[0])

        if migrated_row_count != legacy_row_count:
            raise RuntimeError(
                "Migrated row count does not match legacy row count. "
                f"Legacy={legacy_row_count}, migrated={migrated_row_count}."
            )

        cursor.execute("SELECT mz_array, intensity_array FROM spectra_migrated LIMIT 3")
        validation_rows = cursor.fetchall()
        for validation_row in validation_rows:
            migrated_mz_array = np.frombuffer(
                validation_row["mz_array"], dtype=np.float64
            ).copy()
            migrated_intensity_array = np.frombuffer(
                validation_row["intensity_array"], dtype=np.float64
            ).copy()
            _validate_decoded_legacy_arrays(migrated_mz_array, migrated_intensity_array)

        cursor.execute("DROP TABLE spectra")
        cursor.execute("ALTER TABLE spectra_migrated RENAME TO spectra")

        connection.commit()

        return {
            "status": "migrated",
            "database": str(Path(db_path)),
            "backup_table": backup_table_name,
            "legacy_row_count": legacy_row_count,
            "migrated_row_count": migrated_row_count,
            "sample_checks": migrated_sample_checks,
        }

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def migrate_legacy_peaks_to_arrays(
    db_path: Union[str, Path],
) -> dict[str, Any]:
    """
    Compatibility wrapper for the explicit legacy peak migration helper.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database to migrate.

    Returns
    -------
    dict[str, Any]
        Migration summary from :func:`migrate_legacy_peaks_database`.

    Examples
    --------
    >>> summary = migrate_legacy_peaks_to_arrays("legacy_library.db")
    """
    return migrate_legacy_peaks_database(db_path)


def migrate_blobs_to_zarr(
    db_path: Union[str, Path],
    zarr_path: Union[str, Path, None] = None,
    peak_chunk_size: int = _DEFAULT_PEAK_CHUNK_SIZE,
    boundary_chunk_size: int = _DEFAULT_BOUNDARY_CHUNK_SIZE,
    compressor: Optional[Any] = None,
    null_blobs: bool = True,
    overwrite: bool = False,
    batch_size: int = 5000,
    verify: bool = True,
) -> dict[str, Any]:
    """
    Migrate SQLite BLOB peak arrays into a chunked Zarr store.

    Reads every spectrum row that still carries BLOB arrays (``zarr_index``
    IS NULL), appends the arrays to a :class:`ZarrPeakArrayStore`, and
    updates each row with the ``zarr_ref`` UUID and its ``zarr_index`` in the
    flat Zarr arrays. After a successful, verified write the BLOB columns are
    NULLed (default) so SQLite retains only metadata plus the Zarr reference.

    The migration is:

    - **append-ordered**: rows are processed in ``id`` order so indices are
      stable and re-runs continue where they left off.
    - **verified**: each appended batch is read back from Zarr and compared
      bit-for-bit against the BLOB data before the row references are
      committed.
    - **idempotent**: rows that already reference the Zarr store are skipped;
      a fully migrated database reports ``"already_migrated"``.

    If a previous run was interrupted between a Zarr append and its SQL
    update, the orphaned Zarr tail is detected and truncated automatically.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database to migrate.
    zarr_path : str or Path or None, optional
        Path for the Zarr array store. Defaults to the database path with a
        ``.zarr`` suffix (e.g. ``library.db`` → ``library.zarr``).
    peak_chunk_size : int, optional
        Float64 elements per chunk in the Zarr peak arrays. Small chunks
        increase metadata overhead; large chunks increase decompression
        memory during parallel reads.
    boundary_chunk_size : int, optional
        Spectra per chunk in the Zarr boundaries array.
    compressor : optional
        Zarr compressor (defaults to Blosc+zstd, clevel=3).
    null_blobs : bool, optional
        If ``True`` (default), NULL out the ``mz_array`` / ``intensity_array``
        columns of migrated rows after verification.
    overwrite : bool, optional
        If ``True``, clear any existing ``zarr_ref`` / ``zarr_index`` values
        and rebuild the Zarr store from scratch. Only meaningful while the
        BLOB columns still hold the source data (i.e. before a previous run
        with ``null_blobs=True``); rows whose BLOBs were already NULLed
        cannot be re-migrated.
    batch_size : int, optional
        Number of rows migrated per Zarr append / SQL commit batch.
    verify : bool, optional
        If ``True`` (default), read every migrated batch back from Zarr and
        compare it bit-for-bit against the source BLOB arrays.

    Returns
    -------
    dict[str, Any]
        Migration summary with ``status`` (``"migrated"`` or
        ``"already_migrated"``), ``database``, ``zarr_path``, ``zarr_ref``,
        ``migrated_row_count``, and ``sample_checks``.

    Raises
    ------
    RuntimeError
        If the database uses the legacy ``peaks`` schema (run migration
        0001 first), if verification fails, or if the Zarr store does not
        match the rows that already reference it.

    Examples
    --------
    >>> summary = migrate_blobs_to_zarr("library.db")
    """
    database_path = Path(db_path)
    target_zarr_path = (
        Path(zarr_path) if zarr_path is not None else database_path.with_suffix(".zarr")
    )

    connection = _create_sqlite_connection(database_path)
    store: Optional[ZarrPeakArrayStore] = None

    try:
        if is_legacy_spectra_schema(connection):
            raise RuntimeError(
                "Legacy 'peaks' schema detected. Run "
                "'scripts/migrations/0001_peaks_to_arrays.py' before "
                "migrating BLOBs to Zarr."
            )

        _ensure_zarr_columns(connection)
        columns = set(get_spectra_table_columns(connection))
        if not {"mz_array", "intensity_array"}.issubset(columns):
            return {
                "status": "already_migrated",
                "database": str(database_path),
                "zarr_path": str(target_zarr_path),
                "zarr_ref": None,
                "migrated_row_count": 0,
                "message": "Database has no BLOB columns to migrate.",
            }

        cursor = connection.cursor()
        if overwrite:
            cursor.execute("UPDATE spectra SET zarr_ref = NULL, zarr_index = NULL")
            connection.commit()
            if target_zarr_path.is_dir():
                shutil.rmtree(target_zarr_path)

        store = ZarrPeakArrayStore(
            target_zarr_path,
            peak_chunk_size=peak_chunk_size,
            boundary_chunk_size=boundary_chunk_size,
            compressor=compressor,
        )
        zarr_ref = store.store_uuid

        # Consistency checks against rows that already reference Zarr.
        cursor.execute(
            "SELECT DISTINCT zarr_ref FROM spectra WHERE zarr_ref IS NOT NULL LIMIT 1"
        )
        existing_ref_row = cursor.fetchone()
        if existing_ref_row is not None and str(existing_ref_row[0]) != zarr_ref:
            raise RuntimeError(
                f"Zarr store identity mismatch: rows reference "
                f"'{existing_ref_row[0]}' but '{target_zarr_path}' reports "
                f"'{zarr_ref}'."
            )

        cursor.execute("SELECT COUNT(*) FROM spectra WHERE zarr_index IS NOT NULL")
        referenced_count = int(cursor.fetchone()[0])

        if store.n_spectra > referenced_count:
            # A previous run was interrupted after a Zarr append but before
            # its SQL update: trim the orphaned trailing arrays and continue.
            logger.warning(
                "Zarr store has %d spectra but only %d rows reference it; "
                "truncating %d orphaned arrays.",
                store.n_spectra,
                referenced_count,
                store.n_spectra - referenced_count,
            )
            store.truncate(referenced_count)
        elif store.n_spectra < referenced_count:
            raise RuntimeError(
                f"Zarr store '{target_zarr_path}' is missing referenced "
                f"spectra: {referenced_count} rows reference it but it holds "
                f"only {store.n_spectra}. Rebuild with overwrite=True."
            )

        cursor.execute(
            "SELECT id, mz_array, intensity_array FROM spectra "
            "WHERE zarr_index IS NULL "
            "AND mz_array IS NOT NULL AND intensity_array IS NOT NULL "
            "ORDER BY id"
        )
        rows = cursor.fetchall()
        if not rows:
            return {
                "status": "already_migrated",
                "database": str(database_path),
                "zarr_path": str(target_zarr_path),
                "zarr_ref": zarr_ref,
                "migrated_row_count": 0,
                "message": "All rows already reference the Zarr store.",
            }

        update_cursor = connection.cursor()
        migrated_count = 0
        total_peak_count = 0
        sample_checks: list[dict[str, Any]] = []

        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start : batch_start + batch_size]
            mz_list = [
                np.frombuffer(row["mz_array"], dtype=np.float64) for row in batch
            ]
            intensity_list = [
                np.frombuffer(row["intensity_array"], dtype=np.float64) for row in batch
            ]

            start_index = store.append_spectra(mz_list, intensity_list)

            if verify:
                read_mz, read_intensity = store.read_spectra(
                    list(range(start_index, start_index + len(batch)))
                )
                for offset, (mz_read, intensity_read) in enumerate(
                    zip(read_mz, read_intensity)
                ):
                    if not np.array_equal(mz_read, mz_list[offset]) or not (
                        np.array_equal(intensity_read, intensity_list[offset])
                    ):
                        raise RuntimeError(
                            "Zarr verification failed for row id="
                            f"{int(batch[offset]['id'])} at index "
                            f"{start_index + offset}."
                        )

            update_cursor.executemany(
                "UPDATE spectra SET zarr_ref = ?, zarr_index = ? WHERE id = ?",
                [
                    (zarr_ref, start_index + offset, int(row["id"]))
                    for offset, row in enumerate(batch)
                ],
            )
            connection.commit()

            migrated_count += len(batch)
            total_peak_count += sum(int(arr.size) for arr in mz_list)
            if len(sample_checks) < 3:
                first_mz = mz_list[0]
                first_intensity = intensity_list[0]
                sample_checks.append(
                    {
                        "row_id": int(batch[0]["id"]),
                        "zarr_index": start_index,
                        "peak_count": int(first_mz.size),
                        "first_mz": (float(first_mz[0]) if first_mz.size > 0 else None),
                        "first_intensity": (
                            float(first_intensity[0])
                            if first_intensity.size > 0
                            else None
                        ),
                    }
                )

        if null_blobs:
            update_cursor.execute(
                "UPDATE spectra SET mz_array = NULL, intensity_array = NULL "
                "WHERE zarr_index IS NOT NULL"
            )
            connection.commit()

        return {
            "status": "migrated",
            "database": str(database_path),
            "zarr_path": str(target_zarr_path),
            "zarr_ref": zarr_ref,
            "migrated_row_count": migrated_count,
            "total_peak_count": total_peak_count,
            "blobs_nulled": null_blobs,
            "sample_checks": sample_checks,
        }
    finally:
        if store is not None:
            store.close()
        connection.close()


class SpectralDatabase(SpectralStore):
    """
    Manage a local SQLite database for persistent storage of mass spectra.

    By default the database stores fragment arrays as ``float64`` BLOBs in the
    ``mz_array`` / ``intensity_array`` columns (BLOB mode). Passing
    ``zarr_path`` switches to hybrid mode: SQLite retains only metadata plus a
    ``zarr_ref`` / ``zarr_index`` reference pair, and the fragment arrays are
    persisted in a chunked, compressed Zarr store (:class:`ZarrPeakArrayStore`)
    that supports concurrent, lock-free reads for multiprocessing.

    Parameters
    ----------
    db_path : str or Path
        File system path to the SQLite database.
    allow_destructive_upgrade : bool, optional
        Legacy compatibility flag. If True and a legacy schema is detected,
        initialization raises a ``RuntimeError`` describing the required
        explicit migration path. This flag exists only to make destructive
        upgrade intent explicit and is not used to perform destructive changes.
    zarr_path : str or Path or None, optional
        Path to the Zarr array store used for ``mz_array`` / ``intensity_array``
        in hybrid mode. When ``None`` (default) the database uses BLOB mode.
    peak_chunk_size : int, optional
        Float64 elements per chunk in the Zarr peak arrays (hybrid mode only).
    boundary_chunk_size : int, optional
        Spectra per chunk in the Zarr boundaries array (hybrid mode only).
    compressor : optional
        Zarr compressor for the peak arrays (hybrid mode only).

    Returns
    -------
    None

    Examples
    --------
    >>> db = SpectralDatabase("library.db")
    >>> db = SpectralDatabase("library.db", zarr_path="library.zarr")
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        allow_destructive_upgrade: bool = False,
        zarr_path: Union[str, Path, None] = None,
        peak_chunk_size: int = _DEFAULT_PEAK_CHUNK_SIZE,
        boundary_chunk_size: int = _DEFAULT_BOUNDARY_CHUNK_SIZE,
        compressor: Optional[Any] = None,
    ):
        # Let SpectralStore.__init__ set self.store_path and call
        # self._initialize() (no-op for SpectralDatabase — SQLite setup
        # happens explicitly below).
        super().__init__(Path(db_path))
        # Keep legacy alias for backward compatibility
        self.db_path = self.store_path
        self.conn: Optional[sqlite3.Connection] = None
        self.allow_destructive_upgrade = allow_destructive_upgrade
        self._zarr_path: Optional[Path] = (
            Path(zarr_path) if zarr_path is not None else None
        )
        self._peak_chunk_size = peak_chunk_size
        self._boundary_chunk_size = boundary_chunk_size
        self._compressor = compressor
        self._zarr_arrays: Optional[ZarrPeakArrayStore] = None
        self._connect()
        self._initialize_tables()
        if self._zarr_path is not None:
            self._attach_zarr_store()

    def _initialize(self) -> None:
        """No-op; SQLite setup is handled in __init__ via _connect/_initialize_tables."""

    def _connect(self) -> None:
        """
        Establish a connection to the SQLite database.

        Returns
        -------
        None

        Examples
        --------
        >>> db._connect()
        """
        self.conn = _create_sqlite_connection(self.db_path)

    def _initialize_tables(self) -> None:
        """
        Create the current schema or reject legacy schemas safely.

        Returns
        -------
        None

        Raises
        ------
        ConnectionError
            If the database connection has not been established.
        LegacyDatabaseSchemaError
            If the legacy ``peaks`` schema is detected.

        Examples
        --------
        >>> db._initialize_tables()
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")

        if is_legacy_spectra_schema(self.conn):
            raise LegacyDatabaseSchemaError(
                legacy_migration_error_message(self.db_path)
            )

        create_current_spectra_table(
            self.conn,
            include_peak_blobs=self._zarr_path is None,
        )
        if self._zarr_path is not None:
            _ensure_zarr_columns(self.conn)

    def _attach_zarr_store(self) -> None:
        """
        Open the hybrid-mode Zarr array store and validate its identity.

        Returns
        -------
        None

        Raises
        ------
        ConnectionError
            If the database connection has not been established.
        RuntimeError
            If rows reference a Zarr store whose UUID does not match the
            attached store.

        Examples
        --------
        >>> db._attach_zarr_store()
        """
        if not self.conn or self._zarr_path is None:
            raise ConnectionError("Database not connected or no Zarr path set.")

        self._zarr_arrays = ZarrPeakArrayStore(
            self._zarr_path,
            peak_chunk_size=self._peak_chunk_size,
            boundary_chunk_size=self._boundary_chunk_size,
            compressor=self._compressor,
        )

        # Validate that any referenced rows point at this exact store.
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT zarr_ref FROM spectra WHERE zarr_ref IS NOT NULL LIMIT 1"
        )
        row = cursor.fetchone()
        if row is not None:
            referenced_uuid = str(row[0])
            if referenced_uuid != self._zarr_arrays.store_uuid:
                actual_uuid = self._zarr_arrays.store_uuid
                self._zarr_arrays.close()
                self._zarr_arrays = None
                raise RuntimeError(
                    f"Zarr store identity mismatch: spectra rows reference "
                    f"'{referenced_uuid}' but '{self._zarr_path}' reports "
                    f"'{actual_uuid}'."
                )

        # Verify the array store actually covers every referenced index.
        cursor.execute(
            "SELECT MAX(zarr_index) FROM spectra WHERE zarr_index IS NOT NULL"
        )
        max_index_row = cursor.fetchone()
        if max_index_row is not None and max_index_row[0] is not None:
            if self._zarr_arrays.n_spectra <= int(max_index_row[0]):
                raise RuntimeError(
                    f"Zarr store '{self._zarr_path}' holds "
                    f"{self._zarr_arrays.n_spectra} spectra but rows reference "
                    f"indices up to {int(max_index_row[0])}. The array store "
                    f"is missing data; rebuild it with "
                    f"migrate_blobs_to_zarr(overwrite=True)."
                )

    @property
    def zarr_ref(self) -> Optional[str]:
        """Return the attached Zarr store UUID in hybrid mode, else ``None``."""
        if self._zarr_arrays is None:
            return None
        return self._zarr_arrays.store_uuid

    @property
    def zarr_path(self) -> Optional[Path]:
        """Return the attached Zarr store path in hybrid mode, else ``None``."""
        return self._zarr_path

    def add_spectra(
        self,
        spectra: Iterator[Spectrum],
        category: str = "default",
        batch_size: int = 5000,
    ) -> int:
        """
        Add multiple spectra to the database using batch insertion.

        Parameters
        ----------
        spectra : Iterator[Spectrum]
            Iterator yielding ``matchms.Spectrum`` objects to insert.
        category : str, optional
            Category label for inserted spectra.
        batch_size : int, optional
            Number of rows to accumulate before committing.

        Returns
        -------
        int
            Number of successfully added spectra.

        Raises
        ------
        ConnectionError
            If the database connection is not active.

        Examples
        --------
        >>> count = db.add_spectra(iter([spectrum]), category="library")
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")

        if self._zarr_arrays is not None:
            return self._add_spectra_zarr(spectra, category, batch_size)

        count = 0
        batch: list[tuple[Any, ...]] = []
        cursor = self.conn.cursor()

        insert_query = """
            INSERT INTO spectra (
                original_id, name, precursor_mz, charge, ionmode, adduct, category, metadata, mz_array, intensity_array, triage_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for spectrum in spectra:
            if spectrum is None:
                continue

            mz_array_raw = np.asarray(spectrum.peaks.mz, dtype=np.float64)
            intensity_array_raw = np.asarray(
                spectrum.peaks.intensities, dtype=np.float64
            )

            # Fast scan for Tyrosine immonium ion (136.076 Da)
            # Triage flags are now pre-calculated in processing.py and stored in metadata
            triage_flags = spectrum.get("triage_flags", {})
            triage_json = json.dumps(triage_flags)

            mz_blob, intensity_blob = _serialize_peak_arrays(
                mz_array_raw,
                intensity_array_raw,
            )
            metadata_json = _json_serialize_metadata(spectrum.metadata.copy())

            row = (
                spectrum.get("id"),
                spectrum.get("compound_name") or spectrum.get("name"),
                spectrum.get("precursor_mz"),
                spectrum.get("charge"),
                spectrum.get("ionmode"),
                spectrum.get("adduct"),
                category,
                metadata_json,
                mz_blob,
                intensity_blob,
                triage_json,
            )
            batch.append(row)
            count += 1

            if len(batch) >= batch_size:
                cursor.executemany(insert_query, batch)
                self.conn.commit()
                batch = []

        if batch:
            cursor.executemany(insert_query, batch)
            self.conn.commit()

        return count

    def _zarr_insert_columns(self) -> list[str]:
        """
        Return the column list used by hybrid-mode INSERT statements.

        Returns
        -------
        list[str]
            Preferred columns that actually exist in the ``spectra`` table.

        Examples
        --------
        >>> columns = db._zarr_insert_columns()
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        available = set(get_spectra_table_columns(self.conn))
        preferred = [
            "original_id",
            "name",
            "precursor_mz",
            "charge",
            "ionmode",
            "adduct",
            "category",
            "metadata",
            "triage_flags",
            "zarr_ref",
            "zarr_index",
            "mz_array",
            "intensity_array",
        ]
        return [column for column in preferred if column in available]

    def _add_spectra_zarr(
        self,
        spectra: Iterator[Spectrum],
        category: str,
        batch_size: int,
    ) -> int:
        """
        Insert spectra in hybrid mode: Zarr arrays + metadata rows.

        Fragment arrays are appended to the Zarr store first and the returned
        indices are persisted in the ``zarr_index`` column, so SQLite rows only
        carry metadata plus the ``zarr_ref`` / ``zarr_index`` reference pair.

        Note: if the SQLite insert fails after a Zarr append, the Zarr store
        may contain unreferenced trailing arrays. This is harmless (they are
        never read) and can be removed by rebuilding the store.

        Parameters
        ----------
        spectra : Iterator[Spectrum]
            Iterator yielding spectra to insert.
        category : str
            Category label for inserted spectra.
        batch_size : int
            Number of rows to accumulate before committing.

        Returns
        -------
        int
            Number of successfully added spectra.

        Examples
        --------
        >>> count = db._add_spectra_zarr(iter([spectrum]), "library", 5000)
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        if self._zarr_arrays is None:
            raise RuntimeError("Zarr store is not attached (hybrid mode off).")

        columns = self._zarr_insert_columns()
        insert_query = (
            f"INSERT INTO spectra ({', '.join(columns)}) "
            f"VALUES ({', '.join(['?'] * len(columns))})"
        )
        zarr_ref = self._zarr_arrays.store_uuid

        count = 0
        pending_rows: list[dict[str, Any]] = []
        pending_mz: list[np.ndarray] = []
        pending_intensity: list[np.ndarray] = []
        cursor = self.conn.cursor()

        for spectrum in spectra:
            if spectrum is None:
                continue

            mz_array_raw = np.asarray(spectrum.peaks.mz, dtype=np.float64)
            intensity_array_raw = np.asarray(
                spectrum.peaks.intensities, dtype=np.float64
            )

            # Triage flags are pre-calculated in processing.py and stored in
            # metadata; mirror the BLOB path's storage convention.
            triage_flags = spectrum.get("triage_flags", {})
            triage_json = json.dumps(triage_flags)
            metadata_json = _json_serialize_metadata(spectrum.metadata.copy())

            pending_rows.append(
                {
                    "original_id": spectrum.get("id"),
                    "name": spectrum.get("compound_name") or spectrum.get("name"),
                    "precursor_mz": spectrum.get("precursor_mz"),
                    "charge": spectrum.get("charge"),
                    "ionmode": spectrum.get("ionmode"),
                    "adduct": spectrum.get("adduct"),
                    "category": category,
                    "metadata": metadata_json,
                    "triage_flags": triage_json,
                    "zarr_ref": zarr_ref,
                }
            )
            pending_mz.append(mz_array_raw)
            pending_intensity.append(intensity_array_raw)
            count += 1

            if len(pending_rows) >= batch_size:
                self._flush_zarr_batch(
                    cursor,
                    insert_query,
                    columns,
                    pending_rows,
                    pending_mz,
                    pending_intensity,
                )
                pending_rows = []
                pending_mz = []
                pending_intensity = []

        if pending_rows:
            self._flush_zarr_batch(
                cursor,
                insert_query,
                columns,
                pending_rows,
                pending_mz,
                pending_intensity,
            )

        return count

    def _flush_zarr_batch(
        self,
        cursor: sqlite3.Cursor,
        insert_query: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        mz_arrays: list[np.ndarray],
        intensity_arrays: list[np.ndarray],
    ) -> None:
        """
        Append one batch of arrays to Zarr and persist the metadata rows.

        Parameters
        ----------
        cursor : sqlite3.Cursor
            Active SQLite cursor used for the batch INSERT.
        insert_query : str
            Parameterized INSERT statement with one placeholder per column.
        columns : list[str]
            Column list matching *insert_query*.
        rows : list[dict[str, Any]]
            Per-spectrum metadata dictionaries (already serialized).
        mz_arrays : list[np.ndarray]
            Float64 fragment m/z vectors.
        intensity_arrays : list[np.ndarray]
            Float64 fragment intensity vectors.

        Returns
        -------
        None

        Examples
        --------
        >>> db._flush_zarr_batch(cursor, query, columns, rows, mz, intensities)
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        if self._zarr_arrays is None:
            raise RuntimeError("Zarr store is not attached (hybrid mode off).")

        start_index = self._zarr_arrays.append_spectra(mz_arrays, intensity_arrays)

        values: list[tuple[Any, ...]] = []
        for offset, row in enumerate(rows):
            row["zarr_index"] = start_index + offset
            values.append(tuple(row.get(column) for column in columns))

        cursor.executemany(insert_query, values)
        self.conn.commit()

    def get_spectra(
        self,
        category: Optional[str] = None,
        name_pattern: Optional[str] = None,
    ) -> Iterator[Spectrum]:
        """
        Retrieve spectra from the database matching specific criteria.

        Parameters
        ----------
        category : str or None, optional
            Optional category filter.
        name_pattern : str or None, optional
            Optional SQL ``LIKE`` pattern for spectrum names.

        Yields
        ------
        Spectrum
            Reconstructed ``matchms.Spectrum`` objects.

        Raises
        ------
        ConnectionError
            If the database connection is not active.

        Examples
        --------
        >>> spectra = list(db.get_spectra(category="library"))
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")

        cursor = self.conn.cursor()
        query = "SELECT * FROM spectra"
        conditions: list[str] = []
        params: list[Any] = []

        if category:
            conditions.append("category = ?")
            params.append(category)

        if name_pattern:
            conditions.append("name LIKE ?")
            params.append(name_pattern)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)

        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield self._row_to_spectrum(row)

    def _row_to_spectrum(self, row: sqlite3.Row) -> Spectrum:
        """
        Convert a SQLite row into a ``matchms.Spectrum``.

        Rows that reference the Zarr store (``zarr_index`` is not NULL) are
        read from the attached :class:`ZarrPeakArrayStore`; rows that still
        carry BLOBs fall back to the BLOB read path (pre-migration rows).

        Parameters
        ----------
        row : sqlite3.Row
            Row fetched from the ``spectra`` table.

        Returns
        -------
        Spectrum
            Reconstructed spectrum object.

        Raises
        ------
        RuntimeError
            If the row is Zarr-referenced but no Zarr store is attached, or
            if the row has neither BLOBs nor a Zarr reference.

        Examples
        --------
        >>> spectrum = db._row_to_spectrum(row)
        """
        metadata = json.loads(row["metadata"])
        if "triage_flags" in row.keys() and row["triage_flags"]:
            triage = json.loads(row["triage_flags"])
            metadata.update(triage)

        if "zarr_index" in row.keys() and row["zarr_index"] is not None:
            if self._zarr_arrays is None:
                raise RuntimeError(
                    "This row stores its peak arrays in a Zarr store, but this "
                    "SpectralDatabase was opened without one. Reopen with "
                    "SpectralDatabase(db_path, zarr_path=<path to .zarr store>)."
                )
            mz_array, intensity_array = self._zarr_arrays.read_spectrum(
                int(row["zarr_index"])
            )
            return Spectrum(
                mz=mz_array,
                intensities=intensity_array,
                metadata=metadata,
            )

        mz_blob = row["mz_array"]
        intensity_blob = row["intensity_array"]
        if mz_blob is None or intensity_blob is None:
            raise RuntimeError(
                "Spectrum row has neither BLOB peak arrays nor a Zarr reference."
            )
        mz_array = np.frombuffer(mz_blob, dtype=np.float64).copy()
        intensity_array = np.frombuffer(intensity_blob, dtype=np.float64).copy()
        return Spectrum(mz=mz_array, intensities=intensity_array, metadata=metadata)

    def get_total_spectra_count(self) -> int:
        """
        Get the total number of spectra stored in the database.

        Returns
        -------
        int
            Total spectrum count.

        Raises
        ------
        ConnectionError
            If the database connection is not active.

        Examples
        --------
        >>> total = db.get_total_spectra_count()
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM spectra")
        return int(cursor.fetchone()[0])

    def get_category_counts(self) -> dict[str, int]:
        """
        Get the number of spectra per category.

        Returns
        -------
        dict[str, int]
            Mapping of category name to count.

        Raises
        ------
        ConnectionError
            If the database connection is not active.

        Examples
        --------
        >>> counts = db.get_category_counts()
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        cursor = self.conn.cursor()
        cursor.execute("SELECT category, COUNT(*) FROM spectra GROUP BY category")
        return {str(row[0]): int(row[1]) for row in cursor.fetchall()}

    def get_precursor_mz_range(self) -> tuple[float, float]:
        """
        Get the minimum and maximum precursor m/z values stored in the database.

        Returns
        -------
        tuple[float, float]
            Tuple of ``(min_mz, max_mz)``.

        Raises
        ------
        ConnectionError
            If the database connection is not active.

        Examples
        --------
        >>> mz_min, mz_max = db.get_precursor_mz_range()
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        cursor = self.conn.cursor()
        cursor.execute("SELECT MIN(precursor_mz), MAX(precursor_mz) FROM spectra")
        row = cursor.fetchone()
        return (
            float(row[0]) if row[0] is not None else 0.0,
            float(row[1]) if row[1] is not None else 0.0,
        )

    # Metadata-field mapping shared by :meth:`metadata_query` and the
    # per-row reconstruction path.  Mirrors the Zarr backend's field set so
    # both backends expose the SAME metadata contract.
    _METADATA_FIELD_TO_COLUMN: dict[str, str] = {
        "id": "original_id",
        "name": "name",
        "precursor_mz": "precursor_mz",
        "charge": "charge",
        "ionmode": "ionmode",
        "adduct": "adduct",
        "category": "category",
        "extra_metadata": "metadata",
    }

    def metadata_query(
        self,
        fields: list[str],
        indices: Optional[np.ndarray] = None,
        category: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """Batch-read metadata fields as flat numpy arrays.

        Implements the unified :class:`MassFlow.storage.SpectralStore`
        metadata-lookup contract (identical semantics to the Zarr backend):
        ``indices`` are positional spectrum indices in iteration order
        (``ORDER BY id``), and unknown fields raise ``ValueError``.

        Parameters
        ----------
        fields : list of str
            Metadata field names to retrieve.
        indices : np.ndarray or None
            Positional spectrum indices (``int64``).  ``None`` means all.
        category : str or None
            If provided, only return rows whose ``category`` matches.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of field name to a numpy array aligned by index.

        Raises
        ------
        ValueError
            If any requested field is not present in the store.
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        available = sorted(self._METADATA_FIELD_TO_COLUMN)
        for field in fields:
            if field not in self._METADATA_FIELD_TO_COLUMN:
                raise ValueError(
                    f"Unknown metadata field: '{field}'. Available: {available}"
                )

        if self.get_total_spectra_count() == 0:
            return {f: np.array([], dtype=np.float64) for f in fields}

        columns = [self._METADATA_FIELD_TO_COLUMN[f] for f in fields]
        sql = "SELECT " + ", ".join(columns) + " FROM spectra"
        params: list[Any] = []
        if category is not None:
            sql += " WHERE category = ?"
            params.append(category)
        sql += " ORDER BY id"

        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()

        result: dict[str, np.ndarray] = {}
        for position, field in enumerate(fields):
            values = [row[position] for row in rows]
            if field == "precursor_mz":
                result[field] = np.asarray(
                    [float(v) if v is not None else np.nan for v in values],
                    dtype=np.float64,
                )
            elif field == "charge":
                result[field] = np.asarray(
                    [int(v) if v is not None else 0 for v in values], dtype=np.int64
                )
            else:
                result[field] = np.asarray(
                    ["" if v is None else str(v) for v in values], dtype=object
                )

        if indices is not None:
            idx_arr = np.asarray(indices, dtype=np.int64)
            result = {f: arr[idx_arr] for f, arr in result.items()}
        return result

    def backend_provenance(self) -> dict[str, Any]:
        """Describe this backend's identity for run provenance.

        Returns
        -------
        dict
            ``{"backend": "sqlite" | "hybrid", "path": str,
            "spectrum_count": int}``.
        """
        backend = "hybrid" if self._zarr_path is not None else "sqlite"
        return {
            "backend": backend,
            "path": str(self.store_path),
            "spectrum_count": self.get_total_spectra_count(),
        }

    def get_spectrum_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """
        Retrieve a single spectrum by its unique identifier (``original_id``).

        Parameters
        ----------
        spectrum_id : str
            The ``original_id`` of the desired spectrum.

        Returns
        -------
        Spectrum or None
            The matching spectrum, or ``None`` if not found.
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM spectra WHERE original_id = ? LIMIT 1",
            (spectrum_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_spectrum(row)

    def get_triage_profile(self, spectrum_id: str) -> Optional[TriageProfile]:
        """Retrieve the structured :class:`TriageProfile` for a spectrum.

        Parses the ``triage_flags`` JSON column directly from the database row
        without materialising a full ``matchms.Spectrum``. This is suitable for
        bulk pre-filtering or batch classification where peak arrays are not
        yet needed.

        Parameters
        ----------
        spectrum_id : str
            The ``original_id`` of the spectrum.

        Returns
        -------
        TriageProfile or None
            Populated profile, or ``None`` if the spectrum is not found.
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT triage_flags FROM spectra WHERE original_id = ? LIMIT 1",
            (spectrum_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        raw = row["triage_flags"]
        if raw and isinstance(raw, str) and raw.strip():
            try:
                flags_dict = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                flags_dict = {}
        else:
            flags_dict = {}

        return TriageProfile(**flags_dict)

    def merge_from_sqlite(
        self,
        source_db_path: Path,
        category: str = "merged",
    ) -> int:
        """
        Fast-path merge: bulk INSERT from another SQLite database via ATTACH.

        Bypasses the row-by-row ``get_spectra()`` / ``add_spectra()`` iteration
        loop by using SQLite's ``ATTACH DATABASE`` to perform a single
        ``INSERT INTO ... SELECT FROM`` operation. This is orders of magnitude
        faster for merging multi-gigabyte ``.db`` files.

        The source ``id`` (auto-increment) column is intentionally excluded;
        the target database assigns fresh ``id`` values to avoid conflicts.
        Column mismatches between source and target are handled gracefully:
        missing columns in the source are filled with sensible defaults,
        and extra columns in the source are ignored.

        Parameters
        ----------
        source_db_path : Path
            File-system path to the source SQLite ``.db`` file.
        category : str, optional
            Category label applied to all imported spectra rows.

        Returns
        -------
        int
            Number of spectra rows inserted into the target database.

        Raises
        ------
        ConnectionError
            If the database connection is not active.
        sqlite3.Error
            If the source database cannot be attached or queried.

        Notes
        -----
        This method is only available for the SQLite backend. Cross-backend
        merges (e.g. Zarr → SQLite) should use the iterator-based
        ``get_spectra()`` / ``add_spectra()`` fallback path.

        Examples
        --------
        >>> target = SpectralDatabase("merged.db")
        >>> count = target.merge_from_sqlite(Path("source.db"), category="merged")
        >>> target.close()
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")

        if self._zarr_arrays is not None:
            raise RuntimeError(
                "merge_from_sqlite cannot be used when the target database "
                "stores peak arrays in a Zarr store. Use the iterator-based "
                "merge (get_spectra()/add_spectra()) instead."
            )

        source_abs = str(source_db_path.resolve())
        cursor = self.conn.cursor()

        # Determine columns present in the source (attached) database.
        cursor.execute(f"ATTACH DATABASE '{source_abs}' AS _merge_source")

        try:
            cursor.execute("PRAGMA _merge_source.table_info(spectra)")
            source_cols = [str(row[1]) for row in cursor.fetchall()]

            # A source whose rows reference a Zarr store cannot be merged via
            # the bulk BLOB copy — fall back to the iterator-based merge.
            if "zarr_index" in source_cols:
                cursor.execute(
                    "SELECT COUNT(*) FROM _merge_source.spectra "
                    "WHERE zarr_index IS NOT NULL"
                )
                if int(cursor.fetchone()[0]) > 0:
                    raise RuntimeError(
                        "Source database stores peak arrays in a Zarr store; "
                        "use the iterator-based merge instead."
                    )

            target_cols = get_spectra_table_columns(self.conn)

            # Build the column list for the INSERT.
            # - Exclude 'id' (auto-increment; target assigns fresh values).
            # - Include all other target columns that also exist in the source.
            # - Replace 'category' with the literal value provided by the caller.
            # - COALESCE 'triage_flags' to '{}' when the source column is NULL.
            common_cols = [c for c in target_cols if c in source_cols and c != "id"]

            if not common_cols:
                cursor.execute("DETACH DATABASE _merge_source")
                return 0

            select_parts: list[str] = []
            for col in common_cols:
                if col == "category":
                    # Use the caller-supplied category literal.
                    escaped = category.replace("'", "''")
                    select_parts.append(f"'{escaped}'")
                elif col == "triage_flags":
                    select_parts.append(
                        "COALESCE(_merge_source.spectra.triage_flags, '{}')"
                    )
                else:
                    select_parts.append(f"_merge_source.spectra.{col}")

            insert_cols = ", ".join(common_cols)
            select_expr = ", ".join(select_parts)

            query = (
                f"INSERT INTO spectra ({insert_cols}) "
                f"SELECT {select_expr} "
                f"FROM _merge_source.spectra"
            )

            cursor.execute(query)
            self.conn.commit()

            return cursor.rowcount
        finally:
            # Always detach, even if an error occurred during the merge.
            try:
                cursor.execute("DETACH DATABASE _merge_source")
            except sqlite3.Error:
                pass

    def batch_get_arrays(
        self,
        spectrum_ids: Optional[list[str]] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Retrieve m/z and intensity arrays in batch.

        For the SQLite BLOB backend this iterates over all (or the requested)
        spectra and deserialises each BLOB. The Zarr backend is significantly
        faster for this operation due to zero-copy chunk reads.

        Parameters
        ----------
        spectrum_ids : list of str or None
            Specific spectrum IDs to retrieve. If ``None``, all spectra are
            returned.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            Aligned lists of ``float64`` m/z and intensity arrays.
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")

        if self._zarr_arrays is not None:
            return self._batch_get_arrays_zarr(spectrum_ids)

        mz_arrays: list[np.ndarray] = []
        intensity_arrays: list[np.ndarray] = []

        cursor = self.conn.cursor()
        if spectrum_ids is not None:
            placeholders = ",".join(["?"] * len(spectrum_ids))
            cursor.execute(
                f"SELECT mz_array, intensity_array FROM spectra WHERE original_id IN ({placeholders})",
                spectrum_ids,
            )
        else:
            cursor.execute("SELECT mz_array, intensity_array FROM spectra")

        for row in cursor.fetchall():
            mz_arrays.append(np.frombuffer(row["mz_array"], dtype=np.float64).copy())
            intensity_arrays.append(
                np.frombuffer(row["intensity_array"], dtype=np.float64).copy()
            )

        return mz_arrays, intensity_arrays

    def _batch_get_arrays_zarr(
        self,
        spectrum_ids: Optional[list[str]] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Batch-read peak arrays in hybrid mode.

        Zarr-referenced rows are read in one bulk call against the array
        store; any remaining pre-migration BLOB rows are deserialized
        individually. The returned lists follow the database row order
        (``ORDER BY id``).

        Parameters
        ----------
        spectrum_ids : list of str or None
            Specific spectrum IDs to retrieve. If ``None``, all spectra are
            returned.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            Aligned lists of ``float64`` m/z and intensity arrays.

        Examples
        --------
        >>> mz_arrays, intensity_arrays = db._batch_get_arrays_zarr()
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")
        if self._zarr_arrays is None:
            raise RuntimeError("Zarr store is not attached (hybrid mode off).")

        cursor = self.conn.cursor()
        if spectrum_ids is not None:
            placeholders = ",".join(["?"] * len(spectrum_ids))
            cursor.execute(
                f"SELECT * FROM spectra WHERE original_id IN ({placeholders}) "
                "ORDER BY id",
                spectrum_ids,
            )
            rows_by_id: dict[str, sqlite3.Row] = {
                str(row["original_id"]): row for row in cursor.fetchall()
            }
            # Preserve the requested order and skip unknown ids, matching
            # ZarrSpectralStore.batch_get_arrays semantics.
            ordered_rows = [
                rows_by_id[spectrum_id]
                for spectrum_id in spectrum_ids
                if spectrum_id in rows_by_id
            ]
        else:
            cursor.execute("SELECT * FROM spectra ORDER BY id")
            ordered_rows = cursor.fetchall()

        # Collect row order and classify each row as Zarr- or BLOB-backed.
        order: list[tuple[Any, ...]] = []
        zarr_indices: list[int] = []
        for row in ordered_rows:
            zarr_index = row["zarr_index"] if "zarr_index" in row.keys() else None
            if zarr_index is not None:
                order.append(("zarr", int(zarr_index)))
                zarr_indices.append(int(zarr_index))
                continue

            mz_blob = row["mz_array"] if "mz_array" in row.keys() else None
            intensity_blob = (
                row["intensity_array"] if "intensity_array" in row.keys() else None
            )
            if mz_blob is None or intensity_blob is None:
                raise RuntimeError(
                    "Spectrum row has neither BLOB peak arrays nor a Zarr reference."
                )
            order.append(("blob", mz_blob, intensity_blob))

        zarr_lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if zarr_indices:
            zarr_mz, zarr_intensity = self._zarr_arrays.read_spectra(zarr_indices)
            zarr_lookup = {
                index: (mz_arr, intensity_arr)
                for index, mz_arr, intensity_arr in zip(
                    zarr_indices, zarr_mz, zarr_intensity
                )
            }

        mz_arrays: list[np.ndarray] = []
        intensity_arrays: list[np.ndarray] = []
        for entry in order:
            if entry[0] == "zarr":
                zarr_mz_arr, zarr_intensity_arr = zarr_lookup[int(entry[1])]
            else:
                zarr_mz_arr = np.frombuffer(entry[1], dtype=np.float64).copy()
                zarr_intensity_arr = np.frombuffer(entry[2], dtype=np.float64).copy()
            mz_arrays.append(zarr_mz_arr)
            intensity_arrays.append(zarr_intensity_arr)

        return mz_arrays, intensity_arrays

    def close(self) -> None:
        """
        Close the database connection and any attached Zarr store.

        Returns
        -------
        None

        Examples
        --------
        >>> db.close()
        """
        if self._zarr_arrays is not None:
            self._zarr_arrays.close()
            self._zarr_arrays = None
        if self.conn:
            self.conn.close()
            self.conn = None


__all__ = [
    "CURRENT_SPECTRA_COLUMNS",
    "LEGACY_PEAKS_COLUMN",
    "LegacyDatabaseSchemaError",
    "SpectralDatabase",
    "create_current_spectra_table",
    "create_legacy_backup_table",
    "create_migrated_spectra_table",
    "get_spectra_table_columns",
    "has_table",
    "is_current_spectra_schema",
    "is_legacy_spectra_schema",
    "legacy_migration_error_message",
    "migrate_legacy_peaks_database",
    "migrate_legacy_peaks_to_arrays",
]
