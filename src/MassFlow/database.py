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

Current schema version
----------------------
The active ``spectra`` table stores:

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
"""

from __future__ import annotations

import ast
import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from matchms import Spectrum

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
}
LEGACY_PEAKS_COLUMN = "peaks"


class LegacyDatabaseSchemaError(RuntimeError):
    """
    Raised when a database uses the legacy ``peaks`` schema.

    Parameters
    ----------
    message : str
        Human-readable error message describing the migration requirement.

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
    return bool(columns) and CURRENT_SPECTRA_COLUMNS.issubset(columns)


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


def create_current_spectra_table(connection: sqlite3.Connection) -> None:
    """
    Create the current ``spectra`` table if it does not exist.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection.

    Returns
    -------
    None

    Examples
    --------
    >>> create_current_spectra_table(connection)
    """
    cursor = connection.cursor()
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
            intensity_array BLOB
        )
        """
    )
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
            intensity_array BLOB
        )
        """
    )


def _fetch_legacy_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
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
    return list(cursor.fetchall())


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
                intensity_array
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        migrated_sample_checks: list[dict[str, Any]] = []
        for row_index, row in enumerate(legacy_rows):
            mz_array, intensity_array = _decode_legacy_peaks_payload(row["peaks"])
            _validate_decoded_legacy_arrays(mz_array, intensity_array)
            mz_blob, intensity_blob = _serialize_peak_arrays(mz_array, intensity_array)

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


class SpectralDatabase:
    """
    Manage a local SQLite database for persistent storage of mass spectra.

    Parameters
    ----------
    db_path : str or Path
        File system path to the SQLite database.
    allow_destructive_upgrade : bool, optional
        Legacy compatibility flag. If True and a legacy schema is detected,
        initialization raises a ``RuntimeError`` describing the required
        explicit migration path. This flag exists only to make destructive
        upgrade intent explicit and is not used to perform destructive changes.

    Returns
    -------
    None

    Examples
    --------
    >>> db = SpectralDatabase("library.db")
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        allow_destructive_upgrade: bool = False,
    ):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self.allow_destructive_upgrade = allow_destructive_upgrade
        self._connect()
        self._initialize_tables()

    def _connect(self) -> None:
        """
        Establish a connection to the SQLite database.

        Parameters
        ----------
        None

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

        Parameters
        ----------
        None

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

        create_current_spectra_table(self.conn)

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

        count = 0
        batch: list[tuple[Any, ...]] = []
        cursor = self.conn.cursor()

        insert_query = """
            INSERT INTO spectra (
                original_id, name, precursor_mz, charge, ionmode, adduct, category, metadata, mz_array, intensity_array
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for spectrum in spectra:
            if spectrum is None:
                continue

            mz_blob, intensity_blob = _serialize_peak_arrays(
                np.asarray(spectrum.peaks.mz, dtype=np.float64),
                np.asarray(spectrum.peaks.intensities, dtype=np.float64),
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

        Parameters
        ----------
        row : sqlite3.Row
            Row fetched from the ``spectra`` table.

        Returns
        -------
        Spectrum
            Reconstructed spectrum object.

        Examples
        --------
        >>> spectrum = db._row_to_spectrum(row)
        """
        metadata = json.loads(row["metadata"])
        mz_array = np.frombuffer(row["mz_array"], dtype=np.float64).copy()
        intensity_array = np.frombuffer(row["intensity_array"], dtype=np.float64).copy()
        return Spectrum(mz=mz_array, intensities=intensity_array, metadata=metadata)

    def get_total_spectra_count(self) -> int:
        """
        Get the total number of spectra stored in the database.

        Parameters
        ----------
        None

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

        Parameters
        ----------
        None

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

        Parameters
        ----------
        None

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

    def close(self) -> None:
        """
        Close the database connection.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> db.close()
        """
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
