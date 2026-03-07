"""
Spectral Database Management for MassFlow.

This module implements a SQLite-based storage backend for mass spectral data.
It provides the ``SpectralDatabase`` class to efficiently handle the insertion,
retrieval, and management of large collections of spectra, including their
associated metadata and peak lists. This is particularly useful for managing
reference libraries or caching processed results.
"""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from matchms import Spectrum


class SpectralDatabase:
    """
    SQLite-based database for mass spectra.

    This class provides an interface to a local SQLite database designed to store
    mass spectral data. It handles the serialization of spectral peaks (m/z and
    intensity arrays) and metadata into database tables, allowing for persistent
    storage and efficient retrieval of large spectral libraries.
    """

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the spectral database connection.

        Parameters
        ----------
        db_path : str or Path
            The file system path to the SQLite database file. If the file does
            not exist, it (and any necessary parent directories) will be created.
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._initialize_tables()

    def _connect(self) -> None:
        """
        Establish connection to the SQLite database.

        This method creates the database file and its parent directories if they
        do not exist, and sets up the SQLite connection with a row factory for
        dictionary-like access to query results.

        Returns
        -------
        None
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def _initialize_tables(self) -> None:
        """
        Create necessary database tables if they do not exist.

        This method defines the schema for the 'spectra' table, which includes
        fields for standard metadata (ID, name, precursor m/z, charge, etc.)
        and binary blobs for storing complex peak data and extended metadata.

        Returns
        -------
        None

        Raises
        ------
        ConnectionError
            If the database connection has not been established.
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")

        cursor = self.conn.cursor()

        # Check for legacy schema and drop if necessary to upgrade to v1.0
        cursor.execute("PRAGMA table_info(spectra)")
        columns = [info[1] for info in cursor.fetchall()]
        if columns and "peaks" in columns:
            cursor.execute("DROP TABLE spectra")

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
        self.conn.commit()

    def add_spectra(
        self,
        spectra: Iterator[Spectrum],
        category: str = "default",
        batch_size: int = 5000,
    ) -> int:
        """
        Add multiple spectra to the database using batch insertion.

        This method iterates over a collection of ``matchms.Spectrum`` objects,
        serializes their peak data (m/z and intensities) and metadata, and
        inserts them into the database. It uses transactions to commit records
        in batches, improving performance for large datasets.

        Parameters
        ----------
        spectra : Iterator[Spectrum]
            An iterator yielding ``matchms.Spectrum`` objects to be inserted.
        category : str, optional
            A tag to categorize the spectra within the database (e.g., 'reference',
            'experimental'). Default is 'default'.
        batch_size : int, optional
            The number of records to accumulate before committing a transaction
            to the database. Default is 5000.

        Returns
        -------
        int
            The total number of spectra successfully added to the database.

        Raises
        ------
        ConnectionError
            If the database connection is not active.
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

            # Serialize peaks directly using raw bytes
            mz_blob = spectrum.peaks.mz.astype(np.float64).tobytes()
            intensity_blob = spectrum.peaks.intensities.astype(np.float64).tobytes()

            # Serialize metadata
            metadata_dict = spectrum.metadata.copy()
            metadata_json = json.dumps(metadata_dict, default=str)

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

        # Commit remaining
        if batch:
            cursor.executemany(insert_query, batch)
            self.conn.commit()

        return count

    def get_spectra(
        self, category: Optional[str] = None, name_pattern: Optional[str] = None
    ) -> Iterator[Spectrum]:
        """
        Retrieve spectra from the database matching specific criteria.

        This method queries the database for spectra records, optionally filtering
        by category or name pattern (using SQL LIKE syntax). It deserializes the
        retrieved data back into ``matchms.Spectrum`` objects, yielding them one
        by one to minimize memory usage.

        Parameters
        ----------
        category : str or None, optional
            If provided, only retrieve spectra belonging to this category.
        name_pattern : str or None, optional
            If provided, filter spectra whose names match this SQL LIKE pattern
            (e.g., 'Caffeine%').

        Yields
        ------
        matchms.Spectrum
            Reconstructed spectrum objects containing metadata and peak data.

        Raises
        ------
        ConnectionError
            If the database connection is not active.
        """
        if not self.conn:
            raise ConnectionError("Database not connected.")

        cursor = self.conn.cursor()
        query = "SELECT * FROM spectra"
        conditions = []
        params = []

        if category:
            conditions.append("category = ?")
            params.append(category)

        if name_pattern:
            conditions.append("name LIKE ?")
            params.append(name_pattern)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)

        # Yield rows one by one to save memory
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield self._row_to_spectrum(row)

    def _row_to_spectrum(self, row: sqlite3.Row) -> Spectrum:
        """
        Convert a database row into a matchms Spectrum object.

        This internal helper method handles the deserialization of the JSON
        metadata and binary peak data stored in a database row.

        Parameters
        ----------
        row : sqlite3.Row
            A single row fetched from the 'spectra' table.

        Returns
        -------
        matchms.Spectrum
            The reconstructed spectrum object.
        """
        metadata = json.loads(row["metadata"])

        # Use .copy() to ensure the arrays are writable, as np.frombuffer is read-only
        mz = np.frombuffer(row["mz_array"], dtype=np.float64).copy()
        intensities = np.frombuffer(row["intensity_array"], dtype=np.float64).copy()

        return Spectrum(mz=mz, intensities=intensities, metadata=metadata)

    def close(self) -> None:
        """
        Close the database connection.

        Ensures that any open SQLite connection is properly closed and the
        connection object is reset to None.

        Returns
        -------
        None
        """
        if self.conn:
            self.conn.close()
            self.conn = None
