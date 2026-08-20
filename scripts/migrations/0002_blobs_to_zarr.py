"""
Executable migration wrapper for moving SQLite BLOB peak arrays to Zarr.

This script delegates the actual migration logic to
``MassFlow.database.migrate_blobs_to_zarr`` so that SQL remains centralized in
the database module.

The migration moves the ``float64`` fragment arrays out of the SQLite
``mz_array`` / ``intensity_array`` BLOB columns and into a chunked,
compressed Zarr store. After migration each SQLite row retains metadata plus:

- ``zarr_ref`` — the UUID of the Zarr group that owns its peak arrays.
- ``zarr_index`` — the row's index into the flat Zarr arrays.

The BLOB columns are NULLed after each batch is verified bit-for-bit against
the source data. The migration is idempotent and safe to re-run.

Usage
-----
Run the script from the repository root, for example:

    python scripts/migrations/0002_blobs_to_zarr.py --input path/to/library.db

Optionally override the Zarr store location and chunk sizes:

    python scripts/migrations/0002_blobs_to_zarr.py \\
        --input path/to/library.db \\
        --zarr-output path/to/library.zarr \\
        --peak-chunk-size 262144 \\
        --boundary-chunk-size 4096

Optionally create an external file backup before migrating:

    python scripts/migrations/0002_blobs_to_zarr.py \\
        --input path/to/library.db \\
        --backup-dir path/to/backups
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from MassFlow.database import migrate_blobs_to_zarr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the migration wrapper.

    Parameters
    ----------
    argv : list[str] or None, optional
        Argument vector to parse. If None, ``sys.argv`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="0002_blobs_to_zarr.py",
        description=(
            "Migrate a MassFlow SQLite database from BLOB peak arrays to a "
            "chunked Zarr store, leaving only metadata and a zarr_ref/"
            "zarr_index reference pair in SQLite."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the SQLite database file to migrate.",
    )
    parser.add_argument(
        "--zarr-output",
        required=False,
        default=None,
        help=(
            "Path for the Zarr array store. Defaults to the database path "
            "with a .zarr suffix (e.g. library.db -> library.zarr)."
        ),
    )
    parser.add_argument(
        "--peak-chunk-size",
        required=False,
        default=1_048_576,
        type=int,
        help=(
            "Float64 elements per chunk in the Zarr peak arrays "
            "(default: 1048576, ~8 MB per chunk)."
        ),
    )
    parser.add_argument(
        "--boundary-chunk-size",
        required=False,
        default=4096,
        type=int,
        help="Spectra per chunk in the Zarr boundaries array (default: 4096).",
    )
    parser.add_argument(
        "--keep-blobs",
        required=False,
        action="store_true",
        help=(
            "Keep the SQLite BLOB columns populated after migration "
            "(default: NULL them out)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help=(
            "Clear any existing zarr_ref/zarr_index values and rebuild the "
            "Zarr store from scratch."
        ),
    )
    parser.add_argument(
        "--backup-dir",
        required=False,
        help=(
            "Optional directory where an external file backup copy of the "
            "database should be created before migration."
        ),
    )
    return parser.parse_args(argv)


def create_external_backup(
    database_path: Path,
    backup_dir: Path,
) -> Path:
    """
    Create an external filesystem backup copy of the database.

    Parameters
    ----------
    database_path : Path
        Path to the database file being migrated.
    backup_dir : Path
        Directory where the backup copy should be written.

    Returns
    -------
    Path
        Path to the created backup file.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{database_path.stem}.pre_zarr{database_path.suffix}"
    shutil.copy2(database_path, backup_path)
    return backup_path


def print_migration_summary(summary: dict[str, object]) -> None:
    """
    Print a human-readable migration summary.

    Parameters
    ----------
    summary : dict[str, object]
        Summary dictionary returned by the database migration helper.

    Returns
    -------
    None
    """
    status = str(summary.get("status", "unknown"))

    print(f"Status: {status}")
    print(f"Database: {summary.get('database')}")
    print(f"Zarr store: {summary.get('zarr_path')}")

    if status == "already_migrated":
        print("All rows already reference the Zarr store. No changes were made.")
        return

    print(f"Zarr ref (UUID): {summary.get('zarr_ref')}")
    print(f"Migrated row count: {summary.get('migrated_row_count')}")
    print(f"Total peak count: {summary.get('total_peak_count')}")
    print(f"BLOBs nulled: {summary.get('blobs_nulled')}")

    sample_checks = summary.get("sample_checks")
    if isinstance(sample_checks, list) and sample_checks:
        print("Sample validation rows:")
        for sample in sample_checks:
            if isinstance(sample, dict):
                print(
                    "  - "
                    f"row_id={sample.get('row_id')}, "
                    f"zarr_index={sample.get('zarr_index')}, "
                    f"peak_count={sample.get('peak_count')}, "
                    f"first_mz={sample.get('first_mz')}, "
                    f"first_intensity={sample.get('first_intensity')}"
                )


def main(argv: list[str] | None = None) -> int:
    """
    Execute the migration wrapper.

    Parameters
    ----------
    argv : list[str] or None, optional
        Argument vector to parse. If None, ``sys.argv`` is used.

    Returns
    -------
    int
        Process-style exit code. Returns ``0`` on success and non-zero on
        failure.
    """
    args = parse_args(argv)
    database_path = Path(args.input)

    if not database_path.exists():
        print(
            f"Error: input database does not exist: {database_path}",
            file=sys.stderr,
        )
        return 2

    if args.backup_dir:
        backup_dir = Path(args.backup_dir)
        try:
            backup_path = create_external_backup(database_path, backup_dir)
            print(f"Created external backup copy at: {backup_path}")
        except Exception as error:
            print(f"Failed to create external backup: {error}", file=sys.stderr)
            return 2

    try:
        summary = migrate_blobs_to_zarr(
            database_path,
            zarr_path=args.zarr_output,
            peak_chunk_size=args.peak_chunk_size,
            boundary_chunk_size=args.boundary_chunk_size,
            null_blobs=not args.keep_blobs,
            overwrite=args.overwrite,
        )
    except Exception as error:
        print(
            "Migration failed. Rows that were already committed keep their "
            "zarr_ref/zarr_index references; rerun the script to continue, "
            "or rerun with --overwrite to rebuild the Zarr store from scratch.",
            file=sys.stderr,
        )
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_migration_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
