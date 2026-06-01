"""
Executable migration wrapper for upgrading legacy MassFlow SQLite databases.

This script delegates the actual migration logic to helpers implemented in
``MassFlow.database`` so that SQL remains centralized in the database module.

The migration upgrades legacy databases that still store peak data in a single
``peaks`` column to the current schema that stores:

- ``mz_array``
- ``intensity_array``

as ``float64``-encoded BLOBs.

Usage
-----
Run the script from the repository root, for example:

    python scripts/migrations/0001_peaks_to_arrays.py --input path/to/library.db

Optionally create an external file backup before migrating:

    python scripts/migrations/0001_peaks_to_arrays.py \
        --input path/to/library.db \
        --backup-dir path/to/backups
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from MassFlow.database import migrate_legacy_peaks_to_arrays


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
        prog="0001_peaks_to_arrays.py",
        description=(
            "Migrate a legacy MassFlow SQLite database from the old "
            "'peaks' schema to the current mz_array/intensity_array schema."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the SQLite database file to migrate.",
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
    backup_path = (
        backup_dir / f"{database_path.stem}.pre_migration{database_path.suffix}"
    )
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

    if status == "already_current":
        print("Database already uses the current schema. No changes were made.")
        print(f"Database: {summary.get('database')}")
        return

    print("Migration completed successfully.")
    print(f"Database: {summary.get('database')}")
    print(f"Status: {status}")
    print(f"Backup table: {summary.get('backup_table')}")
    print(f"Legacy row count: {summary.get('legacy_row_count')}")
    print(f"Migrated row count: {summary.get('migrated_row_count')}")

    sample_checks = summary.get("sample_checks")
    if isinstance(sample_checks, list) and sample_checks:
        print("Sample validation rows:")
        for sample in sample_checks:
            if isinstance(sample, dict):
                print(
                    "  - "
                    f"row_index={sample.get('row_index')}, "
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
        print(f"Error: input database does not exist: {database_path}", file=sys.stderr)
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
        summary = migrate_legacy_peaks_to_arrays(database_path)
    except Exception as error:
        print(
            "Migration failed. The database transaction should have been rolled back.",
            file=sys.stderr,
        )
        print(f"Error: {error}", file=sys.stderr)
        print(
            "Review the database state, keep the backup table created during the "
            "migration attempt if present, and rerun the migration once the issue "
            "is resolved.",
            file=sys.stderr,
        )
        return 1

    print_migration_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
