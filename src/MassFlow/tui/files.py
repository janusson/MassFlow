"""
Spectral file discovery, classification, and workspace upload for the console.

The "find" and "upload" verbs of the TUI live here:

- :func:`discover_spectral_files` finds spectral files under a directory.
- :func:`classify_file` labels each file so the UI can explain *why* a file
  is selectable (or, for vendor formats, why it is not loadable).
- :func:`copy_into_workspace` implements "upload" as a local copy with
  collision-safe naming (MassFlow is local-first; nothing leaves the disk).

This module does not parse spectral data — it only reasons about paths,
extensions, and file sizes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Vendor extensions MassFlow refuses to load directly. This mirrors
# ``MassFlow.io.PROPRIETARY_FORMATS`` but is duplicated here so importing the
# console does not drag in the heavy core stack (matchms/pytecopics);
# ``tests/test_tui_files.py`` asserts the two sets stay in sync.
VENDOR_EXTENSIONS = frozenset({".raw", ".d", ".wiff", ".lcd", ".t2d", ".baf"})

# Open formats the core loader accepts as query spectra.
QUERY_EXTENSIONS = frozenset({".mzml", ".mzxml", ".mgf", ".msp"})

# Text formats usable as reference libraries.
LIBRARY_TEXT_EXTENSIONS = frozenset({".msp", ".mgf"})

# MassFlow-native databases (SQLite files; Zarr is a directory).
DATABASE_EXTENSIONS = frozenset({".db", ".sqlite"})

# Marker file present inside Zarr directories.
_ZARR_MARKER = ".zgroup"

# Hidden entries (dotfiles/dot-directories) are skipped during discovery.
_SKIPPED_DIRECTORY_NAMES = frozenset(
    {".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
)


@dataclass(frozen=True)
class FileEntry:
    """Classification of a single filesystem entry."""

    path: Path
    kind: str  # "query" | "library" | "database" | "vendor" | "unsupported"
    format_hint: str | None
    size_bytes: int

    @property
    def display_name(self) -> str:
        return self.path.name


def classify_file(path: Path) -> FileEntry:
    """Classify a filesystem entry for the browser.

    Directories that look like Zarr stores (contain ``.zgroup``) are
    classified as databases; all other directories are ``unsupported`` and
    are handled by the directory-navigation logic of the UI instead.
    """
    path = Path(path)
    if path.is_dir():
        if (path / _ZARR_MARKER).exists():
            return FileEntry(
                path=path,
                kind="database",
                format_hint="zarr",
                size_bytes=_directory_size(path),
            )
        if path.suffix.lower() in VENDOR_EXTENSIONS:
            return FileEntry(
                path=path,
                kind="vendor",
                format_hint=path.suffix.lower().lstrip("."),
                size_bytes=_directory_size(path),
            )
        return FileEntry(path=path, kind="unsupported", format_hint=None, size_bytes=0)

    suffix = path.suffix.lower()
    size_bytes = path.stat().st_size if path.exists() else 0

    if suffix in VENDOR_EXTENSIONS:
        return FileEntry(
            path=path,
            kind="vendor",
            format_hint=suffix.lstrip("."),
            size_bytes=size_bytes,
        )
    if suffix in DATABASE_EXTENSIONS:
        return FileEntry(
            path=path,
            kind="database",
            format_hint=suffix.lstrip("."),
            size_bytes=size_bytes,
        )
    if suffix in LIBRARY_TEXT_EXTENSIONS:
        return FileEntry(
            path=path,
            kind="library",
            format_hint=suffix.lstrip("."),
            size_bytes=size_bytes,
        )
    if suffix in QUERY_EXTENSIONS:
        return FileEntry(
            path=path,
            kind="query",
            format_hint=suffix.lstrip("."),
            size_bytes=size_bytes,
        )
    return FileEntry(
        path=path,
        kind="unsupported",
        format_hint=suffix.lstrip(".") or None,
        size_bytes=size_bytes,
    )


def discover_spectral_files(
    root: Path,
    *,
    max_depth: Optional[int] = None,
    include_hidden: bool = False,
) -> list[FileEntry]:
    """Discover spectral files under ``root``, breadth-first.

    Only *relevant* entries are returned: directories (for navigation), open
    spectral formats, MassFlow databases, and vendor formats. Vendor files are
    included deliberately so the console can explain that they must be
    converted before loading instead of silently hiding them.

    Parameters
    ----------
    root : Path
        Directory to scan.
    max_depth : int, optional
        Maximum traversal depth (``None`` = unlimited).
    include_hidden : bool
        When False (default) hidden files/directories are skipped.

    Returns
    -------
    list[FileEntry]
        Entries sorted by kind-group then name: directories first, then
        supported files, then vendor files, then unsupported files.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    entries: list[FileEntry] = []
    # Stack of (path, depth) for an explicit iterative walk.
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue

        for child in children:
            if not include_hidden and child.name.startswith("."):
                continue
            if child.is_dir():
                if child.name in _SKIPPED_DIRECTORY_NAMES:
                    continue
                entry = classify_file(child)
                if entry.kind == "database" or entry.kind == "vendor":
                    entries.append(entry)
                elif max_depth is None or depth < max_depth:
                    stack.append((child, depth + 1))
                continue
            entry = classify_file(child)
            if entry.kind != "unsupported":
                entries.append(entry)

    return sorted(entries, key=_sort_key)


def _sort_key(entry: FileEntry) -> tuple[int, str]:
    order = {
        "database": 0,
        "library": 1,
        "query": 2,
        "vendor": 3,
        "unsupported": 4,
    }
    return order.get(entry.kind, 9), entry.path.name.lower()


def copy_into_workspace(source: Path, workspace: Path) -> Path:
    """Copy ``source`` into the workspace directory (the "upload" verb).

    The copy is collision-safe: if ``<name>`` already exists the file is
    written as ``<stem>_2.<suffix>``, ``<stem>_3.<suffix>``, ... so an upload
    can never overwrite a previous one. The workspace directory is created on
    demand.

    Parameters
    ----------
    source : Path
        File to copy.
    workspace : Path
        Destination directory.

    Returns
    -------
    Path
        The path of the copied file inside ``workspace``.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    OSError
        If the copy fails (permissions, disk full, ...).
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Upload source does not exist: {source}")
    workspace.mkdir(parents=True, exist_ok=True)

    destination = workspace / source.name
    counter = 2
    while destination.exists():
        destination = workspace / f"{source.stem}_{counter}{source.suffix}"
        counter += 1

    # shutil.copy2 preserves mtime metadata, which matters for provenance.
    import shutil

    shutil.copy2(source, destination)
    return destination


def human_size(num_bytes: Optional[int]) -> str:
    """Format a byte count for humans (``1.5 MB``); ``n/a`` when unknown."""
    if num_bytes is None or num_bytes < 0:
        return "n/a"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def guess_backend(path: Path) -> str:
    """Infer the storage backend of a library path.

    Returns one of ``"zarr"``, ``"hybrid"``, ``"sqlite"``, or ``"text"``.
    """
    path = Path(path)
    if path.is_dir() and (path / _ZARR_MARKER).exists():
        return "zarr"
    if path.suffix.lower() == ".zarr":
        return "zarr"
    if path.suffix.lower() in DATABASE_EXTENSIONS:
        if path.with_suffix(".zarr").is_dir():
            return "hybrid"
        return "sqlite"
    return "text"


def _directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def supported_query_extensions() -> Iterable[str]:
    """Human-readable list of supported query extensions."""
    return sorted(QUERY_EXTENSIONS)
