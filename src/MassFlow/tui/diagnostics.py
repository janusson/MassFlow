"""
Error UX for the MassFlow terminal console.

Mass-spectrometry pipelines fail in colourful ways: vendor formats, legacy
database schemas, missing ML extras, locked SQLite files, malformed MGF
headers. This module translates raw exceptions into :class:`Problem` reports
that a terminal user can actually act on, with:

- a **stage** (which pipeline step failed),
- a **hint** (what to do about it, in plain English),
- the original traceback, kept available but out of the way.

It also reads the core pipeline's quarantine log
(``massflow_quarantine.log``), which records every spectrum the validation
layer rejected and why — surfaced in the console's diagnostics tab.
"""

from __future__ import annotations

import datetime as _datetime
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class TuiError(Exception):
    """An error raised by the TUI pipeline bridge, carrying UX context."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "unknown",
        hint: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.hint = hint

    @classmethod
    def from_exception(
        cls, exception: BaseException, *, stage: str = "unknown"
    ) -> "TuiError":
        """Wrap an arbitrary exception, attaching a suggested fix when known."""
        message = str(exception) or exception.__class__.__name__
        return cls(message, stage=stage, hint=suggest_fix(exception))


@dataclass
class Problem:
    """A user-facing problem report."""

    stage: str
    title: str
    detail: str
    hint: Optional[str] = None
    traceback_text: str = ""
    occurred_at: str = field(
        default_factory=lambda: _datetime.datetime.now().isoformat(timespec="seconds")
    )

    @classmethod
    def from_exception(
        cls, exception: BaseException, *, stage: str = "unknown"
    ) -> "Problem":
        """Build a :class:`Problem` from an exception (with traceback)."""
        if isinstance(exception, TuiError):
            hint = exception.hint
            stage = exception.stage or stage
        else:
            hint = suggest_fix(exception)
        return cls(
            stage=stage,
            title=exception.__class__.__name__,
            detail=str(exception) or "(no message)",
            hint=hint,
            traceback_text="".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            ),
        )

    def to_text(self) -> str:
        """Render the problem as plain text (used by logs and tests)."""
        lines = [
            f"[{self.occurred_at}] {self.stage}: {self.title}",
            f"  {self.detail}",
        ]
        if self.hint:
            lines.append(f"  fix: {self.hint}")
        if self.traceback_text:
            lines.append("  traceback:")
            for tb_line in self.traceback_text.rstrip().splitlines()[-6:]:
                lines.append(f"    {tb_line}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hints: exception → plain-English remediation
# ---------------------------------------------------------------------------

_ML_INSTALL_RE = re.compile(r"pip install (massflow\[ml\])")
_LIBRARY_TOO_SMALL_RE = re.compile(r"small library", re.IGNORECASE)


def suggest_fix(exception: BaseException) -> Optional[str]:
    """Return a remediation hint for a known exception class/message.

    Unknown exceptions return ``None``; the caller is responsible for falling
    back to generic advice (keep the traceback visible, contact the author).
    """
    name = exception.__class__.__name__
    message = str(exception)

    if name == "UnsupportedVendorFormatError" or "vendor" in message.lower():
        return (
            "Vendor raw formats (.raw, .d, .wiff, ...) must be converted to an "
            "open format first. Run: massflow convert --input <dir> --output <dir> "
            "(requires ProteoWizard msconvert)."
        )
    if name == "FileNotFoundError":
        return "Check that the path exists and is spelled correctly. Use the browser tab to navigate instead of typing paths."
    if name in {"IsADirectoryError", "NotADirectoryError"}:
        return "A directory was used where a file was expected (or vice versa). Pick the other kind of path."
    if name == "PermissionError":
        return "MassFlow cannot read/write that path due to filesystem permissions. Check ownership or choose a different location."
    if name == "LegacyDatabaseSchemaError" or "legacy schema" in message.lower():
        return (
            "This database uses a legacy schema. Migrate it first with "
            "scripts/migrations/0001_peaks_to_arrays.py, or rebuild it with "
            "massflow db build."
        )
    if name == "OperationalError" and "locked" in message.lower():
        return "The SQLite library is locked by another process. Close other MassFlow runs and retry."
    if name in {"DatabaseError", "OperationalError", "IntegrityError"}:
        return "The database file is unreadable or corrupt. Inspect it with: massflow db inspect <file>."
    if name == "ValidationError":
        return "The configuration failed schema validation. Fix the YAML (the error lists the offending keys) and reload."
    if "machine-learning" in message.lower() or "massflow[ml]" in message:
        return (
            _ML_INSTALL_RE.sub(r"\1", message)
            or "Install the ML engines with: pip install massflow[ml]"
        )
    if name == "ImportError":
        return "A required package is missing. Install the optional extra shown in the error (e.g. pip install massflow[ml] or massflow[tui])."
    if _LIBRARY_TOO_SMALL_RE.search(message):
        return (
            "Target-decoy FDR is statistically weak on small libraries. Use a "
            "larger reference library, or relax fdr_threshold toward 1.0."
        )
    if name in {"MemoryError"}:
        return "The operation ran out of memory. Reduce the preview size or use a database-backed library (massflow db build)."
    if name == "UnicodeDecodeError":
        return "The file is not valid UTF-8 text. It may be a binary/vendor file with a misleading extension — convert or re-export it."
    return None


@dataclass(frozen=True)
class QuarantineEntry:
    """One line of the quarantine log."""

    message: str
    timestamp: Optional[str] = None


def parse_quarantine_log(
    path: Path, *, max_entries: int = 200
) -> list[QuarantineEntry]:
    """Read the core pipeline's quarantine log, newest entries last.

    The core validation layer writes one ``<timestamp> - <message>`` line per
    rejected spectrum. This reader is tolerant of lines it cannot parse and
    of a missing log file (which yields an empty list).

    Parameters
    ----------
    path : Path
        Path to the quarantine log (typically ``massflow_quarantine.log``).
    max_entries : int
        Keep at most this many entries (the *tail* of the file).

    Returns
    -------
    list[QuarantineEntry]
        Parsed entries in file order (oldest first), truncated to the tail.
    """
    path = Path(path)
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    entries: list[QuarantineEntry] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if " - " in line:
            timestamp, message = line.split(" - ", 1)
            entries.append(QuarantineEntry(message=message, timestamp=timestamp))
        else:
            entries.append(QuarantineEntry(message=line))
    return entries[-max_entries:]
