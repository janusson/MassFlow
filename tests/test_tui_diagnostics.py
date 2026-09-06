"""
Tests for MassFlow.tui.diagnostics — error UX and quarantine log parsing.
"""

from pathlib import Path

from MassFlow.tui.diagnostics import (
    Problem,
    QuarantineEntry,
    TuiError,
    parse_quarantine_log,
    suggest_fix,
)


class TestTuiError:
    def test_plain_construction(self):
        error = TuiError("boom", stage="search", hint="do the thing")
        assert str(error) == "boom"
        assert error.stage == "search"
        assert error.hint == "do the thing"

    def test_from_exception_attaches_hint(self):
        error = TuiError.from_exception(
            FileNotFoundError("no such file: x"), stage="load-query"
        )
        assert error.stage == "load-query"
        assert error.hint is not None
        assert "path" in error.hint

    def test_from_exception_without_known_hint(self):
        error = TuiError.from_exception(
            RuntimeError("something wholly unexpected"), stage="x"
        )
        assert error.hint is None

    def test_empty_message_falls_back_to_class_name(self):
        error = TuiError.from_exception(ValueError(), stage="x")
        assert str(error) == "ValueError"


class TestProblem:
    def test_from_exception_renders_text(self):
        problem = Problem.from_exception(
            TuiError("bad vendor", stage="load-query", hint="convert first"),
            stage="other",
        )
        text = problem.to_text()
        assert "load-query" in text
        assert "bad vendor" in text
        assert "convert first" in text

    def test_stage_preserved_from_tui_error(self):
        problem = Problem.from_exception(TuiError("x", stage="search"))
        assert problem.stage == "search"

    def test_traceback_captured(self):
        try:
            raise RuntimeError("kaboom")
        except RuntimeError as exception:
            problem = Problem.from_exception(exception, stage="worker")
        assert "RuntimeError: kaboom" in problem.traceback_text
        assert "kaboom" in problem.to_text()

    def test_traceback_truncated_to_tail(self):
        try:
            raise ValueError("deep failure")
        except ValueError as exception:
            problem = Problem.from_exception(exception, stage="worker")
        lines = problem.to_text().splitlines()
        # At most the last 6 traceback lines are rendered.
        traceback_lines = [line for line in lines if "    " in line]
        assert len(traceback_lines) <= 6


class TestSuggestFix:
    def test_vendor_format(self):
        from MassFlow.io import UnsupportedVendorFormatError

        hint = suggest_fix(UnsupportedVendorFormatError("convert me"))
        assert hint is not None
        assert "massflow convert" in hint

    def test_file_not_found(self):
        assert "path" in suggest_fix(FileNotFoundError("x"))

    def test_directory_confusion(self):
        assert suggest_fix(IsADirectoryError("x")) is not None
        assert suggest_fix(NotADirectoryError("x")) is not None

    def test_permission_error(self):
        assert "permissions" in suggest_fix(PermissionError("denied"))

    def test_legacy_database_schema(self):
        from MassFlow.database import LegacyDatabaseSchemaError

        hint = suggest_fix(LegacyDatabaseSchemaError("legacy schema"))
        assert hint is not None
        assert "migrat" in hint.lower()

    def test_locked_database(self):
        import sqlite3

        hint = suggest_fix(sqlite3.OperationalError("database is locked"))
        assert hint is not None
        assert "locked" in hint.lower()

    def test_generic_database_error(self):
        import sqlite3

        hint = suggest_fix(sqlite3.DatabaseError("corrupt"))
        assert hint is not None
        assert "db inspect" in hint

    def test_pydantic_validation(self):
        from pydantic import ValidationError

        hint = suggest_fix(ValidationError.from_exception_data("MassFlowConfig", []))
        assert hint is not None
        assert "YAML" in hint

    def test_ml_missing(self):
        hint = suggest_fix(
            RuntimeError(
                "This scoring engine requires the machine-learning extras. "
                "Install them with: pip install massflow[ml]"
            )
        )
        assert hint is not None
        assert "massflow[ml]" in hint

    def test_import_error(self):
        assert suggest_fix(ImportError("No module named 'torch'")) is not None

    def test_memory_error(self):
        assert "memory" in suggest_fix(MemoryError("oom"))

    def test_unicode_error(self):
        assert suggest_fix(UnicodeDecodeError("utf-8", b"", 0, 1, "bad")) is not None

    def test_unknown_exception(self):
        assert suggest_fix(KeyError("missing")) is None


class TestParseQuarantineLog:
    def test_missing_file(self, tmp_path: Path):
        assert parse_quarantine_log(tmp_path / "nope.log") == []

    def test_standard_lines(self, tmp_path: Path):
        log = tmp_path / "massflow_quarantine.log"
        log.write_text(
            "2026-01-01 12:00:00,123 - Quarantined Spectrum | Source: x.mgf | Reason: Missing precursor_mz\n"
        )
        entries = parse_quarantine_log(log)
        assert len(entries) == 1
        entry = entries[0]
        assert isinstance(entry, QuarantineEntry)
        assert entry.timestamp == "2026-01-01 12:00:00,123"
        assert "Missing precursor_mz" in entry.message

    def test_unparseable_lines_tolerated(self, tmp_path: Path):
        log = tmp_path / "massflow_quarantine.log"
        log.write_text("garbage without separator\n2026-01-01 - real message\n")
        entries = parse_quarantine_log(log)
        assert len(entries) == 2
        assert entries[0].timestamp is None
        assert entries[1].timestamp == "2026-01-01"

    def test_tail_truncation(self, tmp_path: Path):
        log = tmp_path / "massflow_quarantine.log"
        log.write_text("\n".join(f"{i} - message {i}" for i in range(100)))
        entries = parse_quarantine_log(log, max_entries=10)
        assert len(entries) == 10
        assert entries[0].message == "message 90"
        assert entries[-1].message == "message 99"

    def test_empty_file(self, tmp_path: Path):
        log = tmp_path / "massflow_quarantine.log"
        log.write_text("")
        assert parse_quarantine_log(log) == []
