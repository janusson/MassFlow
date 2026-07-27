"""Logging configuration for MassFlow.

Provides TTY-aware structured logging that switches between a human-readable
Rich console handler (interactive terminals) and machine-friendly JSON-lines
output (piped/redirected or worker processes).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """
    A dev-friendly JSON-lines logging formatter for MassFlow.

    Captures standard log record attributes and any extra context
    passed in via the ``extra`` dictionary. Used when output is
    piped/redirected or when ``force_json`` is ``True``.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }

        # Capture specific metadata if injected via ``extra``.
        for key in ["spectrum_id", "precursor_mz", "compound_name", "step"]:
            if hasattr(record, key):
                log_record[key] = getattr(record, key)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_structured_logging(
    level: int = logging.INFO,
    force_json: bool = False,
) -> None:
    """Configure process-wide structured logging.

    In an interactive terminal (TTY) the root logger receives a
    :class:`rich.logging.RichHandler` that renders coloured,
    human-readable output.  When output is piped, redirected, or when
    *force_json* is ``True`` (e.g. inside multiprocessing workers),
    JSON-lines formatting is used instead so that log records remain
    machine-parseable.

    Parameters
    ----------
    level : int
        Logging level for the root logger (default: ``logging.INFO``).
    force_json : bool
        Force JSON-lines output regardless of TTY detection.  Set to
        ``True`` in worker processes where interleaved Rich rendering
        would be problematic.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []

    use_rich = sys.stderr.isatty() and not force_json

    if use_rich:
        from rich.console import Console
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())

    root_logger.addHandler(handler)

    # Suppress extremely verbose matchms metadata warnings.
    logging.getLogger("matchms").setLevel(logging.ERROR)

    # --- Quarantine Logger Setup -------------------------------------------
    # Dedicated logger for invalid spectra that should not pollute the
    # main log.
    quarantine_logger = logging.getLogger("quarantine")
    quarantine_logger.setLevel(logging.WARNING)
    quarantine_logger.propagate = False  # Do not forward to the root logger.

    if not quarantine_logger.handlers:
        q_handler = logging.FileHandler("massflow_quarantine.log")
        q_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        quarantine_logger.addHandler(q_handler)
