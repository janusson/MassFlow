import json
import logging
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """
    A dev-friendly JSON-like logging formatter for MassFlow.
    Captures standard log record attributes and any extra context
    passed in via the `extra` dictionary.
    """

    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }

        # Capture specific metadata if injected
        for key in ["spectrum_id", "precursor_mz", "compound_name", "step"]:
            if hasattr(record, key):
                log_record[key] = getattr(record, key)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_structured_logging(level=logging.INFO):
    """Configures process-wide structured JSON logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())

    # We clear existing handlers to avoid duplicates from default config
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []
    root_logger.addHandler(handler)

    # Suppress extremely verbose matchms metadata warnings
    logging.getLogger("matchms").setLevel(logging.ERROR)

    # --- Quarantine Logger Setup ---
    # Dedicated logger for invalid spectra that should not pollute the main log.
    quarantine_logger = logging.getLogger("quarantine")
    quarantine_logger.setLevel(logging.WARNING)
    quarantine_logger.propagate = False  # Do not forward to the root logger

    # Ensure it has a handler, even if the root one is suppressed
    if not quarantine_logger.handlers:
        q_handler = logging.FileHandler("massflow_quarantine.log")
        q_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        quarantine_logger.addHandler(q_handler)
