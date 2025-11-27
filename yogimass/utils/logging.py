"""
Centralized logging utilities for Yogimass.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"
ROOT_LOGGER_NAME = "yogimass"


def _coerce_level(level: str | int | None) -> int:
    if level is None:
        level = os.getenv("YOGIMASS_LOGLEVEL", DEFAULT_LOG_LEVEL)
    if isinstance(level, str):
        numeric = logging.getLevelName(level.upper())
        if isinstance(numeric, int):
            return numeric
        return logging.INFO
    return int(level)


def configure_logging(*, level: str | int | None = None) -> logging.Logger:
    """
    Configure the shared Yogimass logger with a simple console handler.
    Safe to call multiple times; handlers are attached once.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(_coerce_level(level))
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a child logger under the Yogimass root logger.
    Usage: logger = get_logger(__name__)
    """
    root = configure_logging()
    if not name or name == ROOT_LOGGER_NAME:
        return root
    return root.getChild(name)


__all__ = ["configure_logging", "get_logger"]
