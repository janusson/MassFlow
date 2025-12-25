"""Utility helpers exposed at the ``yogimass.utils`` package level."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_base_dir = Path(__file__).resolve().parent
_repo_root = _base_dir.parent.parent
_extra_path = _repo_root / "splinters" / "utils" / "yogimass" / "utils"
if _extra_path.is_dir():
    extra_str = str(_extra_path)
    if extra_str not in __path__:
        __path__.append(extra_str)

from .logging import configure_logging, get_logger

try:
    from .misc import get_spectrum_inchi
except Exception:  # pragma: no cover - fallback if misc is unavailable

    def get_spectrum_inchi(spectrum):
        return spectrum.get("inchi") or spectrum.get("inchikey")


__all__ = ["configure_logging", "get_logger", "get_spectrum_inchi"]
