"""Compatibility package for workflow scripts."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_base_dir = Path(__file__).resolve().parent
_repo_root = _base_dir.parent
_extra_path = _repo_root / "splinters" / "workflows" / "scripts"
if _extra_path.is_dir():
    extra_str = str(_extra_path)
    if extra_str not in __path__:
        __path__.append(extra_str)
