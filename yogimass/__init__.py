"""
Yogimass package initialization.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from pkgutil import extend_path
from typing import Any

__version__ = "0.3.0"

__path__ = extend_path(__path__, __name__)


def _extend_package_paths() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    splinter_dirs = [
        repo_root / "splinters" / "workflows" / "yogimass",
        repo_root / "splinters" / "io-msdial" / "yogimass",
        repo_root / "splinters" / "networking" / "yogimass",
        repo_root / "splinters" / "similarity-search" / "yogimass",
        repo_root / "splinters" / "curation" / "yogimass",
        repo_root / "splinters" / "utils" / "yogimass",
    ]
    for path in splinter_dirs:
        if path.is_dir():
            path_str = str(path)
            if path_str not in __path__:
                __path__.append(path_str)


_extend_package_paths()

_COMPAT_EXPORTS = {"cli", "pipeline", "quickstart", "workflow"}


def __getattr__(name: str) -> Any:
    if name in _COMPAT_EXPORTS:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["__version__", *sorted(_COMPAT_EXPORTS)]
