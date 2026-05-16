"""
MassFlow Package Initialization.

This top-level module exposes the core components of the MassFlow library,
including input/output handling, spectral processing, similarity search,
configuration management, and workflow orchestration. It also defines the
package version and handles sub-module imports for convenient access.
"""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = "1.0.0"
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "MassFlowConfig",
    "SpectralDatabase",
    "load_spectra",
    "process_spectra",
    "run_annotation_pipeline",
    "cli",
    "config",
    "database",
    "io",
    "processing",
    "similarity",
    "workflow",
    "__version__",
]


def __getattr__(name):
    if name == "MassFlowConfig":
        from .config import MassFlowConfig

        return MassFlowConfig
    elif name == "SpectralDatabase":
        from .database import SpectralDatabase

        return SpectralDatabase
    elif name == "load_spectra":
        from .io import load_spectra

        return load_spectra
    elif name == "process_spectra":
        from .processing import process_spectra

        return process_spectra
    elif name == "run_annotation_pipeline":
        from .workflow import run_annotation_pipeline

        return run_annotation_pipeline
    elif name in __all__:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
