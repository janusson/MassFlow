"""
MassFlow Package Initialization.

This top-level module exposes the core components of the MassFlow library,
including input/output handling, spectral processing, similarity search,
configuration management, and workflow orchestration. It also defines the
package version and handles sub-module imports for convenient access.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "MassFlowConfig",
    "MLEngineProtocol",
    "SpectralDatabase",
    "SpectralStore",
    "ZarrPeakArrayStore",
    "ZarrSpectralStore",
    "create_spectral_store",
    "load_spectra",
    "process_spectra",
    "run_annotation_pipeline",
    "cli",
    "config",
    "database",
    "io",
    "ml_client",
    "processing",
    "protocols",
    "similarity",
    "storage",
    "workflow",
    "zarr_store",
    "__version__",
]


def __getattr__(name):
    if name == "MassFlowConfig":
        from .config import MassFlowConfig

        return MassFlowConfig
    elif name == "MLEngineProtocol":
        from .protocols import MLEngineProtocol

        return MLEngineProtocol
    elif name == "SpectralDatabase":
        from .database import SpectralDatabase

        return SpectralDatabase
    elif name == "SpectralStore":
        from .storage import SpectralStore

        return SpectralStore
    elif name == "ZarrSpectralStore":
        from .zarr_store import ZarrSpectralStore

        return ZarrSpectralStore
    elif name == "ZarrPeakArrayStore":
        from .zarr_store import ZarrPeakArrayStore

        return ZarrPeakArrayStore
    elif name == "create_spectral_store":
        from .storage import create_spectral_store

        return create_spectral_store
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
