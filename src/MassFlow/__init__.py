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
    __version__ = importlib.metadata.version("massflow")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from . import cli, config, io, processing, similarity, workflow

__all__ = ["io", "processing", "similarity", "cli", "config", "workflow", "__version__"]
