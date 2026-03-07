"""
MassFlow Package Initialization.

This top-level module exposes the core components of the MassFlow library,
including input/output handling, spectral processing, similarity search,
configuration management, and workflow orchestration. It also defines the
package version and handles sub-module imports for convenient access.
"""

from __future__ import annotations

__version__ = "0.4.0"

from . import cli, config, io, processing, similarity, workflow

__all__ = ["io", "processing", "similarity", "cli", "config", "workflow", "__version__"]
