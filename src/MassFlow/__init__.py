"""
MassFlow Core Package Initialization.
"""

from __future__ import annotations

__version__ = "0.4.0"

from . import cli, config, io, processing, similarity, workflow

__all__ = ["io", "processing", "similarity", "cli", "config", "workflow", "__version__"]
