"""
SpectralMetricMS Core Package.
"""
from __future__ import annotations

__version__ = "0.4.0"

from . import io
from . import processing
from . import similarity

__all__ = ["io", "processing", "similarity", "__version__"]
