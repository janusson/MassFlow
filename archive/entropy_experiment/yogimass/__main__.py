"""Run the Yogimass CLI as a module: python -m yogimass"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - executed via python -m
    raise SystemExit(main())
