"""Benchmark tests: every test under tests/benchmarks/ is a benchmark.

Marked with ``pytest.mark.benchmark`` so the default release suite (which
excludes ``benchmark``) never runs them; they are opt-in via
``-m benchmark``.
"""

import pytest

pytestmark = pytest.mark.benchmark
