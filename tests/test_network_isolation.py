"""
Import-isolation tests for the MassFlow graph data layer.

The graph data layer must be strictly downstream of the annotation engine:
(1) importing it must not pull in the workflow/CLI/database/io surfaces, and
(2) nothing on the stable path may depend on it. Both properties are checked in
a clean subprocess so earlier tests cannot pollute ``sys.modules``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_WHEN_IMPORTING_NETWORK = (
    "MassFlow.workflow",
    "MassFlow.cli",
    "MassFlow.database",
    "MassFlow.io",
)


def _run(code: str) -> str:
    env = dict(os.environ)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return completed.stdout.strip()


def test_importing_network_does_not_import_stable_path() -> None:
    code = (
        "import sys\n"
        "import MassFlow.network  # noqa: F401\n"
        "forbidden = "
        f"{list(_FORBIDDEN_WHEN_IMPORTING_NETWORK)!r}\n"
        "print(','.join(m for m in forbidden if m in sys.modules))\n"
    )
    assert _run(code) == ""


def test_stable_path_does_not_import_network() -> None:
    code = (
        "import sys\n"
        "import MassFlow.workflow  # noqa: F401\n"
        "print('MassFlow.network' in sys.modules)\n"
    )
    assert _run(code) == "False"
