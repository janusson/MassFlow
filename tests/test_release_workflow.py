from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest
import yaml


@pytest.mark.parametrize(
    ("version", "expected_prerelease"),
    [
        ("0.1.0a1", True),
        ("0.1.0b2", True),
        ("0.1.0rc1", True),
        ("0.1.0.dev1", True),
        ("0.1.0", False),
        ("0.1.0.post1", False),
    ],
)
def test_github_release_prerelease_flag(
    version: str, expected_prerelease: bool
) -> None:
    """Check the actual release step passes the appropriate GitHub flag."""
    workflow = yaml.safe_load(Path(".github/workflows/release.yml").read_text())
    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])]
    script = next(
        step["run"] for step in steps if step.get("name") == "Publish GitHub release"
    )
    command = 'gh() { printf "%s\\n" "$@"; }\n' + script
    result = subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "VERSION": version,
            "TAG": f"v{version}",
            "GITHUB_REPOSITORY": "owner/repo",
        },
        check=True,
        text=True,
        capture_output=True,
    )
    assert ("--prerelease" in result.stdout.splitlines()) is expected_prerelease
