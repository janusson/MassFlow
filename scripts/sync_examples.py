"""Sync root-level examples from the workflows splinter."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _collect_example_files(source_dir: Path) -> list[Path]:
    yaml_files = list(source_dir.glob("*.yaml"))
    yml_files = list(source_dir.glob("*.yml"))
    return sorted({*yaml_files, *yml_files})


def sync_examples(*, delete: bool = False) -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    source_dir = repo_root / "splinters" / "workflows" / "examples"
    dest_dir = repo_root / "examples"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source examples at {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    source_files = _collect_example_files(source_dir)
    source_names = {path.name for path in source_files}
    copied: list[Path] = []

    for path in source_files:
        target = dest_dir / path.name
        shutil.copy2(path, target)
        copied.append(target)

    if delete:
        for path in _collect_example_files(dest_dir):
            if path.name not in source_names:
                path.unlink()

    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync root examples/ from splinters/workflows/examples."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Remove root examples not present in the source directory.",
    )
    args = parser.parse_args(argv)

    copied = sync_examples(delete=args.delete)
    for path in copied:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
