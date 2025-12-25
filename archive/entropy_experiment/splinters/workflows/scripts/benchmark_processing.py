"""
Micro-benchmark runner for SpectrumProcessor workflows.

Run a full config-driven workflow and report wall time and rough memory usage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from yogimass import workflow
from yogimass.config import load_config


def _max_rss_mb() -> float | None:
    try:
        import resource
    except Exception:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    # ru_maxrss is bytes on macOS/BSD.
    if sys.platform == "darwin":
        rss_mb = rss / (1024 * 1024)
    else:
        rss_mb = rss / 1024
    return rss_mb


def _count_spectra(config_path: Path) -> int:
    cfg = load_config(config_path)
    data = workflow.load_data(
        cfg.input.paths,
        input_format=cfg.input.format,
        recursive=cfg.input.recursive,
        msdial_output_dir=cfg.input.msdial_output,
    )
    return len(data) if isinstance(data, list) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SpectrumProcessor on a config-driven workflow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/processing_workflow.yaml"),
        help="Path to a YAML/JSON config (default: examples/processing_workflow.yaml).",
    )
    args = parser.parse_args(argv)

    n_spectra = _count_spectra(args.config) if args.config.exists() else 0

    start = time.perf_counter()
    workflow.run_from_config(args.config)
    elapsed = time.perf_counter() - start
    rss_mb = _max_rss_mb()

    rss_str = f"{rss_mb:.1f} MB" if rss_mb is not None else "n/a"
    print(f"elapsed={elapsed:.2f}s, max_rss={rss_str}, n_spectra={n_spectra}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
