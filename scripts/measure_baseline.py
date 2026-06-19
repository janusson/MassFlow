#!/usr/bin/env python3
"""
Measure peak memory usage and execution time of the MassFlow V1.0 pipeline.

Uses the built-in ``tracemalloc`` and ``time`` modules to provide a
self-contained benchmark with zero external dependencies beyond the
MassFlow project itself.
"""

import time
import tracemalloc
from pathlib import Path

from MassFlow.config import MassFlowConfig
from MassFlow.workflow import run_annotation_pipeline


def main() -> None:
    config_path = Path(__file__).resolve().parent.parent / "massflow_config.yaml"
    config = MassFlowConfig.from_yaml(config_path)

    tracemalloc.start()
    t_start = time.perf_counter()

    run_annotation_pipeline(config)

    t_end = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = t_end - t_start
    peak_mb = peak_bytes / (1024 * 1024)

    print(f"Execution Time : {elapsed:.2f} s")
    print(f"Peak RAM Usage : {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
