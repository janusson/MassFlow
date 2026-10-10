"""
Opt-in benchmark for Stage 1 spectral networking (P1, experimental).

This is **not** part of the test suite (``scripts/`` is excluded from pytest
collection). Run it manually to record network-construction scaling:

    uv run python scripts/benchmark_network.py --spectra 500
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from matchms import Spectrum

from MassFlow.config import NetworkConfig
from MassFlow.network.build import build_spectral_graph


def _synthetic_spectra(count: int, rng: np.random.Generator) -> list[Spectrum]:
    spectra: list[Spectrum] = []
    for index in range(count):
        peak_count = int(rng.integers(8, 40))
        mz = np.sort(rng.uniform(50.0, 900.0, peak_count)).astype(np.float64)
        intensities = rng.uniform(0.01, 1.0, peak_count).astype(np.float64)
        spectra.append(
            Spectrum(
                mz=mz,
                intensities=intensities,
                metadata={
                    "id": f"bench_{index}",
                    "precursor_mz": float(rng.uniform(200.0, 800.0)),
                    "retention_time": float(rng.uniform(0.0, 1800.0)),
                    "adduct": "[M+H]+",
                    "charge": 1,
                },
            )
        )
    return spectra


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Stage 1 spectral networking (P1)."
    )
    parser.add_argument(
        "--spectra", type=int, default=500, help="Number of synthetic spectra."
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    spectra = _synthetic_spectra(args.spectra, rng)
    cfg = NetworkConfig(
        enabled=True, min_score=0.7, min_matched_peaks=6, top_k_per_node=10
    )

    started = time.perf_counter()
    graph = build_spectral_graph(spectra, cfg)
    elapsed = time.perf_counter() - started

    print(
        f"spectra={len(spectra)} nodes={len(graph.nodes)} "
        f"edges={len(graph.relationships)} seconds={elapsed:.3f}"
    )


if __name__ == "__main__":
    main()
