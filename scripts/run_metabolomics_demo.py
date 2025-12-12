"""
Run an end-to-end untargeted metabolomics demo workflow and print a brief summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from matchms import Spectrum  # type: ignore

from yogimass import workflow
from yogimass.config import ProcessorConfig, WorkflowConfig, load_config
from yogimass.similarity.processing import SpectrumProcessor
from yogimass.similarity.library import LocalSpectralLibrary


def _processor_from_cfg(processor_cfg: ProcessorConfig) -> SpectrumProcessor:
    return SpectrumProcessor(
        normalization=processor_cfg.normalization,
        min_relative_intensity=processor_cfg.min_relative_intensity,
        min_absolute_intensity=processor_cfg.min_absolute_intensity,
        max_peaks=processor_cfg.max_peaks,
        mz_dedup_tolerance=processor_cfg.mz_dedup_tolerance,
        float_dtype=processor_cfg.float_dtype,
    )


def _summarize_network(summary_path: Path | None) -> tuple[int, int]:
    if not summary_path or not summary_path.exists():
        return 0, 0
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(data.get("nodes", 0)), int(data.get("edges", 0))


def _count_processed(spectra: Sequence[Spectrum], processor: SpectrumProcessor) -> int:
    count = 0
    for spectrum in spectra:
        _ = processor.process(spectrum)
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an untargeted metabolomics demo workflow.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/metabolomics_workflow.yaml"),
        help="Path to a YAML/JSON config (default: examples/metabolomics_workflow.yaml).",
    )
    args = parser.parse_args(argv)

    cfg: WorkflowConfig = load_config(args.config)
    spectra = workflow.load_data(
        cfg.input.paths,
        input_format=cfg.input.format,
        recursive=cfg.input.recursive,
        msdial_output_dir=cfg.input.msdial_output,
    )
    spectra_list = list(spectra) if isinstance(spectra, list) else []
    processor = _processor_from_cfg(cfg.similarity.processor)
    processed_count = _count_processed(spectra_list, processor)

    workflow.run_from_config(cfg)

    library_path = cfg.library.path
    n_library = 0
    if library_path and Path(library_path).exists():
        lib = LocalSpectralLibrary(library_path)
        n_library = len(lib)

    summary_path = cfg.network.summary or cfg.outputs.network_summary
    n_nodes, n_edges = _summarize_network(Path(summary_path) if summary_path else None)

    print(
        f"inputs={len(spectra_list)}, processed={processed_count}, "
        f"library_entries={n_library}, network_nodes={n_nodes}, network_edges={n_edges}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
