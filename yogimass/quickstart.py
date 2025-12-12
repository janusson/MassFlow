"""
Interactive quickstart config generator for Yogimass.

Generates a minimal YAML config based on a small set of presets and user-provided paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Mapping

import yaml

from yogimass.config import ConfigError, load_config

_PRESETS: Mapping[str, dict] = {
    "demo_minimal": {
        "description": "Small demo using a single MGF file for build/search/network.",
        "template": {
            "input": {"path": "{input_path}", "format": "{input_format}"},
            "library": {"path": "{output_dir}/library.json", "build": True},
            "similarity": {
                "search": True,
                "queries": ["{input_path}"],
                "top_n": 3,
                "processor": {
                    "normalization": "basepeak",
                    "min_relative_intensity": 0.02,
                    "max_peaks": 300,
                },
            },
            "network": {
                "enabled": True,
                "metric": "spec2vec",
                "knn": 5,
                "output": "{output_dir}/network.csv",
            },
            "outputs": {
                "search_results": "{output_dir}/search.csv",
                "network_summary": "{output_dir}/network_summary.json",
            },
        },
    },
    "untargeted_metabolomics": {
        "description": "Untargeted LC-MS/MS defaults (basepeak norm, dedup, top-N).",
        "template": {
            "input": {"path": "{input_path}", "format": "{input_format}"},
            "library": {"path": "{output_dir}/metabolomics_library.json", "build": True},
            "similarity": {
                "search": True,
                "queries": ["{input_path}"],
                "top_n": 5,
                "processor": {
                    "normalization": "basepeak",
                    "min_relative_intensity": 0.02,
                    "min_absolute_intensity": 0.0,
                    "max_peaks": 400,
                    "mz_dedup_tolerance": 0.01,
                },
            },
            "network": {
                "enabled": True,
                "metric": "spec2vec",
                "knn": 10,
                "output": "{output_dir}/metabolomics_network.csv",
            },
            "outputs": {
                "search_results": "{output_dir}/metabolomics_search.csv",
                "network_summary": "{output_dir}/metabolomics_network_summary.json",
            },
        },
    },
    "environmental_lcms": {
        "description": "Environmental LC-MS with TIC normalization and stricter filtering.",
        "template": {
            "input": {"path": "{input_path}", "format": "{input_format}"},
            "library": {"path": "{output_dir}/env_library.json", "build": True},
            "similarity": {
                "search": True,
                "queries": ["{input_path}"],
                "top_n": 5,
                "processor": {
                    "normalization": "tic",
                    "min_relative_intensity": 0.01,
                    "min_absolute_intensity": 0.001,
                    "max_peaks": 500,
                },
            },
            "network": {
                "enabled": True,
                "metric": "spec2vec",
                "knn": 8,
                "output": "{output_dir}/env_network.csv",
            },
            "outputs": {
                "search_results": "{output_dir}/env_search.csv",
                "network_summary": "{output_dir}/env_network_summary.json",
            },
        },
    },
}

_INPUT_FORMATS = {"mgf", "msp", "msdial"}


def _prompt(prompt: str, *, input_func: Callable[[str], str]) -> str:
    return input_func(prompt).strip()


def _prompt_choice(prompt: str, choices: list[str], *, input_func: Callable[[str], str]) -> str:
    choice_set = {c.lower() for c in choices}
    while True:
        value = _prompt(prompt, input_func=input_func).lower()
        if value in choice_set:
            return value
        print(f"Please enter one of: {', '.join(choices)}")


def _build_config(preset: str, input_path: str, output_dir: str, input_format: str) -> dict:
    template = _PRESETS[preset]["template"]
    placeholders = {
        "input_path": input_path,
        "output_dir": output_dir,
        "input_format": input_format,
    }
    text = yaml.safe_dump(template)
    filled = text.format(**placeholders)
    return yaml.safe_load(filled)


def _choose_output_path(default_path: Path, *, input_func: Callable[[str], str]) -> Path:
    path = default_path
    if path.exists():
        resp = _prompt(
            f"{path} already exists. Overwrite? [y/N] ",
            input_func=input_func,
        ).lower()
        if resp not in {"y", "yes"}:
            alt = _prompt("Enter a new output filename: ", input_func=input_func)
            path = Path(alt.strip() or path)
    return path


def run_quickstart(*, input_func: Callable[[str], str] | None = None) -> int:
    input_func = input_func or input
    print("=== Yogimass quickstart ===")
    preset_names = list(_PRESETS.keys())
    preset_prompt = "Choose a preset (" + "/".join(preset_names) + "): "
    preset = _prompt_choice(preset_prompt, preset_names, input_func=input_func)

    input_format = _prompt_choice(
        f"Input format ({'/'.join(sorted(_INPUT_FORMATS))}): ",
        sorted(_INPUT_FORMATS),
        input_func=input_func,
    )
    input_path = _prompt("Path to input file or directory: ", input_func=input_func)
    output_dir = _prompt("Output directory (will be created if missing): ", input_func=input_func)
    if not output_dir:
        output_dir = "out"
    output_dir_path = Path(output_dir).expanduser()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    config = _build_config(preset, input_path, str(output_dir_path), input_format)

    output_path = _choose_output_path(Path("yogimass-quickstart.yaml"), input_func=input_func)
    yaml.safe_dump(config, output_path.open("w", encoding="utf-8"))

    try:
        load_config(output_path)
    except ConfigError as exc:
        print(f"Generated config is invalid: {exc}")
        return 1

    print(f"\nConfig written to {output_path}")
    print(f"Run it with: python -m yogimass.cli config run --config {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wrapper
    _ = argv  # unused
    return run_quickstart()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
