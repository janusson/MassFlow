"""
Processing configuration and validation for Yogimass core.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import importlib.util
import sys

import numpy as np

_WORKFLOW_CONFIG_MODULE = "_yogimass_workflows_config"


def _load_workflow_config_module():
    module = sys.modules.get(_WORKFLOW_CONFIG_MODULE)
    if module is not None:
        return module
    config_path = (
        Path(__file__).resolve().parent.parent
        / "splinters"
        / "workflows"
        / "yogimass"
        / "config.py"
    )
    if not config_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(_WORKFLOW_CONFIG_MODULE, config_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_WORKFLOW_CONFIG_MODULE] = module
    spec.loader.exec_module(module)
    return module


_workflow_config = _load_workflow_config_module()

if _workflow_config is not None:
    ConfigError = _workflow_config.ConfigError
    WorkflowConfig = _workflow_config.WorkflowConfig
    load_config = _workflow_config.load_config
else:

    class ConfigError(ValueError):
        """Configuration validation error with dotted-path context."""

        def __init__(self, path: str, message: str):
            self.path = path
            self.message = message
            super().__init__(f"{path}: {message}")

    WorkflowConfig = None

    def load_config(*_: Any, **__: Any):
        raise RuntimeError("Workflow config module not available.")


def _check_unknown_keys(
    section: Mapping[str, Any], allowed: set[str], *, prefix: str
) -> None:
    for key in section:
        if key not in allowed:
            path = key if prefix in {"", "<root>"} else f"{prefix}.{key}"
            raise ConfigError(path, f"Unknown key '{key}'.")


def _coerce_non_negative_float(value: Any, *, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(path, "Expected a number.") from exc
    if number < 0:
        raise ConfigError(path, "Value cannot be negative.")
    return number


def _coerce_positive_int(value: Any, *, path: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(path, "Expected an integer.") from exc
    if number <= 0:
        raise ConfigError(path, "Value must be greater than zero.")
    return number


@dataclass
class ProcessorConfig:
    """Validated defaults for spectrum processing."""

    normalization: str | None = None
    min_relative_intensity: float = 0.01
    min_absolute_intensity: float = 0.0
    max_peaks: int | None = 500
    mz_dedup_tolerance: float | None = None
    float_dtype: type = float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProcessorConfig":
        allowed = {
            "normalization",
            "min_relative_intensity",
            "min_absolute_intensity",
            "max_peaks",
            "mz_dedup_tolerance",
            "float_dtype",
        }
        _check_unknown_keys(data, allowed, prefix="processor")

        normalization_raw = data.get("normalization")
        normalization: str | None = None
        if normalization_raw is not None:
            normalization = str(normalization_raw).lower()
            if normalization == "none":
                normalization = None
            elif normalization not in {"tic", "basepeak"}:
                raise ConfigError(
                    "processor.normalization",
                    "Normalization must be one of: tic, basepeak, none.",
                )

        min_relative_intensity = _coerce_non_negative_float(
            data.get("min_relative_intensity", 0.01),
            path="processor.min_relative_intensity",
        )
        min_absolute_intensity = _coerce_non_negative_float(
            data.get("min_absolute_intensity", 0.0),
            path="processor.min_absolute_intensity",
        )

        max_peaks_raw = data.get("max_peaks", 500)
        max_peaks: int | None
        if max_peaks_raw is None:
            max_peaks = None
        else:
            max_peaks = _coerce_positive_int(
                max_peaks_raw,
                path="processor.max_peaks",
            )

        mz_dedup_tolerance_raw = data.get("mz_dedup_tolerance")
        mz_dedup_tolerance: float | None = None
        if mz_dedup_tolerance_raw is not None:
            mz_dedup_tolerance = _coerce_non_negative_float(
                mz_dedup_tolerance_raw,
                path="processor.mz_dedup_tolerance",
            )

        float_dtype_raw = str(data.get("float_dtype", "float64")).lower()
        if float_dtype_raw not in {"float32", "float64"}:
            raise ConfigError(
                "processor.float_dtype",
                "float_dtype must be 'float32' or 'float64'.",
            )
        float_dtype = np.float64 if float_dtype_raw == "float64" else np.float32

        return cls(
            normalization=normalization,
            min_relative_intensity=min_relative_intensity,
            min_absolute_intensity=min_absolute_intensity,
            max_peaks=max_peaks,
            mz_dedup_tolerance=mz_dedup_tolerance,
            float_dtype=float_dtype,
        )

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "normalization": self.normalization,
            "min_relative_intensity": self.min_relative_intensity,
            "min_absolute_intensity": self.min_absolute_intensity,
            "max_peaks": self.max_peaks,
            "mz_dedup_tolerance": self.mz_dedup_tolerance,
            "float_dtype": self.float_dtype,
        }


__all__ = ["ConfigError", "ProcessorConfig", "WorkflowConfig", "load_config"]
