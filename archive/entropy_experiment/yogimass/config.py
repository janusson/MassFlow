"""
Processing configuration and validation for Yogimass core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class ConfigError(ValueError):
    """Configuration validation error with dotted-path context."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


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


__all__ = ["ConfigError", "ProcessorConfig"]
