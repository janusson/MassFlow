"""Retention-time interpretation shared by processing, scoring, and exports."""

from __future__ import annotations

import math
import re
from typing import Any, Optional

from matchms import Spectrum

# SI time conversions; unit accessions are from the Unit Ontology (UO).
_SECONDS_PER_UNIT = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "uo:0000010": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "uo:0000031": 60.0,
    "ms": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
    "uo:0000028": 0.001,
    "h": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "uo:0000032": 3600.0,
}
_TIME_TEXT = re.compile(
    r"^([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z:0-9]+)?$"
)
# Unlabelled canonical and MSP aliases follow MassFlow's seconds contract.
# An unlabelled mzML scan start time has no such contract: its CV unit is required.
_TIME_FIELDS = (
    ("retention_time", "seconds"),
    ("rtinseconds", "seconds"),
    ("rtinminutes", "minutes"),
    ("scan_start_time", None),
    ("retentiontime", "seconds"),
    ("rt", "seconds"),
)


def _source_value(value: Any) -> Any:
    """Preserve a time input in portable JSON, without losing its unit separately."""
    if isinstance(value, (list, tuple)):
        return [_source_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return numeric_value if math.isfinite(numeric_value) else str(numeric_value)


def _time_input(
    raw_value: Any, default_unit: Optional[str], explicit_unit: Any
) -> tuple[Optional[float], dict[str, Any]]:
    """Interpret one metadata value, retaining the original scalar and unit."""
    value = raw_value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    tagged_unit = getattr(value, "unit_info", None)
    intrinsic_unit = str(tagged_unit).lower() if tagged_unit else None
    numeric_value: Optional[float] = None
    if isinstance(value, str):
        match = _TIME_TEXT.fullmatch(value.strip())
        if match:
            numeric_value = float(match[1].replace(",", "."))
            intrinsic_unit = match[2].lower() if match[2] else intrinsic_unit
    elif not isinstance(value, (bool, list, tuple)):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            pass
    declared_unit = (
        str(explicit_unit).strip().lower() if explicit_unit is not None else None
    )
    unit = declared_unit or intrinsic_unit or default_unit
    record = {
        "value": _source_value(value),
        "unit": unit,
        "declared_unit": declared_unit,
        "intrinsic_unit": intrinsic_unit,
        "unit_origin": "metadata"
        if declared_unit
        else ("value" if intrinsic_unit else "field_contract"),
    }
    factor = _SECONDS_PER_UNIT.get(unit or "")
    if (
        declared_unit
        and intrinsic_unit
        and _SECONDS_PER_UNIT.get(declared_unit)
        != _SECONDS_PER_UNIT.get(intrinsic_unit)
    ):
        record["conflicting_units"] = True
        return None, record
    if numeric_value is None or factor is None:
        return None, record
    retention_time_seconds = numeric_value * factor
    if not math.isfinite(retention_time_seconds) or retention_time_seconds < 0:
        return None, record
    return retention_time_seconds, record


def _interpret_retention_time(
    spectrum: Spectrum, extract_aliases: bool = True
) -> tuple[Optional[float], dict[str, Any]]:
    """Resolve explicit canonical metadata or consistent source aliases."""
    previous_source = spectrum.get("retention_time_source")
    if (
        spectrum.get("retention_time") is None
        and isinstance(previous_source, dict)
        and previous_source.get("retention_time_seconds") is None
    ):
        return None, previous_source
    inputs: list[dict[str, Any]] = []
    values_seconds: list[float] = []
    fields = _TIME_FIELDS if extract_aliases else _TIME_FIELDS[:1]
    # These are a few metadata fields, never a loop over spectral peak arrays.
    for field, default_unit in fields:
        raw_value = spectrum.get(field)
        if raw_value is None or (
            isinstance(raw_value, (list, tuple)) and not raw_value
        ):
            continue
        explicit_unit = (
            spectrum.get("retention_time_unit")
            if field == "retention_time"
            else (default_unit if field in {"rtinseconds", "rtinminutes"} else None)
        )
        value_seconds, record = _time_input(raw_value, default_unit, explicit_unit)
        inputs.append({"field": field, **record})
        if value_seconds is not None:
            values_seconds.append(value_seconds)
        # Previously normalized canonical seconds survive source aliases whose
        # unitfloat tags may have been erased by JSON serialization. Fresh inputs
        # must still reconcile all aliases, even if their canonical field has units.
        if field == "retention_time" and explicit_unit is not None:
            previous_source = spectrum.get("retention_time_source")
            if (
                isinstance(previous_source, dict)
                and previous_source.get("status") == "normalized"
                and previous_source.get("retention_time_seconds") == value_seconds
            ):
                return value_seconds, previous_source
    state = "missing" if not inputs else "invalid"
    retention_time_seconds = None
    if values_seconds:
        retention_time_seconds = values_seconds[0]
        state = "normalized"
        if any(
            not math.isclose(value, retention_time_seconds, rel_tol=1e-12, abs_tol=1e-9)
            for value in values_seconds[1:]
        ):
            retention_time_seconds = None
            state = "conflicting"
    if any(record.get("conflicting_units") for record in inputs):
        retention_time_seconds = None
        state = "conflicting"
    source = {
        "inputs": inputs,
        "status": state,
        "retention_time_seconds": retention_time_seconds,
    }
    original = spectrum.get("retention_time_original")
    if original is not None:
        source["original"] = original
        source["input_origin"] = "importer_metadata"
    return retention_time_seconds, source


def get_retention_time_seconds(spectrum: Spectrum) -> Optional[float]:
    """Read an unambiguous retention time in seconds without mutating metadata.

    Parameters
    ----------
    spectrum : matchms.Spectrum
        Spectrum with canonical metadata or supported source time fields.

    Returns
    -------
    float or None
        Finite, nonnegative seconds; None for absent, invalid, or conflicting data.

    Examples
    --------
    >>> get_retention_time_seconds(spectrum)  # doctest: +SKIP
    120.0
    """
    return _interpret_retention_time(spectrum)[0]


def normalize_retention_time(
    spectrum: Spectrum, extract_aliases: bool = True
) -> Spectrum:
    """Return a cloned spectrum with canonical seconds and original time evidence.

    Parameters
    ----------
    spectrum : matchms.Spectrum
        Original spectrum, which is never modified.
    extract_aliases : bool, optional
        Extract additional source fields when True. Canonical retention_time is
        always interpreted so that a known minute value cannot be stored as seconds.

    Returns
    -------
    matchms.Spectrum
        Clone with retention_time, retention_time_unit and retention_time_source.
        Missing scientific values remain None, including ambiguous aliases.

    Examples
    --------
    >>> normalized = normalize_retention_time(spectrum)  # doctest: +SKIP
    """
    retention_time_seconds, source = _interpret_retention_time(
        spectrum, extract_aliases
    )
    normalized = spectrum.clone()
    normalized.set("retention_time", retention_time_seconds)
    normalized.set(
        "retention_time_unit", "seconds" if retention_time_seconds is not None else None
    )
    normalized.set("retention_time_source", source)
    return normalized
