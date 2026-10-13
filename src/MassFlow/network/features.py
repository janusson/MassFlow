"""
Stage 2 feature-based networking: LC-MS feature identity.

An LC-MS **feature** is one ion — a precursor m/z observed with a given charge,
adduct and ionisation mode — eluting over a retention-time window. This module
groups experimental spectra into features deterministically:

1. **Ion channel** — spectra are partitioned by ``(ion_mode, charge, adduct)``.
   Only ions in the *same* channel can belong to the same feature (an ``[M+H]+``
   and an ``[M+Na]+`` of the same neutral mass are different ions).
2. **Single-linkage clustering** — within a channel, spectra are connected when
   their precursor m/z and retention time fall within the configured windows
   (both windows are configurable; a missing retention time does not block an
   otherwise m/z-consistent grouping).
3. **Representative** — the member with the highest summed intensity (TIC), with
   a deterministic node-id tie-break. Consensus MS/MS generation is deferred.

All outputs are deterministic: grouping is a connected-component computation over
a canonically ordered pair list, member node ids are sorted, and the feature
identifier is derived from the representative's ion metadata.

Confidence discipline: a feature is a *measurement* grouping. It carries no
similarity score and no statistical confidence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from MassFlow.cheminformatics import normalize_adduct
from MassFlow.network.candidates import generate_candidate_pairs
from MassFlow.network.models import Feature, feature_node_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MassFlow.config import NetworkConfig

logger = logging.getLogger(__name__)

__all__ = ["build_features"]

# Metadata keys consulted (in order) for an originating sample identifier.
_SAMPLE_KEYS = ("sample_id", "sample", "filename", "source_file", "source")

_ION_MODES = frozenset({"positive", "negative", "neutral"})


def _optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_adduct(value: object) -> Optional[str]:
    """Canonical adduct spelling, falling back to the raw text."""
    text = _optional_text(value)
    if text is None:
        return None
    normalized = normalize_adduct(text)
    return normalized if normalized is not None else text


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _coerce_int(value: object) -> Optional[int]:
    number = _coerce_float(value)
    return None if number is None else int(number)


def _ion_channel(
    spectrum: object,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Return the ``(ion_mode, charge, adduct)`` channel of a spectrum."""
    ion_mode = _optional_text(
        spectrum.get("ionmode") or spectrum.get("ion_mode")  # type: ignore[attr-defined]
    )
    if ion_mode not in _ION_MODES:
        ion_mode = None
    return (
        ion_mode,
        _coerce_int(spectrum.get("charge")),  # type: ignore[attr-defined]
        _canonical_adduct(spectrum.get("adduct")),  # type: ignore[attr-defined]
    )


def _precursor_mz(spectrum: object) -> float:
    value = _coerce_float(spectrum.get("precursor_mz"))  # type: ignore[attr-defined]
    return float("nan") if value is None else value


def _retention_time_seconds(spectrum: object) -> float:
    value = _coerce_float(spectrum.get("retention_time"))  # type: ignore[attr-defined]
    return float("nan") if value is None else value


def _total_intensity(spectrum: object) -> float:
    peaks = spectrum.peaks  # type: ignore[attr-defined]
    if peaks is None:
        return 0.0
    intensities = np.asarray(peaks.intensities, dtype=np.float64)
    return float(np.nansum(intensities))


def _sample_id(spectrum: object) -> Optional[str]:
    for key in _SAMPLE_KEYS:
        value = _optional_text(spectrum.get(key))  # type: ignore[attr-defined]
        if value is not None:
            return value
    return None


class _UnionFind:
    """Minimal union-find producing the smallest index as the component root."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            high, low = max(root_left, root_right), min(root_left, root_right)
            self._parent[high] = low


def _make_feature(
    indices: Sequence[int],
    spectra: Sequence[object],
    node_ids: Sequence[str],
    channel: tuple[Optional[str], Optional[int], Optional[str]],
) -> Optional[Feature]:
    """Build one :class:`Feature` from a group of member spectrum indices."""
    usable = [
        index
        for index in indices
        if np.isfinite(_precursor_mz(spectra[index]))
        and _precursor_mz(spectra[index]) > 0.0
    ]
    if not usable:
        # A feature cannot exist without a positive precursor m/z.
        return None

    representative = min(
        usable, key=lambda index: (-_total_intensity(spectra[index]), node_ids[index])
    )
    representative_mz = _precursor_mz(spectra[representative])
    retention_value = _retention_time_seconds(spectra[representative])
    retention_time = None if not np.isfinite(retention_value) else retention_value

    ion_mode, charge, adduct = channel
    feature_id = feature_node_id(
        precursor_mz=representative_mz,
        retention_time_seconds=retention_time,
        charge=charge,
        adduct=adduct,
        ion_mode=ion_mode,
    )
    return Feature(
        feature_id=str(feature_id),
        precursor_mz=representative_mz,
        retention_time_seconds=retention_time,
        ion_mode=ion_mode,
        charge=charge,
        adduct=adduct,
        spectrum_node_ids=tuple(sorted(node_ids[index] for index in indices)),
        representative_spectrum_node_id=node_ids[representative],
        sample_id=_sample_id(spectra[representative]),
        abundance=None,
    )


def build_features(
    spectra: Sequence[object],
    node_ids: Sequence[str],
    cfg: "NetworkConfig",
) -> list[Feature]:
    """
    Group spectra into LC-MS features.

    Parameters
    ----------
    spectra : sequence of matchms.Spectrum
        The (already node-aligned) experimental spectra.
    node_ids : sequence of str
        Stable node identifiers aligned with ``spectra``.
    cfg : NetworkConfig
        Provides ``build_features`` (the on/off switch),
        ``feature_precursor_tolerance`` (Da) and ``feature_rt_tolerance``
        (seconds; ``None`` disables retention-time gating).

    Returns
    -------
    list of Feature
        Features ordered by ``feature_id``. Empty when ``cfg.build_features`` is
        ``False`` or no spectra are given.

    Raises
    ------
    ValueError
        If ``node_ids`` is not aligned with ``spectra``.
    """
    if not cfg.build_features or len(spectra) == 0:
        return []
    if len(node_ids) != len(spectra):
        raise ValueError(
            f"node_ids length ({len(node_ids)}) must match spectra length "
            f"({len(spectra)})."
        )

    channels: dict[tuple[Optional[str], Optional[int], Optional[str]], list[int]] = {}
    for index, spectrum in enumerate(spectra):
        channels.setdefault(_ion_channel(spectrum), []).append(index)

    features: list[Feature] = []
    for channel, members in channels.items():
        if len(members) == 1:
            feature = _make_feature(members, spectra, node_ids, channel)
            if feature is not None:
                features.append(feature)
            continue

        precursor_mzs = np.array(
            [_precursor_mz(spectra[index]) for index in members], dtype=np.float64
        )
        retention_times = np.array(
            [_retention_time_seconds(spectra[index]) for index in members],
            dtype=np.float64,
        )
        pairs = generate_candidate_pairs(
            precursor_mzs,
            tolerance=cfg.feature_precursor_tolerance,
            rt_seconds=retention_times,
            rt_tolerance=cfg.feature_rt_tolerance,
            treat_missing_rt_as_compatible=True,
        )
        union_find = _UnionFind(len(members))
        for left, right in pairs:
            union_find.union(left, right)

        components: dict[int, list[int]] = {}
        for local_index in range(len(members)):
            components.setdefault(union_find.find(local_index), []).append(
                members[local_index]
            )
        for group in components.values():
            feature = _make_feature(group, spectra, node_ids, channel)
            if feature is not None:
                features.append(feature)

    features.sort(key=lambda feature: feature.feature_id)

    # Defensive de-duplication: distinct components whose representatives share
    # identical ion metadata would collide on feature_id. Keep the first.
    unique: list[Feature] = []
    seen: set[str] = set()
    for feature in features:
        if feature.feature_id in seen:
            logger.warning(
                "Duplicate feature identifier %s; keeping the first occurrence.",
                feature.feature_id,
            )
            continue
        seen.add(feature.feature_id)
        unique.append(feature)
    return unique
