"""
Spectrum processing utilities used by the similarity pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Literal

import numpy as np
from matchms import Spectrum


def _clone_spectrum(spectrum: Spectrum) -> Spectrum:
    return spectrum.clone()


def _make_empty_spectrum(metadata: dict | None = None, dtype: type = float) -> Spectrum:
    mz = np.asarray([], dtype=dtype)
    intensities = np.asarray([], dtype=dtype)
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata or {})


def _cast_spectrum(spectrum: Spectrum, dtype: type = float) -> Spectrum:
    mz = np.asarray(spectrum.peaks.mz, dtype=dtype)
    intensities = np.asarray(spectrum.peaks.intensities, dtype=dtype)
    return Spectrum(mz=mz, intensities=intensities, metadata=spectrum.metadata)


def _dedup_mz_window(
    mz: np.ndarray, intensities: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate peaks within an m/z window by choosing the apex (max intensity) peak.

    Deterministic grouping: sort by m/z then group adjacent peaks within `tolerance`.
    For each group, choose the m/z corresponding to the maximum intensity (tie-break: lowest original index).
    Returns deduped (mz, intensities) arrays sorted by mz.
    """
    if mz.size == 0:
        return mz.astype(float), intensities.astype(float)

    order = np.argsort(mz)
    mz_sorted = mz[order]
    ints_sorted = intensities[order]

    groups = []  # list of lists of indices
    current_group = [0]
    for i in range(1, mz_sorted.size):
        if mz_sorted[i] - mz_sorted[current_group[-1]] <= tolerance:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    new_mz = []
    new_ints = []
    for grp in groups:
        grp_ints = ints_sorted[grp]
        grp_mz = mz_sorted[grp]
        # choose index of max intensity; if ties, choose the first in group
        max_idx = int(np.argmax(grp_ints))
        new_mz.append(grp_mz[max_idx])
        new_ints.append(grp_ints[max_idx])

    new_mz = np.asarray(new_mz, dtype=float)
    new_ints = np.asarray(new_ints, dtype=float)
    return new_mz, new_ints


@dataclass
class SpectrumProcessor:
    """Processing utilities for MS/MS spectra.

    Features implemented:
    - Intensity normalization (TIC or base-peak)
    - Noise filtering (absolute + relative)
    - Top-N peak selection
    - Optional m/z window deduplication
    - Data model invariants enforced in outputs
    """

    min_relative_intensity: float = 0.01
    min_absolute_intensity: float = 0.0
    max_peaks: Optional[int] = 500
    align_tolerance: float = 0.01
    mz_dedup_tolerance: Optional[float] = None
    normalization: Optional[Literal["tic", "basepeak"]] = None
    float_dtype: type = float  # either float or np.float32/64

    def filter_noise(self, spectrum: Spectrum) -> Spectrum:
        """
        Remove low-intensity peaks and optionally cap peak count.
        Returns a *new* Spectrum instance (no in-place mutation of peaks).

        Steps:
        1. Drop NaN peaks and validate intensities (reject negatives).
        2. Compute a threshold as max(max_intensity * min_relative_intensity, min_absolute_intensity).
        3. Keep peaks above that threshold.
        4. If configured, keep the top-N peaks by intensity (deterministically).
        5. Optionally deduplicate peaks within `mz_dedup_tolerance` by choosing the apex.
        6. Normalize retained intensities to [0, 1] by dividing by the maximum kept intensity.
        7. Return a Spectrum with consistent dtype, sorted m/z and matching intensity lengths.
        """
        clone = _clone_spectrum(spectrum)
        mz = np.asarray(clone.peaks.mz, dtype=float)
        intensities = np.asarray(clone.peaks.intensities, dtype=float)

        if intensities.size == 0:
            return clone

        if np.isnan(intensities).any():
            # Drop NaN peaks from the input
            valid_mask = ~np.isnan(intensities)
            mz = mz[valid_mask]
            intensities = intensities[valid_mask]

        if intensities.size == 0:
            return _make_empty_spectrum(clone.metadata, dtype=self.float_dtype)

        if (intensities < 0).any():
            raise ValueError("Negative intensity values are not allowed")

        max_intensity = float(np.max(intensities))
        if max_intensity == 0.0:
            return clone

        threshold = max(
            max_intensity * self.min_relative_intensity, self.min_absolute_intensity
        )
        keep_mask = intensities >= threshold
        kept_indices = np.nonzero(keep_mask)[0]

        if self.max_peaks is not None and kept_indices.size > self.max_peaks:
            # deterministic top-N by intensity descending; stable sort by index on ties
            order = np.argsort(intensities[kept_indices], kind="stable")[::-1]
            top_by_intensity = order[: self.max_peaks]
            kept_indices = np.sort(kept_indices[top_by_intensity])
        else:
            kept_indices = np.sort(kept_indices)

        # If nothing changes, just return the clone.
        if kept_indices.size == 0:
            return _make_empty_spectrum(clone.metadata, dtype=self.float_dtype)

        new_mz = mz[kept_indices]
        new_intensities = intensities[kept_indices]

        # Optional m/z-window deduplication
        if self.mz_dedup_tolerance is not None and self.mz_dedup_tolerance > 0:
            new_mz, new_intensities = _dedup_mz_window(
                new_mz, new_intensities, self.mz_dedup_tolerance
            )

        # if nothing changed from original (after dedup) and lengths are same
        if new_mz.size == mz.size and np.allclose(new_mz, mz):
            # ensure types/dtype but otherwise return clone
            return _cast_spectrum(clone, dtype=self.float_dtype)

        if new_intensities.size:
            max_kept = float(np.max(new_intensities))
            if max_kept > 0:
                new_intensities = new_intensities / max_kept

        # Ensure sorted by m/z and consistent dtypes
        order = np.argsort(new_mz)
        new_mz = new_mz[order]
        new_intensities = new_intensities[order]

        return Spectrum(
            mz=new_mz.astype(self.float_dtype),
            intensities=new_intensities.astype(self.float_dtype),
            metadata=clone.metadata,
        )

    def normalize(self, spectrum: Spectrum) -> Spectrum:
        """Normalize spectrum intensities according to the chosen normalization.

        Normalizations implemented:
        - 'tic': divide intensities by the sum of all intensities (Total Ion Current).
              This produces intensities that sum to 1.0.
        - 'basepeak': divide intensities by the maximum intensity (base peak), which
                   produces intensities in the [0, 1] range where the largest peak
                   is 1.0.

        Notes on determinism: operations use deterministic numpy routines (stable sorts) and do
        not rely on randomness, ensuring identical input yields identical output.
        """
        if self.normalization not in (None, "tic", "basepeak"):
            raise ValueError("normalization must be 'tic', 'basepeak' or None")

        clone = _clone_spectrum(spectrum)
        mz = np.asarray(clone.peaks.mz, dtype=float)
        intensities = np.asarray(clone.peaks.intensities, dtype=float)

        if intensities.size == 0:
            return _cast_spectrum(clone, dtype=self.float_dtype)

        if np.isnan(intensities).any():
            # Drop NaN peaks
            valid_mask = ~np.isnan(intensities)
            mz = mz[valid_mask]
            intensities = intensities[valid_mask]

        if (intensities < 0).any():
            raise ValueError("Negative intensity values are not allowed")

        if self.normalization == "tic":
            tic = float(np.sum(intensities))
            if tic == 0.0:
                return _cast_spectrum(clone, dtype=self.float_dtype)
            intensities = intensities / tic
        elif self.normalization == "basepeak":
            max_i = float(np.max(intensities))
            if max_i == 0.0:
                return _cast_spectrum(clone, dtype=self.float_dtype)
            intensities = intensities / max_i

        return Spectrum(
            mz=mz.astype(self.float_dtype),
            intensities=intensities.astype(self.float_dtype),
            metadata=clone.metadata,
        )

    def align_to_reference(
        self, spectrum: Spectrum, reference_mz: Sequence[float]
    ) -> Spectrum:
        """
        Align a spectrum to a reference m/z grid within align_tolerance.
        Returns a new Spectrum with intensities on the reference grid.
        """
        if not reference_mz:
            return spectrum

        ref = np.asarray(reference_mz, dtype=float)
        mz = np.asarray(spectrum.peaks.mz, dtype=float)
        intensities = np.asarray(spectrum.peaks.intensities, dtype=float)

        new_intensities = np.zeros_like(ref, dtype=float)
        for i, r in enumerate(ref):
            mask = np.abs(mz - r) <= self.align_tolerance
            if np.any(mask):
                # take max intensity of matching peaks
                new_intensities[i] = float(np.max(intensities[mask]))

        return Spectrum(
            mz=ref,
            intensities=new_intensities,
            metadata=spectrum.metadata,
        )

    def process(
        self, spectrum: Spectrum, reference_mz: Optional[Sequence[float]] = None
    ) -> Spectrum:
        """
        Apply filtering and (optionally) alignment to a spectrum.
        """
        # Apply optional normalization first (if configured), then filtering
        if self.normalization is not None:
            spectrum = self.normalize(spectrum)

        filtered = self.filter_noise(spectrum)
        if reference_mz is not None:
            return self.align_to_reference(filtered, reference_mz=reference_mz)
        return filtered
