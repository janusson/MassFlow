"""
Spectrum processing utilities used by the similarity pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from matchms import Spectrum


def _clone_spectrum(spectrum: Spectrum) -> Spectrum:
    return spectrum.clone()


@dataclass
class SpectrumProcessor:
    min_relative_intensity: float = 0.01
    max_peaks: Optional[int] = 500
    align_tolerance: float = 0.01

    def filter_noise(self, spectrum: Spectrum) -> Spectrum:
        """
        Remove low-intensity peaks and optionally cap peak count.
        Returns a *new* Spectrum instance (no in-place mutation of peaks).
        """
        clone = _clone_spectrum(spectrum)
        mz = np.asarray(clone.peaks.mz, dtype=float)
        intensities = np.asarray(clone.peaks.intensities, dtype=float)

        if intensities.size == 0:
            return clone

        max_intensity = float(np.max(intensities))
        if max_intensity == 0.0:
            return clone

        threshold = max_intensity * self.min_relative_intensity
        keep_mask = intensities >= threshold
        kept_indices = np.nonzero(keep_mask)[0]

        if self.max_peaks is not None and kept_indices.size > self.max_peaks:
            top_by_intensity = np.argsort(intensities[kept_indices])[::-1][: self.max_peaks]
            kept_indices = np.sort(kept_indices[top_by_intensity])
        else:
            kept_indices = np.sort(kept_indices)

        # If nothing changes, just return the clone.
        if kept_indices.size == mz.size:
            return clone

        new_mz = mz[kept_indices]
        new_intensities = intensities[kept_indices]

        if new_intensities.size:
            max_kept = float(np.max(new_intensities))
            if max_kept > 0:
                new_intensities = new_intensities / max_kept

        return Spectrum(
            mz=new_mz,
            intensities=new_intensities,
            metadata=clone.metadata,
        )

    def align_to_reference(self, spectrum: Spectrum, reference_mz: Sequence[float]) -> Spectrum:
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

    def process(self, spectrum: Spectrum, reference_mz: Optional[Sequence[float]] = None) -> Spectrum:
        """
        Apply filtering and (optionally) alignment to a spectrum.
        """
        filtered = self.filter_noise(spectrum)
        if reference_mz is not None:
            return self.align_to_reference(filtered, reference_mz=reference_mz)
        return filtered
