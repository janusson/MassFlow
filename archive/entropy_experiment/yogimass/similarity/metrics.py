"""
Similarity metrics and vectorization helpers for Yogimass.
"""

from __future__ import annotations

from typing import Dict, Hashable, Mapping

import numpy as np
from matchms import Spectrum
from matchms.similarity import ModifiedCosine

from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def _spectrum_to_vector_map(
    spectrum: Spectrum,
    *,
    decimals: int = 3,
) -> Dict[Hashable, float]:
    """Convert a Spectrum into a {rounded_mz: intensity} mapping."""
    mz = np.round(np.asarray(spectrum.peaks.mz, dtype=float), decimals=decimals)
    intensities = np.asarray(spectrum.peaks.intensities, dtype=float)
    vec: Dict[Hashable, float] = {}
    for mass, intensity in zip(mz, intensities):
        vec[mass] = vec.get(mass, 0.0) + float(intensity)
    return vec


def _cosine_from_maps(a: Dict[Hashable, float], b: Dict[Hashable, float]) -> float:
    """Cosine similarity between two sparse maps."""
    if not a or not b:
        return 0.0
    keys = sorted(set(a) | set(b))
    va = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(va.dot(vb) / denom)


def cosine_similarity(
    spectrum_a: Spectrum, spectrum_b: Spectrum, tolerance: float = 0.005
) -> float:
    """
    Compute a simple cosine similarity between two spectra.

    Peaks are binned by rounded m/z; this is sufficient for unit tests and
    small-scale comparisons, and avoids matchms CosineGreedy internals.
    """
    # tolerance is kept in the signature for compatibility, but not used here.
    vec_a = _spectrum_to_vector_map(spectrum_a)
    vec_b = _spectrum_to_vector_map(spectrum_b)
    return _cosine_from_maps(vec_a, vec_b)


def modified_cosine_similarity(
    spectrum_a: Spectrum, spectrum_b: Spectrum, tolerance: float = 0.005
) -> float:
    """
    Compute modified cosine similarity, which accounts for precursor m/z shifts.
    """
    spectrum_a = _ensure_precursor_mz(spectrum_a)
    spectrum_b = _ensure_precursor_mz(spectrum_b)
    similarity = ModifiedCosine(tolerance=tolerance)
    result = similarity.pair(spectrum_a, spectrum_b)
    score = _extract_pair_score(result)
    return float(score or 0.0)


def _ensure_precursor_mz(spectrum: Spectrum) -> Spectrum:
    precursor = spectrum.get("precursor_mz")
    if isinstance(precursor, (int, float)) and precursor > 0:
        return spectrum
    mz_values = np.asarray(spectrum.peaks.mz, dtype=float)
    fallback = float(mz_values.max()) if mz_values.size else 1.0
    if fallback <= 0:
        fallback = 1.0
    metadata = dict(spectrum.metadata or {})
    metadata["precursor_mz"] = fallback
    return Spectrum(
        mz=np.asarray(spectrum.peaks.mz, dtype=float),
        intensities=np.asarray(spectrum.peaks.intensities, dtype=float),
        metadata=metadata,
    )


def _extract_pair_score(result: object) -> float:
    if isinstance(result, np.ndarray):
        if result.shape == ():
            result = result.item()
        else:
            result = result.tolist()
    if isinstance(result, (list, tuple)):
        return float(result[0]) if result else 0.0
    if hasattr(result, "score"):
        return float(getattr(result, "score"))
    try:
        return float(result)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def spec2vec_vectorize(
    spectrum: Spectrum, decimal_places: int = 2, intensity_power: float = 0.5
) -> dict[str, float]:
    """
    Build a Spec2Vec-compatible token-weight mapping from a spectrum.

    Peaks become ``peak@{mz}`` tokens rounded to ``decimal_places``; intensities
    are normalized to 1.0 then raised to ``intensity_power`` (sqrt by default).
    """
    mz = spectrum.peaks.mz
    intensities = spectrum.peaks.intensities
    if mz.size == 0 or intensities.size == 0:
        return {}
    max_intensity = float(np.max(intensities))
    if max_intensity == 0:
        return {}
    normalized = intensities / max_intensity
    weights = normalized**intensity_power
    tokens = [f"peak@{value:.{decimal_places}f}" for value in mz]
    vector: dict[str, float] = {}
    for token, weight in zip(tokens, weights):
        vector[token] = vector.get(token, 0.0) + float(weight)
    return vector


def cosine_from_vectors(
    vector_a: Mapping[str, float], vector_b: Mapping[str, float]
) -> float:
    """
    Compute cosine similarity between sparse token-weight vectors.
    """
    if not vector_a or not vector_b:
        return 0.0
    keys = set(vector_a) | set(vector_b)
    dot = sum(vector_a.get(key, 0.0) * vector_b.get(key, 0.0) for key in keys)
    norm_a = sum(value * value for value in vector_a.values())
    norm_b = sum(value * value for value in vector_b.values())
    denom = (norm_a * norm_b) ** 0.5
    if denom == 0:
        return 0.0
    return float(dot / denom)
