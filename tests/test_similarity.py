import numpy as np
import pytest

pytest.importorskip("matchms", reason="Similarity module depends on matchms.")

from matchms import Spectrum

from yogimass.similarity import (
    SpectrumProcessor,
    cosine_similarity,
    modified_cosine_similarity,
    spec2vec_vectorize,
)
from yogimass.similarity.metrics import cosine_from_vectors


def _spectrum(peaks, intensities, **metadata):
    return Spectrum(
        mz=np.asarray(peaks, dtype="float32"),
        intensities=np.asarray(intensities, dtype="float32"),
        metadata=metadata,
    )


def test_cosine_similarity_identical_spectra():
    spectrum = _spectrum([50.0, 75.0], [100.0, 50.0], name="ref")

    score = cosine_similarity(spectrum, spectrum)

    assert score == pytest.approx(1.0, rel=1e-4)


def test_modified_cosine_similarity_identical_spectra():
    spectrum = _spectrum([50.0, 75.0], [100.0, 50.0], name="ref")

    score = modified_cosine_similarity(spectrum, spectrum)

    assert score >= 0.9


def test_spec2vec_vectorization_generates_tokens():
    spectrum = _spectrum([50.0, 50.01], [10.0, 5.0], name="token-test")

    vector = spec2vec_vectorize(spectrum, decimal_places=1)

    assert set(vector) == {"peak@50.0"}
    assert vector["peak@50.0"] > 0


def test_cosine_from_vectors_handles_empty():
    assert cosine_from_vectors({}, {}) == 0.0


def test_processor_filters_and_aligns_to_reference():
    processor = SpectrumProcessor(min_relative_intensity=0.05, align_tolerance=0.05)
    spectrum = _spectrum([50.0, 60.0], [100.0, 1.0], name="query")

    processed = processor.process(spectrum, reference_mz=[50.0])

    assert processed.peaks.mz.tolist() == [50.0]
    assert processed.peaks.intensities.tolist() == [pytest.approx(1.0)]
