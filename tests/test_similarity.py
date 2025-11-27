import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matchms", reason="Similarity module depends on matchms.")

from matchms import Spectrum

from yogimass.similarity.backends import create_index_backend
from yogimass.similarity.embeddings import build_embeddings, embedding_vectorizer
from yogimass.similarity import (
    LibrarySearcher,
    LocalSpectralLibrary,
    SpectrumProcessor,
    batch_process_folder,
    cosine_similarity,
    spec2vec_vectorize,
)


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


def test_spec2vec_vectorization_generates_tokens():
    spectrum = _spectrum([50.0, 50.01], [10.0, 5.0], name="token-test")

    vector = spec2vec_vectorize(spectrum, decimal_places=1)

    assert set(vector) == {"peak@50.0"}
    assert vector["peak@50.0"] > 0


def test_processor_filters_and_aligns_to_reference():
    processor = SpectrumProcessor(min_relative_intensity=0.05, align_tolerance=0.05)
    spectrum = _spectrum([50.0, 60.0], [100.0, 1.0], name="query")

    processed = processor.process(spectrum, reference_mz=[50.0])

    assert processed.peaks.mz.tolist() == [50.0]
    assert processed.peaks.intensities.tolist() == [pytest.approx(1.0)]


def test_local_library_add_and_search(tmp_path):
    library = LocalSpectralLibrary(tmp_path / "library.json")
    processor = SpectrumProcessor()
    searcher = LibrarySearcher(library, processor=processor)
    reference = _spectrum([10, 20], [1.0, 0.5], name="reference")
    decoy = _spectrum([100, 200], [1.0, 0.5], name="decoy")
    library.add_spectrum(reference, identifier="ref")
    library.add_spectrum(decoy, identifier="decoy")

    results = searcher.search_spectrum(reference, top_n=1)

    assert results
    assert results[0].identifier == "ref"
    assert results[0].score > 0.9


def test_local_library_supports_sqlite(tmp_path):
    library = LocalSpectralLibrary(tmp_path / "library.sqlite")
    spectrum = _spectrum([25, 50], [0.6, 0.4], name="sqlite-spectrum")

    library.add_spectrum(spectrum)
    hits = library.search(spec2vec_vectorize(spectrum), top_n=1)

    assert len(library) == 1
    assert hits[0].entry.metadata["name"] == "sqlite-spectrum"


def test_batch_processing_adds_to_library(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    shutil.copy(Path("tests/data/simple.mgf"), input_dir / "simple.mgf")
    library = LocalSpectralLibrary(tmp_path / "batch.json")
    processor = SpectrumProcessor()

    processed = batch_process_folder(input_dir, processor, library=library)

    assert len(processed) == 1
    assert len(library) == 1


def test_naive_backend_queries(tmp_path):
    library = LocalSpectralLibrary(tmp_path / "backend.json")
    spectrum = _spectrum([10, 20], [1.0, 0.5], name="a")
    other = _spectrum([100, 200], [1.0, 0.5], name="b")
    library.add_spectrum(spectrum, identifier="a")
    library.add_spectrum(other, identifier="b")

    backend = create_index_backend("naive", entries=library.iter_entries())
    hits = backend.query(spec2vec_vectorize(spectrum), top_n=1, min_score=0.0)

    assert hits
    assert hits[0].entry.identifier == "a"


def test_annoy_backend_queries_when_available(tmp_path):
    pytest.importorskip("annoy", reason="Annoy backend is optional.")
    library = LocalSpectralLibrary(tmp_path / "annoy.json")
    spectrum = _spectrum([5, 15], [0.8, 0.2], name="x")
    library.add_spectrum(spectrum, identifier="x")

    backend = create_index_backend("annoy", entries=library.iter_entries())
    hits = backend.query(spec2vec_vectorize(spectrum), top_n=1, min_score=0.0)

    assert hits and hits[0].entry.identifier == "x"


def test_embedding_helpers_produce_vectors():
    spectra = [
        _spectrum([50.0, 75.0], [1.0, 0.5], name="embed-a"),
        _spectrum([60.0], [1.0], name="embed-b"),
    ]

    embeddings = build_embeddings("spec2vec-lite", spectra, dimension=8)
    assert embeddings.shape == (2, 8)
    tokens = embedding_vectorizer(spectra[0], dimension=8)
    assert tokens
    assert any(key.startswith("spec2vec-lite") for key in tokens)
