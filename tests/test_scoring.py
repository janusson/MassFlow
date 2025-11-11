import numpy as np
import pytest

pytest.importorskip("matchms", reason="Scoring tests depend on matchms.")

from matchms import Spectrum

from yogimass.scoring import cosine


def _make_spectrum(peaks, intensities, **metadata):
    return Spectrum(
        mz=np.array(peaks, dtype="float32"),
        intensities=np.array(intensities, dtype="float32"),
        metadata=metadata,
    )


def test_top10_cosine_matches_returns_smiles():
    reference = [
        _make_spectrum([50, 100], [0.7, 0.3], name="ref1", smiles="C"),
        _make_spectrum([60, 120], [0.6, 0.4], name="ref2", smiles="CC"),
    ]
    query = [_make_spectrum([50, 100], [0.7, 0.3], name="query1")]

    smiles = cosine.top10_cosine_matches(reference, query)

    assert smiles[0] == "C"
    assert len(smiles) <= 2


def test_threshold_matches_filters_on_minimum_matches():
    reference = [
        _make_spectrum([50, 100], [0.7, 0.3], name="ref1", smiles="C"),
        _make_spectrum([10, 20], [0.2, 0.8], name="ref2", smiles="CCC"),
    ]
    query = [_make_spectrum([50, 100], [0.7, 0.3], name="query1")]

    matches, smiles = cosine.threshold_matches(reference, query, min_match=2)

    assert len(matches) == 1
    reference_match, score = matches[0]
    assert reference_match.get("name") == "ref1"
    assert score.matches >= 2
    assert smiles == ["C"]


def test_threshold_matches_rejects_out_of_bounds_index():
    reference = [_make_spectrum([50], [1.0], name="ref")]
    query = [_make_spectrum([50], [1.0], name="query")]

    try:
        cosine.threshold_matches(reference, query, min_match=1, query_index=5)
    except IndexError:
        pass
    else:
        raise AssertionError("Expected IndexError for invalid query index")
