"""Experimental consensus must not combine evidence from distinct acquisitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import io
from MassFlow.config import ProcessingConfig, SimilarityConfig
from MassFlow.library import RawFileLibraryStore
from MassFlow.similarity import CascadeEngine, ConsensusEngine
from tests.test_spectral_record_provenance import _records

pytestmark = pytest.mark.scientific  # Consensus remains outside the core contract.


@pytest.mark.parametrize("algorithm", ["consensus", "cascade"])
@pytest.mark.parametrize("identifier", [None, "duplicate", ""])
def test_meta_engines_reject_ambiguous_reference_ids(
    algorithm: str, identifier: str | None
) -> None:
    """Direct callers cannot aggregate or route distinct records under one ID."""
    references = _records((identifier, identifier))
    config = SimilarityConfig(algorithm=algorithm, min_score=0.0)
    engine = (
        ConsensusEngine(config) if algorithm == "consensus" else CascadeEngine(config)
    )
    with pytest.raises(ValueError, match="unique.*id"):
        engine.search([references[0]], references, include_decoys=False)


@pytest.mark.parametrize("algorithm", ["consensus", "cascade"])
def test_meta_engines_reject_generated_decoy_id_collision(algorithm: str) -> None:
    """A real target cannot share an aggregation/routing key with a decoy."""
    references = _records(("x", "x_decoy"))
    config = SimilarityConfig(algorithm=algorithm, min_score=0.0, min_matched_peaks=0)
    engine = (
        ConsensusEngine(config) if algorithm == "consensus" else CascadeEngine(config)
    )
    with pytest.raises(ValueError, match="generated.*decoy.*id"):
        engine.search([references[0]], references, include_decoys=True)


def test_consensus_keeps_scores_with_their_source_record(tmp_path: Path) -> None:
    """Same-structure records must not share a consensus aggregation bucket."""
    originals = _records((None, None))
    first = Spectrum(
        mz=originals[0].peaks.mz,
        intensities=np.array([1.0, 1.0, 0.5], dtype=np.float64),
        metadata=originals[0].metadata,
    )
    second = Spectrum(
        mz=originals[1].peaks.mz,
        intensities=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        metadata=originals[1].metadata,
    )
    source = tmp_path / "library.msp"
    io.save_spectra_to_msp([first, second], source)
    references = list(RawFileLibraryStore(source, ProcessingConfig()).get_spectra())
    query = Spectrum(
        mz=first.peaks.mz,
        intensities=np.ones(3, dtype=np.float64),
        metadata={**first.metadata, "id": "query"},
    )
    engine = ConsensusEngine(
        SimilarityConfig(
            algorithm="consensus",
            consensus_weights={"cosine": 1.0, "modified_cosine": 1.0},
            min_score=0.0,
            min_matched_peaks=3,
        )
    )
    results = engine.search([query], references, include_decoys=False)
    assert len(results) == 2
    by_precursor = {hit["reference_precursor_mz"]: hit for hit in results}
    # H+ record: both engines give 2.5/sqrt(3*2.25). Na+ record:
    # cosine is precursor-gated; modified cosine gives 1, hence mean 0.5.
    assert by_precursor[first.get("precursor_mz")]["score"] == pytest.approx(
        0.9622504486493763
    )
    assert by_precursor[second.get("precursor_mz")]["score"] == pytest.approx(0.5)
    assert len({hit["reference_id"] for hit in results}) == 2
