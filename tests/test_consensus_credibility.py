import pytest

from MassFlow.consensus import generate_consensus
from MassFlow.models import (
    AnnotationHit,
    ConsensusConfig,
    ConsensusInput,
    MassFlowSpectrum,
    SpectralPeaks,
    SpectrumMetadata,
)


@pytest.fixture
def base_config():
    return ConsensusConfig(
        engine_weights={"cosine": 1.0},
        isotopic_credibility_weight=0.5,
        penalize_impossible_neutral_losses=True,
        neutral_loss_penalty_factor=0.1,
    )


def test_neutral_loss_penalty(base_config):
    # Candidate smiles has only C and H
    hit = AnnotationHit(
        engine_id="cosine", reference_id="ref_01", score=1.0, rank=1, smiles="CC"
    )

    # Experimental spectrum has a loss of 18.01 (H2O), which is impossible for C2H6
    exp_spec = MassFlowSpectrum(
        metadata=SpectrumMetadata(spectrum_id="query", precursor_mz=100.0),
        peaks=SpectralPeaks(mz_array=[50.0, 81.99], intensity_array=[50.0, 100.0]),
    )

    # Precursor 100 - 81.99 = 18.01. Impossible for "CC" because there is no oxygen.
    c_in = ConsensusInput(
        query_id="query_01", hits=[hit], experimental_spectrum=exp_spec
    )
    res = generate_consensus(c_in, base_config)

    assert res.best_reference_id == "ref_01"
    # Score should be heavily penalized
    assert res.best_consensus_score == pytest.approx(0.1)  # 1.0 * 0.1
    assert res.flagged_for_review is True
    assert "impossible neutral losses detected" in res.review_reason
    assert "H2O" in res.review_reason or "requires {'O'}" in res.review_reason


def test_isotopic_credibility(base_config):
    hit1 = AnnotationHit(
        engine_id="cosine", reference_id="ref_bad", score=1.0, rank=1, smiles="CC"
    )
    hit2 = AnnotationHit(
        engine_id="cosine",
        reference_id="ref_good",
        score=0.9,
        rank=2,
        smiles="c1ccccc1O",
    )  # Phenol

    # Let's mock the experimental isotopic envelope to be exactly like Phenol
    from MassFlow.cheminformatics import calculate_isotopic_envelope

    theor_env = calculate_isotopic_envelope("c1ccccc1O")

    exp_spec = MassFlowSpectrum(
        metadata=SpectrumMetadata(
            spectrum_id="query",
            precursor_mz=94.04,
            experimental_isotopic_envelope=theor_env,
        ),
        peaks=SpectralPeaks(mz_array=[50.0], intensity_array=[100.0]),
    )

    c_in = ConsensusInput(
        query_id="query_01", hits=[hit1, hit2], experimental_spectrum=exp_spec
    )
    res = generate_consensus(c_in, base_config)

    # ref_good: Base score 0.9 * 1.0 = 0.9. Isotopic similarity = 1.0.
    # Total sum = 0.9 + 1.0 * 0.5 = 1.4. Total weight = 1.5. Consensus = 1.4/1.5 = 0.9333
    # ref_bad: Base score 1.0 * 1.0 = 1.0. Isotopic similarity with CC is going to be something else.

    cand_good = next(c for c in res.candidates if c.reference_id == "ref_good")
    cand_bad = next(c for c in res.candidates if c.reference_id == "ref_bad")

    assert cand_good.consensus_score == pytest.approx(1.4 / 1.5)
    assert cand_bad.consensus_score < cand_good.consensus_score
