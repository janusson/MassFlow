from MassFlow.consensus import ConsensusEngine
from MassFlow.models import (
    AnnotationHit,
    ConsensusConfig,
    ConsensusInput,
    MolecularStructure,
)


def test_molecular_structure_ethanol_envelope():
    # Ethanol SMILES
    smiles = "CCO"

    # Instantiating the model should auto-populate the isotopic_envelope
    mol = MolecularStructure(smiles=smiles)

    envelope = mol.isotopic_envelope
    assert envelope is not None
    assert len(envelope) >= 2  # At least M and M+1

    # Check M peak (Ethanol exact mass is ~46.041865)
    m_peak = envelope[0]
    assert abs(m_peak[0] - 46.04) < 0.01
    assert m_peak[1] == 1.0  # Base peak must be normalized to 1.0

    # Check M+1 peak (~13C contribution, mostly)
    m1_peak = envelope[1]
    assert abs(m1_peak[0] - 47.04) < 0.01

    # Two carbons -> ~2.2% chance of a 13C
    assert 0.015 < m1_peak[1] < 0.03


def test_consensus_flag_rank_discrepancy():
    # Setup mock hits where rank discrepancy exceeds the threshold
    hits = [
        # Engine A (exact_mass) ranks Ref_X as #1
        AnnotationHit(engine_id="exact_mass", reference_id="Ref_X", score=0.99, rank=1),
        AnnotationHit(engine_id="exact_mass", reference_id="Ref_Y", score=0.80, rank=2),
        # Engine B (ms2deepscore) ranks Ref_X as #5, Ref_Y as #1
        AnnotationHit(
            engine_id="ms2deepscore", reference_id="Ref_Y", score=0.95, rank=1
        ),
        AnnotationHit(
            engine_id="ms2deepscore", reference_id="Ref_X", score=0.40, rank=5
        ),
    ]

    # Configure consensus engine with a low discrepancy threshold (e.g., 3)
    config = ConsensusConfig(
        engine_weights={"exact_mass": 0.5, "ms2deepscore": 0.5},
        flag_rank_discrepancy_threshold=3,
        tie_breaker_strategy="highest_rank",
    )

    engine = ConsensusEngine(config)
    c_input = ConsensusInput(query_id="query_ethanol", hits=hits)

    # Resolve consensus
    result = engine.resolve(c_input)

    # Engine 'exact_mass' ranked Ref_X as #1, but 'ms2deepscore' ranked it as #5.
    # 5 > threshold (3), so orthogonal agreement failure should flag the result.
    assert result.flagged_for_review is True
    assert result.review_reason is not None
    assert "Orthogonal Agreement Failure" in result.review_reason
    assert "was ranked #5 by ms2deepscore" in result.review_reason
