import pytest

from MassFlow.cheminformatics import (
    COMMON_NEUTRAL_LOSSES,
    _get_morgan_fingerprint,
    calculate_isotopic_envelope,
    calculate_isotopic_similarity,
    calculate_tanimoto_similarity,
    calculate_theoretical_mass,
    find_impossible_neutral_losses,
    get_isotopic_distribution,
    parse_elements_from_smiles,
)

# NIST monoisotopic element masses (Da), inlined here as literals so this test is an
# independent first-principles cross-check of COMMON_NEUTRAL_LOSSES. cheminformatics
# now derives those masses from pyteomics; defining the expected values from a
# separate source keeps the regression test from becoming circular.
H_MASS = 1.0078250322
C_MASS = 12.0000000000
N_MASS = 14.0030740044
O_MASS = 15.9949146196
S_MASS = 31.9720711374
P_MASS = 30.9737616320
F_MASS = 18.9984031627
CL_MASS = 34.96885268
BR_MASS = 78.9183371000


def test_tanimoto_similarity():
    # Aspirin
    smiles1 = "CC(=O)OC1=CC=CC=C1C(=O)O"
    # Salicylic Acid (structurally similar, differs by acetyl group)
    smiles2 = "C1=CC=C(C(=C1)C(=O)O)O"

    score = calculate_tanimoto_similarity(smiles1, smiles2)
    assert score is not None
    assert 0.4 < score < 1.0  # They share a significant substructure


def test_tanimoto_invalid_smiles():
    score = calculate_tanimoto_similarity("INVALID_SMILES", "C")
    assert score is None


def test_theoretical_mass_protonated():
    # Caffeine: C8H10N4O2
    # Exact mass is ~194.08038
    # Protonated [M+H]+ should be ~195.087
    mass = calculate_theoretical_mass("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "[M+H]+")
    assert mass is not None
    assert abs(mass - 195.08766) < 0.001


def test_theoretical_mass_sodiated():
    # Caffeine [M+Na]+ should be ~217.069
    mass = calculate_theoretical_mass("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "[M+Na]+")
    assert mass is not None
    assert abs(mass - 217.0696) < 0.001


def test_theoretical_mass_doubly_protonated():
    # Caffeine [M+2H]2+: neutral mass ~194.08038, add 2 protons and divide by 2.
    # (194.08038 + 2 × 1.00727646) / 2 = ~98.04746 m/z.
    # This exercises the abs(charge) division for multiply-charged adducts.
    mass = calculate_theoretical_mass("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "[M+2H]2+")
    assert mass is not None
    assert abs(mass - 98.04746) < 0.001
    # Sanity: the doubly-charged m/z must be roughly half the protonated [M+H]+ m/z.
    singly = calculate_theoretical_mass("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "[M+H]+")
    assert mass < singly


def test_theoretical_mass_invalid_adduct():
    with pytest.raises(ValueError, match="is not supported"):
        calculate_theoretical_mass("C", "[M+UNKNOWN]+")


# First-principles expected masses for every entry in COMMON_NEUTRAL_LOSSES,
# computed from the NIST element literals defined above in this test module.
# Table index must stay in sync with the declaration order in cheminformatics.py.
_NEUTRAL_LOSS_FIRST_PRINCIPLES = [
    ("H2O", 0, 2 * H_MASS + 1 * O_MASS),
    ("NH3", 1, 1 * N_MASS + 3 * H_MASS),
    ("CO", 2, 1 * C_MASS + 1 * O_MASS),
    ("CO2", 3, 1 * C_MASS + 2 * O_MASS),
    ("H2S", 4, 2 * H_MASS + 1 * S_MASS),
    ("SO2", 5, 1 * S_MASS + 2 * O_MASS),
    ("PO3", 6, 1 * P_MASS + 3 * O_MASS),
    ("HCl", 7, 1 * H_MASS + 1 * CL_MASS),
    ("HBr", 8, 1 * H_MASS + 1 * BR_MASS),
    ("HF", 9, 1 * H_MASS + 1 * F_MASS),
]


@pytest.mark.parametrize(
    "name,table_index,expected_mass", _NEUTRAL_LOSS_FIRST_PRINCIPLES
)
def test_neutral_loss_mass_first_principles(name, table_index, expected_mass):
    """
    Regression test: every COMMON_NEUTRAL_LOSSES mass must agree with a
    first-principles calculation (element exact masses from NIST) to within
    0.5 mDa.  A failure here means the tabulated value is wrong for the named
    neutral loss fragment.
    """
    recorded_mass = COMMON_NEUTRAL_LOSSES[table_index][0]
    deviation_da = abs(recorded_mass - expected_mass)
    assert deviation_da < 5e-4, (
        f"{name}: recorded mass {recorded_mass:.4f} differs from "
        f"first-principles {expected_mass:.4f} by {deviation_da * 1000:.2f} mDa"
    )


def test_co2_loss_on_one_oxygen_molecule_is_impossible():
    """CO₂ loss requires ≥ 2 oxygen atoms; methanol (1 O) must be flagged."""
    # methanol CH4O: precursor 100, fragment at 56.0102 → NL = 43.9898 (CO2)
    precursor_mz = 100.0
    fragment_mz = precursor_mz - 43.9898
    losses = find_impossible_neutral_losses(
        mz_array=[fragment_mz],
        int_array=[100.0],
        precursor_mz=precursor_mz,
        smiles="CO",  # methanol: C1H4O1
    )
    assert len(losses) == 1
    nl, exact, req = losses[0]
    assert abs(nl - 43.9898) < 0.01
    assert req == {"C": 1, "O": 2}


def test_co2_loss_on_three_oxygen_glycerol_is_possible():
    """CO₂ loss on glycerol (3 O) must NOT be flagged — 3 ≥ 2 required."""
    precursor_mz = 100.0
    fragment_mz = precursor_mz - 43.9898
    losses = find_impossible_neutral_losses(
        mz_array=[fragment_mz],
        int_array=[100.0],
        precursor_mz=precursor_mz,
        smiles="OCC(O)CO",  # glycerol: C3H8O3
    )
    assert losses == []


def test_po3_loss_on_one_oxygen_phospho_compound_is_impossible():
    """PO₃ loss requires ≥ 3 oxygen atoms; trimethylphosphine oxide (1 O) must be flagged."""
    # trimethylphosphine oxide: O=P(C)(C)C → C3H9OP (1 O, 1 P)
    precursor_mz = 150.0
    fragment_mz = precursor_mz - 78.9585  # PO3 mass
    losses = find_impossible_neutral_losses(
        mz_array=[fragment_mz],
        int_array=[100.0],
        precursor_mz=precursor_mz,
        smiles="O=P(C)(C)C",  # 1 O, 1 P → PO3 impossible
    )
    assert len(losses) == 1
    _, _, req = losses[0]
    assert req == {"P": 1, "O": 3}


def test_h2o_loss_on_oxygen_containing_molecule_is_possible():
    """Regression: H₂O loss on a molecule with O must not be flagged."""
    # ethanol CCO: C2H6O → H2O requires H ≥ 2, O ≥ 1 → possible
    precursor_mz = 100.0
    fragment_mz = precursor_mz - 18.0106
    losses = find_impossible_neutral_losses(
        mz_array=[fragment_mz],
        int_array=[100.0],
        precursor_mz=precursor_mz,
        smiles="CCO",  # ethanol: C2H6O
    )
    assert losses == []


def test_parse_elements_from_smiles_returns_counts():
    """parse_elements_from_smiles must return element counts, not just a set."""
    counts = parse_elements_from_smiles("OCC(O)CO")  # glycerol C3H8O3
    assert counts["C"] == 3
    assert counts["O"] == 3
    assert counts["H"] == 8


def test_caching_behavior():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    # Run once to populate cache
    _get_morgan_fingerprint(smiles)

    # Check cache info
    info = _get_morgan_fingerprint.cache_info()
    assert info.hits == 0
    assert info.misses >= 1

    # Run again, should hit cache
    _get_morgan_fingerprint(smiles)
    info2 = _get_morgan_fingerprint.cache_info()
    assert info2.hits == 1


# --- Invalid-input guards on the public mass/structure functions ---------------
# These exercise the "bad SMILES / empty input" branches that protect the pipeline
# from crashing on malformed library or query records.


def test_tanimoto_empty_string_returns_none():
    """An empty SMILES short-circuits to None before fingerprinting."""
    assert calculate_tanimoto_similarity("", "C") is None


def test_theoretical_mass_invalid_smiles_returns_none():
    """A SMILES RDKit cannot parse yields None rather than raising."""
    assert calculate_theoretical_mass("not_a_molecule", "[M+H]+") is None


def test_parse_elements_invalid_smiles_returns_empty_counter():
    """Unparseable SMILES returns an empty Counter, not an exception."""
    counts = parse_elements_from_smiles("???")
    assert counts == {}


def test_get_isotopic_distribution_invalid_smiles_returns_empty():
    """Invalid SMILES yields an empty distribution."""
    assert get_isotopic_distribution("not_a_molecule") == []


def test_isotopic_envelope_invalid_smiles_returns_empty():
    """Invalid SMILES yields an empty isotopic envelope."""
    assert calculate_isotopic_envelope("not_a_molecule") == []


# --- Isotopic envelope: normal and isotope-labelled structures -----------------


def test_isotopic_envelope_normal_molecule():
    """
    A normal (unlabelled) molecule returns a base peak at relative abundance 1.0
    followed by decreasing-abundance M+1, M+2 satellites.
    """
    # Acetic acid C2H4O2, monoisotopic ~60.0211
    envelope = calculate_isotopic_envelope("CC(=O)O")
    assert len(envelope) >= 2
    base_mass, base_abundance = envelope[0]
    assert abs(base_mass - 60.0211) < 0.01
    assert base_abundance == 1.0
    # Satellite peaks must be less abundant than the monoisotopic base peak.
    assert all(abundance <= 1.0 for _, abundance in envelope)


def test_isotopic_envelope_isotope_labelled_smiles_returns_valid_envelope():
    """
    An isotope-labelled SMILES drives the isotope-aware formula-construction
    branch (explicit isotope labels overlaid on the base formula). The branch
    must return a well-formed, normalised envelope: a non-empty list, base peak
    normalised to relative abundance 1.0, ascending masses, and no more than
    ``max_isopeaks`` entries.

    Note: this asserts only the structural contract of the envelope. The exact
    mass shift from the ¹³C label is intentionally not asserted here because
    pyteomics' ``isotopologues`` expansion does not honour a pinned isotope in
    the composition (see the isotope-handling caveat in cheminformatics).
    """
    labelled = calculate_isotopic_envelope("[13CH3]C(=O)O", max_isopeaks=4)

    assert labelled, "Isotope-labelled envelope unexpectedly empty"
    assert len(labelled) <= 4
    masses = [mass for mass, _ in labelled]
    assert masses == sorted(masses), "Envelope peaks must be sorted by ascending mass"
    assert labelled[0][1] == 1.0, "Base peak must be normalised to abundance 1.0"


# --- Isotopic envelope cosine similarity --------------------------------------


def test_isotopic_similarity_identical_envelopes_scores_one():
    """Two identical envelopes are perfectly correlated (cosine == 1.0)."""
    envelope = [(100.0, 1.0), (101.0, 0.5), (102.0, 0.1)]
    score = calculate_isotopic_similarity(envelope, envelope)
    assert abs(score - 1.0) < 1e-9


def test_isotopic_similarity_empty_input_scores_zero():
    """An empty experimental or theoretical envelope scores 0.0."""
    assert calculate_isotopic_similarity([], [(100.0, 1.0)]) == 0.0
    assert calculate_isotopic_similarity([(100.0, 1.0)], []) == 0.0


def test_isotopic_similarity_zero_intensity_scores_zero():
    """
    Envelopes whose peaks align in m/z but carry zero intensity have zero norm,
    so the cosine is defined as 0.0 rather than dividing by zero.
    """
    zero_env = [(100.0, 0.0), (101.0, 0.0)]
    assert calculate_isotopic_similarity(zero_env, zero_env) == 0.0


# --- find_impossible_neutral_losses input guards -------------------------------


def test_impossible_losses_invalid_smiles_returns_empty():
    """An unparseable candidate SMILES cannot be evaluated, so nothing is flagged."""
    losses = find_impossible_neutral_losses(
        mz_array=[50.0],
        int_array=[100.0],
        precursor_mz=100.0,
        smiles="not_a_molecule",
    )
    assert losses == []


def test_impossible_losses_all_zero_intensity_returns_empty():
    """If every fragment has zero intensity there is no signal to evaluate."""
    losses = find_impossible_neutral_losses(
        mz_array=[56.0102],
        int_array=[0.0],
        precursor_mz=100.0,
        smiles="CO",
    )
    assert losses == []


def test_impossible_losses_fragment_above_precursor_is_skipped():
    """A fragment heavier than the precursor implies a negative loss and is ignored."""
    losses = find_impossible_neutral_losses(
        mz_array=[150.0],  # heavier than the 100 Da precursor
        int_array=[100.0],
        precursor_mz=100.0,
        smiles="CO",
    )
    assert losses == []


def test_impossible_losses_below_intensity_threshold_is_skipped():
    """A peak below the relative-intensity threshold is not considered a real loss."""
    precursor_mz = 100.0
    co2_fragment = precursor_mz - 43.9898  # would be a CO2 loss if intense enough
    losses = find_impossible_neutral_losses(
        mz_array=[co2_fragment, 90.0],
        int_array=[1.0, 100.0],  # CO2 fragment is 1% of base peak, below 5% default
        precursor_mz=precursor_mz,
        smiles="CO",
    )
    assert losses == []
