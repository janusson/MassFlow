import pytest

from MassFlow.cheminformatics import (
    _get_morgan_fingerprint,
    calculate_isotopic_envelope,
    calculate_isotopic_similarity,
    calculate_tanimoto_similarity,
    calculate_theoretical_mass,
    get_isotopic_distribution,
)


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


def test_isotopic_envelope_normal_molecule_from_formula():
    """Formula-based isotopic envelope (no RDKit required)."""
    envelope = calculate_isotopic_envelope(formula="C2H4O2")
    assert len(envelope) >= 2
    base_mass, base_abundance = envelope[0]
    assert abs(base_mass - 60.0211) < 0.01
    assert base_abundance == 1.0
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
