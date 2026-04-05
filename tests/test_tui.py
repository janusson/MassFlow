"""
Tests for the unified experimental TUI module and related CLI hooks.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import cli
from MassFlow.tui import (
    Candidate,
    browse_cas_main,
    chemical_similarity_rdkit,
    extract_metadata,
    find_query_spectrum_by_cas,
    fuzzy_text_scores,
    gather_library_summaries,
    normalize_cas,
    parse_args,
    rank_candidates_by_chem_and_spec,
    spectral_cosine_score,
    top_n_peaks,
)


def make_spectrum(
    *,
    spec_id: str,
    precursor_mz: float,
    compound_name: str = "Unknown",
    cas: str | None = None,
    smiles: str | None = None,
    mz_values: list[float] | None = None,
    intensity_values: list[float] | None = None,
) -> Spectrum:
    """
    Create a minimal matchms Spectrum for TUI tests.
    """
    if mz_values is None:
        mz_values = [precursor_mz - 20.0, precursor_mz - 10.0, precursor_mz]
    mz_array = np.array(sorted(mz_values), dtype=np.float64)
    intensity_array = np.array(intensity_values or [0.2, 0.4, 1.0], dtype=np.float64)
    metadata = {
        "id": spec_id,
        "compound_name": compound_name,
        "precursor_mz": precursor_mz,
    }
    if cas is not None:
        metadata["cas"] = cas
    if smiles is not None:
        metadata["smiles"] = smiles
    return Spectrum(mz=mz_array, intensities=intensity_array, metadata=metadata)


def test_normalize_cas_removes_hyphens_and_whitespace():
    """CAS normalization should ignore common formatting differences."""
    assert normalize_cas(" 50-00-0 ") == "50000"
    assert normalize_cas("50 00 0") == "50000"
    assert normalize_cas("") is None
    assert normalize_cas(None) is None


def test_extract_metadata_returns_expected_fields():
    """Metadata extraction should pull the common fields used by the inspector."""
    spectrum = make_spectrum(
        spec_id="spec_1",
        precursor_mz=180.063,
        compound_name="Caffeine",
        cas="58-08-2",
        smiles="Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
    )

    compound_name, cas_value, smiles_value, precursor_mz = extract_metadata(spectrum)

    assert compound_name == "Caffeine"
    assert cas_value == "58-08-2"
    assert smiles_value == "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"
    assert precursor_mz == pytest.approx(180.063)


def test_fuzzy_text_scores_prefers_closer_match():
    """Fuzzy textual fallback should rank the most similar text highest."""
    scores = fuzzy_text_scores("caffeine", ["caffeine", "theobromine", "aspirin"])

    assert scores[0] == pytest.approx(1.0)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_spectral_cosine_score_returns_one_for_identical_spectra():
    """The lightweight cosine helper should return ~1 for identical spectra."""
    spectrum_a = make_spectrum(
        spec_id="a",
        precursor_mz=200.0,
        mz_values=[100.0, 150.0, 200.0],
        intensity_values=[0.2, 0.4, 1.0],
    )
    spectrum_b = make_spectrum(
        spec_id="b",
        precursor_mz=200.0,
        mz_values=[100.0, 150.0, 200.0],
        intensity_values=[0.2, 0.4, 1.0],
    )

    score = spectral_cosine_score(spectrum_a, spectrum_b, mz_tolerance=0.01)

    assert score == pytest.approx(1.0)


def test_top_n_peaks_returns_highest_intensity_peaks_first():
    """Top-N peak extraction should sort by descending intensity."""
    spectrum = make_spectrum(
        spec_id="spec_1",
        precursor_mz=250.0,
        mz_values=[100.0, 150.0, 200.0],
        intensity_values=[0.1, 0.9, 0.5],
    )

    peaks = top_n_peaks(spectrum, n=2)

    assert peaks == [(150.0, 0.9), (200.0, 0.5)]


def test_gather_library_summaries_collects_names_cas_and_smiles():
    """Library summary extraction should preserve ordering across fields."""
    spectra = [
        make_spectrum(
            spec_id="spec_a",
            precursor_mz=100.0,
            compound_name="A",
            cas="50-00-0",
            smiles="CCO",
        ),
        make_spectrum(
            spec_id="spec_b",
            precursor_mz=110.0,
            compound_name="B",
            cas="64-17-5",
            smiles="CCN",
        ),
    ]

    names, cas_values, smiles_values = gather_library_summaries(spectra)

    assert names == ["A", "B"]
    assert cas_values == ["50-00-0", "64-17-5"]
    assert smiles_values == ["CCO", "CCN"]


def test_find_query_spectrum_by_cas_returns_matching_index():
    """CAS lookup should return the index of the first normalized match."""
    spectra = [
        make_spectrum(spec_id="spec_a", precursor_mz=100.0, cas="64-17-5"),
        make_spectrum(spec_id="spec_b", precursor_mz=110.0, cas="50-00-0"),
    ]

    assert find_query_spectrum_by_cas("50 00 0", spectra) == 1
    assert find_query_spectrum_by_cas("999-99-9", spectra) is None


def test_rank_candidates_by_chem_and_spec_returns_ranked_candidates():
    """Ranking should return Candidate objects sorted by combined score."""
    query_spectrum = make_spectrum(
        spec_id="query",
        precursor_mz=150.0,
        compound_name="Target",
        cas="50-00-0",
        smiles=None,
        mz_values=[100.0, 120.0, 150.0],
        intensity_values=[0.2, 0.5, 1.0],
    )
    similar_spectrum = make_spectrum(
        spec_id="similar",
        precursor_mz=150.0,
        compound_name="Target Analog",
        cas="50-00-1",
        smiles=None,
        mz_values=[100.0, 120.0, 150.0],
        intensity_values=[0.2, 0.5, 1.0],
    )
    unrelated_spectrum = make_spectrum(
        spec_id="other",
        precursor_mz=300.0,
        compound_name="Unrelated",
        cas="99-99-9",
        smiles=None,
        mz_values=[300.0, 320.0, 350.0],
        intensity_values=[1.0, 0.4, 0.2],
    )

    candidates = rank_candidates_by_chem_and_spec(
        [query_spectrum, similar_spectrum, unrelated_spectrum],
        cas_query="50-00-0",
        top_k=3,
        spec_weight=0.7,
        chem_weight=0.3,
        mz_tolerance=0.01,
    )

    assert len(candidates) == 3
    assert all(isinstance(candidate, Candidate) for candidate in candidates)
    assert candidates[0].combined_score >= candidates[1].combined_score
    assert candidates[1].combined_score >= candidates[2].combined_score


def test_parse_args_parses_browse_cas_arguments():
    """The unified TUI parser should expose the browse-cas options."""
    args = parse_args(
        [
            "library.msp",
            "--cas",
            "50-00-0",
            "--top-k",
            "5",
            "--chem-weight",
            "0.6",
            "--spec-weight",
            "0.4",
            "--mz-tol",
            "0.02",
        ]
    )

    assert args.library == "library.msp"
    assert args.cas == "50-00-0"
    assert args.top_k == 5
    assert args.chem_weight == pytest.approx(0.6)
    assert args.spec_weight == pytest.approx(0.4)
    assert args.mz_tol == pytest.approx(0.02)


def test_browse_cas_main_returns_success_after_user_selection(capsys):
    """browse_cas_main should load spectra, rank candidates, and plot selections."""
    spectra = [
        make_spectrum(
            spec_id="query",
            precursor_mz=150.0,
            compound_name="Target",
            cas="50-00-0",
        ),
        make_spectrum(
            spec_id="candidate",
            precursor_mz=151.0,
            compound_name="Candidate",
            cas="60-00-0",
        ),
    ]

    with (
        patch("MassFlow.tui.io.load_spectra", return_value=iter(spectra)),
        patch("builtins.input", return_value="1"),
        patch("MassFlow.tui.plot_spectrum_terminal") as mock_plot,
    ):
        exit_code = browse_cas_main(["library.msp", "--cas", "50-00-0", "--top-k", "2"])

    assert exit_code == 0
    mock_plot.assert_called_once()
    captured = capsys.readouterr()
    assert "Top candidates" in captured.out


def test_browse_cas_main_returns_nonzero_when_library_load_fails(capsys):
    """browse_cas_main should surface load failures as a nonzero exit code."""
    with patch(
        "MassFlow.tui.io.load_spectra",
        side_effect=ValueError("bad library"),
    ):
        exit_code = browse_cas_main(["library.msp", "--cas", "50-00-0"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Error loading library" in captured.err


def test_chemical_similarity_rdkit_raises_when_rdkit_unavailable(monkeypatch):
    """RDKit helper should fail clearly when the optional dependency is absent."""
    monkeypatch.setattr("MassFlow.tui._HAS_RDKIT", False)

    with pytest.raises(RuntimeError, match="RDKit is not available"):
        chemical_similarity_rdkit("CCO", ["CCO"])


def test_run_browse_uses_unified_tui_module():
    """CLI browse command should call the unified experimental TUI browser."""
    args = argparse.Namespace(file="library.msp")

    with patch("MassFlow.tui.browse_file") as mock_browse_file:
        ret = cli.run_browse(args)

    assert ret == 0
    mock_browse_file.assert_called_once_with("library.msp")


def test_run_browse_cas_uses_unified_tui_browse_cas_entrypoint():
    """CLI browse-cas command should call browse_cas_main from MassFlow.tui."""
    args = argparse.Namespace(
        library="library.msp",
        cas="50-00-0",
        top_k=5,
        chem_weight=0.6,
        spec_weight=0.4,
        mz_tol=0.02,
    )

    with patch("MassFlow.tui.browse_cas_main", return_value=0) as mock_main:
        ret = cli.run_browse_cas(args)

    assert ret == 0
    mock_main.assert_called_once_with(
        [
            "library.msp",
            "--cas",
            "50-00-0",
            "--top-k",
            "5",
            "--chem-weight",
            "0.6",
            "--spec-weight",
            "0.4",
            "--mz-tol",
            "0.02",
        ]
    )


def test_run_browse_cas_handles_import_error():
    """CLI browse-cas command should return 1 when the TUI import fails."""
    args = argparse.Namespace(
        library="library.msp",
        cas="50-00-0",
        top_k=5,
        chem_weight=0.5,
        spec_weight=0.5,
        mz_tol=0.01,
    )

    with (
        patch("MassFlow.cli.logger") as mock_logger,
        patch(
            "MassFlow.tui.browse_cas_main", side_effect=ImportError("tui unavailable")
        ),
    ):
        ret = cli.run_browse_cas(args)

    assert ret == 1
    mock_logger.error.assert_called_once()


def test_run_browse_handles_runtime_failure():
    """CLI browse command should return 1 when the browser raises."""
    args = argparse.Namespace(file="library.msp")

    with (
        patch("MassFlow.cli.logger") as mock_logger,
        patch("MassFlow.tui.browse_file", side_effect=RuntimeError("boom")),
    ):
        ret = cli.run_browse(args)

    assert ret == 1
    mock_logger.error.assert_called_once()


def test_browse_cas_main_returns_zero_when_user_quits(capsys):
    """browse_cas_main should exit cleanly when the user declines selection."""
    spectra = [
        make_spectrum(
            spec_id="query",
            precursor_mz=150.0,
            compound_name="Target",
            cas="50-00-0",
        ),
        make_spectrum(
            spec_id="candidate",
            precursor_mz=151.0,
            compound_name="Candidate",
            cas="60-00-0",
        ),
    ]

    with (
        patch("MassFlow.tui.io.load_spectra", return_value=iter(spectra)),
        patch("builtins.input", return_value="q"),
        patch("MassFlow.tui.plot_spectrum_terminal") as mock_plot,
    ):
        exit_code = browse_cas_main(["library.msp", "--cas", "50-00-0"])

    assert exit_code == 0
    mock_plot.assert_not_called()
    captured = capsys.readouterr()
    assert "No selection made" in captured.out
